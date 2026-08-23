#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# gsplat_renderer_node.py
#
# High-performance, multi-threaded ROS 2 node for real-time Gaussian Splatting
# rendering on an NVIDIA RTX 4090.
#
# Architecture (Producer–Consumer, 3 threads):
#   Thread 1 — Pose Subscriber : Receives /robot/fused_pose, stores latest
#                                 pose in a lock-protected variable.
#   Thread 2 — Renderer Loop   : Continuously rasterises the gsplat scene
#                                 from the latest pose, entirely in VRAM.
#   Thread 3 — Publisher Loop   : Pops rendered frames from a bounded queue
#                                 and publishes as sensor_msgs/Image.
#
# Coordinate Transform (ROS REP 103 → gsplat graphics):
#   ROS  : X-forward, Y-left,  Z-up
#   gsplat: X-right,   Y-down,  Z-forward  (OpenGL-style)
#
#   W2C = (T_robot_in_gsplat × T_ros_to_graphics)⁻¹
# ─────────────────────────────────────────────────────────────────────────────

import queue
import threading
import time
from typing import Optional

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# gsplat high-level rasterization API
from gsplat import rasterization


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate frame alignment: Virtual Camera (Graphics) → Tag (OpenCV)
#
# Virtual Camera: X-right, Y-down, Z-forward
# Tag Surface:    X-right, Y-forward, Z-up (normal to tag)
#
# We map:
#   Camera Z (fwd)  → Tag Y (fwd)
#   Camera Y (down) → Tag -Z (down)
#   Camera X (right)→ Tag X (right)
# And add a 0.2m Z-offset to elevate the camera above the floor.
# ─────────────────────────────────────────────────────────────────────────────
T_CAM_TO_TAG = torch.tensor([
    [1.0,  0.0,  0.0, 0.0],
    [0.0,  0.0,  1.0, 0.0],
    [0.0, -1.0,  0.0, 0.2],
    [0.0,  0.0,  0.0, 1.0],
], dtype=torch.float32)


class GsplatRendererNode(Node):
    """Real-time Gaussian Splatting renderer driven by fused robot pose."""

    def __init__(self):
        super().__init__("gsplat_renderer")

        # ── Declare parameters ───────────────────────────────────────────
        self.declare_parameter("model_checkpoint", "/workspace/data/model.ckpt")
        # Quaternion component order as stored in the checkpoint.
        # gsplat requires (w, x, y, z); many exporters write (x, y, z, w).
        self.declare_parameter("quat_order", "xyzw")
        self.declare_parameter("render_width", 1280)
        self.declare_parameter("render_height", 720)
        # Virtual camera intrinsics (pinhole)
        self.declare_parameter("fx", 600.0)
        self.declare_parameter("fy", 600.0)
        self.declare_parameter("cx", 640.0)
        self.declare_parameter("cy", 360.0)

        self.width = self.get_parameter("render_width").value
        self.height = self.get_parameter("render_height").value

        # ── CUDA device ──────────────────────────────────────────────────
        assert torch.cuda.is_available(), "CUDA is required for gsplat rendering"
        self.device = torch.device("cuda:0")
        self.get_logger().info(
            f"Using GPU: {torch.cuda.get_device_name(self.device)}"
        )

        # ── Build virtual camera intrinsics matrix K (3×3) ──────────────
        fx = self.get_parameter("fx").value
        fy = self.get_parameter("fy").value
        cx = self.get_parameter("cx").value
        cy = self.get_parameter("cy").value

        self.K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 3, 3]

        # ── Load gsplat model ────────────────────────────────────────────
        self._load_model()

        # ── Thread-safe latest pose ──────────────────────────────────────
        self._pose_lock = threading.Lock()
        self._latest_pose: Optional[torch.Tensor] = None  # 4×4 SE3

        # ── Bounded frame queue (drop-oldest policy via maxsize) ─────────
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # ── ROS 2 QoS: Reliable ─────────────────────────────────────────────
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Thread 1: Pose subscriber ───────────────────────────────────
        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/robot/fused_pose",
            self._pose_callback,
            qos,
        )

        # ── Image publisher ──────────────────────────────────────────────
        self.image_pub = self.create_publisher(Image, "/gsplat/raw_image", qos)
        self.bridge = CvBridge()

        # ── Thread 2: Renderer loop ─────────────────────────────────────
        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name="gsplat_render"
        )
        self._render_thread.start()

        # ── Thread 3: Publisher loop ────────────────────────────────────
        self._publish_thread = threading.Thread(
            target=self._publish_loop, daemon=True, name="gsplat_publish"
        )
        self._publish_thread.start()

        self.get_logger().info(
            f"GsplatRenderer started — {self.width}×{self.height} @ VRAM-only pipeline"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────
    def _load_model(self):
        """Load pre-trained gsplat model checkpoint into GPU memory."""
        ckpt_path = self.get_parameter("model_checkpoint").value
        self.get_logger().info(f"Loading gsplat model from {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device,
                                weights_only=False)

        # Ensure we know the base number of Gaussians
        raw_means = checkpoint["means"]
        N = raw_means.numel() // 3

        # Extract Gaussian parameters — enforce exact shapes, float32, and contiguity
        self.means = raw_means.to(self.device).float().contiguous().view(N, 3)
        
        # Quaternions must be [N, 4], in (w, x, y, z) order, and unit-length.
        # NOTE: a misordered quaternion still normalises to a valid unit
        # quaternion, so this cannot be auto-detected — it must be declared.
        # Getting it wrong rotates every Gaussian arbitrarily, which renders
        # as a uniform haze with the scene structure still recognisable.
        quats = checkpoint["quats"].to(self.device).float().contiguous().view(N, 4)
        quat_order = self.get_parameter("quat_order").value
        if quat_order == "xyzw":
            quats = torch.roll(quats, shifts=1, dims=-1)
        elif quat_order != "wxyz":
            raise ValueError(
                f"quat_order must be 'wxyz' or 'xyzw', got {quat_order!r}"
            )
        self.quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
        
        scales = checkpoint["scales"].to(self.device).float().contiguous().view(N, 3)
        # Auto-detect if scales are log-scales (log-scales typically have negative values)
        if scales.min() < 0.0:
            self.scales = torch.exp(scales)
        else:
            self.scales = scales
        
        opacities = checkpoint["opacities"].to(self.device).float().contiguous().view(N)
        # Auto-detect if opacities are logits (logits go outside [0, 1])
        if opacities.min() < 0.0 or opacities.max() > 1.0:
            self.opacities = torch.sigmoid(opacities)
        else:
            self.opacities = opacities

        # Colors: support both raw RGB and spherical harmonics
        self.sh_degree = None
        if "colors" in checkpoint:
            colors = checkpoint["colors"].to(self.device).float()
            if colors.numel() == N * 3:
                self.colors = colors.contiguous().view(N, 3)
            else:
                K = colors.numel() // (N * 3)
                if colors.dim() == 3 and colors.shape[1] == 3 and colors.shape[2] == K:
                    colors = colors.transpose(1, 2)
                self.colors = colors.contiguous().view(N, K, 3)
                import math
                self.sh_degree = int(math.sqrt(K)) - 1
        elif "sh_coefficients" in checkpoint:
            colors = checkpoint["sh_coefficients"].to(self.device).float()
            K = colors.numel() // (N * 3)
            if colors.dim() == 3 and colors.shape[1] == 3 and colors.shape[2] == K:
                colors = colors.transpose(1, 2)
            self.colors = colors.contiguous().view(N, K, 3)
            import math
            self.sh_degree = int(math.sqrt(K)) - 1
        else:
            raise KeyError(
                "Checkpoint must contain 'colors' or 'sh_coefficients'"
            )

        n_gaussians = self.means.shape[0]
        self.get_logger().info(f"Quaternion order: {quat_order} (gsplat needs wxyz)")
        self.get_logger().info(
            f"Loaded {n_gaussians:,} Gaussians into VRAM "
            f"({self.means.element_size() * self.means.nelement() / 1e6:.1f} MB means)"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Thread 1: Pose subscriber callback
    # ─────────────────────────────────────────────────────────────────────
    def _pose_callback(self, msg: PoseStamped):
        """Store the latest pose as a 4×4 SE3 tensor (on GPU)."""
        p = msg.pose.position
        q = msg.pose.orientation

        # Build SE3 from quaternion + translation
        # quaternion order: (w, x, y, z)
        T = self._quat_pos_to_se3(q.w, q.x, q.y, q.z, p.x, p.y, p.z)

        with self._pose_lock:
            self._latest_pose = T

    @staticmethod
    @torch.no_grad()
    def _quat_pos_to_se3(
        qw: float, qx: float, qy: float, qz: float,
        tx: float, ty: float, tz: float,
    ) -> torch.Tensor:
        """Convert quaternion (w,x,y,z) + translation to a 4×4 SE3 matrix."""
        # Rotation matrix from quaternion (Hamilton convention)
        q = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
        q = q / q.norm()
        w, x, y, z = q

        R = torch.tensor([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=torch.float32)

        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = R
        T[0, 3] = tx
        T[1, 3] = ty
        T[2, 3] = tz

        return T

    # ─────────────────────────────────────────────────────────────────────
    # Thread 2: Renderer loop (runs entirely in VRAM)
    # ─────────────────────────────────────────────────────────────────────
    def _render_loop(self):
        """Infinite loop: grab pose → rasterise → enqueue frame (throttled to 60 FPS)."""
        self.get_logger().info("Render thread started (60 FPS cap)")
        target_interval = 1.0 / 60.0  # Cap at 60 FPS
        last_render_time = time.perf_counter()

        while rclpy.ok():
            # ── Frame rate governor (60 FPS max) ───────────────────────────
            now = time.perf_counter()
            elapsed = now - last_render_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            last_render_time = time.perf_counter()

            # ── Grab the latest pose ─────────────────────────────────────
            with self._pose_lock:
                pose_T = self._latest_pose

            if pose_T is None:
                continue

            try:
                frame = self._rasterise(pose_T)
                # Non-blocking put: if queue is full, drop the oldest frame
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()  # Drop oldest
                    except queue.Empty:
                        pass
                    self._frame_queue.put_nowait(frame)

            except Exception as e:
                self.get_logger().error(f"Render error: {e}", throttle_duration_sec=1.0)

    @torch.no_grad()
    def _rasterise(self, pose_T: torch.Tensor) -> np.ndarray:
        """
        Run the full gsplat rasterization pipeline on the GPU.

        Coordinate math:
          T_tag_to_world = pose_T (tag pose in OpenCV world frame)
          T_cam_to_tag = alignment + 0.2m elevation matrix
          C2W = T_tag_to_world @ T_cam_to_tag
          W2C = C2W⁻¹

        gsplat.rasterization() expects viewmats as W2C [C, 4, 4].
        """
        # ── Move pose to GPU if not already there ────────────────────────
        pose_gpu = pose_T.to(self.device)
        T_cam_to_tag_gpu = T_CAM_TO_TAG.to(self.device)

        # ── Compute camera-to-world in gsplat coordinate frame ───────────
        #    C2W = T_tag_to_world × T_cam_to_tag
        C2W = pose_gpu @ T_cam_to_tag_gpu

        # ── World-to-camera = inverse of C2W ─────────────────────────────
        W2C = torch.inverse(C2W)

        # ── Batch dimension [1, 4, 4] for gsplat API ─────────────────────
        viewmats = W2C.unsqueeze(0).contiguous()

        # ── Rasterise ────────────────────────────────────────────────────
        rasterization_kwargs = {
            "means": self.means,
            "quats": self.quats,
            "scales": self.scales,
            "opacities": self.opacities,
            "colors": self.colors,
            "viewmats": viewmats,
            "Ks": self.K.contiguous(),
            "width": self.width,
            "height": self.height,
            "absgrad": False,
        }
        if self.sh_degree is not None:
            rasterization_kwargs["sh_degree"] = self.sh_degree
            
        rendered, _, _ = rasterization(**rasterization_kwargs)

        # rendered shape: [1, H, W, 3] (float32)
        # NOTE: gsplat.rasterization() evaluates SH internally (including DC bias).
        # Do NOT add a manual +0.5 offset here — it would over-brighten the image.

        # ── Convert to uint8 — still on GPU ──────────────────────────────
        frame_gpu = (rendered[0].clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        # ── Transfer to CPU as NumPy array ───────────────────────────────
        frame_np = frame_gpu.cpu().numpy()  # shape: [H, W, 3], dtype: uint8

        return frame_np

    # ─────────────────────────────────────────────────────────────────────
    # Thread 3: Publisher loop
    # ─────────────────────────────────────────────────────────────────────
    def _publish_loop(self):
        """Pop rendered frames from the queue and publish as ROS Image."""
        self.get_logger().info("Publisher thread started")

        while rclpy.ok():
            try:
                frame: np.ndarray = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # ── Build sensor_msgs/Image ──────────────────────────────────
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "gsplat_virtual_camera"

            self.image_pub.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = GsplatRendererNode()

    try:
        # Spin on the main thread (handles the pose subscription callback).
        # The render and publish threads run independently.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
