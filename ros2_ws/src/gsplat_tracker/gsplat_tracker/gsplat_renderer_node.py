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

# gsplat high-level rasterization API
from gsplat import rasterization


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate frame alignment matrix: ROS REP 103 → gsplat/OpenGL convention
#
# ROS:    X-fwd,  Y-left, Z-up
# gsplat: X-right, Y-down, Z-fwd
#
# T_ros_to_graphics = | 0  -1   0  0 |   (ROS Y → graphics -X)
#                     | 0   0  -1  0 |   (ROS Z → graphics -Y)
#                     | 1   0   0  0 |   (ROS X → graphics  Z)
#                     | 0   0   0  1 |
# ─────────────────────────────────────────────────────────────────────────────
T_ROS_TO_GRAPHICS = torch.tensor([
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [1.0,  0.0,  0.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=torch.float32)


class GsplatRendererNode(Node):
    """Real-time Gaussian Splatting renderer driven by fused robot pose."""

    def __init__(self):
        super().__init__("gsplat_renderer")

        # ── Declare parameters ───────────────────────────────────────────
        self.declare_parameter("model_checkpoint", "/workspace/data/model.ckpt")
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

        # ── Coordinate transform (pre-computed, on GPU) ─────────────────
        self.T_ros_to_graphics = T_ROS_TO_GRAPHICS.to(self.device)

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

        # Extract Gaussian parameters — adapt keys to your training pipeline
        self.means = checkpoint["means"].to(self.device).contiguous()
        self.quats = checkpoint["quats"].to(self.device).contiguous()
        self.scales = checkpoint["scales"].to(self.device).contiguous()
        
        # gsplat often expects opacities to be of shape [N], but checkpoints might save them as [N, 1]
        opacities = checkpoint["opacities"].to(self.device).contiguous()
        if opacities.dim() == 2 and opacities.shape[1] == 1:
            opacities = opacities.squeeze(-1)
        self.opacities = opacities

        # Colors: support both raw RGB and spherical harmonics
        if "colors" in checkpoint:
            self.colors = checkpoint["colors"].to(self.device).contiguous()
        elif "sh_coefficients" in checkpoint:
            self.colors = checkpoint["sh_coefficients"].to(self.device).contiguous()
        else:
            raise KeyError(
                "Checkpoint must contain 'colors' or 'sh_coefficients'"
            )

        n_gaussians = self.means.shape[0]
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
          T_robot_in_gsplat = pose_T (robot pose in ROS world frame)
          T_ros_to_graphics = axis flip matrix
          C2W = T_robot_in_gsplat @ T_ros_to_graphics
          W2C = C2W⁻¹

        gsplat.rasterization() expects viewmats as W2C [C, 4, 4].
        """
        # ── Move pose to GPU if not already there ────────────────────────
        pose_gpu = pose_T.to(self.device)

        # ── Compute camera-to-world in gsplat coordinate frame ───────────
        #    C2W = T_robot_in_gsplat × T_ros_to_graphics
        C2W = pose_gpu @ self.T_ros_to_graphics

        # ── World-to-camera = inverse of C2W ─────────────────────────────
        W2C = torch.inverse(C2W)

        # ── Batch dimension [1, 4, 4] for gsplat API ─────────────────────
        viewmats = W2C.unsqueeze(0)

        # ── Rasterise ────────────────────────────────────────────────────
        rendered, _, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=viewmats,
            Ks=self.K,
            width=self.width,
            height=self.height,
            # Performance: disable gradient tracking, we're inference-only
            absgrad=False,
        )

        # rendered shape: [1, H, W, 3] (float32, range [0, 1])

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
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "gsplat_virtual_camera"
            msg.height = frame.shape[0]
            msg.width = frame.shape[1]
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = frame.shape[1] * 3  # 3 bytes per pixel (RGB)
            msg.data = frame.tobytes()

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
