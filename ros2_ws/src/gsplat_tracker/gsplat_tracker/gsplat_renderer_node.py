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

import inspect
import math
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
#
# NOTE: rotation ONLY — no translation. The camera elevation is applied
# separately, along the WORLD vertical, in _rasterise(). Putting it here
# would express the offset in the tag frame, so any tag tilt would both
# tilt the camera and shorten its rise. A tag reading 16 deg off vertical
# lifted the camera only 9 cm instead of 20 cm, burying it in the floor.
# ─────────────────────────────────────────────────────────────────────────────
T_CAM_TO_TAG = torch.tensor([
    [1.0,  0.0,  0.0, 0.0],
    [0.0,  0.0,  1.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=torch.float32)

# World "up" axis index (REP 103 / this scene: +Z is the floor normal).
WORLD_UP_AXIS = 2


class GsplatRendererNode(Node):
    """Real-time Gaussian Splatting renderer driven by fused robot pose."""

    def __init__(self):
        super().__init__("gsplat_renderer")

        # ── Declare parameters ───────────────────────────────────────────
        self.declare_parameter("model_checkpoint", "/workspace/data/model.ckpt")
        # Quaternion component order as stored in the checkpoint.
        # gsplat requires (w, x, y, z). This checkpoint already stores wxyz —
        # verified by rendering both orders at a live tracker pose. Set to
        # "xyzw" only for a checkpoint that genuinely stores (x, y, z, w).
        self.declare_parameter("quat_order", "wxyz")
        # Extra virtual camera elevation along the WORLD vertical (floor
        # normal), applied on top of pose_offset. Defaults to 0.0 because
        # pose_offset already carries the full rig height in its z term.
        self.declare_parameter("camera_height", 0.0)

        # Fixed translation applied to every /robot/fused_pose sample, e.g. the
        # rig offset from the AprilTag to the camera mount. [x, y, z] in metres.
        self.declare_parameter("pose_offset", [0.41, -0.3, 0.4])

        # Frame the offset is expressed in:
        #   "tag"   — the tag/robot body frame. Rotates with the robot, so it
        #             stays fixed relative to the chassis. Correct for a
        #             physical camera mount. Note a tilted tag tilts it too.
        #   "world" — fixed world axes. Never rotates with the robot. Correct
        #             for correcting a world-frame misalignment.
        self.declare_parameter("pose_offset_frame", "world")
        # Output resolution. Defaults approximate a JetRacer CSI camera feed
        # (540p) rather than the 720p the node originally rendered.
        self.declare_parameter("render_width", 960)
        self.declare_parameter("render_height", 540)

        # Field of view drives the focal length, so changing resolution alone
        # can never silently change how wide the virtual camera sees.
        # JetRacer IMX219 variants: ~77 (standard), ~136 or ~160 (wide/fisheye).
        self.declare_parameter("camera_hfov_deg", 93.7)

        # "pinhole" or "fisheye" — fisheye needs a gsplat build that supports
        # the camera_model argument; the node checks and warns if it does not.
        self.declare_parameter("camera_model", "pinhole")

        # Cap the spherical-harmonics degree used at render time. The bands
        # above 0 encode view-dependence fitted at the TRAINING camera poses;
        # rendering far outside that distribution makes them extrapolate, which
        # shows up as smooth magenta/green casts drifting across flat surfaces.
        # -1 uses every band stored in the checkpoint. 0 = flat albedo only.
        self.declare_parameter("max_sh_degree", -1)

        # Explicit intrinsics override camera_hfov_deg. Leave at -1.0 to derive
        # fx/fy from the FOV and put the principal point at the image centre.
        self.declare_parameter("fx", -1.0)
        self.declare_parameter("fy", -1.0)
        self.declare_parameter("cx", -1.0)
        self.declare_parameter("cy", -1.0)

        self.width = self.get_parameter("render_width").value
        self.height = self.get_parameter("render_height").value
        self.camera_height = float(self.get_parameter("camera_height").value)

        # ── CUDA device ──────────────────────────────────────────────────
        assert torch.cuda.is_available(), "CUDA is required for gsplat rendering"
        self.device = torch.device("cuda:0")
        self.get_logger().info(
            f"Using GPU: {torch.cuda.get_device_name(self.device)}"
        )

        # ── Build virtual camera intrinsics matrix K (3×3) ──────────────
        # fx/fy/cx/cy left at -1.0 are derived from the resolution and FOV, so
        # the framing stays consistent whenever render_width/height change.
        hfov_deg = float(self.get_parameter("camera_hfov_deg").value)
        if not 1.0 < hfov_deg < 179.0:
            raise ValueError(
                f"camera_hfov_deg must be in (1, 179), got {hfov_deg}"
            )

        fx = float(self.get_parameter("fx").value)
        fy = float(self.get_parameter("fy").value)
        cx = float(self.get_parameter("cx").value)
        cy = float(self.get_parameter("cy").value)

        if fx <= 0.0:
            fx = (self.width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        if fy <= 0.0:
            fy = fx                      # square pixels
        if cx < 0.0:
            cx = self.width / 2.0
        if cy < 0.0:
            cy = self.height / 2.0

        self.K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 3, 3]

        # ── Camera projection model ─────────────────────────────────────
        self.camera_model = str(self.get_parameter("camera_model").value)
        supported = "camera_model" in inspect.signature(rasterization).parameters
        if self.camera_model != "pinhole" and not supported:
            self.get_logger().warn(
                f"gsplat build does not accept camera_model="
                f"{self.camera_model!r}; falling back to pinhole"
            )
            self.camera_model = "pinhole"
        self._pass_camera_model = supported and self.camera_model != "pinhole"

        v_hfov = 2.0 * math.degrees(math.atan((self.width / 2.0) / fx))
        v_vfov = 2.0 * math.degrees(math.atan((self.height / 2.0) / fy))
        self.get_logger().info(
            f"Virtual camera: {self.width}x{self.height} {self.camera_model} — "
            f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} "
            f"(hfov={v_hfov:.1f} vfov={v_vfov:.1f})"
        )

        # Keep the alignment matrix resident on the GPU — _rasterise() runs at
        # 60 FPS and re-uploading a 4x4 every frame is pure overhead.
        self._T_cam_to_tag = T_CAM_TO_TAG.to(self.device)

        # ── Fixed pose offset ────────────────────────────────────────────
        offset = [float(v) for v in self.get_parameter("pose_offset").value]
        if len(offset) != 3:
            raise ValueError(
                f"pose_offset must be [x, y, z], got {offset!r}"
            )
        self.pose_offset_frame = str(self.get_parameter("pose_offset_frame").value)
        if self.pose_offset_frame not in ("tag", "world"):
            raise ValueError(
                f"pose_offset_frame must be 'tag' or 'world', "
                f"got {self.pose_offset_frame!r}"
            )
        self._pose_offset = torch.tensor(
            offset, dtype=torch.float32, device=self.device
        )
        self._has_pose_offset = any(v != 0.0 for v in offset)
        self.get_logger().info(
            f"Pose offset: [{offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f}] m "
            f"in the {self.pose_offset_frame} frame"
        )

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
                self.sh_degree = int(math.sqrt(K)) - 1
        elif "sh_coefficients" in checkpoint:
            colors = checkpoint["sh_coefficients"].to(self.device).float()
            K = colors.numel() // (N * 3)
            if colors.dim() == 3 and colors.shape[1] == 3 and colors.shape[2] == K:
                colors = colors.transpose(1, 2)
            self.colors = colors.contiguous().view(N, K, 3)
            self.sh_degree = int(math.sqrt(K)) - 1
        else:
            raise KeyError(
                "Checkpoint must contain 'colors' or 'sh_coefficients'"
            )

        # ── Resolve the SH degree actually used for rendering ────────────
        max_sh = int(self.get_parameter("max_sh_degree").value)
        if self.sh_degree is None:
            self.render_sh_degree = None
        elif max_sh < 0:
            self.render_sh_degree = self.sh_degree
        else:
            self.render_sh_degree = min(max_sh, self.sh_degree)
            if self.render_sh_degree < self.sh_degree:
                self.get_logger().info(
                    f"Clamping SH degree {self.sh_degree} -> "
                    f"{self.render_sh_degree} (suppresses view-dependent "
                    f"colour casts at novel viewpoints)"
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
          T_cam_to_tag = alignment rotation ONLY
          C2W = T_tag_to_world @ T_cam_to_tag
          C2W[up] += camera_height        (elevation along the world vertical)
          W2C = C2W⁻¹

        gsplat.rasterization() expects viewmats as W2C [C, 4, 4].
        """
        # ── Move pose to GPU if not already there ────────────────────────
        pose_gpu = pose_T.to(self.device)

        # ── Apply the fixed rig offset to the tracked pose ───────────────
        # "tag" rotates the offset into world space by the tag's own attitude,
        # so it stays fixed relative to the chassis as the robot turns.
        # "world" adds it along fixed world axes, ignoring robot heading.
        if self._has_pose_offset:
            pose_gpu = pose_gpu.clone()
            if self.pose_offset_frame == "tag":
                pose_gpu[:3, 3] += pose_gpu[:3, :3] @ self._pose_offset
            else:
                pose_gpu[:3, 3] += self._pose_offset

        # ── Compute camera-to-world in gsplat coordinate frame ───────────
        #    C2W = T_tag_to_world × T_cam_to_tag   (rotation alignment only)
        C2W = pose_gpu @ self._T_cam_to_tag

        # ── Elevate along the WORLD vertical, not the tag normal ─────────
        # Writing straight into the translation column keeps the offset in
        # world coordinates, so a tilted tag no longer drags the virtual
        # camera down into the floor geometry.
        C2W = C2W.clone()
        C2W[WORLD_UP_AXIS, 3] += self.camera_height

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
            rasterization_kwargs["sh_degree"] = self.render_sh_degree
        if self._pass_camera_model:
            rasterization_kwargs["camera_model"] = self.camera_model
            
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
