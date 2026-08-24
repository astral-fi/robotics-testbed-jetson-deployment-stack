#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# full_pipeline.launch.py
#
# Orchestrates the complete tracking + rendering + streaming pipeline:
#
#   1. multi_view_tracker   (C++)   — Multi-camera AprilTag triangulation
#   2. gsplat_renderer      (Python) — Real-time Gaussian Splatting in VRAM
#   3. image_transport republish     — H.264 NVENC hardware-accelerated stream
#
# Usage:
#   ros2 launch gsplat_tracker full_pipeline.launch.py \
#       calibration_file:=/workspace/data/camera_calibration.yaml \
#       model_checkpoint:=/workspace/data/model.ckpt \
#       tag_size:=0.25 \
#       render_width:=1280 \
#       render_height:=720
# ─────────────────────────────────────────────────────────────────────────────

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Launch arguments ─────────────────────────────────────────────────
    calibration_file_arg = DeclareLaunchArgument(
        "calibration_file",
        default_value="/workspace/data/camera_calibration.yaml",
        description="Path to the multi-camera calibration YAML file.",
    )

    model_checkpoint_arg = DeclareLaunchArgument(
        "model_checkpoint",
        default_value="/workspace/data/model.ckpt",
        description="Path to the trained gsplat model checkpoint.",
    )

    quat_order_arg = DeclareLaunchArgument(
        "quat_order",
        default_value="wxyz",
        description="Quaternion order stored in the checkpoint: 'wxyz' or 'xyzw'. "
                    "gsplat needs wxyz; a wrong setting renders as a spiky haze.",
    )

    tag_size_arg = DeclareLaunchArgument(
        "tag_size",
        default_value="0.21",
        description="Physical side length of the AprilTag in metres.",
    )

    camera_height_arg = DeclareLaunchArgument(
        "camera_height",
        default_value="0.0",
        description="Extra virtual camera elevation in metres along the WORLD "
                    "vertical, applied on top of pose_offset. Defaults to 0.0 "
                    "because pose_offset already carries the rig height.",
    )

    pose_offset_frame_arg = DeclareLaunchArgument(
        "pose_offset_frame",
        default_value="world",
        description="Frame for pose_offset: 'tag' (rotates with the robot — a "
                    "physical camera mount) or 'world' (fixed world axes).",
    )

    render_width_arg = DeclareLaunchArgument(
        "render_width",
        default_value="960",
        description="Render output width in pixels (540p ~ JetRacer CSI feed).",
    )

    render_height_arg = DeclareLaunchArgument(
        "render_height",
        default_value="540",
        description="Render output height in pixels.",
    )

    camera_hfov_arg = DeclareLaunchArgument(
        "camera_hfov_deg",
        default_value="93.7",
        description="Horizontal field of view in degrees. Drives fx/fy so that "
                    "changing resolution never changes framing. JetRacer IMX219 "
                    "variants are roughly 77 (standard), 136 or 160 (wide).",
    )

    max_sh_degree_arg = DeclareLaunchArgument(
        "max_sh_degree",
        default_value="-1",
        description="Cap the SH degree used at render time. -1 uses every band "
                    "in the checkpoint. Lower values suppress view-dependent "
                    "magenta/green casts when rendering far from training views; "
                    "0 is flat albedo.",
    )

    camera_model_arg = DeclareLaunchArgument(
        "camera_model",
        default_value="pinhole",
        description="Projection model: 'pinhole' or 'fisheye'. Fisheye requires "
                    "a gsplat build exposing camera_model; the node warns and "
                    "falls back to pinhole otherwise.",
    )

    ekf_process_noise_arg = DeclareLaunchArgument(
        "ekf_process_noise",
        default_value="0.01",
        description="EKF process noise scalar (jerk model).",
    )

    ekf_measurement_noise_arg = DeclareLaunchArgument(
        "ekf_measurement_noise",
        default_value="0.005",
        description="EKF measurement noise scalar.",
    )

    # ── Tracker robustness / smoothing ───────────────────────────────────
    rotation_tau_arg = DeclareLaunchArgument(
        "rotation_tau",
        default_value="0.3",
        description="Orientation smoothing time constant in seconds. Larger is "
                    "smoother but laggier. Unlike a fixed SLERP alpha this holds "
                    "its meaning when the detection rate changes.",
    )

    ekf_gate_chi2_arg = DeclareLaunchArgument(
        "ekf_gate_chi2",
        default_value="16.27",
        description="Chi-square gate on the 3-DoF position innovation. "
                    "11.34 = 99 pct, 16.27 = 99.9 pct. Lower rejects harder.",
    )

    single_view_noise_scale_arg = DeclareLaunchArgument(
        "single_view_noise_scale",
        default_value="6.0",
        description="Measurement-noise multiplier when only one camera sees the "
                    "tag, so a single-view PnP fix cannot yank the estimate the "
                    "way a 3-view triangulation can.",
    )

    use_ippe_square_arg = DeclareLaunchArgument(
        "use_ippe_square",
        default_value="true",
        description="Use cv::SOLVEPNP_IPPE_SQUARE instead of SOLVEPNP_ITERATIVE. "
                    "Avoids the two-fold ambiguity that makes a planar tag flip "
                    "between mirrored poses.",
    )

    # Virtual camera intrinsics. -1.0 means "derive from resolution + FOV",
    # which is what you want unless you are matching a real calibrated camera.
    fx_arg = DeclareLaunchArgument("fx", default_value="-1.0")
    fy_arg = DeclareLaunchArgument("fy", default_value="-1.0")
    cx_arg = DeclareLaunchArgument("cx", default_value="-1.0")
    cy_arg = DeclareLaunchArgument("cy", default_value="-1.0")

    # ── Node 1: C++ Multi-View Tracker ───────────────────────────────────
    tracker_node = Node(
        package="gsplat_tracker",
        executable="multi_view_tracker",
        name="multi_view_tracker",
        output="screen",
        parameters=[
            {
                "use_intra_process_comms": True,
                "calibration_file": LaunchConfiguration("calibration_file"),
                "tag_size": LaunchConfiguration("tag_size"),
                "ekf_process_noise": LaunchConfiguration("ekf_process_noise"),
                "ekf_measurement_noise": LaunchConfiguration("ekf_measurement_noise"),
                "rotation_tau": LaunchConfiguration("rotation_tau"),
                "ekf_gate_chi2": LaunchConfiguration("ekf_gate_chi2"),
                "single_view_noise_scale":
                    LaunchConfiguration("single_view_noise_scale"),
                "use_ippe_square": LaunchConfiguration("use_ippe_square"),
            }
        ],
    )

    # ── Node 2: Python gsplat Renderer ───────────────────────────────────
    renderer_node = Node(
        package="gsplat_tracker",
        executable="gsplat_renderer_node.py",
        name="gsplat_renderer",
        output="screen",
        parameters=[
            {
                "use_intra_process_comms": True,
                "model_checkpoint": LaunchConfiguration("model_checkpoint"),
                "quat_order": LaunchConfiguration("quat_order"),
                "camera_height": LaunchConfiguration("camera_height"),
                "pose_offset_frame": LaunchConfiguration("pose_offset_frame"),
                "render_width": LaunchConfiguration("render_width"),
                "render_height": LaunchConfiguration("render_height"),
                "camera_hfov_deg": LaunchConfiguration("camera_hfov_deg"),
                "camera_model": LaunchConfiguration("camera_model"),
                "max_sh_degree": LaunchConfiguration("max_sh_degree"),
                "fx": LaunchConfiguration("fx"),
                "fy": LaunchConfiguration("fy"),
                "cx": LaunchConfiguration("cx"),
                "cy": LaunchConfiguration("cy"),
            }
        ],
    )

    # ── Node 3: image_transport republish — Raw → H.264 NVENC stream ────
    republish_node = Node(
        package="image_transport",
        executable="republish",
        name="ffmpeg_republisher",
        output="screen",
        arguments=[
            "raw",                              # input transport
            "ffmpeg",                           # output transport
        ],
        remappings=[
            ("in", "/gsplat/raw_image"),
            ("out", "/gsplat/rendered_stream"),
        ],
        parameters=[
            {
                "use_intra_process_comms": True,
                # Output encoder configuration for ffmpeg_image_transport
                "out.ffmpeg.encoder": "h264_nvenc",
                "out.ffmpeg.bit_rate": 8000000,
                "out.ffmpeg.gop_size": 5,
                "out.ffmpeg.qmax": 15,
                # NVENC-specific advanced options (key:value comma-separated)
                "out.ffmpeg.encoder_av_options":
                    "preset:ll,tune:ull,delay:0,zerolatency:1",
            }
        ],
    )

    # ── Assemble launch description ──────────────────────────────────────
    return LaunchDescription([
        # Arguments
        calibration_file_arg,
        model_checkpoint_arg,
        quat_order_arg,
        camera_height_arg,
        pose_offset_frame_arg,
        tag_size_arg,
        render_width_arg,
        render_height_arg,
        camera_hfov_arg,
        camera_model_arg,
        max_sh_degree_arg,
        ekf_process_noise_arg,
        ekf_measurement_noise_arg,
        rotation_tau_arg,
        ekf_gate_chi2_arg,
        single_view_noise_scale_arg,
        use_ippe_square_arg,
        fx_arg,
        fy_arg,
        cx_arg,
        cy_arg,

        # Startup log
        LogInfo(msg="═══ Starting Multi-Camera Tracker + gsplat Renderer + NVENC Streamer ═══"),

        # Nodes
        tracker_node,
        renderer_node,
        republish_node,
    ])
