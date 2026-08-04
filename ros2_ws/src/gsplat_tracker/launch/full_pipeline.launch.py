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

    tag_size_arg = DeclareLaunchArgument(
        "tag_size",
        default_value="0.25",
        description="Physical side length of the AprilTag in metres.",
    )

    render_width_arg = DeclareLaunchArgument(
        "render_width",
        default_value="1280",
        description="Render output width in pixels.",
    )

    render_height_arg = DeclareLaunchArgument(
        "render_height",
        default_value="720",
        description="Render output height in pixels.",
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

    # Virtual camera intrinsics for the gsplat renderer
    fx_arg = DeclareLaunchArgument("fx", default_value="600.0")
    fy_arg = DeclareLaunchArgument("fy", default_value="600.0")
    cx_arg = DeclareLaunchArgument("cx", default_value="640.0")
    cy_arg = DeclareLaunchArgument("cy", default_value="360.0")

    # ── Node 1: C++ Multi-View Tracker ───────────────────────────────────
    tracker_node = Node(
        package="gsplat_tracker",
        executable="multi_view_tracker",
        name="multi_view_tracker",
        output="screen",
        extra_arguments=[{"use_intra_process_comms": True}],
        parameters=[
            {
                "calibration_file": LaunchConfiguration("calibration_file"),
                "tag_size": LaunchConfiguration("tag_size"),
                "ekf_process_noise": LaunchConfiguration("ekf_process_noise"),
                "ekf_measurement_noise": LaunchConfiguration("ekf_measurement_noise"),
            }
        ],
    )

    # ── Node 2: Python gsplat Renderer ───────────────────────────────────
    renderer_node = Node(
        package="gsplat_tracker",
        executable="gsplat_renderer_node.py",
        name="gsplat_renderer",
        output="screen",
        extra_arguments=[{"use_intra_process_comms": True}],
        parameters=[
            {
                "model_checkpoint": LaunchConfiguration("model_checkpoint"),
                "render_width": LaunchConfiguration("render_width"),
                "render_height": LaunchConfiguration("render_height"),
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
        extra_arguments=[{"use_intra_process_comms": True}],
        arguments=[
            "raw",                              # input transport
            "ffmpeg",                           # output transport
            "--ros-args",
            "--remap", "in:=/gsplat/raw_image",
            "--remap", "out:=/gsplat/rendered_stream",
        ],
        parameters=[
            {
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
        tag_size_arg,
        render_width_arg,
        render_height_arg,
        ekf_process_noise_arg,
        ekf_measurement_noise_arg,
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
