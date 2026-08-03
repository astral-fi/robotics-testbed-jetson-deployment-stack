# ──────────────────────────────────────────────────────────────────────────────
# Multi-Camera Tracking + Real-Time Gaussian Splatting Renderer
# Base: CUDA 12.2 + Ubuntu 22.04, layered with ROS 2 Humble + PyTorch + gsplat
# Target: NVIDIA RTX 4090
# ──────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS base

# ── Prevent interactive prompts during apt installs ──────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ── System-level dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg2 \
        lsb-release \
        software-properties-common \
        ca-certificates \
        build-essential \
        cmake \
        git \
        wget \
        pkg-config \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        libeigen3-dev \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Install ROS 2 Humble from OSRF apt repo ─────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2-latest.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-image-transport \
        ros-humble-ffmpeg-image-transport \
        ros-humble-cv-bridge \
        ros-humble-rmw-fastrtps-cpp \
        ros-humble-rosidl-default-generators \
        ros-humble-rosidl-default-runtime \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# ── Source ROS 2 in every shell ──────────────────────────────────────────────
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc
SHELL ["/bin/bash", "-c"]

# ── PyTorch + gsplat (CUDA 12.1 wheels are forward-compatible with 12.2) ────
RUN pip3 install --no-cache-dir \
        torch==2.3.1+cu121 \
        torchvision==0.18.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
        gsplat \
        numpy \
        scipy \
        pyyaml \
        transforms3d

# ── Copy & build the ROS 2 workspace ────────────────────────────────────────
FROM base AS builder

WORKDIR /ros2_ws
COPY ros2_ws/src /ros2_ws/src

RUN source /opt/ros/humble/setup.bash \
    && colcon build \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --parallel-workers $(nproc) \
    && rm -rf build log

# ── Final runtime image ─────────────────────────────────────────────────────
FROM base AS runtime

COPY --from=builder /ros2_ws/install /ros2_ws/install

# ── Environment for runtime ─────────────────────────────────────────────────
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# Ensure gsplat CUDA kernels are cached inside the container
ENV GSPLAT_BUILD_DIR=/tmp/gsplat_build

WORKDIR /ros2_ws

# ── Entrypoint: source both ROS 2 underlay + overlay, then launch ───────────
COPY <<'ENTRYPOINT_SCRIPT' /ros2_ws/entrypoint.sh
#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
exec "$@"
ENTRYPOINT_SCRIPT
RUN chmod +x /ros2_ws/entrypoint.sh

ENTRYPOINT ["/ros2_ws/entrypoint.sh"]
CMD ["ros2", "launch", "gsplat_tracker", "full_pipeline.launch.py"]
