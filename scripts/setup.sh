#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPENDENCIES_DIR="$REPO_DIR/.dependencies"

GREEN="\033[0;32m"
BLUE="\033[0;34m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
NC="\033[0m"

echo -e "${BLUE}"
echo "======================================"
echo "      Jetson Stack Setup"
echo "======================================"
echo -e "${NC}"

############################################
# Check Docker
############################################

command -v docker >/dev/null 2>&1 || {
    echo -e "${RED}Docker is not installed.${NC}"
    exit 1
}

echo -e "${GREEN}✓ Docker Installed${NC}"

############################################
# Check Docker Compose
############################################

docker compose version >/dev/null 2>&1 || {
    echo -e "${RED}Docker Compose not found.${NC}"
    exit 1
}

echo -e "${GREEN}✓ Docker Compose Installed${NC}"

############################################
# Create dependency directory
############################################

mkdir -p "$DEPENDENCIES_DIR"

############################################
# Clone Isaac ROS Common (release-3.2)
############################################

if [ ! -d "$DEPENDENCIES_DIR/isaac_ros_common" ]; then

    echo
    echo "Cloning Isaac ROS Common (release-3.2)..."

    git clone \
        --branch release-3.2 \
        https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git \
        "$DEPENDENCIES_DIR/isaac_ros_common"

else

    echo
    echo "Isaac ROS Common already exists."

fi

############################################
# Build isaac_ros_dev
############################################

if docker image inspect isaac_ros_dev:latest >/dev/null 2>&1
then

    echo
    echo -e "${GREEN}✓ isaac_ros_dev image already exists${NC}"

else

    echo
    echo "Building isaac_ros_dev..."

    cd "$DEPENDENCIES_DIR/isaac_ros_common"

    ./scripts/run_dev.sh --build-only

    cd "$REPO_DIR"

fi

############################################
# Build project image
############################################

echo
echo "Building robotics-testbed image..."

cd "$REPO_DIR"

docker compose build

echo
echo -e "${GREEN}"
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo -e "${NC}"

echo
echo "Run:"
echo "    docker compose up"