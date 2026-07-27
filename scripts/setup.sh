#!/usr/bin/env bash

set -e

REPO_DIR=$(pwd)
DEPENDENCIES_DIR="$REPO_DIR/.dependencies"

GREEN="\033[0;32m"
BLUE="\033[0;34m"
RED="\033[0;31m"
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

docker compose version >/dev/null

echo -e "${GREEN}✓ Docker Compose Installed${NC}"

############################################
# Clone Isaac ROS Common
############################################

mkdir -p "$DEPENDENCIES_DIR"

if [ ! -d "$DEPENDENCIES_DIR/isaac_ros_common" ]; then

    echo
    echo "Downloading Isaac ROS Common..."

    git clone \
        https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git \
        "$DEPENDENCIES_DIR/isaac_ros_common"

fi

############################################
# Build isaac_ros_dev image
############################################

if docker image inspect isaac_ros_dev:latest >/dev/null 2>&1
then

    echo
    echo -e "${GREEN}✓ isaac_ros_dev already exists${NC}"

else

    echo
    echo "Building isaac_ros_dev..."

    cd "$DEPENDENCIES_DIR/isaac_ros_common"

    ./scripts/run_dev.sh --build-only

    cd "$REPO_DIR"

fi

############################################
# Build Jetson Stack
############################################

echo
echo "Building jetson-stack..."

docker compose build

echo
echo -e "${GREEN}"
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo -e "${NC}"

echo
echo "Next:"
echo
echo "docker compose up"