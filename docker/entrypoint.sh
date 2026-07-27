#!/usr/bin/env bash

set -e

echo "======================================="
echo "      Jetson Stack Container"
echo "======================================="

# ROS
source /opt/ros/humble/setup.bash

# Workspace (if already built)
if [ -f /workspaces/jetson-stack/install/setup.bash ]; then
    source /workspaces/jetson-stack/install/setup.bash
fi

echo ""
echo "Workspace:"
echo "  /workspaces/jetson-stack"
echo ""

exec "$@"