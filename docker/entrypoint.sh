#!/bin/bash

source /opt/ros/humble/setup.bash

if [ -f /workspaces/jetson-stack/install/setup.bash ]; then
    source /workspaces/jetson-stack/install/setup.bash
fi

exec "$@"