FROM isaac_ros_dev:latest

SHELL ["/bin/bash", "-c"]

WORKDIR /workspaces/jetson-stack

COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]