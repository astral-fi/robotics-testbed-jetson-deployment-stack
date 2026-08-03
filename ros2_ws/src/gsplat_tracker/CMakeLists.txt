cmake_minimum_required(VERSION 3.16)
project(gsplat_tracker)

# ── C++ Standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Performance flags for release builds ─────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native -DNDEBUG)
endif()

# ── Find dependencies ───────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(isaac_ros_apriltag_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# ── C++ Multi-View Tracker executable ────────────────────────────────────────
add_executable(multi_view_tracker
  src/multi_view_tracker.cpp
)

target_include_directories(multi_view_tracker PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(multi_view_tracker
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

ament_target_dependencies(multi_view_tracker
  rclcpp
  geometry_msgs
  sensor_msgs
  isaac_ros_apriltag_interfaces
  cv_bridge
  image_transport
  std_msgs
)

# ── Install C++ targets ─────────────────────────────────────────────────────
install(TARGETS multi_view_tracker
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# ── Install Python nodes ────────────────────────────────────────────────────
install(PROGRAMS
  gsplat_tracker/gsplat_renderer_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Install launch files ────────────────────────────────────────────────────
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# ── Install config files ────────────────────────────────────────────────────
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# ── Testing ──────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
