// ─────────────────────────────────────────────────────────────────────────────
// multi_view_tracker.hpp
//
// High-performance multi-view AprilTag triangulation node.
// Fuses detections from 4 static cameras using DLT + Kabsch + EKF
// to publish an ambiguity-free 6-DoF pose on /robot/fused_pose.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace gsplat_tracker
{

// ── Number of cameras in the rig ────────────────────────────────────────────
constexpr int NUM_CAMERAS = 4;

// ── Number of corners per AprilTag ──────────────────────────────────────────
constexpr int NUM_CORNERS = 4;

// ── Time-window width for accumulating synchronous detections (seconds) ─────
constexpr double WINDOW_SEC = 0.005;  // 5 ms

// ─────────────────────────────────────────────────────────────────────────────
// Structs
// ─────────────────────────────────────────────────────────────────────────────

/// Per-camera calibration data (loaded from YAML).
struct CameraCalibration
{
  std::string frame_id;
  Eigen::Matrix3d K;            // 3×3 intrinsic matrix
  Eigen::Matrix3d R;            // 3×3 rotation (world → camera)
  Eigen::Vector3d t;            // 3×1 translation (world → camera)
  Eigen::Matrix<double, 3, 4> P; // Projection matrix P = K * [R | t]
};

/// A single tag detection from one camera with its 2D corner pixels.
struct CameraDetection
{
  int camera_id;
  rclcpp::Time stamp;
  std::array<Eigen::Vector2d, NUM_CORNERS> corners_px;
};

/// EKF state for a single tracked tag: [x, y, z, vx, vy, vz].
struct EkfState
{
  Eigen::Matrix<double, 6, 1> x = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Identity();
  rclcpp::Time last_update_time;
  bool initialized = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// MultiViewTracker Node
// ─────────────────────────────────────────────────────────────────────────────

class MultiViewTracker : public rclcpp::Node
{
public:
  explicit MultiViewTracker(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // ── Initialization ────────────────────────────────────────────────────────
  void loadCalibration(const std::string & yaml_path);
  void declareParameters();

  // ── Callbacks ─────────────────────────────────────────────────────────────
  void detectionCallback(
    const apriltag_msgs::msg::AprilTagDetectionArray::ConstSharedPtr msg,
    int camera_id);

  /// Timer-driven: evaluate the time-window and triangulate.
  void windowEvaluationCallback();

  // ── Core algorithms ───────────────────────────────────────────────────────

  /// DLT triangulation of a single 3D point from ≥2 views.
  /// Builds the 2N×4 system A from projection matrices and pixel coordinates,
  /// solves AX = 0 via SVD (last column of V, dehomogenised).
  Eigen::Vector3d dltTriangulate(
    const std::vector<Eigen::Matrix<double, 3, 4>> & Ps,
    const std::vector<Eigen::Vector2d> & pixels) const;

  /// Kabsch / Procrustes alignment.
  /// Given 4 triangulated 3D points (measured) and the known physical tag
  /// corner positions (model), finds the optimal rotation R and translation t
  /// minimising RMSD via SVD of the cross-covariance matrix.
  /// Returns the 6-DoF pose as an Eigen::Isometry3d.
  Eigen::Isometry3d kabschAlign(
    const std::array<Eigen::Vector3d, NUM_CORNERS> & measured,
    const std::array<Eigen::Vector3d, NUM_CORNERS> & model) const;

  /// Single-camera fallback using cv::solvePnP (EPnP).
  Eigen::Isometry3d solvePnPFallback(
    const CameraDetection & det) const;

  // ── EKF ───────────────────────────────────────────────────────────────────

  /// Constant-velocity prediction step.
  void ekfPredict(EkfState & state, double dt) const;

  /// Measurement update with a new position observation.
  void ekfUpdate(
    EkfState & state,
    const Eigen::Vector3d & z_pos,
    const rclcpp::Time & stamp);

  // ── Utilities ─────────────────────────────────────────────────────────────

  /// Convert Eigen rotation + translation to geometry_msgs::PoseStamped.
  geometry_msgs::msg::PoseStamped toPoseStamped(
    const Eigen::Isometry3d & pose,
    const rclcpp::Time & stamp) const;

  /// Build the model tag corners in the tag-local frame (z = 0 plane).
  std::array<Eigen::Vector3d, NUM_CORNERS> buildTagModel() const;

  // ── Members ───────────────────────────────────────────────────────────────

  // Camera calibrations indexed by camera_id (0–3).
  std::array<CameraCalibration, NUM_CAMERAS> calibrations_;

  // Sliding-window detection cache: tag_id → detections within window.
  std::map<int, std::vector<CameraDetection>> window_cache_;
  std::mutex cache_mutex_;

  // EKF states per tag_id.
  std::map<int, EkfState> ekf_states_;
  std::mutex ekf_mutex_;

  // Tag physical half-size (metres). Full side = 2 * half_size.
  double tag_half_size_;

  // EKF tuning
  double ekf_process_noise_;
  double ekf_measurement_noise_;

  // ROS interface
  std::array<rclcpp::Subscription<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr,
             NUM_CAMERAS> subs_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::TimerBase::SharedPtr window_timer_;

  // Reentrant callback group (allows concurrent callbacks on the executor).
  rclcpp::CallbackGroup::SharedPtr cb_group_;
};

}  // namespace gsplat_tracker
