// ─────────────────────────────────────────────────────────────────────────────
// multi_view_tracker.cpp
//
// Production-grade multi-view AprilTag tracking pipeline.
//
// Algorithm overview:
//   1. Receive AprilTag detections from 4 cameras concurrently.
//   2. Accumulate detections in a 5 ms sliding time-window per tag ID.
//   3. When ≥2 cameras see the same tag in the window → DLT triangulate
//      each of the 4 corners to 3D, then Kabsch-align to known geometry.
//   4. When only 1 camera sees the tag → cv::solvePnP (EPnP) fallback.
//   5. Feed the resulting position into a lightweight 6-state EKF for
//      temporal smoothing, then publish PoseStamped to /robot/fused_pose.
//
// Threading model:
//   MultiThreadedExecutor with Reentrant callback group.
//   4 subscription callbacks + 1 timer callback run concurrently.
//   Shared state is protected by std::mutex.
// ─────────────────────────────────────────────────────────────────────────────

#include "gsplat_tracker/multi_view_tracker.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>
#include <chrono>

// For YAML parsing we use a simple manual parser since yaml-cpp may not be
// available. In production you would use yaml-cpp; here we use OpenCV's
// FileStorage which is universally available via libopencv-dev.
#include <opencv2/core.hpp>

using namespace std::chrono_literals;
using std::placeholders::_1;

namespace gsplat_tracker
{

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
MultiViewTracker::MultiViewTracker(const rclcpp::NodeOptions & options)
: Node("multi_view_tracker", options)
{
  declareParameters();

  // ── Load calibration ────────────────────────────────────────────────────
  const std::string calib_path =
    this->get_parameter("calibration_file").as_string();
  loadCalibration(calib_path);

  // ── Tag geometry ────────────────────────────────────────────────────────
  const double tag_size = this->get_parameter("tag_size").as_double();
  tag_half_size_ = tag_size / 2.0;

  // ── Pre-allocate solvePnP constants (H2 fix: avoid per-frame heap alloc)
  const auto model = buildTagModel();
  obj_pts_.resize(NUM_CORNERS);
  for (int i = 0; i < NUM_CORNERS; ++i) {
    obj_pts_[i] = cv::Point3d(model[i].x(), model[i].y(), model[i].z());
  }
  dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64F);

  // ── EKF tuning ──────────────────────────────────────────────────────────
  ekf_process_noise_     = this->get_parameter("ekf_process_noise").as_double();
  ekf_measurement_noise_ = this->get_parameter("ekf_measurement_noise").as_double();

  // ── Callback group: Reentrant allows concurrent execution ─────────────
  cb_group_ = this->create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);

  // ── QoS: RELIABLE with depth 1 to match Jetson's RELIABLE publisher.
  // Now safe because the Jetson runs per-camera isolated containers,
  // so RELIABLE ACKs cannot cause cross-camera thread starvation.
  rclcpp::QoS qos_profile = rclcpp::QoS(
    rclcpp::KeepLast(1))
    .reliable()
    .durability_volatile();

  // ── Create subscriptions for 4 cameras ────────────────────────────────
  rclcpp::SubscriptionOptions sub_opts;
  sub_opts.callback_group = cb_group_;

  const std::array<std::string, NUM_CAMERAS> topics = {
    "/cam_25251947/tag_detections",
    "/cam_25251937/tag_detections",
    "/cam_25251925/tag_detections",
    "/cam_25251936/tag_detections"
  };

  for (int i = 0; i < NUM_CAMERAS; ++i) {
    subs_[i] = this->create_subscription<
      isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray>(
      topics[i],
      qos_profile,
      [this, i](const isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray::ConstSharedPtr msg) {
        this->detectionCallback(msg, i);
      },
      sub_opts);
    RCLCPP_INFO(this->get_logger(), "Subscribed to %s", topics[i].c_str());
  }

  // ── Pose publisher ────────────────────────────────────────────────────
  rclcpp::QoS pub_qos(1);
  pub_qos.best_effort();
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "/robot/fused_pose", pub_qos);

  // ── Window evaluation timer (50 ms period = 20 Hz) ─────────────────────
  window_timer_ = this->create_wall_timer(
    50ms,
    std::bind(&MultiViewTracker::windowEvaluationCallback, this),
    cb_group_);

  RCLCPP_INFO(this->get_logger(),
    "MultiViewTracker initialised — tag_size=%.3fm, window=%.1fms",
    tag_size, WINDOW_SEC * 1000.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter declaration
// ─────────────────────────────────────────────────────────────────────────────
void MultiViewTracker::declareParameters()
{
  this->declare_parameter<std::string>(
    "calibration_file", "/workspace/data/camera_calibration.yaml");
  this->declare_parameter<double>("tag_size", 0.25);              // metres
  this->declare_parameter<double>("ekf_process_noise", 0.01);
  this->declare_parameter<double>("ekf_measurement_noise", 0.005);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load camera calibrations from YAML via OpenCV FileStorage
// ─────────────────────────────────────────────────────────────────────────────
void MultiViewTracker::loadCalibration(const std::string & yaml_path)
{
  if (yaml_path.empty()) {
    RCLCPP_FATAL(this->get_logger(),
      "calibration_file parameter is empty! Specify path to camera_calibration.yaml");
    throw std::runtime_error("calibration_file parameter is empty!");
  }

  cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    RCLCPP_FATAL(this->get_logger(),
      "Cannot open calibration file: %s", yaml_path.c_str());
    throw std::runtime_error("Calibration file not found: " + yaml_path);
  }

  cv::FileNode cameras_node = fs["cameras"];
  const std::array<std::string, NUM_CAMERAS> cam_names =
    {"cam0", "cam1", "cam2", "cam3"};

  for (int i = 0; i < NUM_CAMERAS; ++i) {
    cv::FileNode cam = cameras_node[cam_names[i]];

    // ── Frame ID ──────────────────────────────────────────────────────────
    calibrations_[i].frame_id = static_cast<std::string>(cam["frame_id"]);

    // ── Intrinsic matrix K (3×3, row-major flat list) ────────────────────
    std::vector<double> k_vec;
    cam["K"] >> k_vec;
    calibrations_[i].K = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
      k_vec.data());

    // ── Rotation R (3×3, world → camera) ────────────────────────────────
    std::vector<double> r_vec;
    cam["R"] >> r_vec;
    calibrations_[i].R = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
      r_vec.data());

    // ── Translation t (3×1) ─────────────────────────────────────────────
    std::vector<double> t_vec;
    cam["t"] >> t_vec;
    calibrations_[i].t = Eigen::Map<Eigen::Vector3d>(t_vec.data());

    // ── Build projection matrix P = K * [R | t] ────────────────────────
    Eigen::Matrix<double, 3, 4> Rt;
    Rt.block<3, 3>(0, 0) = calibrations_[i].R;
    Rt.col(3) = calibrations_[i].t;
    calibrations_[i].P = calibrations_[i].K * Rt;

    // Pre-compute OpenCV intrinsics (H2 fix: avoids per-frame eigen2cv)
    cv::eigen2cv(calibrations_[i].K, calibrations_[i].K_cv);

    RCLCPP_INFO(this->get_logger(), "Loaded calibration for %s (%s)",
      cam_names[i].c_str(), calibrations_[i].frame_id.c_str());
  }

  fs.release();
}

// ─────────────────────────────────────────────────────────────────────────────
// Detection callback — runs concurrently for each camera
// ─────────────────────────────────────────────────────────────────────────────
void MultiViewTracker::detectionCallback(
  const isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray::ConstSharedPtr msg,
  int camera_id)
{
  if (msg->detections.empty()) return;

  // Use local arrival stamp for time-windowing to avoid cross-machine clock drift issues
  const rclcpp::Time stamp = this->now();

  std::lock_guard<std::mutex> lock(cache_mutex_);
  for (const auto & det : msg->detections) {
    CameraDetection cd;
    cd.camera_id = camera_id;
    cd.stamp = stamp;

    // Extract 4 corner pixel coordinates from the detection message.
    for (int c = 0; c < NUM_CORNERS; ++c) {
      cd.corners_px[c] = Eigen::Vector2d(
        det.corners[c].x,
        det.corners[c].y);
    }

    // Insert into sliding-window cache
    window_cache_[det.id].push_back(std::move(cd));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Window evaluation timer callback (5 ms cadence)
// ─────────────────────────────────────────────────────────────────────────────
void MultiViewTracker::windowEvaluationCallback()
{
  const rclcpp::Time now = this->now();
  const auto model_corners = buildTagModel();

  // ── Swap out the cache under lock to minimise hold time ───────────────
  std::map<int, std::vector<CameraDetection>> local_cache;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    local_cache.swap(window_cache_);
  }

  // ── DIAGNOSTIC: log cache contents before pruning ─────────────────────
  {
    size_t total_dets = 0;
    for (const auto & [tid, dets] : local_cache) total_dets += dets.size();
    if (total_dets > 0) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "DIAG: cache has %zu tags, %zu total detections", local_cache.size(), total_dets);
    }
  }

  // Collect poses to publish OUTSIDE the ekf_mutex_ (H1 fix: never
  // call publish() while holding a mutex — it can block in DDS).
  std::vector<geometry_msgs::msg::PoseStamped> poses_to_publish;

  for (auto & [tag_id, detections] : local_cache) {
    size_t before_prune = detections.size();

    // ── Prune stale detections outside the window ──────────────────────
    detections.erase(
      std::remove_if(detections.begin(), detections.end(),
        [&](const CameraDetection & d) {
          return std::abs((now - d.stamp).seconds()) > WINDOW_SEC;
        }),
      detections.end());

    if (before_prune > 0 && detections.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "DIAG: Tag %d — ALL %zu detections pruned as stale!", tag_id, before_prune);
    }

    if (detections.empty()) continue;

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    rclcpp::Time best_stamp = detections.front().stamp;
    bool valid = false;

    if (detections.size() >= 2) {
      // ── Multi-view DLT triangulation for each corner ──────────────────
      std::array<Eigen::Vector3d, NUM_CORNERS> pts_3d;

      for (int c = 0; c < NUM_CORNERS; ++c) {
        std::vector<Eigen::Matrix<double, 3, 4>> Ps;
        std::vector<Eigen::Vector2d> pixels;

        for (const auto & d : detections) {
          Ps.push_back(calibrations_[d.camera_id].P);
          pixels.push_back(d.corners_px[c]);
        }

        pts_3d[c] = dltTriangulate(Ps, pixels);
      }

      // ── Kabsch alignment: measured 3D corners → known tag model ───────
      pose = kabschAlign(pts_3d, model_corners);
      valid = true;

      RCLCPP_DEBUG(this->get_logger(),
        "Tag %d: DLT+Kabsch from %zu views", tag_id, detections.size());

    } else {
      // ── Single-camera fallback: EPnP ──────────────────────────────────
      pose = solvePnPFallback(detections.front());
      valid = true;

      RCLCPP_DEBUG(this->get_logger(),
        "Tag %d: solvePnP fallback (cam%d)", tag_id,
        detections.front().camera_id);
    }

    if (!valid) continue;

    // ── EKF smooth the position (minimised critical section — H1 fix) ───
    {
      std::lock_guard<std::mutex> lock(ekf_mutex_);
      auto & state = ekf_states_[tag_id];

      if (!state.initialized) {
        state.x.head<3>() = pose.translation();
        state.x.tail<3>().setZero();
        state.P = Eigen::Matrix<double, 6, 6>::Identity() * 1.0;
        state.last_update_time = best_stamp;
        state.initialized = true;
      } else {
        const double dt = (best_stamp - state.last_update_time).seconds();
        if (dt > 0.0 && dt < 1.0) {
          ekfPredict(state, dt);
        }
        ekfUpdate(state, pose.translation(), best_stamp);
      }

      // Copy smoothed result before releasing lock.
      Eigen::Isometry3d smoothed = pose;
      smoothed.translation() = state.x.head<3>();
      poses_to_publish.push_back(toPoseStamped(smoothed, best_stamp));
    }  // ekf_mutex_ released BEFORE publish
  }

  // ── Publish all poses OUTSIDE any mutex (H1 fix) ──────────────────────
  for (const auto & msg : poses_to_publish) {
    pose_pub_->publish(msg);
  }

  // ── Prune stale EKF states (C2 fix: prevent monotonic map growth) ─────
  {
    std::lock_guard<std::mutex> lock(ekf_mutex_);
    for (auto it = ekf_states_.begin(); it != ekf_states_.end(); ) {
      if (it->second.initialized &&
          std::abs((now - it->second.last_update_time).seconds()) > EKF_STALE_SEC) {
        it = ekf_states_.erase(it);
      } else {
        ++it;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DLT Triangulation  —  AX = 0 solved via SVD
//
// For N views, build a (2N × 4) matrix A where each view contributes 2 rows:
//   row 2i   :  u_i * P_i.row(2) - P_i.row(0)
//   row 2i+1 :  v_i * P_i.row(2) - P_i.row(1)
//
// The 3D point X is the last column of V from SVD(A), dehomogenised by w.
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Vector3d MultiViewTracker::dltTriangulate(
  const std::vector<Eigen::Matrix<double, 3, 4>> & Ps,
  const std::vector<Eigen::Vector2d> & pixels) const
{
  const int n = static_cast<int>(Ps.size());

  // H3 fix: fixed-max-capacity matrix (stack-allocated, max 8×4 for 4 cameras).
  Eigen::Matrix<double, Eigen::Dynamic, 4, 0, 2 * NUM_CAMERAS, 4> A(2 * n, 4);

  for (int i = 0; i < n; ++i) {
    const double u = pixels[i].x();
    const double v = pixels[i].y();

    A.row(2 * i)     = u * Ps[i].row(2) - Ps[i].row(0);
    A.row(2 * i + 1) = v * Ps[i].row(2) - Ps[i].row(1);
  }

  // SVD decomposition — we only need V.
  Eigen::JacobiSVD<decltype(A)> svd(A, Eigen::ComputeFullV);
  Eigen::Vector4d X_homogeneous = svd.matrixV().col(3);

  // Dehomogenise: divide by the 4th coordinate.
  if (std::abs(X_homogeneous(3)) < 1e-12) {
    // M2 fix: throttle to 1 Hz to prevent log flooding from bad data.
    static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
    RCLCPP_WARN_THROTTLE(this->get_logger(), steady_clock, 1000,
      "DLT: degenerate point (w ≈ 0), returning origin");
    return Eigen::Vector3d::Zero();
  }

  return X_homogeneous.head<3>() / X_homogeneous(3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kabsch / Procrustes Alignment
//
// Given:
//   measured[4]  — triangulated 3D corners (noisy)
//   model[4]     — known tag corner positions (ground truth geometry)
//
// Returns optimal rigid transform (R, t) minimising Σ||R·model_i + t - meas_i||²
//
// Steps:
//   1. Compute centroids of both point sets.
//   2. Centre both sets.
//   3. Build cross-covariance H = (centred_model)ᵀ · (centred_measured)
//   4. SVD(H) → U Σ Vᵀ
//   5. R = V · diag(1, 1, det(V·Uᵀ)) · Uᵀ   (ensures proper rotation)
//   6. t = centroid_measured - R · centroid_model
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Isometry3d MultiViewTracker::kabschAlign(
  const std::array<Eigen::Vector3d, NUM_CORNERS> & measured,
  const std::array<Eigen::Vector3d, NUM_CORNERS> & model) const
{
  // ── 1. Compute centroids ──────────────────────────────────────────────
  Eigen::Vector3d centroid_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d centroid_p = Eigen::Vector3d::Zero();

  for (int i = 0; i < NUM_CORNERS; ++i) {
    centroid_p += measured[i];  // P = measured ("points")
    centroid_m += model[i];     // Q = model   ("query")
  }
  centroid_p /= NUM_CORNERS;
  centroid_m /= NUM_CORNERS;

  // ── 2. Centre both sets ───────────────────────────────────────────────
  Eigen::Matrix<double, 3, NUM_CORNERS> P_centred, Q_centred;
  for (int i = 0; i < NUM_CORNERS; ++i) {
    P_centred.col(i) = measured[i] - centroid_p;
    Q_centred.col(i) = model[i] - centroid_m;
  }

  // ── 3. Cross-covariance matrix H = Qᵀ · P ────────────────────────────
  //    (Using columns: H = Q_centred · P_centredᵀ, which is 3×3)
  Eigen::Matrix3d H = Q_centred * P_centred.transpose();

  // ── 4. SVD of H ───────────────────────────────────────────────────────
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
    H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  // ── 5. Ensure proper rotation (det = +1, not reflection) ──────────────
  double d = (V * U.transpose()).determinant();
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  D(2, 2) = (d > 0.0) ? 1.0 : -1.0;

  Eigen::Matrix3d R = V * D * U.transpose();

  // ── 6. Translation ───────────────────────────────────────────────────
  Eigen::Vector3d t = centroid_p - R * centroid_m;

  // ── Assemble the isometry ─────────────────────────────────────────────
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = R;
  pose.translation() = t;

  return pose;
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-camera fallback: cv::solvePnP (EPnP)
//
// When only 1 camera sees the tag, we use the classic PnP solution with
// the known tag geometry to estimate pose in that camera's frame, then
// transform to world frame using the camera's extrinsics.
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Isometry3d MultiViewTracker::solvePnPFallback(
  const CameraDetection & det) const
{
  const auto & cal = calibrations_[det.camera_id];

  // H2 fix: obj_pts_, dist_coeffs_, and cal.K_cv are pre-allocated at
  // construction time. Only img_pts needs per-call construction.
  std::array<cv::Point2d, NUM_CORNERS> img_pts;
  for (int i = 0; i < NUM_CORNERS; ++i) {
    img_pts[i] = cv::Point2d(det.corners_px[i].x(), det.corners_px[i].y());
  }

  // ── Solve PnP (EPnP algorithm) ───────────────────────────────────────
  cv::Mat rvec, tvec;
  bool ok = cv::solvePnP(obj_pts_, img_pts, cal.K_cv, dist_coeffs_,
                          rvec, tvec, false, cv::SOLVEPNP_EPNP);
  if (!ok) {
    // M3 fix: throttle to 1 Hz to prevent log flooding.
    static rclcpp::Clock steady_clock_pnp(RCL_STEADY_TIME);
    RCLCPP_WARN_THROTTLE(this->get_logger(), steady_clock_pnp, 1000,
      "solvePnP failed for cam%d", det.camera_id);
    return Eigen::Isometry3d::Identity();
  }

  // ── Convert rotation vector to matrix ─────────────────────────────────
  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);

  Eigen::Matrix3d R_tag_in_cam;
  cv::cv2eigen(R_cv, R_tag_in_cam);
  Eigen::Vector3d t_tag_in_cam;
  cv::cv2eigen(tvec, t_tag_in_cam);

  // ── Transform from camera frame to world frame ────────────────────────
  Eigen::Matrix3d R_cam_to_world = cal.R.transpose();
  Eigen::Vector3d t_cam_to_world = -R_cam_to_world * cal.t;

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = R_cam_to_world * R_tag_in_cam;
  pose.translation() = R_cam_to_world * t_tag_in_cam + t_cam_to_world;

  return pose;
}

// ─────────────────────────────────────────────────────────────────────────────
// EKF Predict — constant-velocity model
//
// State: x = [px, py, pz, vx, vy, vz]ᵀ
// F = | I₃  dt·I₃ |    Q = q · | dt³/3·I₃  dt²/2·I₃ |
//     | 0    I₃   |            | dt²/2·I₃  dt·I₃    |
// ─────────────────────────────────────────────────────────────────────────────
void MultiViewTracker::ekfPredict(EkfState & state, double dt) const
{
  // ── State transition matrix F ─────────────────────────────────────────
  Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
  F(0, 3) = dt;
  F(1, 4) = dt;
  F(2, 5) = dt;

  // ── Process noise Q (continuous white-noise jerk model) ───────────────
  Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
  const double q = ekf_process_noise_;
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;

  for (int i = 0; i < 3; ++i) {
    Q(i, i)         = q * dt3 / 3.0;
    Q(i, i + 3)     = q * dt2 / 2.0;
    Q(i + 3, i)     = q * dt2 / 2.0;
    Q(i + 3, i + 3) = q * dt;
  }

  // ── Predict ───────────────────────────────────────────────────────────
  state.x = F * state.x;
  state.P = F * state.P * F.transpose() + Q;
}

// ─────────────────────────────────────────────────────────────────────────────
// EKF Update — position-only measurement
//
// H = | I₃  0₃ |    R = r · I₃
// ─────────────────────────────────────────────────────────────────────────────
void MultiViewTracker::ekfUpdate(
  EkfState & state,
  const Eigen::Vector3d & z_pos,
  const rclcpp::Time & stamp)
{
  // ── Measurement matrix H (observe position only) ──────────────────────
  Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
  H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  // ── Measurement noise R ───────────────────────────────────────────────
  Eigen::Matrix3d R_meas =
    Eigen::Matrix3d::Identity() * ekf_measurement_noise_;

  // ── Innovation ────────────────────────────────────────────────────────
  Eigen::Vector3d y = z_pos - H * state.x;

  // ── Innovation covariance ─────────────────────────────────────────────
  Eigen::Matrix3d S = H * state.P * H.transpose() + R_meas;

  // ── Kalman gain ───────────────────────────────────────────────────────
  Eigen::Matrix<double, 6, 3> K = state.P * H.transpose() * S.inverse();

  // ── State update ──────────────────────────────────────────────────────
  state.x = state.x + K * y;

  // ── Covariance update (Joseph form for numerical stability) ───────────
  Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 6> IKH = I6 - K * H;
  state.P = IKH * state.P * IKH.transpose() + K * R_meas * K.transpose();

  state.last_update_time = stamp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build the known tag model corners (in tag-local frame, z = 0)
//
// AprilTag corner ordering (same as isaac_ros_apriltag_interfaces):
//   0: bottom-left, 1: bottom-right, 2: top-right, 3: top-left
//
// Origin at tag centre, side length = 2 * tag_half_size_
// ─────────────────────────────────────────────────────────────────────────────
std::array<Eigen::Vector3d, NUM_CORNERS> MultiViewTracker::buildTagModel() const
{
  const double h = tag_half_size_;
  return {
    Eigen::Vector3d(-h, -h, 0.0),  // corner 0: bottom-left
    Eigen::Vector3d( h, -h, 0.0),  // corner 1: bottom-right
    Eigen::Vector3d( h,  h, 0.0),  // corner 2: top-right
    Eigen::Vector3d(-h,  h, 0.0)   // corner 3: top-left
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Convert Eigen::Isometry3d to geometry_msgs::PoseStamped
// ─────────────────────────────────────────────────────────────────────────────
geometry_msgs::msg::PoseStamped MultiViewTracker::toPoseStamped(
  const Eigen::Isometry3d & pose,
  const rclcpp::Time & stamp) const
{
  geometry_msgs::msg::PoseStamped msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = "world";

  // Translation
  msg.pose.position.x = pose.translation().x();
  msg.pose.position.y = pose.translation().y();
  msg.pose.position.z = pose.translation().z();

  // Rotation → quaternion
  Eigen::Quaterniond q(pose.rotation());
  q.normalize();
  msg.pose.orientation.w = q.w();
  msg.pose.orientation.x = q.x();
  msg.pose.orientation.y = q.y();
  msg.pose.orientation.z = q.z();

  return msg;
}

}  // namespace gsplat_tracker

// ─────────────────────────────────────────────────────────────────────────────
// main() — MultiThreadedExecutor with 4 threads
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<gsplat_tracker::MultiViewTracker>();

  // Use a MultiThreadedExecutor so that the 4 camera callbacks + timer
  // can execute concurrently on separate threads.
  // M1 fix: 6 threads for 4 sub callbacks + 1 timer + headroom.
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), /* num_threads = */ 6);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
