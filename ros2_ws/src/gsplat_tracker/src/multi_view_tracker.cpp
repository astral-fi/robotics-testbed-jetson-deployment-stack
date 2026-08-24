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
#include <limits>
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

  // SOLVEPNP_IPPE_SQUARE mandates the order
  //   [(-h,+h,0), (+h,+h,0), (+h,-h,0), (-h,-h,0)]
  // while buildTagModel() emits [(-h,-h), (-h,+h), (+h,+h), (+h,-h)].
  // IPPE index i therefore corresponds to model index (i + 1) % 4, and the
  // image points are permuted by the same rule so correspondence survives.
  obj_pts_ippe_.resize(NUM_CORNERS);
  for (int i = 0; i < NUM_CORNERS; ++i) {
    const auto & m = model[(i + 1) % NUM_CORNERS];
    obj_pts_ippe_[i] = cv::Point3d(m.x(), m.y(), m.z());
  }

  dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64F);

  // ── EKF tuning ──────────────────────────────────────────────────────────
  ekf_process_noise_     = this->get_parameter("ekf_process_noise").as_double();
  ekf_measurement_noise_ = this->get_parameter("ekf_measurement_noise").as_double();

  ekf_gate_chi2_           = this->get_parameter("ekf_gate_chi2").as_double();
  ekf_max_rejects_         = this->get_parameter("ekf_max_rejects").as_int();
  single_view_noise_scale_ = this->get_parameter("single_view_noise_scale").as_double();
  rotation_tau_            = this->get_parameter("rotation_tau").as_double();
  rotation_max_step_deg_   = this->get_parameter("rotation_max_step_deg").as_double();
  rotation_max_rejects_    = this->get_parameter("rotation_max_rejects").as_int();
  use_ippe_square_         = this->get_parameter("use_ippe_square").as_bool();
  sync_tolerance_          = this->get_parameter("sync_tolerance").as_double();
  max_reproj_error_px_     = this->get_parameter("max_reproj_error_px").as_double();

  if (rotation_tau_ <= 0.0) {
    throw std::invalid_argument("rotation_tau must be > 0");
  }

  RCLCPP_INFO(this->get_logger(),
    "Robustness: gate_chi2=%.2f single_view_scale=%.1f rot_tau=%.2fs "
    "rot_max_step=%.0fdeg pnp=%s",
    ekf_gate_chi2_, single_view_noise_scale_, rotation_tau_,
    rotation_max_step_deg_, use_ippe_square_ ? "IPPE_SQUARE" : "ITERATIVE");

  // ── Callback group: Reentrant allows concurrent execution ─────────────
  cb_group_ = this->create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);

  // ── QoS: BEST_EFFORT with depth 1 to match Jetson's publisher.
  // The Jetson uses fastdds_nonblocking.xml which enforces BEST_EFFORT.
  rclcpp::QoS qos_profile = rclcpp::QoS(
    rclcpp::KeepLast(1))
    .best_effort()
    .durability_volatile();

  // ── Create subscriptions for 4 cameras ────────────────────────────────
  rclcpp::SubscriptionOptions sub_opts;
  sub_opts.callback_group = cb_group_;

  const std::array<std::string, NUM_CAMERAS> topics = {
    "/cam_25251947/tag_detections",
    "/cam_25251937/tag_detections",
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
  pub_qos.reliable();
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
  this->declare_parameter<double>("tag_size", 0.21);              // metres
  this->declare_parameter<double>("ekf_process_noise", 0.01);
  this->declare_parameter<double>("ekf_measurement_noise", 0.05);
  this->declare_parameter<bool>("gsplat_opengl_convention", false);

  // ── Robustness / smoothing ──────────────────────────────────────────────
  // Chi-square gate on the 3-DoF position innovation: 11.34 = 99%,
  // 16.27 = 99.9%. Default is deliberately permissive — it should discard
  // gross outliers (mis-detections, PnP flips) without fighting real motion.
  this->declare_parameter<double>("ekf_gate_chi2", 16.27);
  this->declare_parameter<int>("ekf_max_rejects", 10);
  // A single-camera PnP fix is far less accurate than 3-view triangulation,
  // so it must not enter the filter with the same weight.
  this->declare_parameter<double>("single_view_noise_scale", 6.0);
  // Orientation smoothing time constant. tau=0.3 s at a 50 ms tick gives
  // alpha ~= 0.15; unlike the old fixed alpha it holds its meaning when the
  // update rate changes.
  this->declare_parameter<double>("rotation_tau", 0.3);
  this->declare_parameter<double>("rotation_max_step_deg", 45.0);
  this->declare_parameter<int>("rotation_max_rejects", 5);
  // SOLVEPNP_IPPE_SQUARE is purpose-built for square planar markers and does
  // not suffer the two-fold ambiguity that makes ITERATIVE flip between
  // mirrored poses on a coplanar 4-point target.
  this->declare_parameter<bool>("use_ippe_square", true);
  // Views are fused only if their stamps agree to within this many seconds.
  // WINDOW_SEC decides how long evidence lives; this decides what counts as
  // simultaneous. Loosen it if cameras are badly unsynchronised and you would
  // rather have a smeared multi-view fix than frequent single-view fallbacks.
  this->declare_parameter<double>("sync_tolerance", 0.05);
  // RMS reprojection error above which a fused pose is discarded, in pixels.
  this->declare_parameter<double>("max_reproj_error_px", 8.0);
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
    {"cam0", "cam1", "cam2"};

  for (int i = 0; i < NUM_CAMERAS; ++i) {
    cv::FileNode cam = cameras_node[cam_names[i]];

    // ── Frame ID ──────────────────────────────────────────────────────────
    calibrations_[i].frame_id = static_cast<std::string>(cam["frame_id"]);

    // ── Intrinsic matrix K (3×3, row-major flat list) ────────────────────
    std::vector<double> k_vec;
    cam["K"] >> k_vec;
    calibrations_[i].K = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
      k_vec.data());

    // ── Rotation R (3×3, camera → world) ────────────────────────────────
    std::vector<double> r_vec;
    cam["R"] >> r_vec;
    Eigen::Matrix3d R_c2w = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
      r_vec.data());

    if (this->get_parameter("gsplat_opengl_convention").as_bool()) {
      // gsplat uses OpenGL camera convention (Y up, Z back).
      // OpenCV (used by DLT and solvePnP) uses Y down, Z forward.
      // Convert C2W rotation from OpenGL to OpenCV by flipping Y and Z axes.
      R_c2w.col(1) *= -1.0;
      R_c2w.col(2) *= -1.0;
    }

    // ── Translation t (3×1, camera → world) ─────────────────────────────
    std::vector<double> t_vec;
    cam["t"] >> t_vec;
    Eigen::Vector3d t_c2w = Eigen::Map<Eigen::Vector3d>(t_vec.data());

    // ── Convert Camera-to-World to World-to-Camera ──────────────────────
    calibrations_[i].R = R_c2w.transpose();
    calibrations_[i].t = -calibrations_[i].R * t_c2w;

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

  // ── Age out the window under lock, then take a copy ───────────────────
  //
  // This used to swap() the cache, draining it on every 50 ms tick. That made
  // the 200 ms WINDOW_SEC purely decorative: two cameras only ever fused if
  // their detections happened to land in the same tick, so with unsynchronised
  // cameras most frames fell through to single-view solvePnP. Alternating
  // between 3-view triangulation and single-view PnP — which have very
  // different error characteristics — was itself a major source of jitter.
  //
  // Now the window genuinely slides: entries persist for WINDOW_SEC and are
  // pruned by age. Re-consuming the same detection is prevented per tag by
  // EkfState::last_measurement_stamp rather than by destroying the evidence.
  std::map<int, std::vector<CameraDetection>> local_cache;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (auto it = window_cache_.begin(); it != window_cache_.end(); ) {
      auto & dets = it->second;
      dets.erase(
        std::remove_if(dets.begin(), dets.end(),
          [&](const CameraDetection & d) {
            return std::abs((now - d.stamp).seconds()) > WINDOW_SEC;
          }),
        dets.end());
      if (dets.empty()) {
        it = window_cache_.erase(it);
      } else {
        ++it;
      }
    }
    local_cache = window_cache_;
  }

  if (!local_cache.empty()) {
    size_t total_dets = 0;
    for (const auto & [tid, dets] : local_cache) total_dets += dets.size();
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "window: %zu tags, %zu detections", local_cache.size(), total_dets);
  }

  // Collect poses to publish OUTSIDE the ekf_mutex_ (H1 fix: never
  // call publish() while holding a mutex — it can block in DDS).
  std::vector<geometry_msgs::msg::PoseStamped> poses_to_publish;

  for (auto & [tag_id, detections] : local_cache) {
    // ── Keep only the most recent detection per camera ────────────────
    // (Ageing already happened against the shared window above.)
    std::map<int, CameraDetection> unique_cams;
    for (const auto & d : detections) {
      auto it = unique_cams.find(d.camera_id);
      if (it == unique_cams.end() || d.stamp > it->second.stamp) {
        unique_cams[d.camera_id] = d;
      }
    }
    detections.clear();
    for (const auto & [cid, d] : unique_cams) {
      detections.push_back(d);
    }

    if (detections.empty()) continue;

    // ── Newest detection in this batch drives the filter timestamp ─────
    rclcpp::Time best_stamp = detections.front().stamp;
    for (const auto & d : detections) {
      if (d.stamp > best_stamp) best_stamp = d.stamp;
    }

    // ── Discard views that are not contemporaneous with the newest ─────
    // Ageing above only guarantees each detection is within WINDOW_SEC of
    // NOW — not that the views agree with each other. Triangulating a view
    // from 180 ms ago against a fresh one reconstructs a quad the tag never
    // occupied, which Kabsch then dutifully fits, yielding both position
    // error and spurious tilt. Better a clean single-view fix than a
    // smeared multi-view one.
    if (detections.size() > 1) {
      const size_t before_sync = detections.size();
      detections.erase(
        std::remove_if(detections.begin(), detections.end(),
          [&](const CameraDetection & d) {
            return (best_stamp - d.stamp).seconds() > sync_tolerance_;
          }),
        detections.end());
      if (detections.size() < before_sync) {
        static rclcpp::Clock steady_clock_sync(RCL_STEADY_TIME);
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), steady_clock_sync, 2000,
          "Tag %d: dropped %zu of %zu views as non-contemporaneous "
          "(sync_tolerance=%.3fs)",
          tag_id, before_sync - detections.size(), before_sync,
          sync_tolerance_);
      }
    }

    if (detections.empty()) continue;

    // ── Skip tags with no new evidence since the last update ───────────
    // Without this the sliding window would re-feed the same measurement on
    // every tick, artificially collapsing the covariance and making the
    // filter over-confident in stale data.
    {
      std::lock_guard<std::mutex> lock(ekf_mutex_);
      const auto it = ekf_states_.find(tag_id);
      if (it != ekf_states_.end() && it->second.initialized &&
          best_stamp <= it->second.last_measurement_stamp)
      {
        continue;
      }
    }

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    bool valid = false;

    if (detections.size() >= 2) {
      // ── Multi-view DLT triangulation for each corner ──────────────────
      std::array<Eigen::Vector3d, NUM_CORNERS> pts_3d;

      bool dlt_ok = true;

      for (int c = 0; c < NUM_CORNERS; ++c) {
        std::vector<Eigen::Matrix<double, 3, 4>> Ps;
        std::vector<Eigen::Vector2d> pixels;

        for (const auto & d : detections) {
          Ps.push_back(calibrations_[d.camera_id].P);
          pixels.push_back(d.corners_px[c]);
        }

        pts_3d[c] = dltTriangulate(Ps, pixels);
        if (std::isnan(pts_3d[c].x())) {
          dlt_ok = false;
          break;
        }
      }

      if (dlt_ok) {
        // ── Kabsch alignment: measured 3D corners → known tag model ───────
        pose = kabschAlign(pts_3d, model_corners);
        valid = true;

        RCLCPP_DEBUG(this->get_logger(),
          "Tag %d: DLT+Kabsch from %zu views", tag_id, detections.size());
      } else {
        // ── DLT Failed: Fallback to EPnP ──────────────────────────────────
        pose = solvePnPFallback(detections.front());
        valid = true;
        
        static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
        RCLCPP_WARN_THROTTLE(this->get_logger(), steady_clock, 1000,
          "Tag %d: DLT triangulation degenerate, fallback to solvePnP (cam%d)", 
          tag_id, detections.front().camera_id);
      }

    } else {
      // ── Single-camera fallback: EPnP ──────────────────────────────────
      pose = solvePnPFallback(detections.front());
      valid = true;

      RCLCPP_DEBUG(this->get_logger(),
        "Tag %d: solvePnP fallback (cam%d)", tag_id,
        detections.front().camera_id);
    }

    // ── Reprojection check: does this pose explain the pixels? ──────────
    // Measured in detector units, so it catches failures a metric threshold
    // cannot: mis-ordered corners, degenerate triangulation, ambiguity flips.
    if (valid) {
      const double rms_px = reprojectionRmsPx(pose, detections, model_corners);
      if (!std::isfinite(rms_px) || rms_px > max_reproj_error_px_) {
        static rclcpp::Clock steady_clock_rp(RCL_STEADY_TIME);
        RCLCPP_WARN_THROTTLE(this->get_logger(), steady_clock_rp, 1000,
          "Tag %d: pose rejected, reprojection RMS %.2f px > %.2f px "
          "(%zu view(s))", tag_id, rms_px, max_reproj_error_px_,
          detections.size());
        valid = false;
      }
    }

    // ── Hard Sanity Check (Prevent 9e11 Explosions) ─────────────────────
    if (valid) {
      if (!std::isfinite(pose.translation().x()) || pose.translation().norm() > 100.0) {
        static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
        RCLCPP_WARN_THROTTLE(this->get_logger(), steady_clock, 1000,
          "Tag %d: Computed pose is physically impossible (norm = %g). Rejecting.", 
          tag_id, pose.translation().norm());
        valid = false;
      }
    }

    if (!valid) continue;

    // ── Weight the measurement by how it was obtained ───────────────────
    // 3-view DLT triangulation is far tighter than a single-camera PnP fix;
    // feeding both in with identical R is what let single-view frames yank
    // the estimate around.
    const size_t n_views = detections.size();
    double meas_noise = ekf_measurement_noise_;
    if (n_views == 1) {
      meas_noise *= single_view_noise_scale_;
    } else if (n_views >= 3) {
      meas_noise *= 0.5;
    }

    // ── EKF smooth the position (minimised critical section — H1 fix) ───
    {
      std::lock_guard<std::mutex> lock(ekf_mutex_);
      auto & state = ekf_states_[tag_id];

      if (!state.initialized) {
        state.x.head<3>() = pose.translation();
        state.x.tail<3>().setZero();
        state.P = Eigen::Matrix<double, 6, 6>::Identity() * 1.0;
        state.last_update_time = best_stamp;
        state.last_measurement_stamp = best_stamp;
        state.smoothed_q = Eigen::Quaterniond(pose.rotation());
        state.consecutive_rejects = 0;
        state.consecutive_rot_rejects = 0;
        state.initialized = true;
      } else {
        const double dt = (best_stamp - state.last_update_time).seconds();
        if (dt > 0.0 && dt < 1.0) {
          ekfPredict(state, dt);
        }

        const bool accepted =
          ekfUpdate(state, pose.translation(), best_stamp, meas_noise);
        state.last_measurement_stamp = best_stamp;

        if (!accepted) {
          static rclcpp::Clock steady_clock_gate(RCL_STEADY_TIME);
          RCLCPP_WARN_THROTTLE(this->get_logger(), steady_clock_gate, 1000,
            "Tag %d: position outlier gated out (%d consecutive, %zu view(s))",
            tag_id, state.consecutive_rejects, n_views);
        }

        // ── Orientation: reject flips, then smooth with a real time constant
        // A rejected position measurement almost always carries a rubbish
        // orientation too, so orientation only advances on accepted frames.
        if (accepted) {
          Eigen::Quaterniond raw_q(pose.rotation());
          if (state.smoothed_q.dot(raw_q) < 0.0) {
            raw_q.coeffs() = -raw_q.coeffs();  // shortest path
          }

          const double step_rad = state.smoothed_q.angularDistance(raw_q);
          const double max_step_rad = rotation_max_step_deg_ * M_PI / 180.0;

          if (step_rad > max_step_rad &&
              state.consecutive_rot_rejects < rotation_max_rejects_)
          {
            // A single large step is far more likely a PnP ambiguity flip
            // than genuine motion. Blending it in at a fixed alpha — as the
            // old code did — is what produced the visible swinging.
            ++state.consecutive_rot_rejects;
            static rclcpp::Clock steady_clock_rot(RCL_STEADY_TIME);
            RCLCPP_WARN_THROTTLE(this->get_logger(), steady_clock_rot, 1000,
              "Tag %d: rotation jump %.1f deg rejected (%d consecutive)",
              tag_id, step_rad * 180.0 / M_PI, state.consecutive_rot_rejects);
          } else {
            // Sustained large steps mean the tag really did move: stop
            // fighting it and let the filter follow.
            state.consecutive_rot_rejects = 0;

            // alpha = 1 - exp(-dt / tau) keeps the smoothing time constant
            // fixed in seconds, so a change in update rate no longer silently
            // changes how heavily orientation is filtered.
            const double eff_dt = (dt > 0.0 && dt < 1.0) ? dt : 0.05;
            const double alpha =
              1.0 - std::exp(-eff_dt / rotation_tau_);
            state.smoothed_q = state.smoothed_q.slerp(alpha, raw_q);
            state.smoothed_q.normalize();
          }
        }
      }

      // Copy smoothed result before releasing lock.
      Eigen::Isometry3d smoothed = pose;
      smoothed.translation() = state.x.head<3>();
      smoothed.linear() = state.smoothed_q.toRotationMatrix();
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
  // We use 1e-4 because if w < 1e-4, the point is > 10,000 units away.
  if (std::abs(X_homogeneous(3)) < 1e-4) {
    // M2 fix: throttle to 1 Hz to prevent log flooding from bad data.
    static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
    RCLCPP_WARN_THROTTLE(this->get_logger(), steady_clock, 1000,
      "DLT: degenerate point (w = %g), returning NaN", X_homogeneous(3));
    return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
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

  // ── Solve PnP ───────────────────────────────────────────────────────
  // ITERATIVE on four coplanar points has a two-fold ambiguity: the mirrored
  // pose fits the projection nearly as well, and the solver can hop between
  // them frame to frame. That shows up as a tag reading tens of degrees off
  // true and flickering. IPPE_SQUARE is derived specifically for square
  // planar markers and returns the lower-reprojection-error solution.
  cv::Mat rvec, tvec;
  bool ok;
  if (use_ippe_square_) {
    // Permute image points into IPPE's mandated corner order, matching the
    // permutation applied to obj_pts_ippe_ at construction.
    std::array<cv::Point2d, NUM_CORNERS> ippe_pts;
    for (int i = 0; i < NUM_CORNERS; ++i) {
      ippe_pts[i] = img_pts[(i + 1) % NUM_CORNERS];
    }
    ok = cv::solvePnP(obj_pts_ippe_, ippe_pts, cal.K_cv, dist_coeffs_,
                      rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
  } else {
    ok = cv::solvePnP(obj_pts_, img_pts, cal.K_cv, dist_coeffs_,
                      rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
  }
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
bool MultiViewTracker::ekfUpdate(
  EkfState & state,
  const Eigen::Vector3d & z_pos,
  const rclcpp::Time & stamp,
  double meas_noise)
{
  // ── Measurement matrix H (observe position only) ──────────────────────
  Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
  H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  // ── Measurement noise R (per-measurement, set by the caller) ──────────
  Eigen::Matrix3d R_meas = Eigen::Matrix3d::Identity() * meas_noise;

  // ── Innovation ────────────────────────────────────────────────────────
  Eigen::Vector3d y = z_pos - H * state.x;

  // ── Innovation covariance ─────────────────────────────────────────────
  Eigen::Matrix3d S = H * state.P * H.transpose() + R_meas;
  const Eigen::Matrix3d S_inv = S.inverse();

  // ── Chi-square gate on the normalised innovation ──────────────────────
  // Squared Mahalanobis distance is chi-square distributed with 3 DoF. A
  // mis-detected tag or a flipped PnP solution lands far out in that tail;
  // previously any such value went straight into the state at full weight.
  const double mahalanobis2 = y.dot(S_inv * y);

  if (mahalanobis2 > ekf_gate_chi2_) {
    ++state.consecutive_rejects;

    if (state.consecutive_rejects < ekf_max_rejects_) {
      // Keep coasting on the prediction; do not fold the outlier in.
      state.last_update_time = stamp;
      return false;
    }

    // Persistent rejection means the filter, not the measurement, is wrong
    // (a teleport, a long dropout, or divergence). Re-seed from the
    // observation rather than gating ourselves into a permanent freeze.
    state.x.head<3>() = z_pos;
    state.x.tail<3>().setZero();
    state.P = Eigen::Matrix<double, 6, 6>::Identity();
    state.consecutive_rejects = 0;
    state.last_update_time = stamp;
    return true;
  }

  state.consecutive_rejects = 0;

  // ── Kalman gain ───────────────────────────────────────────────────────
  Eigen::Matrix<double, 6, 3> K = state.P * H.transpose() * S_inv;

  // ── State update ──────────────────────────────────────────────────────
  state.x = state.x + K * y;

  // ── Covariance update (Joseph form for numerical stability) ───────────
  Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 6> IKH = I6 - K * H;
  state.P = IKH * state.P * IKH.transpose() + K * R_meas * K.transpose();

  state.last_update_time = stamp;
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// RMS reprojection error of a candidate pose against its own observations
// ─────────────────────────────────────────────────────────────────────────────
double MultiViewTracker::reprojectionRmsPx(
  const Eigen::Isometry3d & pose,
  const std::vector<CameraDetection> & detections,
  const std::array<Eigen::Vector3d, NUM_CORNERS> & model) const
{
  double sum_sq = 0.0;
  int n = 0;

  for (const auto & d : detections) {
    if (d.camera_id < 0 || d.camera_id >= NUM_CAMERAS) continue;
    const Eigen::Matrix<double, 3, 4> & P = calibrations_[d.camera_id].P;

    for (int c = 0; c < NUM_CORNERS; ++c) {
      // Tag-local corner → world → homogeneous pixel
      const Eigen::Vector3d Xw = pose * model[c];
      Eigen::Vector4d Xh;
      Xh << Xw.x(), Xw.y(), Xw.z(), 1.0;
      const Eigen::Vector3d xh = P * Xh;

      // Behind the camera, or on the principal plane: the pose cannot be
      // explaining this observation at all.
      if (xh.z() <= 1e-9) {
        return std::numeric_limits<double>::infinity();
      }

      const Eigen::Vector2d px(xh.x() / xh.z(), xh.y() / xh.z());
      sum_sq += (px - d.corners_px[c]).squaredNorm();
      ++n;
    }
  }

  if (n == 0) return std::numeric_limits<double>::infinity();
  return std::sqrt(sum_sq / static_cast<double>(n));
}

// ─────────────────────────────────────────────────────────────────────────────
// Build the known tag model corners (in tag-local frame, z = 0)
//
// AprilTag corner ordering (must match isaac_ros_apriltag_interfaces):
//   0: bottom-left, 1: top-left, 2: top-right, 3: bottom-right
// NOTE: the previous comment here read "bottom-left, bottom-right, top-right,
// top-left", which is the reverse winding and does NOT describe the code
// below. If the detector's true order ever turns out to be that reversed
// winding, the recovered rotation is mirrored — worth verifying against a
// tag at a known attitude.
//
// Origin at tag centre, side length = 2 * tag_half_size_
// ─────────────────────────────────────────────────────────────────────────────
std::array<Eigen::Vector3d, NUM_CORNERS> MultiViewTracker::buildTagModel() const
{
  const double h = tag_half_size_;
  return {
    Eigen::Vector3d(-h, -h, 0.0),  // corner 0: bottom-left
    Eigen::Vector3d(-h,  h, 0.0),  // corner 1: top-left
    Eigen::Vector3d( h,  h, 0.0),  // corner 2: top-right
    Eigen::Vector3d( h, -h, 0.0)   // corner 3: bottom-right
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
