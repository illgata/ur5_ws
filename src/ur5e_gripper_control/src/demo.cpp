#include "ur5e_gripper_control/ur5e_gripper.h"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

struct ClassifiedGraspTask {
  std::string class_name;
  std::string frame_name;
  std::vector<double> grasp_pose;
  std::vector<double> place_pose;
};

std::vector<double> offset_z(const std::vector<double> &pose, double dz) {
  std::vector<double> result = pose;
  if (result.size() >= 3) {
    result[2] += dz;
  }
  return result;
}

void short_pause() {
  rclcpp::sleep_for(std::chrono::milliseconds(300));
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<UR5eGripper>(node_options);
  node->init();


  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread([&executor]() { executor.spin(); }).detach();

  std::vector<std::vector<double>> target_pose_list;
  node->get_target_pose_list(target_pose_list);
  std::string from_frame = "base_link";

  const std::vector<std::pair<std::string, size_t>> class_place_indices = {
      {"red_box", 0},
      {"blue_box", 1},
  };

  std::vector<ClassifiedGraspTask> grasp_tasks;
  for (const auto &[class_name, place_index] : class_place_indices) {
    if (place_index >= target_pose_list.size()) {
      RCLCPP_ERROR(rclcpp::get_logger("classified_grasp"),
                   "No target pose configured for class %s at index %zu",
                   class_name.c_str(), place_index);
      continue;
    }

    for (size_t i = 1; i <= 6; i++) {
      const std::string frame_name = class_name + std::to_string(i);
      std::vector<double> object_pose;
      node->get_cube_pose(from_frame, frame_name, object_pose);
      if (object_pose.empty()) {
        continue;
      }

      object_pose[0] -= 0.012;
      object_pose[1] += 0.01;
      object_pose[2] += 0.14;
      object_pose[3] = 0.0;
      object_pose[4] = M_PI;
      object_pose[5] = 0.0;

      RCLCPP_INFO(rclcpp::get_logger("classified_grasp"),
                  "Detected %s as %s: x=%f, y=%f, z=%f",
                  class_name.c_str(), frame_name.c_str(),
                  object_pose[0], object_pose[1], object_pose[2]);
      grasp_tasks.push_back({class_name, frame_name, object_pose, target_pose_list[place_index]});
    }
  }

  for (const auto &task : grasp_tasks) {
    RCLCPP_INFO(rclcpp::get_logger("classified_grasp"),
                "Grasping %s (%s)", task.frame_name.c_str(), task.class_name.c_str());

    bool grasp_success = node->plan_and_execute(task.grasp_pose);
    if (!grasp_success) {
      continue;
    }
    node->grasp(0.36);
    short_pause();

    std::vector<double> lift_pose = offset_z(task.grasp_pose, 0.12);
    std::vector<double> place_approach_pose = offset_z(task.place_pose, 0.12);

    bool lift_success = node->plan_and_execute(lift_pose);
    if (!lift_success) {
      RCLCPP_WARN(rclcpp::get_logger("classified_grasp"),
                  "Failed to lift %s after grasp, opening gripper and skipping",
                  task.frame_name.c_str());
      node->grasp(0);
      short_pause();
      continue;
    }

    bool approach_success = node->plan_and_execute(place_approach_pose);
    if (!approach_success) {
      RCLCPP_WARN(rclcpp::get_logger("classified_grasp"),
                  "Failed to reach place approach pose for %s, opening gripper and skipping",
                  task.frame_name.c_str());
      node->grasp(0);
      short_pause();
      continue;
    }

    bool place_success = node->plan_and_execute(task.place_pose);
    if (!place_success) {
      RCLCPP_WARN(rclcpp::get_logger("classified_grasp"),
                  "Failed to reach final place pose for %s, opening gripper at approach pose",
                  task.frame_name.c_str());
    }

    node->grasp(0);
    short_pause();
    node->plan_and_execute(place_approach_pose);
  }

  node->go_to_ready_position();
  rclcpp::shutdown();
  return 0;
}
