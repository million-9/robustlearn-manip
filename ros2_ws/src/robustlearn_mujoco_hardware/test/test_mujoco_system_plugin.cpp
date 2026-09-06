// Copyright 2026 Mohamed Musthafa
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "hardware_interface/hardware_component.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_component_params.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "mujoco/mujoco.h"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/logger.hpp"
#include "robustlearn_mujoco_hardware/mujoco_system.hpp"

namespace robustlearn_mujoco_hardware
{

class MuJoCoSystemTestPeer
{
public:
  static void set_joint_state(
    MuJoCoSystem & system,
    const std::array<double, 7> & positions,
    const std::array<double, 7> & velocities)
  {
    for (std::size_t index = 0; index < positions.size(); ++index) {
      system.data_->qpos[system.qpos_addresses_[index]] =
        positions[index];

      system.data_->qvel[system.dof_addresses_[index]] =
        velocities[index];
    }
  }

  static std::array<double, 7> panda_actuator_controls(
    const MuJoCoSystem & system)
  {
    std::array<double, 7> controls{};

    for (std::size_t index = 0; index < controls.size(); ++index) {
      controls[index] =
        system.data_->ctrl[system.actuator_ids_[index]];
    }

    return controls;
  }

  static double actuator_control(
    const MuJoCoSystem & system,
    const char * actuator_name)
  {
    const int actuator_id = mj_name2id(
      system.model_.get(),
      mjOBJ_ACTUATOR,
      actuator_name);

    if (actuator_id < 0) {
      throw std::runtime_error(
              std::string("Missing MuJoCo actuator: ") + actuator_name);
    }

    return system.data_->ctrl[actuator_id];
  }

  static double simulation_time(const MuJoCoSystem & system)
  {
    return system.data_->time;
  }

  static double physics_timestep(const MuJoCoSystem & system)
  {
    return system.model_->opt.timestep;
  }
};

}  // namespace robustlearn_mujoco_hardware

namespace
{

hardware_interface::InterfaceInfo make_interface(const std::string & name)
{
  hardware_interface::InterfaceInfo interface_info{};
  interface_info.name = name;
  return interface_info;
}

hardware_interface::ComponentInfo make_panda_joint(const int joint_number)
{
  hardware_interface::ComponentInfo joint{};
  joint.name = "panda_joint" + std::to_string(joint_number);
  joint.type = "joint";

  joint.command_interfaces = {
    make_interface(hardware_interface::HW_IF_POSITION),
  };

  joint.state_interfaces = {
    make_interface(hardware_interface::HW_IF_POSITION),
    make_interface(hardware_interface::HW_IF_VELOCITY),
  };

  return joint;
}

hardware_interface::HardwareInfo make_valid_hardware_info()
{
  hardware_interface::HardwareInfo hardware_info{};
  hardware_info.name = "PandaMuJoCoSystem";
  hardware_info.type = "system";
  hardware_info.hardware_plugin_name =
    "robustlearn_mujoco_hardware/MuJoCoSystem";

  hardware_info.hardware_parameters["model_path"] =
    ROBUSTLEARN_TEST_MODEL_PATH;

  for (int joint_number = 1; joint_number <= 7; ++joint_number) {
    hardware_info.joints.push_back(make_panda_joint(joint_number));
  }

  return hardware_info;
}

hardware_interface::HardwareComponentParams make_component_params(
  hardware_interface::HardwareInfo hardware_info)
{
  hardware_interface::HardwareComponentParams params{};
  params.hardware_info = std::move(hardware_info);
  params.logger = rclcpp::get_logger("mujoco_system_test");
  params.clock = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  return params;
}

hardware_interface::CallbackReturn initialize(
  hardware_interface::HardwareInfo hardware_info)
{
  robustlearn_mujoco_hardware::MuJoCoSystem system;

  return system.init(
    make_component_params(std::move(hardware_info)));
}

struct SyntheticModelOptions
{
  int missing_joint = 0;
  int slide_joint = 0;
  int missing_actuator = 0;
  int remapped_actuator = 0;
  int remapped_target_joint = 0;
};

std::filesystem::path write_synthetic_model(
  const std::string & filename,
  const SyntheticModelOptions & options)
{
  const auto model_path =
    std::filesystem::temp_directory_path() / filename;

  const auto fixed_peg_path =
    std::filesystem::path(ROBUSTLEARN_TEST_MODEL_PATH)
    .parent_path() / "fixed_peg.xml";

  std::ofstream output(model_path);

  if (!output) {
    throw std::runtime_error(
            "Failed to create synthetic MuJoCo test model");
  }

  output
    << "<mujoco model=\"robustlearn synthetic panda test\">\n"
    << "  <asset>\n"
    << "    <model name=\"fixed_peg\" file=\""
    << fixed_peg_path.string()
    << "\"/>\n"
    << "  </asset>\n"
    << "  <worldbody>\n"
    << "    <body name=\"hand\"/>\n";

  for (int joint_number = 1; joint_number <= 7; ++joint_number) {
    if (joint_number == options.missing_joint) {
      continue;
    }

    const char * joint_type =
      joint_number == options.slide_joint ? "slide" : "hinge";

    output
      << "    <body name=\"test_body"
      << joint_number
      << "\">\n"
      << "      <joint name=\"joint"
      << joint_number
      << "\" type=\""
      << joint_type
      << "\"/>\n"
      << "      <geom type=\"sphere\" size=\"0.01\" mass=\"1\"/>\n"
      << "    </body>\n";
  }

  output
    << "  </worldbody>\n"
    << "  <actuator>\n";

  for (int actuator_number = 1; actuator_number <= 7; ++actuator_number) {
    if (actuator_number == options.missing_actuator) {
      continue;
    }

    int target_joint = actuator_number;

    if (actuator_number == options.remapped_actuator) {
      target_joint = options.remapped_target_joint;
    }

    if (target_joint == options.missing_joint) {
      continue;
    }

    output
      << "    <position name=\"actuator"
      << actuator_number
      << "\" joint=\"joint"
      << target_joint
      << "\" kp=\"1\"/>\n";
  }

  output
    << "  </actuator>\n"
    << "</mujoco>\n";

  output.close();

  return model_path;
}

hardware_interface::CallbackReturn initialize_with_model(
  const SyntheticModelOptions & options,
  const std::string & filename)
{
  const auto model_path =
    write_synthetic_model(filename, options);

  auto hardware_info = make_valid_hardware_info();
  hardware_info.hardware_parameters["model_path"] =
    model_path.string();

  const auto result =
    initialize(std::move(hardware_info));

  std::filesystem::remove(model_path);

  return result;
}

TEST(MuJoCoSystemPluginTest, plugin_is_discoverable_and_loadable)
{
  pluginlib::ClassLoader<hardware_interface::SystemInterface> loader(
    "hardware_interface",
    "hardware_interface::SystemInterface");

  const std::string plugin_name =
    "robustlearn_mujoco_hardware/MuJoCoSystem";

  EXPECT_TRUE(loader.isClassAvailable(plugin_name));

  const auto instance = loader.createSharedInstance(plugin_name);

  ASSERT_NE(instance, nullptr);
}

TEST(MuJoCoSystemInitializationTest, valid_panda_description_initializes)
{
  EXPECT_EQ(
    initialize(make_valid_hardware_info()),
    hardware_interface::CallbackReturn::SUCCESS);
}

TEST(MuJoCoSystemInitializationTest, missing_model_path_fails)
{
  auto hardware_info = make_valid_hardware_info();
  hardware_info.hardware_parameters.erase("model_path");

  EXPECT_EQ(
    initialize(std::move(hardware_info)),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(MuJoCoSystemInitializationTest, nonexistent_model_path_fails)
{
  auto hardware_info = make_valid_hardware_info();

  hardware_info.hardware_parameters["model_path"] =
    "/tmp/robustlearn_model_that_does_not_exist.xml";

  EXPECT_EQ(
    initialize(std::move(hardware_info)),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(MuJoCoSystemInitializationTest, wrong_joint_count_fails)
{
  auto hardware_info = make_valid_hardware_info();
  hardware_info.joints.pop_back();

  EXPECT_EQ(
    initialize(std::move(hardware_info)),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(MuJoCoSystemInitializationTest, missing_required_joint_name_fails)
{
  auto hardware_info = make_valid_hardware_info();
  hardware_info.joints.at(3).name = "wrong_joint4";

  EXPECT_EQ(
    initialize(std::move(hardware_info)),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(MuJoCoSystemInitializationTest, wrong_command_interface_fails)
{
  auto hardware_info = make_valid_hardware_info();

  hardware_info.joints.at(0).command_interfaces = {
    make_interface(hardware_interface::HW_IF_VELOCITY),
  };

  EXPECT_EQ(
    initialize(std::move(hardware_info)),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(MuJoCoSystemInitializationTest, missing_state_interface_fails)
{
  auto hardware_info = make_valid_hardware_info();

  hardware_info.joints.at(1).state_interfaces = {
    make_interface(hardware_interface::HW_IF_POSITION),
  };

  EXPECT_EQ(
    initialize(std::move(hardware_info)),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(MuJoCoSystemInitializationTest, extra_state_interface_fails)
{
  auto hardware_info = make_valid_hardware_info();

  hardware_info.joints.at(2).state_interfaces.push_back(
    make_interface("effort"));

  EXPECT_EQ(
    initialize(std::move(hardware_info)),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(MuJoCoSystemInitializationTest, missing_mujoco_joint_fails)
{
  SyntheticModelOptions options{};
  options.missing_joint = 7;
  options.missing_actuator = 7;

  EXPECT_EQ(
    initialize_with_model(
      options,
      "robustlearn_missing_joint_test.xml"),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(MuJoCoSystemInitializationTest, non_hinge_panda_joint_fails)
{
  SyntheticModelOptions options{};
  options.slide_joint = 4;

  EXPECT_EQ(
    initialize_with_model(
      options,
      "robustlearn_slide_joint_test.xml"),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(MuJoCoSystemInitializationTest, missing_mujoco_actuator_fails)
{
  SyntheticModelOptions options{};
  options.missing_actuator = 6;

  EXPECT_EQ(
    initialize_with_model(
      options,
      "robustlearn_missing_actuator_test.xml"),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(MuJoCoSystemInitializationTest, actuator_targeting_wrong_joint_fails)
{
  SyntheticModelOptions options{};
  options.remapped_actuator = 3;
  options.remapped_target_joint = 4;

  EXPECT_EQ(
    initialize_with_model(
      options,
      "robustlearn_wrong_actuator_mapping_test.xml"),
    hardware_interface::CallbackReturn::ERROR);
}


TEST(MuJoCoSystemReadTest, known_mujoco_state_updates_ros_state_interfaces)
{
  auto system =
    std::make_unique<robustlearn_mujoco_hardware::MuJoCoSystem>();

  auto * system_ptr = system.get();

  hardware_interface::HardwareComponent component(
    std::move(system));

  component.initialize(
    make_component_params(make_valid_hardware_info()));

  auto state_interfaces =
    component.export_state_interfaces();

  auto command_interfaces =
    component.export_command_interfaces();

  ASSERT_EQ(state_interfaces.size(), 14u);
  ASSERT_EQ(command_interfaces.size(), 7u);

  component.configure();
  component.activate();

  ASSERT_EQ(
    component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  const std::array<double, 7> expected_positions = {
    0.11,
    -0.22,
    0.33,
    -0.44,
    0.55,
    -0.66,
    0.77,
  };

  const std::array<double, 7> expected_velocities = {
    -1.1,
    1.2,
    -1.3,
    1.4,
    -1.5,
    1.6,
    -1.7,
  };

  robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::set_joint_state(
    *system_ptr,
    expected_positions,
    expected_velocities);

  const auto first_read =
    system_ptr->read(
    rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
    rclcpp::Duration::from_seconds(0.001));

  ASSERT_EQ(
    first_read,
    hardware_interface::return_type::OK);

  for (std::size_t index = 0; index < expected_positions.size(); ++index) {
    const std::string joint_name =
      "panda_joint" + std::to_string(index + 1);

    const auto position_it =
      std::find_if(
        state_interfaces.begin(),
        state_interfaces.end(),
      [&joint_name](const auto & interface)
      {
        return
          interface->get_prefix_name() == joint_name &&
          interface->get_interface_name() ==
          hardware_interface::HW_IF_POSITION;
        });

    const auto velocity_it =
      std::find_if(
        state_interfaces.begin(),
        state_interfaces.end(),
      [&joint_name](const auto & interface)
      {
        return
          interface->get_prefix_name() == joint_name &&
          interface->get_interface_name() ==
          hardware_interface::HW_IF_VELOCITY;
        });

    ASSERT_NE(position_it, state_interfaces.end());
    ASSERT_NE(velocity_it, state_interfaces.end());

    const auto position =
      (*position_it)->get_optional<double>();

    const auto velocity =
      (*velocity_it)->get_optional<double>();

    ASSERT_TRUE(position.has_value());
    ASSERT_TRUE(velocity.has_value());

    EXPECT_DOUBLE_EQ(
      *position,
      expected_positions[index]);

    EXPECT_DOUBLE_EQ(
      *velocity,
      expected_velocities[index]);
  }

  const auto second_read =
    system_ptr->read(
    rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
    rclcpp::Duration::from_seconds(0.001));

  EXPECT_EQ(
    second_read,
    hardware_interface::return_type::OK);

  for (std::size_t index = 0; index < expected_positions.size(); ++index) {
    const std::string joint_name =
      "panda_joint" + std::to_string(index + 1);

    for (const auto & state_interface : state_interfaces) {
      if (state_interface->get_prefix_name() != joint_name) {
        continue;
      }

      const auto value =
        state_interface->get_optional<double>();

      ASSERT_TRUE(value.has_value());

      if (
        state_interface->get_interface_name() ==
        hardware_interface::HW_IF_POSITION)
      {
        EXPECT_DOUBLE_EQ(
          *value,
          expected_positions[index]);
      } else if (
        state_interface->get_interface_name() ==
        hardware_interface::HW_IF_VELOCITY)
      {
        EXPECT_DOUBLE_EQ(
          *value,
          expected_velocities[index]);
      }
    }
  }
}

TEST(MuJoCoSystemReadTest, non_finite_mujoco_state_is_rejected_transactionally)
{
  auto system =
    std::make_unique<robustlearn_mujoco_hardware::MuJoCoSystem>();

  auto * system_ptr = system.get();

  hardware_interface::HardwareComponent component(
    std::move(system));

  component.initialize(
    make_component_params(make_valid_hardware_info()));

  auto state_interfaces =
    component.export_state_interfaces();

  auto command_interfaces =
    component.export_command_interfaces();

  ASSERT_EQ(state_interfaces.size(), 14u);
  ASSERT_EQ(command_interfaces.size(), 7u);

  component.configure();
  component.activate();

  ASSERT_EQ(
    component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  const std::array<double, 7> valid_positions = {
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
  };

  const std::array<double, 7> valid_velocities = {
    -0.10,
    -0.20,
    -0.30,
    -0.40,
    -0.50,
    -0.60,
    -0.70,
  };

  robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::set_joint_state(
    *system_ptr,
    valid_positions,
    valid_velocities);

  ASSERT_EQ(
    system_ptr->read(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(0.001)),
    hardware_interface::return_type::OK);

  std::array<double, 7> invalid_positions = {
    1.10,
    1.20,
    1.30,
    1.40,
    1.50,
    1.60,
    1.70,
  };

  const std::array<double, 7> changed_velocities = {
    -1.10,
    -1.20,
    -1.30,
    -1.40,
    -1.50,
    -1.60,
    -1.70,
  };

  invalid_positions[3] =
    std::numeric_limits<double>::quiet_NaN();

  robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::set_joint_state(
    *system_ptr,
    invalid_positions,
    changed_velocities);

  EXPECT_EQ(
    system_ptr->read(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(0.001)),
    hardware_interface::return_type::ERROR);

  for (std::size_t index = 0; index < valid_positions.size(); ++index) {
    const std::string joint_name =
      "panda_joint" + std::to_string(index + 1);

    for (const auto & state_interface : state_interfaces) {
      if (state_interface->get_prefix_name() != joint_name) {
        continue;
      }

      const auto value =
        state_interface->get_optional<double>();

      ASSERT_TRUE(value.has_value());
      EXPECT_TRUE(std::isfinite(*value));

      if (
        state_interface->get_interface_name() ==
        hardware_interface::HW_IF_POSITION)
      {
        EXPECT_DOUBLE_EQ(
          *value,
          valid_positions[index]);
      } else if (
        state_interface->get_interface_name() ==
        hardware_interface::HW_IF_VELOCITY)
      {
        EXPECT_DOUBLE_EQ(
          *value,
          valid_velocities[index]);
      }
    }
  }
}

TEST(MuJoCoSystemWriteTest, valid_ros_commands_reach_named_panda_actuators)
{
  auto system =
    std::make_unique<robustlearn_mujoco_hardware::MuJoCoSystem>();

  auto * system_ptr = system.get();

  hardware_interface::HardwareComponent component(
    std::move(system));

  component.initialize(
    make_component_params(make_valid_hardware_info()));

  auto state_interfaces =
    component.export_state_interfaces();

  auto command_interfaces =
    component.export_command_interfaces();

  ASSERT_EQ(state_interfaces.size(), 14u);
  ASSERT_EQ(command_interfaces.size(), 7u);

  component.configure();
  component.activate();

  ASSERT_EQ(
    component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  const std::array<double, 7> expected_commands = {
    0.20,
    -0.30,
    0.40,
    -1.20,
    0.50,
    1.30,
    -0.60,
  };

  const double gripper_control_before =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::actuator_control(
    *system_ptr,
    "actuator8");

  for (std::size_t index = 0; index < expected_commands.size(); ++index) {
    const std::string joint_name =
      "panda_joint" + std::to_string(index + 1);

    const auto command_it =
      std::find_if(
      command_interfaces.begin(),
      command_interfaces.end(),
      [&joint_name](const auto & interface)
      {
        return
          interface->get_prefix_name() == joint_name &&
          interface->get_interface_name() ==
          hardware_interface::HW_IF_POSITION;
      });

    ASSERT_NE(command_it, command_interfaces.end());

    ASSERT_TRUE(
      (*command_it)->set_value(
        expected_commands[index],
        true));
  }

  EXPECT_EQ(
    system_ptr->write(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(0.002)),
    hardware_interface::return_type::OK);

  const auto actual_controls =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    panda_actuator_controls(*system_ptr);

  for (std::size_t index = 0; index < expected_commands.size(); ++index) {
    EXPECT_DOUBLE_EQ(
      actual_controls[index],
      expected_commands[index]);
  }

  const double gripper_control_after =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::actuator_control(
    *system_ptr,
    "actuator8");

  EXPECT_DOUBLE_EQ(
    gripper_control_after,
    gripper_control_before);
}

TEST(MuJoCoSystemWriteTest, non_finite_command_is_rejected_transactionally)
{
  auto system =
    std::make_unique<robustlearn_mujoco_hardware::MuJoCoSystem>();

  auto * system_ptr = system.get();

  hardware_interface::HardwareComponent component(
    std::move(system));

  component.initialize(
    make_component_params(make_valid_hardware_info()));

  auto state_interfaces =
    component.export_state_interfaces();

  auto command_interfaces =
    component.export_command_interfaces();

  ASSERT_EQ(state_interfaces.size(), 14u);
  ASSERT_EQ(command_interfaces.size(), 7u);

  component.configure();
  component.activate();

  ASSERT_EQ(
    component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  const std::array<double, 7> baseline_commands = {
    0.10,
    -0.20,
    0.30,
    -1.00,
    0.40,
    1.20,
    -0.50,
  };

  for (std::size_t index = 0; index < baseline_commands.size(); ++index) {
    const std::string joint_name =
      "panda_joint" + std::to_string(index + 1);

    const auto command_it =
      std::find_if(
      command_interfaces.begin(),
      command_interfaces.end(),
      [&joint_name](const auto & interface)
      {
        return
          interface->get_prefix_name() == joint_name &&
          interface->get_interface_name() ==
          hardware_interface::HW_IF_POSITION;
      });

    ASSERT_NE(command_it, command_interfaces.end());

    ASSERT_TRUE(
      (*command_it)->set_value(
        baseline_commands[index],
        true));
  }

  ASSERT_EQ(
    system_ptr->write(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(0.002)),
    hardware_interface::return_type::OK);

  const auto controls_before =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    panda_actuator_controls(*system_ptr);

  std::array<double, 7> invalid_commands = {
    -0.60,
    0.70,
    -0.80,
    -1.40,
    0.90,
    1.50,
    -1.00,
  };

  invalid_commands[4] =
    std::numeric_limits<double>::quiet_NaN();

  for (std::size_t index = 0; index < invalid_commands.size(); ++index) {
    const std::string joint_name =
      "panda_joint" + std::to_string(index + 1);

    const auto command_it =
      std::find_if(
      command_interfaces.begin(),
      command_interfaces.end(),
      [&joint_name](const auto & interface)
      {
        return
          interface->get_prefix_name() == joint_name &&
          interface->get_interface_name() ==
          hardware_interface::HW_IF_POSITION;
      });

    ASSERT_NE(command_it, command_interfaces.end());

    ASSERT_TRUE(
      (*command_it)->set_value(
        invalid_commands[index],
        true));
  }

  EXPECT_EQ(
    system_ptr->write(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(0.002)),
    hardware_interface::return_type::ERROR);

  const auto controls_after =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    panda_actuator_controls(*system_ptr);

  for (std::size_t index = 0; index < controls_before.size(); ++index) {
    EXPECT_DOUBLE_EQ(
      controls_after[index],
      controls_before[index]);
  }
}

TEST(MuJoCoSystemWriteTest, out_of_range_command_is_rejected_transactionally)
{
  auto system =
    std::make_unique<robustlearn_mujoco_hardware::MuJoCoSystem>();

  auto * system_ptr = system.get();

  hardware_interface::HardwareComponent component(
    std::move(system));

  component.initialize(
    make_component_params(make_valid_hardware_info()));

  auto state_interfaces =
    component.export_state_interfaces();

  auto command_interfaces =
    component.export_command_interfaces();

  ASSERT_EQ(state_interfaces.size(), 14u);
  ASSERT_EQ(command_interfaces.size(), 7u);

  component.configure();
  component.activate();

  ASSERT_EQ(
    component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  const std::array<double, 7> baseline_commands = {
    0.10,
    -0.20,
    0.30,
    -1.00,
    0.40,
    1.20,
    -0.50,
  };

  for (std::size_t index = 0; index < baseline_commands.size(); ++index) {
    const std::string joint_name =
      "panda_joint" + std::to_string(index + 1);

    const auto command_it =
      std::find_if(
      command_interfaces.begin(),
      command_interfaces.end(),
      [&joint_name](const auto & interface)
      {
        return
          interface->get_prefix_name() == joint_name &&
          interface->get_interface_name() ==
          hardware_interface::HW_IF_POSITION;
      });

    ASSERT_NE(command_it, command_interfaces.end());

    ASSERT_TRUE(
      (*command_it)->set_value(
        baseline_commands[index],
        true));
  }

  ASSERT_EQ(
    system_ptr->write(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(0.002)),
    hardware_interface::return_type::OK);

  const auto controls_before =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    panda_actuator_controls(*system_ptr);

  std::array<double, 7> invalid_commands = {
    -0.60,
    0.70,
    -0.80,
    -1.40,
    0.90,
    1.50,
    -1.00,
  };

  // Panda joint 4 has a compiled MuJoCo control range of
  // [-3.0718, -0.0698], so zero is finite but invalid.
  invalid_commands[3] = 0.0;

  for (std::size_t index = 0; index < invalid_commands.size(); ++index) {
    const std::string joint_name =
      "panda_joint" + std::to_string(index + 1);

    const auto command_it =
      std::find_if(
      command_interfaces.begin(),
      command_interfaces.end(),
      [&joint_name](const auto & interface)
      {
        return
          interface->get_prefix_name() == joint_name &&
          interface->get_interface_name() ==
          hardware_interface::HW_IF_POSITION;
      });

    ASSERT_NE(command_it, command_interfaces.end());

    ASSERT_TRUE(
      (*command_it)->set_value(
        invalid_commands[index],
        true));
  }

  EXPECT_EQ(
    system_ptr->write(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(0.002)),
    hardware_interface::return_type::ERROR);

  const auto controls_after =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    panda_actuator_controls(*system_ptr);

  for (std::size_t index = 0; index < controls_before.size(); ++index) {
    EXPECT_DOUBLE_EQ(
      controls_after[index],
      controls_before[index]);
  }
}

TEST(MuJoCoSystemWriteTest, successful_write_advances_one_step_and_read_stays_finite)
{
  auto system =
    std::make_unique<robustlearn_mujoco_hardware::MuJoCoSystem>();

  auto * system_ptr = system.get();

  hardware_interface::HardwareComponent component(
    std::move(system));

  component.initialize(
    make_component_params(make_valid_hardware_info()));

  auto state_interfaces =
    component.export_state_interfaces();

  auto command_interfaces =
    component.export_command_interfaces();

  ASSERT_EQ(state_interfaces.size(), 14u);
  ASSERT_EQ(command_interfaces.size(), 7u);

  component.configure();
  component.activate();

  ASSERT_EQ(
    component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  const std::array<double, 7> commands = {
    0.10,
    -0.20,
    0.30,
    -1.20,
    0.40,
    1.20,
    -0.50,
  };

  for (std::size_t index = 0; index < commands.size(); ++index) {
    const std::string joint_name =
      "panda_joint" + std::to_string(index + 1);

    const auto command_it =
      std::find_if(
      command_interfaces.begin(),
      command_interfaces.end(),
      [&joint_name](const auto & interface)
      {
        return
          interface->get_prefix_name() == joint_name &&
          interface->get_interface_name() ==
          hardware_interface::HW_IF_POSITION;
      });

    ASSERT_NE(command_it, command_interfaces.end());

    ASSERT_TRUE(
      (*command_it)->set_value(
        commands[index],
        true));
  }

  const double timestep =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    physics_timestep(*system_ptr);

  const double time_before =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    simulation_time(*system_ptr);

  ASSERT_GT(timestep, 0.0);
  EXPECT_DOUBLE_EQ(timestep, 0.002);

  ASSERT_EQ(
    system_ptr->write(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(timestep)),
    hardware_interface::return_type::OK);

  const double time_after =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    simulation_time(*system_ptr);

  EXPECT_NEAR(
    time_after,
    time_before + timestep,
    1e-12);

  ASSERT_EQ(
    system_ptr->read(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(timestep)),
    hardware_interface::return_type::OK);

  for (const auto & state_interface : state_interfaces) {
    ASSERT_NE(state_interface, nullptr);

    const auto value =
      state_interface->get_optional<double>();

    ASSERT_TRUE(value.has_value());
    EXPECT_TRUE(std::isfinite(*value));
  }
}

TEST(MuJoCoSystemWriteTest, fixed_command_sequence_is_deterministic)
{
  auto first_system =
    std::make_unique<robustlearn_mujoco_hardware::MuJoCoSystem>();

  auto * first_system_ptr = first_system.get();

  hardware_interface::HardwareComponent first_component(
    std::move(first_system));

  first_component.initialize(
    make_component_params(make_valid_hardware_info()));

  auto first_state_interfaces =
    first_component.export_state_interfaces();

  auto first_command_interfaces =
    first_component.export_command_interfaces();

  ASSERT_EQ(first_state_interfaces.size(), 14u);
  ASSERT_EQ(first_command_interfaces.size(), 7u);

  first_component.configure();
  first_component.activate();

  ASSERT_EQ(
    first_component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  auto second_system =
    std::make_unique<robustlearn_mujoco_hardware::MuJoCoSystem>();

  auto * second_system_ptr = second_system.get();

  hardware_interface::HardwareComponent second_component(
    std::move(second_system));

  second_component.initialize(
    make_component_params(make_valid_hardware_info()));

  auto second_state_interfaces =
    second_component.export_state_interfaces();

  auto second_command_interfaces =
    second_component.export_command_interfaces();

  ASSERT_EQ(second_state_interfaces.size(), 14u);
  ASSERT_EQ(second_command_interfaces.size(), 7u);

  second_component.configure();
  second_component.activate();

  ASSERT_EQ(
    second_component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  const std::array<std::array<double, 7>, 3> command_sequence = {{
    {
      0.05,
      -0.10,
      0.15,
      -1.40,
      0.20,
      1.40,
      -0.25,
    },
    {
      0.10,
      -0.20,
      0.25,
      -1.20,
      0.30,
      1.30,
      -0.40,
    },
    {
      0.15,
      -0.30,
      0.35,
      -1.00,
      0.40,
      1.20,
      -0.55,
    },
  }};

  const double timestep =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    physics_timestep(*first_system_ptr);

  ASSERT_DOUBLE_EQ(
    timestep,
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    physics_timestep(*second_system_ptr));

  for (const auto & commands : command_sequence) {
    for (std::size_t index = 0; index < commands.size(); ++index) {
      const std::string joint_name =
        "panda_joint" + std::to_string(index + 1);

      const auto first_command_it =
        std::find_if(
        first_command_interfaces.begin(),
        first_command_interfaces.end(),
        [&joint_name](const auto & interface)
        {
          return
            interface->get_prefix_name() == joint_name &&
            interface->get_interface_name() ==
            hardware_interface::HW_IF_POSITION;
        });

      const auto second_command_it =
        std::find_if(
        second_command_interfaces.begin(),
        second_command_interfaces.end(),
        [&joint_name](const auto & interface)
        {
          return
            interface->get_prefix_name() == joint_name &&
            interface->get_interface_name() ==
            hardware_interface::HW_IF_POSITION;
        });

      ASSERT_NE(
        first_command_it,
        first_command_interfaces.end());

      ASSERT_NE(
        second_command_it,
        second_command_interfaces.end());

      ASSERT_TRUE(
        (*first_command_it)->set_value(
          commands[index],
          true));

      ASSERT_TRUE(
        (*second_command_it)->set_value(
          commands[index],
          true));
    }

    ASSERT_EQ(
      first_system_ptr->write(
        rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
        rclcpp::Duration::from_seconds(timestep)),
      hardware_interface::return_type::OK);

    ASSERT_EQ(
      second_system_ptr->write(
        rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
        rclcpp::Duration::from_seconds(timestep)),
      hardware_interface::return_type::OK);
  }

  const double first_time =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    simulation_time(*first_system_ptr);

  const double second_time =
    robustlearn_mujoco_hardware::MuJoCoSystemTestPeer::
    simulation_time(*second_system_ptr);

  EXPECT_NEAR(
    first_time,
    static_cast<double>(command_sequence.size()) * timestep,
    1e-12);

  EXPECT_NEAR(
    second_time,
    first_time,
    1e-12);

  ASSERT_EQ(
    first_system_ptr->read(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(timestep)),
    hardware_interface::return_type::OK);

  ASSERT_EQ(
    second_system_ptr->read(
      rclcpp::Time(0, 0, RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(timestep)),
    hardware_interface::return_type::OK);

  std::unordered_map<std::string, double> first_states;

  for (const auto & state_interface : first_state_interfaces) {
    ASSERT_NE(state_interface, nullptr);

    const auto value =
      state_interface->get_optional<double>();

    ASSERT_TRUE(value.has_value());
    ASSERT_TRUE(std::isfinite(*value));

    first_states[state_interface->get_name()] = *value;
  }

  ASSERT_EQ(first_states.size(), 14u);

  for (const auto & state_interface : second_state_interfaces) {
    ASSERT_NE(state_interface, nullptr);

    const auto value =
      state_interface->get_optional<double>();

    ASSERT_TRUE(value.has_value());
    ASSERT_TRUE(std::isfinite(*value));

    const auto first_state_it =
      first_states.find(state_interface->get_name());

    ASSERT_NE(
      first_state_it,
      first_states.end());

    EXPECT_NEAR(
      *value,
      first_state_it->second,
      1e-12);
  }
}

TEST(MuJoCoSystemLifecycleTest, activation_and_deactivation_keep_interfaces_finite)
{
  auto system =
    std::make_unique<robustlearn_mujoco_hardware::MuJoCoSystem>();

  hardware_interface::HardwareComponent component(
    std::move(system));

  const auto params =
    make_component_params(make_valid_hardware_info());

  component.initialize(params);

  ASSERT_EQ(
    component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  auto state_interfaces =
    component.export_state_interfaces();

  auto command_interfaces =
    component.export_command_interfaces();

  ASSERT_EQ(state_interfaces.size(), 14u);
  ASSERT_EQ(command_interfaces.size(), 7u);

  component.configure();

  ASSERT_EQ(
    component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  component.activate();

  ASSERT_EQ(
    component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  std::unordered_map<std::string, double> position_states;

  for (const auto & state_interface : state_interfaces) {
    ASSERT_NE(state_interface, nullptr);

    const auto value =
      state_interface->get_optional<double>();

    ASSERT_TRUE(value.has_value());
    EXPECT_TRUE(std::isfinite(*value));

    if (
      state_interface->get_interface_name() ==
      hardware_interface::HW_IF_POSITION)
    {
      position_states[state_interface->get_name()] = *value;
    }
  }

  ASSERT_EQ(position_states.size(), 7u);

  for (const auto & command_interface : command_interfaces) {
    ASSERT_NE(command_interface, nullptr);

    const auto value =
      command_interface->get_optional<double>();

    ASSERT_TRUE(value.has_value());
    EXPECT_TRUE(std::isfinite(*value));

    ASSERT_TRUE(
      command_interface->set_value(
        1.0,
        true));
  }

  component.deactivate();

  ASSERT_EQ(
    component.get_lifecycle_id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  for (const auto & command_interface : command_interfaces) {
    const auto value =
      command_interface->get_optional<double>();

    ASSERT_TRUE(value.has_value());
    EXPECT_TRUE(std::isfinite(*value));

    const auto position_it =
      position_states.find(command_interface->get_name());

    ASSERT_NE(position_it, position_states.end());

    EXPECT_DOUBLE_EQ(
      *value,
      position_it->second);
  }
}

}  // namespace
