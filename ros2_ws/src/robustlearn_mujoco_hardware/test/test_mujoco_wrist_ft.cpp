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

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "hardware_interface/hardware_component.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/types/hardware_component_params.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/logger.hpp"
#include "robustlearn_mujoco_hardware/mujoco_system.hpp"

namespace
{

constexpr std::array<const char *, 6> kWristFtInterfaceNames = {
  "force.x",
  "force.y",
  "force.z",
  "torque.x",
  "torque.y",
  "torque.z",
};

hardware_interface::InterfaceInfo make_interface(
  const std::string & name)
{
  hardware_interface::InterfaceInfo interface_info{};
  interface_info.name = name;
  return interface_info;
}

hardware_interface::ComponentInfo make_panda_joint(
  const int joint_number)
{
  hardware_interface::ComponentInfo joint{};

  joint.name =
    "panda_joint" +
    std::to_string(joint_number);

  joint.type =
    "joint";

  joint.command_interfaces = {
    make_interface(
      hardware_interface::HW_IF_POSITION),
  };

  joint.state_interfaces = {
    make_interface(
      hardware_interface::HW_IF_POSITION),
    make_interface(
      hardware_interface::HW_IF_VELOCITY),
  };

  return joint;
}

hardware_interface::ComponentInfo make_wrist_ft_sensor()
{
  hardware_interface::ComponentInfo sensor{};

  sensor.name =
    "wrist_ft";

  sensor.type =
    "sensor";

  for (
    const char * interface_name :
    kWristFtInterfaceNames)
  {
    sensor.state_interfaces.push_back(
      make_interface(interface_name));
  }

  return sensor;
}

hardware_interface::HardwareInfo make_wrist_ft_hardware_info()
{
  hardware_interface::HardwareInfo hardware_info{};

  hardware_info.name =
    "PandaMuJoCoSystem";

  hardware_info.type =
    "system";

  hardware_info.hardware_plugin_name =
    "robustlearn_mujoco_hardware/MuJoCoSystem";

  hardware_info.hardware_parameters["model_path"] =
    ROBUSTLEARN_TEST_MODEL_PATH;

  for (
    int joint_number = 1;
    joint_number <= 7;
    ++joint_number)
  {
    hardware_info.joints.push_back(
      make_panda_joint(joint_number));
  }

  hardware_info.sensors.push_back(
    make_wrist_ft_sensor());

  return hardware_info;
}

hardware_interface::HardwareComponentParams make_component_params(
  hardware_interface::HardwareInfo hardware_info)
{
  hardware_interface::HardwareComponentParams params{};

  params.hardware_info =
    std::move(hardware_info);

  params.logger =
    rclcpp::get_logger(
    "mujoco_wrist_ft_test");

  params.clock =
    std::make_shared<rclcpp::Clock>(
    RCL_SYSTEM_TIME);

  return params;
}

hardware_interface::CallbackReturn initialize(
  hardware_interface::HardwareInfo hardware_info)
{
  robustlearn_mujoco_hardware::MuJoCoSystem system;

  return system.init(
    make_component_params(
      std::move(hardware_info)));
}

TEST(
  MuJoCoWristFtInitializationTest,
  valid_wrist_ft_description_initializes)
{
  EXPECT_EQ(
    initialize(
      make_wrist_ft_hardware_info()),
    hardware_interface::CallbackReturn::SUCCESS);
}

TEST(
  MuJoCoWristFtInitializationTest,
  missing_force_interface_is_rejected)
{
  auto hardware_info =
    make_wrist_ft_hardware_info();

  ASSERT_EQ(
    hardware_info.sensors.size(),
    1u);

  auto & state_interfaces =
    hardware_info.sensors.front().state_interfaces;

  ASSERT_EQ(
    state_interfaces.size(),
    6u);

  state_interfaces.erase(
    state_interfaces.begin());

  EXPECT_EQ(
    initialize(
      std::move(hardware_info)),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(
  MuJoCoWristFtInitializationTest,
  extra_sensor_interface_is_rejected)
{
  auto hardware_info =
    make_wrist_ft_hardware_info();

  ASSERT_EQ(
    hardware_info.sensors.size(),
    1u);

  hardware_info.sensors.front().state_interfaces.push_back(
    make_interface(
      "temperature"));

  EXPECT_EQ(
    initialize(
      std::move(hardware_info)),
    hardware_interface::CallbackReturn::ERROR);
}

TEST(
  MuJoCoWristFtReadTest,
  exports_six_finite_wrist_interfaces_and_seven_commands)
{
  auto system =
    std::make_unique<
    robustlearn_mujoco_hardware::MuJoCoSystem>();

  auto * system_ptr =
    system.get();

  hardware_interface::HardwareComponent component(
    std::move(system));

  component.initialize(
    make_component_params(
      make_wrist_ft_hardware_info()));

  auto state_interfaces =
    component.export_state_interfaces();

  auto command_interfaces =
    component.export_command_interfaces();

  ASSERT_EQ(
    state_interfaces.size(),
    20u);

  ASSERT_EQ(
    command_interfaces.size(),
    7u);

  std::size_t panda_joint_state_count =
    0;

  std::size_t wrist_ft_state_count =
    0;

  for (
    const auto & state_interface :
    state_interfaces)
  {
    ASSERT_NE(
      state_interface,
      nullptr);

    if (
      state_interface->get_prefix_name() ==
      "wrist_ft")
    {
      ++wrist_ft_state_count;
    } else if (
      state_interface->get_prefix_name().rfind(
        "panda_joint",
        0) == 0)
    {
      ++panda_joint_state_count;
    }
  }

  EXPECT_EQ(
    panda_joint_state_count,
    14u);

  EXPECT_EQ(
    wrist_ft_state_count,
    6u);

  for (
    const auto & command_interface :
    command_interfaces)
  {
    ASSERT_NE(
      command_interface,
      nullptr);

    EXPECT_EQ(
      command_interface->get_interface_name(),
      hardware_interface::HW_IF_POSITION);

    EXPECT_EQ(
      command_interface->get_prefix_name().rfind(
        "panda_joint",
        0),
      0u);
  }

  component.configure();
  component.activate();

  ASSERT_EQ(
    system_ptr->read(
      rclcpp::Time(
        0,
        0,
        RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(
        0.001)),
    hardware_interface::return_type::OK);

  for (
    const char * expected_interface_name :
    kWristFtInterfaceNames)
  {
    const auto wrist_interface_it =
      std::find_if(
      state_interfaces.begin(),
      state_interfaces.end(),
      [expected_interface_name](
        const auto & interface)
      {
        return
          interface->get_prefix_name() ==
          "wrist_ft" &&
          interface->get_interface_name() ==
          expected_interface_name;
      });

    ASSERT_NE(
      wrist_interface_it,
      state_interfaces.end());

    const auto value =
      (*wrist_interface_it)->get_optional<double>();

    ASSERT_TRUE(
      value.has_value());

    EXPECT_TRUE(
      std::isfinite(*value));
  }
}

TEST(
  MuJoCoWristFtReadTest,
  all_exported_states_remain_finite_after_physics_step)
{
  auto system =
    std::make_unique<
    robustlearn_mujoco_hardware::MuJoCoSystem>();

  auto * system_ptr =
    system.get();

  hardware_interface::HardwareComponent component(
    std::move(system));

  component.initialize(
    make_component_params(
      make_wrist_ft_hardware_info()));

  auto state_interfaces =
    component.export_state_interfaces();

  auto command_interfaces =
    component.export_command_interfaces();

  ASSERT_EQ(
    state_interfaces.size(),
    20u);

  ASSERT_EQ(
    command_interfaces.size(),
    7u);

  component.configure();
  component.activate();

  for (
    const auto & command_interface :
    command_interfaces)
  {
    ASSERT_NE(
      command_interface,
      nullptr);

    const auto command_value =
      command_interface->get_optional<double>();

    ASSERT_TRUE(
      command_value.has_value());

    ASSERT_TRUE(
      std::isfinite(*command_value));
  }

  ASSERT_EQ(
    system_ptr->write(
      rclcpp::Time(
        0,
        0,
        RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(
        0.002)),
    hardware_interface::return_type::OK);

  ASSERT_EQ(
    system_ptr->read(
      rclcpp::Time(
        0,
        0,
        RCL_SYSTEM_TIME),
      rclcpp::Duration::from_seconds(
        0.002)),
    hardware_interface::return_type::OK);

  for (
    const auto & state_interface :
    state_interfaces)
  {
    ASSERT_NE(
      state_interface,
      nullptr);

    const auto value =
      state_interface->get_optional<double>();

    ASSERT_TRUE(
      value.has_value());

    EXPECT_TRUE(
      std::isfinite(*value));
  }
}

}  // namespace
