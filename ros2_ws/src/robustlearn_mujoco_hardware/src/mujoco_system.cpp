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

#include "robustlearn_mujoco_hardware/mujoco_system.hpp"

#include <mujoco/mujoco.h>

#include "pluginlib/class_list_macros.hpp"

namespace robustlearn_mujoco_hardware
{

hardware_interface::CallbackReturn MuJoCoSystem::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (
    hardware_interface::SystemInterface::on_init(params) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Keep the scaffold linked against the configured MuJoCo runtime.
  // Full model initialization belongs to the following Week 7 issue.
  const int mujoco_version = mj_version();
  if (mujoco_version <= 0) {
    return hardware_interface::CallbackReturn::ERROR;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type MuJoCoSystem::read(
  const rclcpp::Time & /* time */,
  const rclcpp::Duration & /* period */)
{
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type MuJoCoSystem::write(
  const rclcpp::Time & /* time */,
  const rclcpp::Duration & /* period */)
{
  return hardware_interface::return_type::OK;
}

}  // namespace robustlearn_mujoco_hardware

PLUGINLIB_EXPORT_CLASS(
  robustlearn_mujoco_hardware::MuJoCoSystem,
  hardware_interface::SystemInterface)
