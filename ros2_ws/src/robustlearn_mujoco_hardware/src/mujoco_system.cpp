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

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <string>
#include <filesystem>
#include <utility>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"

namespace robustlearn_mujoco_hardware
{

namespace
{

constexpr std::array<const char *, 7> kRosPandaJointNames = {
  "panda_joint1",
  "panda_joint2",
  "panda_joint3",
  "panda_joint4",
  "panda_joint5",
  "panda_joint6",
  "panda_joint7",
};

constexpr std::array<const char *, 7> kMujocoPandaJointNames = {
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
  "joint7",
};

constexpr std::array<const char *, 7> kMujocoPandaActuatorNames = {
  "actuator1",
  "actuator2",
  "actuator3",
  "actuator4",
  "actuator5",
  "actuator6",
  "actuator7",
};

bool has_interface(
  const std::vector<hardware_interface::InterfaceInfo> & interfaces,
  const char * name)
{
  return std::any_of(
    interfaces.begin(),
    interfaces.end(),
    [name](const auto & interface_info) {
      return interface_info.name == name;
    });
}

}  // namespace

void MuJoCoSystem::MjSpecDeleter::operator()(mjSpec * spec) const noexcept
{
  mj_deleteSpec(spec);
}

void MuJoCoSystem::MjModelDeleter::operator()(mjModel * model) const noexcept
{
  mj_deleteModel(model);
}

void MuJoCoSystem::MjDataDeleter::operator()(mjData * data) const noexcept
{
  mj_deleteData(data);
}

hardware_interface::CallbackReturn MuJoCoSystem::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (
    hardware_interface::SystemInterface::on_init(params) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  const auto & hardware_info = params.hardware_info;
  const auto logger = get_logger();

  if (hardware_info.joints.size() != kRosPandaJointNames.size()) {
    RCLCPP_ERROR(
      logger,
      "Expected exactly %zu Panda arm joints, but hardware description contains %zu",
      kRosPandaJointNames.size(),
      hardware_info.joints.size());

    return hardware_interface::CallbackReturn::ERROR;
  }

  for (const char * expected_joint_name : kRosPandaJointNames) {
    const auto joint_it = std::find_if(
      hardware_info.joints.begin(),
      hardware_info.joints.end(),
      [expected_joint_name](const auto & joint) {
        return joint.name == expected_joint_name;
      });

    if (joint_it == hardware_info.joints.end()) {
      RCLCPP_ERROR(
        logger,
        "Hardware description is missing required Panda joint '%s'",
        expected_joint_name);

      return hardware_interface::CallbackReturn::ERROR;
    }

    const auto & joint = *joint_it;

    if (
      joint.command_interfaces.size() != 1 ||
      joint.command_interfaces.front().name !=
      hardware_interface::HW_IF_POSITION)
    {
      RCLCPP_ERROR(
        logger,
        "Joint '%s' must expose exactly one '%s' command interface",
        expected_joint_name,
        hardware_interface::HW_IF_POSITION);

      return hardware_interface::CallbackReturn::ERROR;
    }

    if (
      joint.state_interfaces.size() != 2 ||
      !has_interface(
        joint.state_interfaces,
        hardware_interface::HW_IF_POSITION) ||
      !has_interface(
        joint.state_interfaces,
        hardware_interface::HW_IF_VELOCITY))
    {
      RCLCPP_ERROR(
        logger,
        "Joint '%s' must expose exactly '%s' and '%s' state interfaces",
        expected_joint_name,
        hardware_interface::HW_IF_POSITION,
        hardware_interface::HW_IF_VELOCITY);

      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  const int mujoco_version = mj_version();

  if (mujoco_version <= 0) {
    RCLCPP_ERROR(logger, "MuJoCo runtime reported an invalid version");
    return hardware_interface::CallbackReturn::ERROR;
  }

  const auto model_path_it =
    hardware_info.hardware_parameters.find("model_path");

  if (
    model_path_it == hardware_info.hardware_parameters.end() ||
    model_path_it->second.empty())
  {
    RCLCPP_ERROR(
      logger,
      "Required hardware parameter 'model_path' is missing or empty");

    return hardware_interface::CallbackReturn::ERROR;
  }

  const std::filesystem::path model_path(model_path_it->second);

  if (!model_path.is_absolute()) {
    RCLCPP_ERROR(
      logger,
      "Hardware parameter 'model_path' must be an absolute path: '%s'",
      model_path.string().c_str());

    return hardware_interface::CallbackReturn::ERROR;
  }

  if (!std::filesystem::is_regular_file(model_path)) {
    RCLCPP_ERROR(
      logger,
      "Configured MuJoCo model does not exist or is not a regular file: '%s'",
      model_path.string().c_str());

    return hardware_interface::CallbackReturn::ERROR;
  }

  std::array<char, 1024> parse_error{};

  MjSpecPtr parsed_spec{
    mj_parseXML(
      model_path.string().c_str(),
      nullptr,
      parse_error.data(),
      static_cast<int>(parse_error.size()))};

  if (!parsed_spec) {
    RCLCPP_ERROR(
      logger,
      "Failed to parse MuJoCo model '%s': %s",
      model_path.string().c_str(),
      parse_error.data());

    return hardware_interface::CallbackReturn::ERROR;
  }

  mjsBody * hand = mjs_findBody(parsed_spec.get(), "hand");

  if (hand == nullptr) {
    RCLCPP_ERROR(
      logger,
      "MuJoCo model does not contain required Panda body 'hand'");

    return hardware_interface::CallbackReturn::ERROR;
  }

  mjSpec * fixed_peg_spec =
    mjs_findSpec(parsed_spec.get(), "fixed_peg");

  if (fixed_peg_spec == nullptr) {
    RCLCPP_ERROR(
      logger,
      "MuJoCo model does not contain required model asset 'fixed_peg'");

    return hardware_interface::CallbackReturn::ERROR;
  }

  mjsFrame * tool_mount =
    mjs_findFrame(fixed_peg_spec, "tool_mount");

  if (tool_mount == nullptr) {
    RCLCPP_ERROR(
      logger,
      "MuJoCo fixed_peg model does not contain required frame 'tool_mount'");

    return hardware_interface::CallbackReturn::ERROR;
  }

  if (
    mjs_attach(
      hand->element,
      tool_mount->element,
      "",
      "") == nullptr)
  {
    const char * spec_error = mjs_getError(parsed_spec.get());

    RCLCPP_ERROR(
      logger,
      "Failed to attach fixed_peg tool to Panda hand: %s",
      spec_error != nullptr ? spec_error : "unknown MuJoCo specification error");

    return hardware_interface::CallbackReturn::ERROR;
  }

  MjModelPtr compiled_model{
    mj_compile(parsed_spec.get(), nullptr)};

  if (!compiled_model) {
    const char * spec_error = mjs_getError(parsed_spec.get());

    RCLCPP_ERROR(
      logger,
      "Failed to compile MuJoCo model '%s': %s",
      model_path.string().c_str(),
      spec_error != nullptr ? spec_error : "unknown MuJoCo compilation error");

    return hardware_interface::CallbackReturn::ERROR;
  }

  MjDataPtr data{mj_makeData(compiled_model.get())};

  if (!data) {
    RCLCPP_ERROR(
      logger,
      "Failed to allocate MuJoCo data for model '%s'",
      model_path.string().c_str());

    return hardware_interface::CallbackReturn::ERROR;
  }

  const int home_keyframe_id = mj_name2id(
    compiled_model.get(),
    mjOBJ_KEY,
    "home");

  if (home_keyframe_id >= 0) {
    mj_resetDataKeyframe(
      compiled_model.get(),
      data.get(),
      home_keyframe_id);

    mj_forward(
      compiled_model.get(),
      data.get());
  }

  std::array<int, 7> qpos_addresses{};
  std::array<int, 7> dof_addresses{};
  std::array<int, 7> actuator_ids{};

  std::array<double, 7> joint_positions{};
  std::array<double, 7> joint_velocities{};
  std::array<double, 7> joint_commands{};

  for (std::size_t index = 0; index < kMujocoPandaJointNames.size(); ++index) {
    const char * joint_name = kMujocoPandaJointNames[index];
    const char * actuator_name = kMujocoPandaActuatorNames[index];

    const int joint_id = mj_name2id(
      compiled_model.get(),
      mjOBJ_JOINT,
      joint_name);

    if (joint_id < 0) {
      RCLCPP_ERROR(
        logger,
        "MuJoCo model does not contain required Panda joint '%s'",
        joint_name);

      return hardware_interface::CallbackReturn::ERROR;
    }

    if (compiled_model->jnt_type[joint_id] != mjJNT_HINGE) {
      RCLCPP_ERROR(
        logger,
        "MuJoCo Panda joint '%s' must be a one-DoF hinge joint",
        joint_name);

      return hardware_interface::CallbackReturn::ERROR;
    }

    const int actuator_id = mj_name2id(
      compiled_model.get(),
      mjOBJ_ACTUATOR,
      actuator_name);

    if (actuator_id < 0) {
      RCLCPP_ERROR(
        logger,
        "MuJoCo model does not contain required Panda actuator '%s'",
        actuator_name);

      return hardware_interface::CallbackReturn::ERROR;
    }

    if (
      compiled_model->actuator_trntype[actuator_id] != mjTRN_JOINT ||
      compiled_model->actuator_trnid[2 * actuator_id] != joint_id)
    {
      RCLCPP_ERROR(
        logger,
        "MuJoCo actuator '%s' does not target expected joint '%s'",
        actuator_name,
        joint_name);

      return hardware_interface::CallbackReturn::ERROR;
    }

    const int qpos_address = compiled_model->jnt_qposadr[joint_id];
    const int dof_address = compiled_model->jnt_dofadr[joint_id];

    const double position = data->qpos[qpos_address];
    const double velocity = data->qvel[dof_address];

    if (!std::isfinite(position) || !std::isfinite(velocity)) {
      RCLCPP_ERROR(
        logger,
        "MuJoCo joint '%s' has non-finite initial state",
        joint_name);

      return hardware_interface::CallbackReturn::ERROR;
    }

    qpos_addresses[index] = qpos_address;
    dof_addresses[index] = dof_address;
    actuator_ids[index] = actuator_id;

    joint_positions[index] = position;
    joint_velocities[index] = velocity;
    joint_commands[index] = position;
  }

  qpos_addresses_ = qpos_addresses;
  dof_addresses_ = dof_addresses;
  actuator_ids_ = actuator_ids;

  joint_positions_ = joint_positions;
  joint_velocities_ = joint_velocities;
  joint_commands_ = joint_commands;

  spec_ = std::move(parsed_spec);
  model_ = std::move(compiled_model);
  data_ = std::move(data);

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MuJoCoSystem::on_activate(
  const rclcpp_lifecycle::State & /* previous_state */)
{
  const auto logger = get_logger();

  if (!model_ || !data_) {
    RCLCPP_ERROR(
      logger,
      "Cannot activate MuJoCo hardware before model initialization");

    return hardware_interface::CallbackReturn::ERROR;
  }

  try {
    for (std::size_t index = 0; index < kRosPandaJointNames.size(); ++index) {
      const double position = joint_positions_[index];
      const double velocity = joint_velocities_[index];
      const double command = joint_commands_[index];

      if (
        !std::isfinite(position) ||
        !std::isfinite(velocity) ||
        !std::isfinite(command))
      {
        RCLCPP_ERROR(
          logger,
          "Cannot activate with non-finite values for joint '%s'",
          kRosPandaJointNames[index]);

        return hardware_interface::CallbackReturn::ERROR;
      }

      const std::string joint_prefix =
        kRosPandaJointNames[index];

      set_state(
        joint_prefix + "/" + hardware_interface::HW_IF_POSITION,
        position);

      set_state(
        joint_prefix + "/" + hardware_interface::HW_IF_VELOCITY,
        velocity);

      set_command(
        joint_prefix + "/" + hardware_interface::HW_IF_POSITION,
        command);
    }
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      logger,
      "Failed to initialize ROS 2 control interfaces during activation: %s",
      error.what());

    return hardware_interface::CallbackReturn::ERROR;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MuJoCoSystem::on_deactivate(
  const rclcpp_lifecycle::State & /* previous_state */)
{
  const auto logger = get_logger();

  if (!model_ || !data_) {
    RCLCPP_ERROR(
      logger,
      "Cannot deactivate MuJoCo hardware before model initialization");

    return hardware_interface::CallbackReturn::ERROR;
  }

  try {
    for (std::size_t index = 0; index < kRosPandaJointNames.size(); ++index) {
      const double hold_position = joint_positions_[index];

      if (!std::isfinite(hold_position)) {
        RCLCPP_ERROR(
          logger,
          "Cannot deactivate with non-finite position for joint '%s'",
          kRosPandaJointNames[index]);

        return hardware_interface::CallbackReturn::ERROR;
      }

      joint_commands_[index] = hold_position;

      const std::string command_interface =
        std::string(kRosPandaJointNames[index]) +
        "/" +
        hardware_interface::HW_IF_POSITION;

      set_command(
        command_interface,
        hold_position);
    }
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      logger,
      "Failed to reset ROS 2 control commands during deactivation: %s",
      error.what());

    return hardware_interface::CallbackReturn::ERROR;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type MuJoCoSystem::read(
  const rclcpp::Time & /* time */,
  const rclcpp::Duration & /* period */)
{
  const auto logger = get_logger();

  if (!model_ || !data_) {
    RCLCPP_ERROR(
      logger,
      "Cannot read MuJoCo hardware state before model initialization");

    return hardware_interface::return_type::ERROR;
  }

  std::array<double, 7> positions{};
  std::array<double, 7> velocities{};

  for (std::size_t index = 0; index < kRosPandaJointNames.size(); ++index) {
    const double position =
      data_->qpos[qpos_addresses_[index]];

    const double velocity =
      data_->qvel[dof_addresses_[index]];

    if (!std::isfinite(position) || !std::isfinite(velocity)) {
      RCLCPP_ERROR(
        logger,
        "MuJoCo joint '%s' has non-finite state during read",
        kMujocoPandaJointNames[index]);

      return hardware_interface::return_type::ERROR;
    }

    positions[index] = position;
    velocities[index] = velocity;
  }

  try {
    for (std::size_t index = 0; index < kRosPandaJointNames.size(); ++index) {
      const std::string joint_prefix =
        kRosPandaJointNames[index];

      set_state(
        joint_prefix + "/" + hardware_interface::HW_IF_POSITION,
        positions[index]);

      set_state(
        joint_prefix + "/" + hardware_interface::HW_IF_VELOCITY,
        velocities[index]);
    }
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      logger,
      "Failed to update ROS 2 control state interfaces during read: %s",
      error.what());

    return hardware_interface::return_type::ERROR;
  }

  joint_positions_ = positions;
  joint_velocities_ = velocities;

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type MuJoCoSystem::write(
  const rclcpp::Time & /* time */,
  const rclcpp::Duration & /* period */)
{
  const auto logger = get_logger();

  if (!model_ || !data_) {
    RCLCPP_ERROR(
      logger,
      "Cannot write MuJoCo hardware commands before model initialization");

    return hardware_interface::return_type::ERROR;
  }

  std::array<double, 7> commands{};

  try {
    for (std::size_t index = 0; index < kRosPandaJointNames.size(); ++index) {
      const std::string command_interface =
        std::string(kRosPandaJointNames[index]) +
        "/" +
        hardware_interface::HW_IF_POSITION;

      const double command =
        get_command<double>(command_interface);

      if (!std::isfinite(command)) {
        RCLCPP_ERROR(
          logger,
          "ROS 2 control command for joint '%s' is non-finite",
          kRosPandaJointNames[index]);

        return hardware_interface::return_type::ERROR;
      }

      const int actuator_id = actuator_ids_[index];

      if (model_->actuator_ctrllimited[actuator_id]) {
        const double lower_limit =
          model_->actuator_ctrlrange[2 * actuator_id];

        const double upper_limit =
          model_->actuator_ctrlrange[2 * actuator_id + 1];

        if (command < lower_limit || command > upper_limit) {
          RCLCPP_ERROR(
            logger,
            "Command %.17g for joint '%s' is outside MuJoCo actuator "
            "control range [%.17g, %.17g]",
            command,
            kRosPandaJointNames[index],
            lower_limit,
            upper_limit);

          return hardware_interface::return_type::ERROR;
        }
      }

      commands[index] = command;
    }
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      logger,
      "Failed to read ROS 2 control position commands: %s",
      error.what());

    return hardware_interface::return_type::ERROR;
  }

  for (std::size_t index = 0; index < kRosPandaJointNames.size(); ++index) {
    data_->ctrl[actuator_ids_[index]] = commands[index];
  }

  joint_commands_ = commands;

  // A successful ros2_control write advances exactly one compiled MuJoCo
  // physics timestep. The ROS control period intentionally does not scale
  // the number of physics steps, keeping fixed command sequences deterministic.
  mj_step(
    model_.get(),
    data_.get());

  return hardware_interface::return_type::OK;
}

}  // namespace robustlearn_mujoco_hardware

PLUGINLIB_EXPORT_CLASS(
  robustlearn_mujoco_hardware::MuJoCoSystem,
  hardware_interface::SystemInterface)
