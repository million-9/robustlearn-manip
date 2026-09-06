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

#define BOOST_BIND_GLOBAL_PLACEHOLDERS

#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/property_tree/ptree.hpp"
#include "gtest/gtest.h"
#include "moveit/robot_model/robot_model.hpp"
#include "moveit/robot_state/robot_state.hpp"
#include "mujoco/mujoco.h"
#include "srdfdom/model.h"
#include "urdf_parser/urdf_parser.h"

namespace
{

constexpr std::size_t kPandaJointCount = 7;

constexpr double kTranslationToleranceMeters = 1.0e-6;
constexpr double kOrientationToleranceRadians = 1.0e-6;

struct JointMapping
{
  std::string ros_joint;
  std::string mujoco_joint;
  double position_sign;
  double position_offset_rad;
};

struct MappingContract
{
  std::vector<JointMapping> joints;
  std::array<double, kPandaJointCount> home_positions{};
  std::string ros_base_frame;
  std::string mujoco_base_body;
  std::string ros_tip_frame;
  std::string mujoco_tip_body;
};

struct Configuration
{
  std::string name;
  std::array<double, kPandaJointCount> positions;
};

struct FramePair
{
  std::string ros_frame;
  std::string mujoco_body;
};

struct MjModelDeleter
{
  void operator()(mjModel * model) const
  {
    if (model != nullptr) {
      mj_deleteModel(model);
    }
  }
};

struct MjDataDeleter
{
  void operator()(mjData * data) const
  {
    if (data != nullptr) {
      mj_deleteData(data);
    }
  }
};

using MjModelPtr = std::unique_ptr<mjModel, MjModelDeleter>;
using MjDataPtr = std::unique_ptr<mjData, MjDataDeleter>;

MappingContract load_mapping_contract()
{
  boost::property_tree::ptree root;

  boost::property_tree::read_json(
    ROBUSTLEARN_MAPPING_CONTRACT_PATH,
    root);

  MappingContract contract;

  for (const auto & entry : root.get_child("joints")) {
    const auto & joint = entry.second;

    contract.joints.push_back(
      JointMapping{
        joint.get<std::string>("ros_joint"),
        joint.get<std::string>("mujoco_joint"),
        joint.get<double>("position_sign"),
        joint.get<double>("position_offset_rad"),
      });
  }

  if (contract.joints.size() != kPandaJointCount) {
    throw std::runtime_error(
            "Panda mapping contract must contain exactly seven joints");
  }

  const auto & home_positions =
    root.get_child(
    "home_reference.positions_rad_by_ros_joint");

  for (std::size_t index = 0; index < contract.joints.size(); ++index) {
    contract.home_positions[index] =
      home_positions.get<double>(
      contract.joints[index].ros_joint);
  }

  contract.ros_base_frame =
    root.get<std::string>(
    "cross_model_fk.ros_base_frame");

  contract.mujoco_base_body =
    root.get<std::string>(
    "cross_model_fk.mujoco_base_body");

  contract.ros_tip_frame =
    root.get<std::string>(
    "cross_model_fk.ros_tip_frame");

  contract.mujoco_tip_body =
    root.get<std::string>(
    "cross_model_fk.mujoco_tip_body");

  return contract;
}

std::string configuration_string(
  const std::array<double, kPandaJointCount> & positions)
{
  std::ostringstream stream;
  stream << "[";

  for (std::size_t index = 0; index < positions.size(); ++index) {
    if (index != 0) {
      stream << ", ";
    }

    stream << positions[index];
  }

  stream << "]";
  return stream.str();
}

moveit::core::RobotModelPtr load_moveit_panda_model()
{
  const auto description_share =
    ament_index_cpp::get_package_share_directory(
    "moveit_resources_panda_description");

  const auto config_share =
    ament_index_cpp::get_package_share_directory(
    "moveit_resources_panda_moveit_config");

  const auto urdf_path =
    std::filesystem::path(description_share) /
    "urdf" /
    "panda.urdf";

  const auto srdf_path =
    std::filesystem::path(config_share) /
    "config" /
    "panda.srdf";

  const auto urdf_model =
    urdf::parseURDFFile(urdf_path.string());

  if (!urdf_model) {
    throw std::runtime_error(
            "Failed to parse MoveIt Panda URDF: " +
            urdf_path.string());
  }

  auto srdf_model =
    std::make_shared<srdf::Model>();

  if (!srdf_model->initFile(
      *urdf_model,
      srdf_path.string()))
  {
    throw std::runtime_error(
            "Failed to parse MoveIt Panda SRDF: " +
            srdf_path.string());
  }

  return std::make_shared<moveit::core::RobotModel>(
    urdf_model,
    srdf_model);
}

MjModelPtr load_mujoco_model()
{
  std::array<char, 1024> error{};

  MjModelPtr model(
    mj_loadXML(
      ROBUSTLEARN_TEST_MODEL_PATH,
      nullptr,
      error.data(),
      static_cast<int>(error.size())));

  if (!model) {
    throw std::runtime_error(
            "Failed to compile MuJoCo Panda model: " +
            std::string(error.data()));
  }

  return model;
}

Eigen::Isometry3d mujoco_body_transform(
  const mjModel * model,
  const mjData * data,
  const std::string & body_name)
{
  const int body_id =
    mj_name2id(
    model,
    mjOBJ_BODY,
    body_name.c_str());

  if (body_id < 0) {
    throw std::runtime_error(
            "Missing MuJoCo body: " + body_name);
  }

  Eigen::Isometry3d transform =
    Eigen::Isometry3d::Identity();

  transform.translation() =
    Eigen::Vector3d(
    data->xpos[3 * body_id],
    data->xpos[3 * body_id + 1],
    data->xpos[3 * body_id + 2]);

  Eigen::Matrix3d rotation;

  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      rotation(row, column) =
        data->xmat[
        9 * body_id +
        3 * row +
        column];
    }
  }

  transform.linear() = rotation;

  return transform;
}

Eigen::Isometry3d mujoco_relative_transform(
  const mjModel * model,
  const mjData * data,
  const std::string & base_body,
  const std::string & tip_body)
{
  const Eigen::Isometry3d world_to_base =
    mujoco_body_transform(
    model,
    data,
    base_body);

  const Eigen::Isometry3d world_to_tip =
    mujoco_body_transform(
    model,
    data,
    tip_body);

  return world_to_base.inverse() * world_to_tip;
}

Eigen::Isometry3d moveit_relative_transform(
  moveit::core::RobotState & state,
  const std::string & base_frame,
  const std::string & tip_frame)
{
  const Eigen::Isometry3d world_to_base =
    state.getGlobalLinkTransform(base_frame);

  const Eigen::Isometry3d world_to_tip =
    state.getGlobalLinkTransform(tip_frame);

  return world_to_base.inverse() * world_to_tip;
}

void set_mujoco_configuration(
  const mjModel * model,
  mjData * data,
  const MappingContract & contract,
  const std::array<double, kPandaJointCount> & ros_positions)
{
  for (std::size_t index = 0; index < contract.joints.size(); ++index) {
    const auto & mapping =
      contract.joints[index];

    if (std::abs(mapping.position_sign) < 1.0e-12) {
      throw std::runtime_error(
              "Mapping position_sign must be non-zero for " +
              mapping.ros_joint);
    }

    const int joint_id =
      mj_name2id(
      model,
      mjOBJ_JOINT,
      mapping.mujoco_joint.c_str());

    if (joint_id < 0) {
      throw std::runtime_error(
              "Missing mapped MuJoCo joint: " +
              mapping.mujoco_joint);
    }

    const int qpos_address =
      model->jnt_qposadr[joint_id];

    const double mujoco_position =
      (
      ros_positions[index] -
      mapping.position_offset_rad
      ) /
      mapping.position_sign;

    data->qpos[qpos_address] =
      mujoco_position;
  }

  mj_forward(model, data);
}

double orientation_error_radians(
  const Eigen::Matrix3d & first,
  const Eigen::Matrix3d & second)
{
  const Eigen::Matrix3d relative =
    first.transpose() * second;

  return Eigen::AngleAxisd(relative).angle();
}

TEST(
  CrossModelPandaFkTest,
  moveit_and_mujoco_forward_kinematics_match)
{
  const MappingContract contract =
    load_mapping_contract();

  ASSERT_EQ(
    contract.joints.size(),
    kPandaJointCount);

  const auto robot_model =
    load_moveit_panda_model();

  ASSERT_NE(robot_model, nullptr);

  const auto * panda_group =
    robot_model->getJointModelGroup("panda_arm");

  ASSERT_NE(panda_group, nullptr);

  std::vector<std::string> expected_joint_names;

  for (const auto & mapping : contract.joints) {
    expected_joint_names.push_back(
      mapping.ros_joint);
  }

  EXPECT_EQ(
    panda_group->getVariableNames(),
    expected_joint_names);

  ASSERT_EQ(
    panda_group->getVariableCount(),
    kPandaJointCount);

  EXPECT_NE(
    robot_model->getLinkModel(
      contract.ros_base_frame),
    nullptr);

  EXPECT_NE(
    robot_model->getLinkModel(
      contract.ros_tip_frame),
    nullptr);

  EXPECT_NE(
    robot_model->getLinkModel("panda_hand"),
    nullptr);

  auto mujoco_model =
    load_mujoco_model();

  ASSERT_NE(mujoco_model, nullptr);

  MjDataPtr mujoco_data(
    mj_makeData(mujoco_model.get()));

  ASSERT_NE(mujoco_data, nullptr);

  const std::array<FramePair, 2> frame_pairs = {{
    {
      contract.ros_tip_frame,
      contract.mujoco_tip_body,
    },
    {
      "panda_hand",
      "hand",
    },
  }};

  const std::array<Configuration, 4> configurations = {{
    {
      "home",
      contract.home_positions,
    },
    {
      "offset_a",
      {
        0.3,
        -0.5,
        0.4,
        -1.2,
        0.5,
        1.8,
        -0.4,
      },
    },
    {
      "offset_b",
      {
        -0.6,
        0.7,
        -0.8,
        -2.2,
        0.9,
        0.8,
        1.1,
      },
    },
    {
      "offset_c",
      {
        1.0,
        -1.0,
        0.7,
        -0.8,
        -1.2,
        2.5,
        -1.0,
      },
    },
  }};

  for (const auto & configuration : configurations) {
    SCOPED_TRACE(
      std::string("configuration=") +
      configuration.name +
      " positions=" +
      configuration_string(
        configuration.positions));

    std::vector<double> moveit_positions(
      configuration.positions.begin(),
      configuration.positions.end());

    moveit::core::RobotState moveit_state(
      robot_model);

    moveit_state.setToDefaultValues();

    moveit_state.setJointGroupPositions(
      panda_group,
      moveit_positions);

    moveit_state.update();

    set_mujoco_configuration(
      mujoco_model.get(),
      mujoco_data.get(),
      contract,
      configuration.positions);

    for (const auto & frame_pair : frame_pairs) {
      SCOPED_TRACE(
        std::string("frame_pair=") +
        frame_pair.ros_frame +
        "<->" +
        frame_pair.mujoco_body);

      const Eigen::Isometry3d moveit_transform =
        moveit_relative_transform(
        moveit_state,
        contract.ros_base_frame,
        frame_pair.ros_frame);

      const Eigen::Isometry3d mujoco_transform =
        mujoco_relative_transform(
        mujoco_model.get(),
        mujoco_data.get(),
        contract.mujoco_base_body,
        frame_pair.mujoco_body);

      const double translation_error =
        (
        moveit_transform.translation() -
        mujoco_transform.translation()
        ).norm();

      const double orientation_error =
        orientation_error_radians(
        moveit_transform.rotation(),
        mujoco_transform.rotation());

      EXPECT_LE(
        translation_error,
        kTranslationToleranceMeters)
        << "translation_error_m="
        << translation_error
        << "\nMoveIt translation="
        << moveit_transform.translation().transpose()
        << "\nMuJoCo translation="
        << mujoco_transform.translation().transpose();

      EXPECT_LE(
        orientation_error,
        kOrientationToleranceRadians)
        << "orientation_error_rad="
        << orientation_error
        << "\nMoveIt rotation=\n"
        << moveit_transform.rotation()
        << "\nMuJoCo rotation=\n"
        << mujoco_transform.rotation();
    }
  }
}

}  // namespace
