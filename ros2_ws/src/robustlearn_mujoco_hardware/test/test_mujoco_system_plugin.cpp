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

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "hardware_interface/system_interface.hpp"
#include "pluginlib/class_loader.hpp"

namespace
{

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

}  // namespace
