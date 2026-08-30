cmake_minimum_required(VERSION 3.8)
project(robustlearn_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(robustlearn_interfaces REQUIRED)
find_package(rclcpp_action REQUIRED)

add_executable(status_publisher src/status_publisher.cpp)

ament_target_dependencies(
  status_publisher
  rclcpp
  std_msgs
)

add_executable(status_subscriber src/status_subscriber.cpp)

ament_target_dependencies(
  status_subscriber
  rclcpp
  std_msgs
)

add_executable(system_mode_server src/system_mode_server.cpp)

ament_target_dependencies(
  system_mode_server
  rclcpp
  robustlearn_interfaces
)

add_executable(system_mode_client src/system_mode_client.cpp)

ament_target_dependencies(
  system_mode_client
  rclcpp
  robustlearn_interfaces
)

add_executable(
  system_check_action_server
  src/system_check_action_server.cpp
)

ament_target_dependencies(
  system_check_action_server
  rclcpp
  rclcpp_action
  robustlearn_interfaces
)

add_executable(
  system_check_action_client
  src/system_check_action_client.cpp
)

ament_target_dependencies(
  system_check_action_client
  rclcpp
  rclcpp_action
  robustlearn_interfaces
)

install(
  TARGETS
    status_publisher
    status_subscriber
    system_mode_server
    system_mode_client
    system_check_action_server
    system_check_action_client
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    config
    launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
