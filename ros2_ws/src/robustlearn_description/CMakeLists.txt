cmake_minimum_required(VERSION 3.8)
project(robustlearn_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    config
    launch
    urdf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  ament_lint_auto_find_test_dependencies()

  add_launch_test(
    test/test_mujoco_panda_launch.py
    TIMEOUT 25
  )

  add_launch_test(
    test/test_mujoco_trajectory_launch.py
    TIMEOUT 30
  )
endif()

ament_package()