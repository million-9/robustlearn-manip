cmake_minimum_required(VERSION 3.8)
project(robustlearn_moveit_config)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  ament_lint_auto_find_test_dependencies()

  add_launch_test(
    test/test_week3_acceptance_launch.py
    TIMEOUT 60
  )
endif()

ament_package()