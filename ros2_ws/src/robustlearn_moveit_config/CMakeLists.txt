cmake_minimum_required(VERSION 3.8)
project(robustlearn_moveit_config)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/hybrid_insertion_coordinator.py
  DESTINATION lib/${PROJECT_NAME}
)

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

  add_launch_test(
    test/test_moveit_mujoco_execution_launch.py
    TIMEOUT 90
  )

  add_launch_test(
    test/test_hybrid_moveit_readiness_timeout_launch.py
    TIMEOUT 20
  )

  add_launch_test(
    test/test_hybrid_insertion_e2e_launch.py
    TIMEOUT 90
  )

  add_launch_test(
    test/test_hybrid_coordinator_failures_launch.py
    TIMEOUT 30
  )
endif()

ament_package()
