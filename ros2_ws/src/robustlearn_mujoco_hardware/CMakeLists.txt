cmake_minimum_required(VERSION 3.8)
project(robustlearn_mujoco_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# MuJoCo is not a ROS package and therefore is not resolved through rosdep.
#
# Discovery order:
#   1. -DROBUSTLEARN_MUJOCO_ROOT=/path/to/mujoco/package
#   2. ROBUSTLEARN_MUJOCO_ROOT environment variable
#   3. import the Python "mujoco" package and use its installation directory
#
# The root must contain:
#   include/mujoco/mujoco.h
#   libmujoco.so or libmujoco.so.<version>
set(
  ROBUSTLEARN_MUJOCO_ROOT
  ""
  CACHE PATH
  "MuJoCo package root containing include/mujoco/mujoco.h and libmujoco.so.*"
)

if(
  NOT ROBUSTLEARN_MUJOCO_ROOT
  AND DEFINED ENV{ROBUSTLEARN_MUJOCO_ROOT}
)
  set(
    ROBUSTLEARN_MUJOCO_ROOT
    "$ENV{ROBUSTLEARN_MUJOCO_ROOT}"
  )
endif()

if(NOT ROBUSTLEARN_MUJOCO_ROOT)
  find_package(Python3 COMPONENTS Interpreter QUIET)

  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND
        "${Python3_EXECUTABLE}"
        -c
        "import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)"
      RESULT_VARIABLE MUJOCO_PYTHON_RESULT
      OUTPUT_VARIABLE MUJOCO_PYTHON_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )

    if(MUJOCO_PYTHON_RESULT EQUAL 0)
      set(
        ROBUSTLEARN_MUJOCO_ROOT
        "${MUJOCO_PYTHON_ROOT}"
      )
    endif()
  endif()
endif()

if(NOT ROBUSTLEARN_MUJOCO_ROOT)
  message(
    FATAL_ERROR
    "MuJoCo C/C++ installation not found. "
    "Set ROBUSTLEARN_MUJOCO_ROOT to the MuJoCo Python package directory "
    "containing include/mujoco/mujoco.h and libmujoco.so.*."
  )
endif()

set(
  MUJOCO_INCLUDE_DIR
  "${ROBUSTLEARN_MUJOCO_ROOT}/include"
)

if(NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
  message(
    FATAL_ERROR
    "MuJoCo header not found at "
    "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h"
  )
endif()

file(
  GLOB
  MUJOCO_LIBRARY_CANDIDATES
  "${ROBUSTLEARN_MUJOCO_ROOT}/libmujoco.so"
  "${ROBUSTLEARN_MUJOCO_ROOT}/libmujoco.so.*"
)

if(NOT MUJOCO_LIBRARY_CANDIDATES)
  message(
    FATAL_ERROR
    "MuJoCo shared library not found under "
    "${ROBUSTLEARN_MUJOCO_ROOT}"
  )
endif()

list(
  SORT
  MUJOCO_LIBRARY_CANDIDATES
  COMPARE NATURAL
  ORDER DESCENDING
)

list(
  GET
  MUJOCO_LIBRARY_CANDIDATES
  0
  MUJOCO_LIBRARY
)

add_library(
  MuJoCo::mujoco
  SHARED
  IMPORTED
)

set_target_properties(
  MuJoCo::mujoco
  PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
)

message(
  STATUS
  "RobustLearn MuJoCo root: ${ROBUSTLEARN_MUJOCO_ROOT}"
)
message(
  STATUS
  "RobustLearn MuJoCo library: ${MUJOCO_LIBRARY}"
)

add_library(
  ${PROJECT_NAME}
  SHARED
  src/mujoco_system.cpp
)

target_compile_features(
  ${PROJECT_NAME}
  PUBLIC
    cxx_std_17
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(
  ${PROJECT_NAME}
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

target_link_libraries(
  ${PROJECT_NAME}
  MuJoCo::mujoco
)

# ament_target_dependencies() uses the plain target_link_libraries()
# signature for this target, so keep that signature consistently.
# MuJoCo is a private implementation dependency and must not be exposed
# through the installed ROS package's INTERFACE_LINK_LIBRARIES.
get_target_property(
  ROBUSTLEARN_INTERFACE_LINK_LIBRARIES
  ${PROJECT_NAME}
  INTERFACE_LINK_LIBRARIES
)

if(ROBUSTLEARN_INTERFACE_LINK_LIBRARIES)
  list(
    REMOVE_ITEM
    ROBUSTLEARN_INTERFACE_LINK_LIBRARIES
    MuJoCo::mujoco
  )

  set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
      INTERFACE_LINK_LIBRARIES "${ROBUSTLEARN_INTERFACE_LINK_LIBRARIES}"
  )
endif()

# MuJoCo may come from the project's Python environment rather than a
# system library directory. Preserve that resolved location in the installed
# plugin so pluginlib can load its MuJoCo dependency at runtime.
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
    INSTALL_RPATH "${ROBUSTLEARN_MUJOCO_ROOT}"
)

pluginlib_export_plugin_description_file(
  hardware_interface
  hardware_plugin.xml
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_targets(
  export_${PROJECT_NAME}
  HAS_LIBRARY_TARGET
)

ament_export_dependencies(
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_index_cpp REQUIRED)
  find_package(Boost REQUIRED)
  find_package(moveit_core REQUIRED)
  find_package(srdfdom REQUIRED)
  find_package(urdf REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(
    test_mujoco_system_plugin
    test/test_mujoco_system_plugin.cpp
  )

  target_link_libraries(
    test_mujoco_system_plugin
    ${PROJECT_NAME}
    MuJoCo::mujoco
  )

  get_filename_component(
    ROBUSTLEARN_REPOSITORY_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    ABSOLUTE
  )

  target_compile_definitions(
    test_mujoco_system_plugin
    PRIVATE
      ROBUSTLEARN_TEST_MODEL_PATH="${ROBUSTLEARN_REPOSITORY_ROOT}/robot_description/mjcf/insertion/panda_insertion.xml"
  )

  ament_target_dependencies(
    test_mujoco_system_plugin
    hardware_interface
    pluginlib
    rclcpp
  )

  ament_add_gtest(
    test_cross_model_fk
    test/test_cross_model_fk.cpp
  )

  target_link_libraries(
    test_cross_model_fk
    MuJoCo::mujoco
  )

  target_compile_definitions(
    test_cross_model_fk
    PRIVATE
      ROBUSTLEARN_TEST_MODEL_PATH="${ROBUSTLEARN_REPOSITORY_ROOT}/robot_description/mjcf/insertion/panda_insertion.xml"
      ROBUSTLEARN_MAPPING_CONTRACT_PATH="${ROBUSTLEARN_REPOSITORY_ROOT}/robot_description/panda_joint_mapping.json"
  )

  target_include_directories(
    test_cross_model_fk
    PRIVATE
      ${Boost_INCLUDE_DIRS}
  )

  ament_target_dependencies(
    test_cross_model_fk
    ament_index_cpp
    moveit_core
    srdfdom
    urdf
  )
endif()

ament_package()
