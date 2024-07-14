cmake_minimum_required(VERSION 2.8.3)
project(sensor_driver_msgs)

set(PACKAGE_DEPENDENCIES
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
)

find_package(catkin REQUIRED COMPONENTS
  ${PACKAGE_DEPENDENCIES}
  roscpp
  rospy
  message_generation
)


add_message_files()

add_service_files()

# file(GLOB_RECURSE ALL_SRV_FILE_ABS "srv/*.srv" )
# set(ALL_SRV_FILE_REL)
# foreach(ABS_FIL ${ALL_SRV_FILE_ABS})
#   file(RELATIVE_PATH REL_FIL ${PROJECT_SOURCE_DIR}/srv ${ABS_FIL})
#   list(APPEND ALL_SRV_FILE_REL ${REL_FIL})
# endforeach()

# add_service_files(FILES ${ALL_SRV_FILE_REL})

generate_messages(
DEPENDENCIES
${PACKAGE_DEPENDENCIES}
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    # ${PACKAGE_DEPENDENCIES}
)
