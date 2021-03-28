# CMake config file for TLX
#
# It defines the following variables
#  TLX_VERSION      - library version
#  TLX_CXX_FLAGS    - C++ flags for TLX
#  TLX_INCLUDE_DIRS - include directories for TLX
#  TLX_LIBRARIES    - libraries to link against


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was tlx-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(TLX_VERSION "0.5.20200222")

# compute paths from current cmake file path
get_filename_component(TLX_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# additional compiler flags
set(TLX_CXX_FLAGS "")

# additional include directories for tlx dependencies
set(TLX_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")

# load our library dependencies (contains definitions for IMPORTED targets)
include("${TLX_CMAKE_DIR}/tlx-targets.cmake")

# these are IMPORTED targets created by tlx-targets.cmake, link these with
# your program.
set(TLX_LIBRARIES "tlx")
