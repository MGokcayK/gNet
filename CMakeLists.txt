cmake_minimum_required(VERSION 3.18)
project(gNet)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(compiler_common)

# Check which compiler is being used, we support MSVC and GCC for now.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
include(compiler_msvc)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GCC")
include(compiler_gcc)
else()
message(FATAL_ERROR "Unsupported compiler.")
endif()

# Add sub projects
add_subdirectory(platform)
add_subdirectory(thirdparty)
