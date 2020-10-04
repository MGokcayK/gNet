# Set minimum required version of CMAKE
cmake_minimum_required(VERSION 3.17.0)

# Set project and version
PROJECT(gNet VERSION 0.3.0)

# Set Eigen Directory as EIGEN_DIR
SET(EIGEN_DIR "C:/eigen")

# Include Eigen
include_directories(${EIGEN_DIR})
include_directories("${PROJECT_SOURCE_DIR}/include")

# Set basic properties 
SET(CMAKE_BUILD_SHARED_LISB TRUE)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_COMPILER g++)

# Create Runtime bin directory to get output in that directory
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Set each files and headers seperately
SET(tensor              include/tensor.hpp)
SET(tensor_ops          include/tensor_ops.hpp src/tensor_ops.cpp)
SET(ops                 include/ops.hpp)
SET(dependency          include/dependency.hpp)
SET(graph               include/graph.hpp src/graph.cpp)
SET(py_wrapper          test/py_wrapper.cpp)
SET(cpp_ops_test        test/cpp_ops_test.cpp)

# Add library
ADD_LIBRARY(tensor INTERFACE)
TARGET_SOURCES(tensor INTERFACE tensor.hpp)

ADD_LIBRARY(ops INTERFACE)
TARGET_SOURCES(ops INTERFACE ops.hpp)

ADD_LIBRARY(dependency INTERFACE)
TARGET_SOURCES(dependency INTERFACE dependency.hpp)

ADD_LIBRARY(tensor_ops ${tensor_ops})
ADD_LIBRARY(graph ${graph})
ADD_LIBRARY(cpp_ops_test SHARED ${cpp_ops_test})
ADD_LIBRARY(py_wrapper SHARED ${py_wrapper})

# Set suffix as '.so'
SET_TARGET_PROPERTIES(cpp_ops_test PROPERTIES SUFFIX ".so")
SET_TARGET_PROPERTIES(py_wrapper PROPERTIES SUFFIX ".so")

# Set compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC")

# Link libraries to target
TARGET_LINK_LIBRARIES(cpp_ops_test tensor_ops)
TARGET_LINK_LIBRARIES(tensor_ops graph)
