cmake_minimum_required(VERSION 3.1)

project(hmp)

set(CMAKE_CXX_STANDARD 14)

find_package(ONNX 1.9 REQUIRED)
include_directories(${ONNX_INCLUDE_DIRS})

