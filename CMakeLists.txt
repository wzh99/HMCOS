cmake_minimum_required(VERSION 3.0)

project(hmcos)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

if(MSVC)
    add_compile_options(/wd4251)
endif(MSVC)

find_package(Protobuf REQUIRED)
find_package(ONNX 1.9 REQUIRED)
find_package(glog REQUIRED)
find_package(fmt REQUIRED)

set(HMCOS_COMMON_LIBS onnx glog::glog fmt::fmt)

set(HMCOS_LIB_SRC)
file(GLOB HMCOS_SRC_CORE src/core/*.cpp)
file(GLOB HMCOS_SRC_SCHED src/sched/*.cpp)
file(GLOB HMCOS_SRC_UTIL src/util/*.cpp)
list(APPEND HMCOS_SRC ${HMCOS_SRC_CORE} ${HMCOS_SRC_SCHED} ${HMCOS_SRC_UTIL})

add_library(hmcos STATIC ${HMCOS_SRC})
target_link_libraries(hmcos ${HMCOS_COMMON_LIBS})

add_executable(model_info src/bin/model_info.cpp)
target_link_libraries(model_info hmcos)

add_executable(op_sched src/bin/op_sched.cpp)
target_link_libraries(op_sched hmcos)
