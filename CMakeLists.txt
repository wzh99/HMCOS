cmake_minimum_required(VERSION 3.0)

project(hmp)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

if(MSVC)
    add_compile_options(/wd4251)
endif(MSVC)

find_package(Protobuf REQUIRED)
find_package(ONNX 1.9 REQUIRED)
find_package(glog REQUIRED)
find_package(fmt REQUIRED)

set(HMP_COMMON_LIBS onnx glog::glog fmt::fmt)

add_executable(model_info src/bin/model_info.cpp)
target_link_libraries(model_info ${HMP_COMMON_LIBS})
