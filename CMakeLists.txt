cmake_minimum_required(VERSION 3.0)

project(hos)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

if(MSVC)
    add_compile_options(/wd4251)
endif(MSVC)

find_package(Protobuf REQUIRED)
find_package(ONNX 1.9 REQUIRED)
find_package(glog REQUIRED)
find_package(fmt REQUIRED)

set(HOS_COMMON_LIBS onnx glog::glog fmt::fmt)

set(HOS_LIB_SRC)
file(GLOB HOS_LIB_UTIL src/util/*.cpp)
list(APPEND HOS_LIB_SRC ${HOS_LIB_UTIL})

add_library(hos STATIC ${HOS_LIB_SRC})
target_link_libraries(hos ${HOS_COMMON_LIBS})

add_executable(model_info src/bin/model_info.cpp)
target_link_libraries(model_info hos)
