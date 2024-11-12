cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(save_model)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen version: ${EIGEN_WORLD_VERSION}")
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(save_model save_model.cpp)
target_link_libraries(save_model PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(save_model PRIVATE Eigen3::Eigen)
set_property(TARGET save_model PROPERTY CXX_STANDARD 17)