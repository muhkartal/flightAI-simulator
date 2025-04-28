cmake_minimum_required(VERSION 3.16)
project(RL-DroneSim VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
find_package(prometheus-cpp CONFIG REQUIRED)
find_package(AirSim REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

# Generate gRPC code
get_filename_component(airsim_proto_path "${CMAKE_CURRENT_SOURCE_DIR}/proto/airsim_bridge.proto" ABSOLUTE)
get_filename_component(airsim_proto_dir "${airsim_proto_path}" DIRECTORY)
set(airsim_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/airsim_bridge.pb.cc")
set(airsim_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/airsim_bridge.pb.h")
set(airsim_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/airsim_bridge.grpc.pb.cc")
set(airsim_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/airsim_bridge.grpc.pb.h")

add_custom_command(
    OUTPUT "${airsim_proto_srcs}" "${airsim_proto_hdrs}" "${airsim_grpc_srcs}" "${airsim_grpc_hdrs}"
    COMMAND protobuf::protoc
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
         -I "${airsim_proto_dir}"
         --plugin=protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         "${airsim_proto_path}"
    DEPENDS "${airsim_proto_path}"
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Main library
add_library(rldronesim_lib
    src/agent.cpp
    src/environment.cpp
    src/grpc_client.cpp
    src/grpc_server.cpp
    src/metrics.cpp
    src/models.cpp
    src/ppo.cpp
    src/utils.cpp
    ${airsim_proto_srcs}
    ${airsim_grpc_srcs}
)

target_link_libraries(rldronesim_lib
    ${TORCH_LIBRARIES}
    gRPC::grpc++
    gRPC::grpc++_reflection
    protobuf::libprotobuf
    prometheus-cpp::core
    prometheus-cpp::push
    AirSim::AirLib
    Eigen3::Eigen
    yaml-cpp::yaml-cpp
    spdlog::spdlog
)

# Executable
add_executable(rldronesim src/main.cpp)
target_link_libraries(rldronesim
    rldronesim_lib
    CLI11::CLI11
)

# Testing
enable_testing()
add_subdirectory(test)

# Installation
install(TARGETS rldronesim
    RUNTIME DESTINATION bin
)
