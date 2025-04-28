#pragma once

#include "airsim_bridge.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <api/RpcLibServerBase.hpp>

using airsim_bridge::DroneControl;
using airsim_bridge::DroneAction;
using airsim_bridge::DroneObservation;
using airsim_bridge::ResetRequest;

class DroneControlServiceImpl final : public DroneControl::Service {
public:
    DroneControlServiceImpl();
    ~DroneControlServiceImpl();

    grpc::Status ControlLoop(
        grpc::ServerReaderWriter<DroneObservation, DroneAction>* stream) override;

    grpc::Status ResetEnvironment(
        grpc::ServerContext* context,
        const ResetRequest* request,
        DroneObservation* response) override;

private:
    DroneObservation get_observation();
    void apply_action(const DroneAction& action);

    std::unique_ptr<msr::airlib::RpcLibServerBase> airsim_client_;
    std::string vehicle_name_;
    int image_width_;
    int image_height_;

    std::atomic<bool> recording_;
};

class GrpcServer {
public:
    GrpcServer(const std::string& server_address = "0.0.0.0:50051");
    ~GrpcServer();

    void start();
    void stop();

private:
    void run();

    std::string server_address_;
    std::unique_ptr<DroneControlServiceImpl> service_;
    std::unique_ptr<grpc::Server> server_;
    std::thread server_thread_;
    std::atomic<bool> running_;
};
