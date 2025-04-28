#pragma once

#include "airsim_bridge.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

using airsim_bridge::DroneControl;
using airsim_bridge::DroneAction;
using airsim_bridge::DroneObservation;
using airsim_bridge::ResetRequest;

class GrpcClient {
public:
    GrpcClient(const std::string& server_address = "localhost:50051");
    ~GrpcClient();

    bool connect();
    void disconnect();

    DroneObservation reset(bool random_start = true);
    DroneObservation step(const DroneAction& action);

    bool start_recording(const std::string& filename);
    bool stop_recording();

private:
    void run_control_loop();

    std::string server_address_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<DroneControl::Stub> stub_;

    std::unique_ptr<grpc::ClientReaderWriter<DroneAction, DroneObservation>> stream_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<DroneObservation> observation_queue_;
    bool connected_;
    bool running_;

    std::thread control_loop_thread_;
};
