#include "grpc_client.h"
#include "utils.h"
#include <spdlog/spdlog.h>

GrpcClient::GrpcClient(const std::string& server_address)
    : server_address_(server_address), connected_(false), running_(false) {}

GrpcClient::~GrpcClient() {
    disconnect();
}

bool GrpcClient::connect() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (connected_) {
        return true;
    }

    channel_ = grpc::CreateChannel(server_address_, grpc::InsecureChannelCredentials());
    stub_ = DroneControl::NewStub(channel_);

    grpc::ClientContext context;
    stream_ = stub_->ControlLoop(&context);

    if (!stream_) {
        spdlog::error("Failed to create control stream");
        return false;
    }

    running_ = true;
    connected_ = true;

    control_loop_thread_ = std::thread(&GrpcClient::run_control_loop, this);

    spdlog::info("Connected to AirSim gRPC server at {}", server_address_);
    return true;
}

void GrpcClient::disconnect() {
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!connected_) {
            return;
        }

        running_ = false;

        if (stream_) {
            stream_->WritesDone();
        }
    }

    if (control_loop_thread_.joinable()) {
        control_loop_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        connected_ = false;
        stream_.reset();
        stub_.reset();
        channel_.reset();
    }

    spdlog::info("Disconnected from AirSim gRPC server");
}

DroneObservation GrpcClient::reset(bool random_start) {
    grpc::ClientContext context;
    ResetRequest request;
    DroneObservation response;

    request.set_random_start(random_start);

    auto status = stub_->ResetEnvironment(&context, request, &response);

    if (!status.ok()) {
        throw std::runtime_error("Reset environment failed: " + status.error_message());
    }

    return response;
}

DroneObservation GrpcClient::step(const DroneAction& action) {
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!connected_ || !running_) {
            throw std::runtime_error("Not connected to server");
        }

        if (!stream_->Write(action)) {
            throw std::runtime_error("Failed to send action to server");
        }
    }

    DroneObservation observation;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !observation_queue_.empty() || !running_; });

        if (!running_) {
            throw std::runtime_error("Control loop stopped");
        }

        observation = observation_queue_.front();
        observation_queue_.pop();
    }

    return observation;
}

bool GrpcClient::start_recording(const std::string& filename) {
    // In a real implementation, this would send a request to the server to start recording
    // For simplicity, we'll just log it here
    spdlog::info("Requested recording start with filename: {}", filename);
    return true;
}

bool GrpcClient::stop_recording() {
    // In a real implementation, this would send a request to the server to stop recording
    // For simplicity, we'll just log it here
    spdlog::info("Requested recording stop");
    return true;
}

void GrpcClient::run_control_loop() {
    DroneObservation observation;

    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_) {
                break;
            }
        }

        if (!stream_->Read(&observation)) {
            spdlog::error("Failed to read observation from server");

            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
            break;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            observation_queue_.push(observation);
        }

        cv_.notify_one();
    }

    spdlog::info("Control loop stopped");
}
