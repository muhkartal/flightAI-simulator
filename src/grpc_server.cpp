#include "grpc_server.h"
#include "utils.h"
#include <spdlog/spdlog.h>

DroneControlServiceImpl::DroneControlServiceImpl()
    : vehicle_name_("PX4"), image_width_(84), image_height_(84), recording_(false) {

    airsim_client_ = std::make_unique<msr::airlib::RpcLibServerBase>();

    airsim_client_->confirmConnection();
    airsim_client_->enableApiControl(true, vehicle_name_);
    airsim_client_->armDisarm(true, vehicle_name_);

    spdlog::info("Connected to AirSim");
}

DroneControlServiceImpl::~DroneControlServiceImpl() {
    if (recording_) {
        airsim_client_->stopRecording();
    }

    airsim_client_->armDisarm(false, vehicle_name_);
    airsim_client_->enableApiControl(false, vehicle_name_);
}

grpc::Status DroneControlServiceImpl::ControlLoop(
    grpc::ServerReaderWriter<DroneObservation, DroneAction>* stream) {

    DroneAction action;

    // Send initial observation
    auto initial_obs = get_observation();
    stream->Write(initial_obs);

    // Main control loop
    while (stream->Read(&action)) {
        apply_action(action);
        auto obs = get_observation();

        if (!stream->Write(obs)) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Failed to send observation");
        }
    }

    return grpc::Status::OK;
}

grpc::Status DroneControlServiceImpl::ResetEnvironment(
    grpc::ServerContext* context,
    const ResetRequest* request,
    DroneObservation* response) {

    airsim_client_->reset();

    if (request->random_start()) {
        // Teleport to random position
        float x = (std::rand() % 100) - 50.0f;
        float y = (std::rand() % 100) - 50.0f;
        float z = -10.0f - (std::rand() % 10);

        msr::airlib::Pose pose;
        pose.position = msr::airlib::Vector3r(x, y, z);
        airsim_client_->simSetVehiclePose(pose, true, vehicle_name_);
    } else if (request->has_initial_position()) {
        // Teleport to specified position
        msr::airlib::Pose pose;
        pose.position.x() = request->initial_position().x();
        pose.position.y() = request->initial_position().y();
        pose.position.z() = request->initial_position().z();

        if (request->has_initial_orientation()) {
            pose.orientation.w() = request->initial_orientation().w();
            pose.orientation.x() = request->initial_orientation().x();
            pose.orientation.y() = request->initial_orientation().y();
            pose.orientation.z() = request->initial_orientation().z();
        }

        airsim_client_->simSetVehiclePose(pose, true, vehicle_name_);
    }

    *response = get_observation();
    return grpc::Status::OK;
}

DroneObservation DroneControlServiceImpl::get_observation() {
    auto state = airsim_client_->getMultirotorState(vehicle_name_);
    auto collision = airsim_client_->simGetCollisionInfo(vehicle_name_);

    DroneObservation obs;

    // Position
    obs.mutable_position()->set_x(state.getPosition().x());
    obs.mutable_position()->set_y(state.getPosition().y());
    obs.mutable_position()->set_z(state.getPosition().z());

    // Linear velocity
    obs.mutable_linear_velocity()->set_x(state.kinematics_estimated.linear_velocity.x());
    obs.mutable_linear_velocity()->set_y(state.kinematics_estimated.linear_velocity.y());
    obs.mutable_linear_velocity()->set_z(state.kinematics_estimated.linear_velocity.z());

    // Angular velocity
    obs.mutable_angular_velocity()->set_x(state.kinematics_estimated.angular_velocity.x());
    obs.mutable_angular_velocity()->set_y(state.kinematics_estimated.angular_velocity.y());
    obs.mutable_angular_velocity()->set_z(state.kinematics_estimated.angular_velocity.z());

    // Orientation
    auto q = state.getOrientation();
    obs.mutable_orientation()->set_w(q.w());
    obs.mutable_orientation()->set_x(q.x());
    obs.mutable_orientation()->set_y(q.y());
    obs.mutable_orientation()->set_z(q.z());

    // Collision
    obs.set_collision_info(collision.has_collided ? 1.0f : 0.0f);

    // Images
    auto responses = airsim_client_->simGetImages({
        msr::airlib::ImageRequest("0", msr::airlib::ImageType::Scene, false, false)
    }, vehicle_name_);

    if (!responses.empty()) {
        auto* image = obs.add_images();
        image->set_width(image_width_);
        image->set_height(image_height_);
        image->set_format("rgb");
        image->set_data(responses[0].image_data_uint8.data(), responses[0].image_data_uint8.size());
    }

    // Reward (computed based on position, velocity, and collision)
    float reward = 0.0f;

    // Penalty for collision
    if (collision.has_collided) {
        reward -= 10.0f;
        obs.set_done(true);
    } else {
        obs.set_done(false);
    }

    // Reward for being close to target position
    msr::airlib::Vector3r target_pos(0, 0, -10); // 10 meters above ground
    float dist = (state.getPosition() - target_pos).norm();
    reward += std::max(0.0f, 20.0f - dist) / 20.0f;

    // Penalty for high velocity
    float vel_norm = state.kinematics_estimated.linear_velocity.norm();
    reward -= std::min(1.0f, vel_norm / 10.0f);

    obs.set_reward(reward);
    obs.set_timestamp_ns(utils::current_time_ns());

    return obs;
}

void DroneControlServiceImpl::apply_action(const DroneAction& action) {
    float roll = std::clamp(action.roll(), -1.0f, 1.0f);
    float pitch = std::clamp(action.pitch(), -1.0f, 1.0f);
    float yaw_rate = std::clamp(action.yaw_rate(), -1.0f, 1.0f);
    float throttle = std::clamp(action.throttle(), 0.0f, 1.0f);

    airsim_client_->moveByRollPitchYawThrottleAsync(
        roll, pitch, yaw_rate, throttle, 1.0f / 120.0f, vehicle_name_);
}

GrpcServer::GrpcServer(const std::string& server_address)
    : server_address_(server_address), running_(false) {}

GrpcServer::~GrpcServer() {
    stop();
}

void GrpcServer::start() {
    if (running_) {
        return;
    }

    running_ = true;
    server_thread_ = std::thread(&GrpcServer::run, this);

    spdlog::info("gRPC server started on {}", server_address_);
}

void GrpcServer::stop() {
    if (!running_) {
        return;
    }

    running_ = false;

    if (server_) {
        server_->Shutdown();
    }

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    spdlog::info("gRPC server stopped");
}

void GrpcServer::run() {
    service_ = std::make_unique<DroneControlServiceImpl>();

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());

    server_ = builder.BuildAndStart();

    if (!server_) {
        spdlog::error("Failed to start gRPC server");
        running_ = false;
        return;
    }

    spdlog::info("gRPC server listening on {}", server_address_);

    server_->Wait();
}
