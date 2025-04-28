#include "environment.h"
#include "utils.h"
#include <spdlog/spdlog.h>

Environment::Environment(const std::string& config_path, int env_id) : env_id_(env_id), recording_(false) {
    config_ = YAML::LoadFile(config_path);

    auto server_address = config_["environment"]["server_address"].as<std::string>();
    client_ = std::make_unique<GrpcClient>(server_address);

    if (!client_->connect()) {
        throw std::runtime_error("Failed to connect to AirSim gRPC server");
    }

    use_image_observations_ = config_["environment"]["use_images"].as<bool>();
    if (use_image_observations_) {
        image_types_ = config_["environment"]["image_types"].as<std::vector<std::string>>();
        image_width_ = config_["environment"]["image_width"].as<int>();
        image_height_ = config_["environment"]["image_height"].as<int>();
    }

    observation_space_shape_ = torch::tensor(
        config_["environment"]["observation_shape"].as<std::vector<int64_t>>());
    action_space_shape_ = torch::tensor(
        config_["environment"]["action_shape"].as<std::vector<int64_t>>());
}

Environment::~Environment() {
    if (recording_) {
        stop_recording();
    }
    client_->disconnect();
}

torch::Tensor Environment::reset(bool random_start) {
    auto obs = client_->reset(random_start);
    return preprocess_observation(obs);
}

std::tuple<torch::Tensor, float, bool, float> Environment::step(const torch::Tensor& action) {
    auto action_proto = action_to_proto(action);
    auto obs = client_->step(action_proto);

    auto processed_obs = preprocess_observation(obs);
    float reward = obs.reward();
    bool done = obs.done();

    // Time in seconds since last step
    float time_delta = obs.timestamp_ns() / 1e9;

    return {processed_obs, reward, done, time_delta};
}

torch::Tensor Environment::preprocess_observation(const DroneObservation& obs) {
    if (use_image_observations_) {
        std::vector<torch::Tensor> image_tensors;

        for (const auto& image : obs.images()) {
            auto tensor = utils::preprocess_image(
                std::vector<uint8_t>(image.data().begin(), image.data().end()),
                image_width_, image_height_, image.format()
            );
            image_tensors.push_back(tensor);
        }

        return torch::cat(image_tensors, 0);
    } else {
        std::vector<float> state;

        // Position
        state.push_back(obs.position().x());
        state.push_back(obs.position().y());
        state.push_back(obs.position().z());

        // Linear velocity
        state.push_back(obs.linear_velocity().x());
        state.push_back(obs.linear_velocity().y());
        state.push_back(obs.linear_velocity().z());

        // Angular velocity
        state.push_back(obs.angular_velocity().x());
        state.push_back(obs.angular_velocity().y());
        state.push_back(obs.angular_velocity().z());

        // Orientation as quaternion
        state.push_back(obs.orientation().w());
        state.push_back(obs.orientation().x());
        state.push_back(obs.orientation().y());
        state.push_back(obs.orientation().z());

        // Collision info
        state.push_back(obs.collision_info());

        return torch::tensor(state).reshape({1, -1});
    }
}

DroneAction Environment::action_to_proto(const torch::Tensor& action) {
    DroneAction action_proto;

    auto action_vec = utils::to_vector(action);
    action_proto.set_roll(action_vec[0]);
    action_proto.set_pitch(action_vec[1]);
    action_proto.set_yaw_rate(action_vec[2]);
    action_proto.set_throttle(action_vec[3]);
    action_proto.set_timestamp_ns(utils::current_time_ns());

    return action_proto;
}

void Environment::start_recording(const std::string& filename) {
    if (!recording_) {
        recording_ = client_->start_recording(filename);
        record_filename_ = filename;
        spdlog::info("Started recording to {}", filename);
    }
}

void Environment::stop_recording() {
    if (recording_) {
        client_->stop_recording();
        recording_ = false;
        spdlog::info("Stopped recording to {}", record_filename_);
    }
}

torch::IntArrayRef Environment::observation_shape() const {
    return observation_space_shape_.sizes();
}

torch::IntArrayRef Environment::action_shape() const {
    return action_space_shape_.sizes();
}
