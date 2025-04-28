#pragma once

#include "grpc_client.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>

class Environment {
public:
    Environment(const std::string& config_path, int env_id);
    ~Environment();

    torch::Tensor reset(bool random_start = true);
    std::tuple<torch::Tensor, float, bool, float> step(const torch::Tensor& action);

    torch::Tensor preprocess_observation(const DroneObservation& obs);
    DroneAction action_to_proto(const torch::Tensor& action);

    void start_recording(const std::string& filename);
    void stop_recording();

    torch::IntArrayRef observation_shape() const;
    torch::IntArrayRef action_shape() const;

private:
    std::unique_ptr<GrpcClient> client_;
    YAML::Node config_;
    int env_id_;

    bool recording_;
    std::string record_filename_;

    torch::Tensor observation_space_shape_;
    torch::Tensor action_space_shape_;

    bool use_image_observations_;
    std::vector<std::string> image_types_;
    int image_width_;
    int image_height_;
};
