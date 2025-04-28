#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

class ConvolutionalBackbone : public torch::nn::Module {
public:
    ConvolutionalBackbone(const std::vector<int64_t>& input_shape);

    torch::Tensor forward(torch::Tensor x);
    int64_t output_size() const { return output_size_; }

private:
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr}, conv3_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr}, bn3_{nullptr};
    int64_t output_size_;
};

class PolicyNetwork : public torch::nn::Module {
public:
    PolicyNetwork(const std::vector<int64_t>& input_shape, int64_t num_actions);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    torch::Tensor act(torch::Tensor x, bool deterministic = false);

private:
    std::shared_ptr<ConvolutionalBackbone> backbone_;
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, fc_mean_{nullptr};
    torch::nn::Parameter log_std_{nullptr};
    int64_t num_actions_;

    void init_weights();
};

class ValueNetwork : public torch::nn::Module {
public:
    ValueNetwork(const std::vector<int64_t>& input_shape);

    torch::Tensor forward(torch::Tensor x);

private:
    std::shared_ptr<ConvolutionalBackbone> backbone_;
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, fc_value_{nullptr};

    void init_weights();
};
