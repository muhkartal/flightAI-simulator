#include "models.h"
#include <spdlog/spdlog.h>

ConvolutionalBackbone::ConvolutionalBackbone(const std::vector<int64_t>& input_shape) {
    // input_shape should be [C, H, W]
    if (input_shape.size() != 3) {
        throw std::runtime_error("Input shape must have 3 dimensions (C, H, W)");
    }

    int64_t c = input_shape[0];
    int64_t h = input_shape[1];
    int64_t w = input_shape[2];

    conv1_ = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(c, 32, 8).stride(4).padding(2)
    ));

    bn1_ = register_module("bn1", torch::nn::BatchNorm2d(32));

    conv2_ = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(32, 64, 4).stride(2).padding(1)
    ));

    bn2_ = register_module("bn2", torch::nn::BatchNorm2d(64));

    conv3_ = register_module("conv3", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)
    ));

    bn3_ = register_module("bn3", torch::nn::BatchNorm2d(64));

    // Calculate output size
    int64_t h_out = (h + 2*2 - 8) / 4 + 1;
    int64_t w_out = (w + 2*2 - 8) / 4 + 1;

    h_out = (h_out + 2*1 - 4) / 2 + 1;
    w_out = (w_out + 2*1 - 4) / 2 + 1;

    h_out = (h_out + 2*1 - 3) / 1 + 1;
    w_out = (w_out + 2*1 - 3) / 1 + 1;

    output_size_ = 64 * h_out * w_out;
}

torch::Tensor ConvolutionalBackbone::forward(torch::Tensor x) {
    x = torch::relu(bn1_(conv1_(x)));
    x = torch::relu(bn2_(conv2_(x)));
    x = torch::relu(bn3_(conv3_(x)));
    return x.view({x.size(0), -1});
}

PolicyNetwork::PolicyNetwork(const std::vector<int64_t>& input_shape, int64_t num_actions)
    : num_actions_(num_actions) {

    backbone_ = std::make_shared<ConvolutionalBackbone>(input_shape);
    register_module("backbone", backbone_);

    fc1_ = register_module("fc1", torch::nn::Linear(backbone_->output_size(), 512));
    fc2_ = register_module("fc2", torch::nn::Linear(512, 256));
    fc_mean_ = register_module("fc_mean", torch::nn::Linear(256, num_actions));

    // Initialize log_std as a learnable parameter
    log_std_ = register_parameter("log_std", torch::zeros({num_actions}));

    init_weights();
}

void PolicyNetwork::init_weights() {
    // Use orthogonal initialization
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto* linear = dynamic_cast<torch::nn::Linear*>(module.get())) {
            torch::nn::init::orthogonal_(linear->weight, 0.01);
            torch::nn::init::constant_(linear->bias, 0.0);
        } else if (auto* conv = dynamic_cast<torch::nn::Conv2d*>(module.get())) {
            torch::nn::init::orthogonal_(conv->weight, 0.01);
            torch::nn::init::constant_(conv->bias, 0.0);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> PolicyNetwork::forward(torch::Tensor x) {
    x = backbone_->forward(x);
    x = torch::relu(fc1_(x));
    x = torch::relu(fc2_(x));

    auto mu = fc_mean_(x);
    auto std = log_std_.exp().expand({x.size(0), num_actions_});

    return {mu, std};
}

torch::Tensor PolicyNetwork::act(torch::Tensor x, bool deterministic) {
    auto [mu, std] = forward(x);

    if (deterministic) {
        return torch::cat({mu, torch::zeros_like(mu)}, 1); // action, log_prob
    }

    auto normal = torch::distributions::Normal(mu, std);
    auto action = normal.sample();
    auto log_prob = normal.log_prob(action).sum(-1, /*keepdim=*/true);

    return torch::cat({action, log_prob}, 1); // action, log_prob
}

ValueNetwork::ValueNetwork(const std::vector<int64_t>& input_shape) {
    backbone_ = std::make_shared<ConvolutionalBackbone>(input_shape);
    register_module("backbone", backbone_);

    fc1_ = register_module("fc1", torch::nn::Linear(backbone_->output_size(), 512));
    fc2_ = register_module("fc2", torch::nn::Linear(512, 256));
    fc_value_ = register_module("fc_value", torch::nn::Linear(256, 1));

    init_weights();
}

void ValueNetwork::init_weights() {
    // Use orthogonal initialization
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto* linear = dynamic_cast<torch::nn::Linear*>(module.get())) {
            torch::nn::init::orthogonal_(linear->weight, 0.01);
            torch::nn::init::constant_(linear->bias, 0.0);
        } else if (auto* conv = dynamic_cast<torch::nn::Conv2d*>(module.get())) {
            torch::nn::init::orthogonal_(conv->weight, 0.01);
            torch::nn::init::constant_(conv->bias, 0.0);
        }
    }
}

torch::Tensor ValueNetwork::forward(torch::Tensor x) {
    x = backbone_->forward(x);
    x = torch::relu(fc1_(x));
    x = torch::relu(fc2_(x));
    return fc_value_(x);
}
