#pragma once

#include <torch/torch.h>
#include "models.h"
#include "metrics.h"
#include <memory>

class PPO {
public:
    PPO(
        std::shared_ptr<PolicyNetwork> policy,
        std::shared_ptr<ValueNetwork> value,
        torch::optim::Optimizer& optimizer,
        std::shared_ptr<Metrics> metrics,
        float learning_rate = 3e-4,
        float clip_param = 0.2,
        float value_coef = 0.5,
        float entropy_coef = 0.01,
        float max_grad_norm = 0.5,
        int num_mini_batches = 4,
        int num_epochs = 4,
        c10::optional<torch::TensorboardWriter> tb_writer = c10::nullopt
    );

    void update(
        const torch::Tensor& observations,
        const torch::Tensor& actions,
        const torch::Tensor& old_log_probs,
        const torch::Tensor& returns,
        const torch::Tensor& values,
        const torch::Tensor& advantages
    );

    std::tuple<torch::Tensor, torch::Tensor> compute_gae(
        const torch::Tensor& rewards,
        const torch::Tensor& values,
        const torch::Tensor& dones,
        float gamma,
        float lambda
    );

private:
    std::shared_ptr<PolicyNetwork> policy_;
    std::shared_ptr<ValueNetwork> value_;
    torch::optim::Optimizer& optimizer_;
    std::shared_ptr<Metrics> metrics_;

    float learning_rate_;
    float clip_param_;
    float value_coef_;
    float entropy_coef_;
    float max_grad_norm_;
    int num_mini_batches_;
    int num_epochs_;

    c10::optional<torch::TensorboardWriter> tb_writer_;
    int update_counter_;
};
