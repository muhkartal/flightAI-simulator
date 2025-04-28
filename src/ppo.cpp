#include "ppo.h"
#include "utils.h"
#include <spdlog/spdlog.h>

PPO::PPO(
    std::shared_ptr<PolicyNetwork> policy,
    std::shared_ptr<ValueNetwork> value,
    torch::optim::Optimizer& optimizer,
    std::shared_ptr<Metrics> metrics,
    float learning_rate,
    float clip_param,
    float value_coef,
    float entropy_coef,
    float max_grad_norm,
    int num_mini_batches,
    int num_epochs,
    c10::optional<torch::TensorboardWriter> tb_writer
) : policy_(policy),
    value_(value),
    optimizer_(optimizer),
    metrics_(metrics),
    learning_rate_(learning_rate),
    clip_param_(clip_param),
    value_coef_(value_coef),
    entropy_coef_(entropy_coef),
    max_grad_norm_(max_grad_norm),
    num_mini_batches_(num_mini_batches),
    num_epochs_(num_epochs),
    tb_writer_(tb_writer),
    update_counter_(0) {}

void PPO::update(
    const torch::Tensor& observations,
    const torch::Tensor& actions,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& returns,
    const torch::Tensor& values,
    const torch::Tensor& advantages
) {
    int batch_size = observations.size(0);
    int mini_batch_size = batch_size / num_mini_batches_;

    // Normalize advantages
    auto adv_mean = advantages.mean();
    auto adv_std = advantages.std() + 1e-8;
    auto normalized_advantages = (advantages - adv_mean) / adv_std;

    metrics_->record_advantage(adv_mean.item<float>());

    for (int epoch = 0; epoch < num_epochs_; ++epoch) {
        auto mini_batches = utils::generate_mini_batches(
            observations, actions, old_log_probs, returns, values, normalized_advantages, mini_batch_size
        );

        for (const auto& batch : mini_batches) {
            auto [
                batch_obs,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_values,
                batch_advantages
            ] = batch;

            // Get new action distributions
            auto [mu, std] = policy_->forward(batch_obs);
            auto dist = torch::distributions::Normal(mu, std);

            // Get log probs of actions
            auto new_log_probs = dist.log_prob(batch_actions).sum(-1, /*keepdim=*/true);

            // Get entropy
            auto entropy = dist.entropy().mean();

            // Get new value predictions
            auto new_values = value_->forward(batch_obs);

            // Calculate policy loss
            auto ratio = torch::exp(new_log_probs - batch_old_log_probs);
            auto surr1 = ratio * batch_advantages;
            auto surr2 = torch::clamp(ratio, 1.0f - clip_param_, 1.0f + clip_param_) * batch_advantages;
            auto policy_loss = -torch::min(surr1, surr2).mean();

            // Calculate value loss
            auto value_loss = torch::nn::functional::mse_loss(new_values, batch_returns);

            // Calculate total loss
            auto loss = policy_loss + value_coef_ * value_loss - entropy_coef_ * entropy;

            // Backprop and optimize
            optimizer_.zero_grad();
            loss.backward();

            // Gradient clipping
            torch::nn::utils::clip_grad_norm_(
                policy_->parameters(), max_grad_norm_
            );
            torch::nn::utils::clip_grad_norm_(
                value_->parameters(), max_grad_norm_
            );

            optimizer_.step();

            // Record metrics
            metrics_->record_policy_loss(policy_loss.item<float>());
            metrics_->record_value_loss(value_loss.item<float>());
            metrics_->record_entropy(entropy.item<float>());
            metrics_->record_training_step();

            // Record to TensorBoard
            if (tb_writer_.has_value()) {
                tb_writer_->add_scalar("losses/policy_loss", policy_loss.item<float>(), update_counter_);
                tb_writer_->add_scalar("losses/value_loss", value_loss.item<float>(), update_counter_);
                tb_writer_->add_scalar("losses/entropy", entropy.item<float>(), update_counter_);
                tb_writer_->add_scalar("losses/total_loss", loss.item<float>(), update_counter_);
                tb_writer_->add_scalar("policy/ratio_mean", ratio.mean().item<float>(), update_counter_);
                tb_writer_->add_scalar("policy/ratio_min", ratio.min().item<float>(), update_counter_);
                tb_writer_->add_scalar("policy/ratio_max", ratio.max().item<float>(), update_counter_);

                update_counter_++;
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> PPO::compute_gae(
    const torch::Tensor& rewards,
    const torch::Tensor& values,
    const torch::Tensor& dones,
    float gamma,
    float lambda
) {
    int num_steps = rewards.size(0);

    auto returns = torch::zeros_like(rewards);
    auto advantages = torch::zeros_like(rewards);

    float gae = 0;
    float next_value = 0; // Assuming episode ends at the last step

    for (int t = num_steps - 1; t >= 0; --t) {
        float next_non_terminal = 1.0f - (t == num_steps - 1 ? 1.0f : dones[t + 1].item<float>());
        float next_val = t == num_steps - 1 ? next_value : values[t + 1].item<float>();

        float delta = rewards[t].item<float>() + gamma * next_val * next_non_terminal - values[t].item<float>();
        gae = delta + gamma * lambda * next_non_terminal * gae;

        returns[t] = gae + values[t];
        advantages[t] = gae;
    }

    return {returns, advantages};
}
