#include "agent.h"
#include "utils.h"
#include <spdlog/spdlog.h>
#include <chrono>

void RolloutBuffer::clear() {
    observations.clear();
    actions.clear();
    rewards.clear();
    values.clear();
    dones.clear();
    log_probs.clear();
}

void RolloutBuffer::add(const torch::Tensor& obs, const torch::Tensor& action, float reward,
                       const torch::Tensor& value, bool done, const torch::Tensor& log_prob) {
    observations.push_back(obs);
    actions.push_back(action);
    rewards.push_back(torch::tensor(reward));
    values.push_back(value);
    dones.push_back(torch::tensor(static_cast<float>(done)));
    log_probs.push_back(log_prob);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RolloutBuffer::get_tensors(const torch::Device& device) {
    auto cat_and_to = [&device](const std::vector<torch::Tensor>& tensors) {
        return torch::cat(tensors, 0).to(device);
    };

    return std::make_tuple(
        cat_and_to(observations),
        cat_and_to(actions),
        cat_and_to(rewards),
        cat_and_to(values),
        cat_and_to(dones),
        cat_and_to(log_probs)
    );
}

EnvironmentWorker::EnvironmentWorker(const std::string& config_path, int worker_id)
    : config_path_(config_path), worker_id_(worker_id), running_(false), collect_requested_(false),
      training_mode_(false), steps_to_collect_(0) {}

EnvironmentWorker::~EnvironmentWorker() {
    stop();
}

void EnvironmentWorker::start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (running_) return;
        running_ = true;
    }
    worker_thread_ = std::thread(&EnvironmentWorker::worker_loop, this);
}

void EnvironmentWorker::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) return;
        running_ = false;
    }
    cv_.notify_one();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

RolloutBuffer EnvironmentWorker::collect_rollout(
    std::shared_ptr<PolicyNetwork> policy,
    std::shared_ptr<ValueNetwork> value,
    int steps,
    bool training
) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        policy_ = policy;
        value_ = value;
        steps_to_collect_ = steps;
        training_mode_ = training;
        collect_requested_ = true;
    }
    cv_.notify_one();

    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return !collect_requested_ || !running_; });

    RolloutBuffer result;
    if (running_) {
        result = std::move(rollout_buffer_);
        rollout_buffer_.clear();
    }
    return result;
}

void EnvironmentWorker::worker_loop() {
    try {
        env_ = std::make_unique<Environment>(config_path_, worker_id_);

        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return collect_requested_ || !running_; });

            if (!running_) break;

            auto policy = policy_;
            auto value = value_;
            auto steps = steps_to_collect_;
            auto training = training_mode_;

            lock.unlock();

            torch::NoGradGuard no_grad_guard;

            rollout_buffer_.clear();
            auto obs = env_->reset();

            for (int step = 0; step < steps; ++step) {
                auto action = policy->act(obs.unsqueeze(0), !training);
                auto value_pred = value->forward(obs.unsqueeze(0));

                auto [next_obs, reward, done, _] = env_->step(action);

                rollout_buffer_.add(
                    obs,
                    action,
                    reward,
                    value_pred.squeeze(),
                    done,
                    action[1] // log_prob is second return value from act()
                );

                if (done) {
                    obs = env_->reset();
                } else {
                    obs = next_obs;
                }
            }

            lock.lock();
            collect_requested_ = false;
            lock.unlock();
            cv_.notify_one();
        }
    } catch (const std::exception& e) {
        spdlog::error("Worker {} error: {}", worker_id_, e.what());
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        collect_requested_ = false;
        cv_.notify_all();
    }
}

Agent::Agent(const std::string& config_path) : config_path_(config_path) {
    config_ = YAML::LoadFile(config_path);

    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    spdlog::info("Using device: {}", device_ == torch::kCUDA ? "CUDA" : "CPU");

    num_workers_ = config_["training"]["num_workers"].as<int>();
    steps_per_epoch_ = config_["training"]["steps_per_epoch"].as<int>();
    epochs_ = config_["training"]["epochs"].as<int>();
    batch_size_ = config_["training"]["batch_size"].as<int>();
    gamma_ = config_["training"]["gamma"].as<float>();
    lambda_ = config_["training"]["lambda"].as<float>();

    auto input_shape = config_["environment"]["observation_shape"].as<std::vector<int64_t>>();
    auto num_actions = config_["environment"]["action_shape"][0].as<int64_t>();

    policy_ = std::make_shared<PolicyNetwork>(input_shape, num_actions);
    value_ = std::make_shared<ValueNetwork>(input_shape);

    policy_->to(device_);
    value_->to(device_);

    auto learning_rate = config_["training"]["learning_rate"].as<float>();
    std::vector<torch::optim::OptimizerParamGroup> param_groups;
    param_groups.push_back(torch::optim::OptimizerParamGroup(policy_->parameters()));
    param_groups.push_back(torch::optim::OptimizerParamGroup(value_->parameters()));
    optimizer_ = std::make_unique<torch::optim::Adam>(param_groups, torch::optim::AdamOptions(learning_rate));

    metrics_ = std::make_shared<Metrics>();

    ppo_ = std::make_shared<PPO>(
        policy_,
        value_,
        *optimizer_,
        metrics_,
        learning_rate,
        config_["training"]["clip_param"].as<float>(),
        config_["training"]["value_coef"].as<float>(),
        config_["training"]["entropy_coef"].as<float>(),
        config_["training"]["max_grad_norm"].as<float>(),
        config_["training"]["num_mini_batches"].as<int>(),
        config_["training"]["num_epochs"].as<int>(),
        tb_writer_
    );

    for (int i = 0; i < num_workers_; ++i) {
        workers_.push_back(std::make_unique<EnvironmentWorker>(config_path_, i));
        workers_[i]->start();
    }

    create_tensorboard_writer();
}

void Agent::create_tensorboard_writer() {
    auto log_dir = config_["training"]["log_dir"].as<std::string>();
    tb_writer_ = torch::TensorboardWriter(log_dir);
}

void Agent::train(int epochs) {
    spdlog::info("Starting training for {} epochs", epochs);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        spdlog::info("Epoch {}/{}", epoch + 1, epochs);

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<RolloutBuffer> rollouts;
        for (auto& worker : workers_) {
            rollouts.push_back(worker->collect_rollout(
                policy_, value_, steps_per_epoch_ / num_workers_, true
            ));
        }

        // Combine rollouts
        torch::Tensor all_obs, all_actions, all_rewards, all_values, all_dones, all_log_probs;
        std::vector<torch::Tensor> obs_vec, actions_vec, rewards_vec, values_vec, dones_vec, log_probs_vec;

        for (auto& rollout : rollouts) {
            auto [obs, actions, rewards, values, dones, log_probs] = rollout.get_tensors(device_);
            obs_vec.push_back(obs);
            actions_vec.push_back(actions);
            rewards_vec.push_back(rewards);
            values_vec.push_back(values);
            dones_vec.push_back(dones);
            log_probs_vec.push_back(log_probs);
        }

        all_obs = torch::cat(obs_vec, 0);
        all_actions = torch::cat(actions_vec, 0);
        all_rewards = torch::cat(rewards_vec, 0);
        all_values = torch::cat(values_vec, 0);
        all_dones = torch::cat(dones_vec, 0);
        all_log_probs = torch::cat(log_probs_vec, 0);

        // Compute returns and advantages
        auto [returns, advantages] = ppo_->compute_gae(
            all_rewards, all_values, all_dones, gamma_, lambda_
        );

        // Update policy and value networks
        ppo_->update(all_obs, all_actions, all_log_probs, returns, all_values, advantages);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

        auto steps_per_second = steps_per_epoch_ / duration;
        metrics_->record_steps_per_second(steps_per_second);

        spdlog::info("Epoch {} completed in {:.2f}s ({:.2f} steps/s)",
                     epoch + 1, duration, steps_per_second);

        // Save checkpoint every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            save_checkpoint(config_["training"]["checkpoint_dir"].as<std::string>() +
                           "/checkpoint_" + std::to_string(epoch + 1) + ".pt");
        }
    }

    // Export final model to ONNX
    export_model(config_["training"]["model_dir"].as<std::string>() + "/final_model.onnx");
}

void Agent::evaluate(int episodes, bool record) {
    spdlog::info("Evaluating for {} episodes", episodes);

    torch::NoGradGuard no_grad_guard;

    auto env = std::make_unique<Environment>(config_path_, 0);
    float total_reward = 0.0f;

    if (record) {
        env->start_recording("evaluation.mp4");
    }

    for (int ep = 0; ep < episodes; ++ep) {
        auto obs = env->reset();
        float episode_reward = 0.0f;
        bool done = false;

        while (!done) {
            auto action = policy_->act(obs.unsqueeze(0), true);
            auto [next_obs, reward, episode_done, _] = env->step(action);

            episode_reward += reward;
            done = episode_done;
            obs = next_obs;
        }

        total_reward += episode_reward;
        spdlog::info("Episode {} reward: {:.2f}", ep + 1, episode_reward);
    }

    if (record) {
        env->stop_recording();
    }

    spdlog::info("Average reward over {} episodes: {:.2f}", episodes, total_reward / episodes);
}

void Agent::play_interactive() {
    spdlog::info("Starting interactive play mode");

    // This would be implemented with keyboard input handling
    // For simplicity, we'll just simulate a few actions here

    auto env = std::make_unique<Environment>(config_path_, 0);
    auto obs = env->reset();

    // In a real implementation, this would be a loop that reads keyboard input
    // and sends the corresponding actions to the environment
    for (int i = 0; i < 100; ++i) {
        // Create a synthetic action (this would come from keyboard in reality)
        auto action = torch::zeros({1, 4});
        action[0][0] = std::sin(i * 0.1); // Roll
        action[0][1] = std::cos(i * 0.1); // Pitch
        action[0][2] = 0.0; // Yaw rate
        action[0][3] = 0.5; // Throttle

        auto [next_obs, reward, done, _] = env->step(action);

        if (done) {
            obs = env->reset();
        } else {
            obs = next_obs;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void Agent::save_checkpoint(const std::string& path) {
    torch::save(policy_, path + ".policy");
    torch::save(value_, path + ".value");
    torch::save(*optimizer_, path + ".optim");

    spdlog::info("Saved checkpoint to {}", path);
}

void Agent::load_checkpoint(const std::string& path) {
    torch::load(policy_, path + ".policy", device_);
    torch::load(value_, path + ".value", device_);
    torch::load(*optimizer_, path + ".optim");

    spdlog::info("Loaded checkpoint from {}", path);
}

void Agent::export_model(const std::string& path) {
    utils::export_to_onnx(*policy_, torch::zeros({1, 3, 84, 84}, device_), path);
    spdlog::info("Exported model to {}", path);
}
