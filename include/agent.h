#pragma once

#include "models.h"
#include "environment.h"
#include "ppo.h"
#include "metrics.h"

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <yaml-cpp/yaml.h>

struct RolloutBuffer {
    std::vector<torch::Tensor> observations;
    std::vector<torch::Tensor> actions;
    std::vector<torch::Tensor> rewards;
    std::vector<torch::Tensor> values;
    std::vector<torch::Tensor> dones;
    std::vector<torch::Tensor> log_probs;

    void clear();
    void add(const torch::Tensor& obs, const torch::Tensor& action, float reward,
             const torch::Tensor& value, bool done, const torch::Tensor& log_prob);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    get_tensors(const torch::Device& device);
};

class EnvironmentWorker {
public:
    EnvironmentWorker(const std::string& config_path, int worker_id);
    ~EnvironmentWorker();

    void start();
    void stop();
    RolloutBuffer collect_rollout(
        std::shared_ptr<PolicyNetwork> policy,
        std::shared_ptr<ValueNetwork> value,
        int steps,
        bool training
    );

private:
    void worker_loop();

    std::unique_ptr<Environment> env_;
    int worker_id_;
    std::string config_path_;

    std::thread worker_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool running_;
    bool collect_requested_;
    bool training_mode_;
    int steps_to_collect_;

    std::shared_ptr<PolicyNetwork> policy_;
    std::shared_ptr<ValueNetwork> value_;
    RolloutBuffer rollout_buffer_;
};

class Agent {
public:
    Agent(const std::string& config_path);

    void train(int epochs);
    void evaluate(int episodes, bool record = false);
    void play_interactive();

    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    void export_model(const std::string& path);

private:
    void create_tensorboard_writer();

    std::string config_path_;
    YAML::Node config_;

    std::shared_ptr<PolicyNetwork> policy_;
    std::shared_ptr<ValueNetwork> value_;
    std::shared_ptr<PPO> ppo_;
    std::shared_ptr<Metrics> metrics_;

    std::vector<std::unique_ptr<EnvironmentWorker>> workers_;

    int num_workers_;
    int steps_per_epoch_;
    int epochs_;
    int batch_size_;
    float gamma_;
    float lambda_;

    torch::Device device_;
    std::unique_ptr<torch::optim::Adam> optimizer_;

    c10::optional<torch::TensorboardWriter> tb_writer_;
};
