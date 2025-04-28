#include <gtest/gtest.h>
#include "ppo.h"
#include "models.h"
#include "metrics.h"

class PPOTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple environment for testing
        input_shape = {3, 84, 84};
        num_actions = 4;

        policy = std::make_shared<PolicyNetwork>(input_shape, num_actions);
        value = std::make_shared<ValueNetwork>(input_shape);
        metrics = std::make_shared<Metrics>("localhost:9092");

        std::vector<torch::optim::OptimizerParamGroup> param_groups;
        param_groups.push_back(torch::optim::OptimizerParamGroup(policy->parameters()));
        param_groups.push_back(torch::optim::OptimizerParamGroup(value->parameters()));
        optimizer = std::make_unique<torch::optim::Adam>(param_groups, torch::optim::AdamOptions(3e-4));

        ppo = std::make_shared<PPO>(
            policy, value, *optimizer, metrics,
            3e-4f, 0.2f, 0.5f, 0.01f, 0.5f, 2, 2
        );
    }

    std::vector<int64_t> input_shape;
    int64_t num_actions;
    std::shared_ptr<PolicyNetwork> policy;
    std::shared_ptr<ValueNetwork> value;
    std::shared_ptr<Metrics> metrics;
    std::unique_ptr<torch::optim::Adam> optimizer;
    std::shared_ptr<PPO> ppo;
};

TEST_F(PPOTest, UpdateStep) {
    int batch_size = 8;

    // Create dummy observations
    auto observations = torch::randn({batch_size, input_shape[0], input_shape[1], input_shape[2]});

    // Get actions and values for our dummy observations
    torch::NoGradGuard no_grad;
    auto actions_log_probs = policy->act(observations);
    auto actions = actions_log_probs.slice(1, 0, num_actions);
    auto log_probs = actions_log_probs.slice(1, num_actions, num_actions + 1);
    auto values = value->forward(observations);

    // Create dummy rewards and dones
    auto rewards = torch::randn({batch_size, 1});
    auto dones = torch::zeros({batch_size, 1});

    // Compute returns and advantages
    auto [returns, advantages] = ppo->compute_gae(rewards, values, dones, 0.99f, 0.95f);

    // Verify shapes
    EXPECT_EQ(returns.sizes()[0], batch_size);
    EXPECT_EQ(advantages.sizes()[0], batch_size);

    // Test the update step
    torch::InferenceMode guard(false);

    // Save initial parameters
    std::vector<torch::Tensor> initial_policy_params;
    for (const auto& param : policy->parameters()) {
        initial_policy_params.push_back(param.clone());
    }

    std::vector<torch::Tensor> initial_value_params;
    for (const auto& param : value->parameters()) {
        initial_value_params.push_back(param.clone());
    }

    // Perform an update
    ppo->update(observations, actions, log_probs, returns, values, advantages);

    // Check that parameters have changed
    bool policy_params_changed = false;
    for (size_t i = 0; i < initial_policy_params.size(); ++i) {
        if (!torch::allclose(initial_policy_params[i], policy->parameters()[i])) {
            policy_params_changed = true;
            break;
        }
    }
    EXPECT_TRUE(policy_params_changed);

    bool value_params_changed = false;
    for (size_t i = 0; i < initial_value_params.size(); ++i) {
        if (!torch::allclose(initial_value_params[i], value->parameters()[i])) {
            value_params_changed = true;
            break;
        }
    }
    EXPECT_TRUE(value_params_changed);
}
