#include <gtest/gtest.h>
#include "ppo.h"
#include "models.h"
#include "metrics.h"

class AdvantageTest : public ::testing::Test {
protected:
    void SetUp() override {
        policy = std::make_shared<PolicyNetwork>(std::vector<int64_t>{3, 84, 84}, 4);
        value = std::make_shared<ValueNetwork>(std::vector<int64_t>{3, 84, 84});
        metrics = std::make_shared<Metrics>("localhost:9091");
        optimizer = std::make_unique<torch::optim::Adam>(
            policy->parameters(), torch::optim::AdamOptions(3e-4));
        ppo = std::make_shared<PPO>(policy, value, *optimizer, metrics);
    }

    std::shared_ptr<PolicyNetwork> policy;
    std::shared_ptr<ValueNetwork> value;
    std::shared_ptr<Metrics> metrics;
    std::unique_ptr<torch::optim::Adam> optimizer;
    std::shared_ptr<PPO> ppo;
};

TEST_F(AdvantageTest, ComputeGAE) {
    // Create test data
    auto rewards = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto values = torch::tensor({0.5f, 1.5f, 2.5f, 3.5f, 4.5f});
    auto dones = torch::zeros_like(rewards);
    dones[2] = 1.0f; // Episode terminates after step 2

    float gamma = 0.99f;
    float lambda = 0.95f;

    auto [returns, advantages] = ppo->compute_gae(rewards, values, dones, gamma, lambda);

    // Verify shape
    EXPECT_EQ(returns.sizes(), rewards.sizes());
    EXPECT_EQ(advantages.sizes(), rewards.sizes());

    // Manually calculate expected returns and advantages for verification
    std::vector<float> expected_advantages = {
        0.595f, 0.5f, 0.5f, 0.5f, 0.5f
    };

    std::vector<float> expected_returns = {
        1.095f, 2.0f, 3.0f, 4.0f, 5.0f
    };

    for (int i = 0; i < rewards.size(0); ++i) {
        EXPECT_NEAR(advantages[i].item<float>(), expected_advantages[i], 1e-2f);
        EXPECT_NEAR(returns[i].item<float>(), expected_returns[i], 1e-2f);
    }
}

TEST_F(AdvantageTest, ZeroAdvantages) {
    // Edge case: all rewards and values are zero
    auto rewards = torch::zeros({5});
    auto values = torch::zeros({5});
    auto dones = torch::zeros_like(rewards);

    float gamma = 0.99f;
    float lambda = 0.95f;

    auto [returns, advantages] = ppo->compute_gae(rewards, values, dones, gamma, lambda);

    // All advantages should be zero
    for (int i = 0; i < rewards.size(0); ++i) {
        EXPECT_NEAR(advantages[i].item<float>(), 0.0f, 1e-6f);
        EXPECT_NEAR(returns[i].item<float>(), 0.0f, 1e-6f);
    }
}
