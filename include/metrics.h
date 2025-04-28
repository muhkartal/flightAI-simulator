#pragma once

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <memory>
#include <string>

class Metrics {
public:
    Metrics(const std::string& bind_address = "0.0.0.0:9091");
    ~Metrics();

    void record_episode_reward(float reward);
    void record_policy_loss(float loss);
    void record_value_loss(float loss);
    void record_entropy(float entropy);
    void record_advantage(float advantage);
    void record_steps_per_second(float steps_per_second);
    void record_training_step();
    void record_episode();

    std::shared_ptr<prometheus::Registry> registry() { return registry_; }

private:
    std::unique_ptr<prometheus::Exposer> exposer_;
    std::shared_ptr<prometheus::Registry> registry_;

    prometheus::Family<prometheus::Gauge>& episode_reward_family_;
    prometheus::Family<prometheus::Gauge>& policy_loss_family_;
    prometheus::Family<prometheus::Gauge>& value_loss_family_;
    prometheus::Family<prometheus::Gauge>& entropy_family_;
    prometheus::Family<prometheus::Gauge>& advantage_family_;
    prometheus::Family<prometheus::Gauge>& steps_per_second_family_;
    prometheus::Family<prometheus::Counter>& training_steps_family_;
    prometheus::Family<prometheus::Counter>& episodes_family_;

    prometheus::Gauge& episode_reward_;
    prometheus::Gauge& policy_loss_;
    prometheus::Gauge& value_loss_;
    prometheus::Gauge& entropy_;
    prometheus::Gauge& advantage_;
    prometheus::Gauge& steps_per_second_;
    prometheus::Counter& training_steps_;
    prometheus::Counter& episodes_;
};
