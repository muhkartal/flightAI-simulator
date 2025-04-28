#include "metrics.h"
#include <spdlog/spdlog.h>

Metrics::Metrics(const std::string& bind_address)
    : exposer_(std::make_unique<prometheus::Exposer>(bind_address)),
      registry_(std::make_shared<prometheus::Registry>()),
      episode_reward_family_(prometheus::BuildGauge()
          .Name("rldronesim_episode_reward")
          .Help("Episode reward")
          .Register(*registry_)),
      policy_loss_family_(prometheus::BuildGauge()
          .Name("rldronesim_policy_loss")
          .Help("Policy loss")
          .Register(*registry_)),
      value_loss_family_(prometheus::BuildGauge()
          .Name("rldronesim_value_loss")
          .Help("Value loss")
          .Register(*registry_)),
      entropy_family_(prometheus::BuildGauge()
          .Name("rldronesim_entropy")
          .Help("Policy entropy")
          .Register(*registry_)),
      advantage_family_(prometheus::BuildGauge()
          .Name("rldronesim_advantage")
          .Help("Advantage")
          .Register(*registry_)),
      steps_per_second_family_(prometheus::BuildGauge()
          .Name("rldronesim_steps_per_second")
          .Help("Steps per second")
          .Register(*registry_)),
      training_steps_family_(prometheus::BuildCounter()
          .Name("rldronesim_training_steps_total")
          .Help("Total number of training steps")
          .Register(*registry_)),
      episodes_family_(prometheus::BuildCounter()
          .Name("rldronesim_episodes_total")
          .Help("Total number of episodes")
          .Register(*registry_)),
      episode_reward_(episode_reward_family_.Add({})),
      policy_loss_(policy_loss_family_.Add({})),
      value_loss_(value_loss_family_.Add({})),
      entropy_(entropy_family_.Add({})),
      advantage_(advantage_family_.Add({})),
      steps_per_second_(steps_per_second_family_.Add({})),
      training_steps_(training_steps_family_.Add({})),
      episodes_(episodes_family_.Add({})) {

    exposer_->RegisterCollectable(registry_);
    spdlog::info("Metrics server started on {}", bind_address);
}

Metrics::~Metrics() {
    spdlog::info("Metrics server stopped");
}

void Metrics::record_episode_reward(float reward) {
    episode_reward_.Set(reward);
}

void Metrics::record_policy_loss(float loss) {
    policy_loss_.Set(loss);
}

void Metrics::record_value_loss(float loss) {
    value_loss_.Set(loss);
}

void Metrics::record_entropy(float entropy) {
    entropy_.Set(entropy);
}

void Metrics::record_advantage(float advantage) {
    advantage_.Set(advantage);
}

void Metrics::record_steps_per_second(float steps_per_second) {
    steps_per_second_.Set(steps_per_second);
}

void Metrics::record_training_step() {
    training_steps_.Increment();
}

void Metrics::record_episode() {
    episodes_.Increment();
}
