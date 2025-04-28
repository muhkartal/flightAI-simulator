#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <random>

namespace utils {

// Random number generation
std::mt19937& get_generator();
void set_seed(int seed);

// Tensor utilities
torch::Tensor to_tensor(const std::vector<float>& data, const torch::TensorOptions& options = torch::TensorOptions());
std::vector<float> to_vector(const torch::Tensor& tensor);

// Mini-batch generation
std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
generate_mini_batches(
    const torch::Tensor& observations,
    const torch::Tensor& actions,
    const torch::Tensor& log_probs,
    const torch::Tensor& returns,
    const torch::Tensor& values,
    const torch::Tensor& advantages,
    int mini_batch_size
);

// ONNX export utilities
void export_to_onnx(torch::nn::Module& module, const torch::Tensor& dummy_input, const std::string& filename);

// Time utilities
int64_t current_time_ns();
float calculate_steps_per_second(int64_t start_time_ns, int64_t end_time_ns, int num_steps);

// Image processing
torch::Tensor preprocess_image(const std::vector<uint8_t>& image_data, int width, int height, const std::string& format);

} // namespace utils
