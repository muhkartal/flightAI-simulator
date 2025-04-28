#include "utils.h"
#include <spdlog/spdlog.h>
#include <random>
#include <chrono>

namespace utils {

std::mt19937& get_generator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

void set_seed(int seed) {
    get_generator().seed(seed);
    torch::manual_seed(seed);
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed_all(seed);
    }
}

torch::Tensor to_tensor(const std::vector<float>& data, const torch::TensorOptions& options) {
    return torch::tensor(data, options);
}

std::vector<float> to_vector(const torch::Tensor& tensor) {
    auto tensor_cpu = tensor.to(torch::kCPU).contiguous();
    std::vector<float> vec(tensor_cpu.data_ptr<float>(),
                         tensor_cpu.data_ptr<float>() + tensor_cpu.numel());
    return vec;
}

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
generate_mini_batches(
    const torch::Tensor& observations,
    const torch::Tensor& actions,
    const torch::Tensor& log_probs,
    const torch::Tensor& returns,
    const torch::Tensor& values,
    const torch::Tensor& advantages,
    int mini_batch_size
) {
    int batch_size = observations.size(0);
    int num_mini_batches = batch_size / mini_batch_size;

    // Generate random permutation
    auto indices = torch::randperm(batch_size, observations.options().dtype(torch::kLong));

    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> mini_batches;

    for (int i = 0; i < num_mini_batches; ++i) {
        auto start = i * mini_batch_size;
        auto end = (i + 1) * mini_batch_size;
        auto idx = indices.slice(0, start, end);

        mini_batches.push_back(std::make_tuple(
            observations.index_select(0, idx),
            actions.index_select(0, idx),
            log_probs.index_select(0, idx),
            returns.index_select(0, idx),
            values.index_select(0, idx),
            advantages.index_select(0, idx)
        ));
    }

    return mini_batches;
}

void export_to_onnx(torch::nn::Module& module, const torch::Tensor& dummy_input, const std::string& filename) {
    module.eval();

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(dummy_input);

    torch::jit::trace_module(module, inputs).save(filename);

    spdlog::info("Exported model to {}", filename);
}

int64_t current_time_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
}

float calculate_steps_per_second(int64_t start_time_ns, int64_t end_time_ns, int num_steps) {
    float duration_sec = (end_time_ns - start_time_ns) / 1e9f;
    return num_steps / duration_sec;
}

torch::Tensor preprocess_image(const std::vector<uint8_t>& image_data, int width, int height, const std::string& format) {
    if (image_data.empty()) {
        return torch::zeros({3, height, width});
    }

    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    torch::Tensor image = torch::from_blob(
        const_cast<uint8_t*>(image_data.data()),
        {static_cast<long>(image_data.size())},
        options
    ).clone();

    if (format == "rgb") {
        image = image.reshape({height, width, 3});
        // Convert HWC to CHW
        image = image.permute({2, 0, 1});
    } else if (format == "depth") {
        image = image.reshape({height, width, 1});
        // Convert HW1 to 1HW
        image = image.permute({2, 0, 1});
    }

    // Normalize to [0, 1]
    image = image.to(torch::kFloat32) / 255.0f;

    return image;
}

} // namespace utils
