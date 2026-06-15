#pragma once

#include <cstddef>
#include <string>

namespace mini_llm::model {

struct ModelConfig {
    std::string model_family{"qwen3"};
    std::size_t vocab_size{0};
    std::size_t num_hidden_layers{0};
    std::size_t hidden_size{0};
    std::size_t intermediate_size{0};
    std::size_t num_attention_heads{0};
    std::size_t num_key_value_heads{0};
    std::size_t max_position_embeddings{0};
    float rms_norm_eps{1e-6f};
    float rope_theta{1000000.0f};
    float rope_scale{1.0f};
    bool tie_word_embeddings{false};

    [[nodiscard]] bool valid() const;
    [[nodiscard]] std::size_t head_dim() const;
    [[nodiscard]] std::size_t kv_hidden_size() const;
};

} // namespace mini_llm::model
