#pragma once

#include <string>
#include <vector>

namespace mini_llm::model {

struct ModelConfig {
    // TODO: 补全模型结构参数（层数、隐藏维度、头数、上下文长度等）
    std::string model_family;
    int n_layers = 0;
    int hidden_size = 0;
    int n_heads = 0;
    int n_ctx = 0;
};

} // namespace mini_llm::model
