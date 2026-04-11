#pragma once

#include <string>

#include "runtime/generation_config.h"
#include "runtime/sampling_config.h"

namespace mini_llm::runtime {

class ModelRunner {
public:
    // TODO: 提供基于 manifest 的模型加载入口
    bool load_model(const std::string& manifest_path);

    // TODO: 实现最小推理闭环（prompt -> text）
    std::string generate(const std::string& prompt,
                         const GenerationConfig& generation,
                         const SamplingConfig& sampling);
};

} // namespace mini_llm::runtime
