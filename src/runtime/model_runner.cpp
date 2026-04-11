#include "runtime/model_runner.h"

namespace mini_llm::runtime {

bool ModelRunner::load_model(const std::string& manifest_path) {
    // TODO: 校验 manifest 并调用 backend 加载模型
    (void)manifest_path;
    return false;
}

std::string ModelRunner::generate(const std::string& prompt,
                                  const GenerationConfig& generation,
                                  const SamplingConfig& sampling) {
    // TODO: 串联 tokenizer/prefill/decode/sampling 生成文本
    (void)prompt;
    (void)generation;
    (void)sampling;
    return {};
}

} // namespace mini_llm::runtime
