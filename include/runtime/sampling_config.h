#pragma once

namespace mini_llm::runtime {

struct SamplingConfig {
    // TODO: 增加 seed、top_k、top_p、min_p、repeat_penalty 等采样参数
    int top_k = 40;
    float top_p = 0.9f;
    float repeat_penalty = 1.1f;
};

} // namespace mini_llm::runtime
