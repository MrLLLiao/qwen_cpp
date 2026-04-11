#pragma once

#include <cstdint>

namespace mini_llm::runtime {

struct GenerationConfig {
    // TODO: 增加 stop words、streaming、max context truncate 策略
    std::int32_t max_new_tokens = 256;
    float temperature = 0.7f;
};

} // namespace mini_llm::runtime
