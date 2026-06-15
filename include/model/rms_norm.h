//
// Created by killua on 2026/4/14.
//

#ifndef QWEN_CPP_RMS_NORM_H
#define QWEN_CPP_RMS_NORM_H

#include <cstddef>

#include "tensor.h"

namespace mini_llm::model {

struct RMSNormConfig {
    float epsilon{1e-6f};
};

class RMSNorm final {
public:
    explicit RMSNorm(RMSNormConfig config = {});

    [[nodiscard]] Tensor forward(const Tensor& input, const Tensor& weight) const;
    [[nodiscard]] const RMSNormConfig& config() const;

private:
    RMSNormConfig config_{};
};

[[nodiscard]] Tensor rms_norm(const Tensor& input,
                                const Tensor& weight,
                                float epsilon = 1e-6f);

} // namespace mini_llm::model

#endif //QWEN_CPP_RMS_NORM_H
