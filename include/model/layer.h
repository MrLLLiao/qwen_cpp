#pragma once

#include <cstddef>

#include "model/model_config.h"
#include "tensor.h"

namespace mini_llm::model {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& hidden_states,
                             const Tensor* additive_mask = nullptr) const = 0;
    [[nodiscard]] virtual std::size_t layer_id() const = 0;
    [[nodiscard]] virtual const ModelConfig& config() const = 0;
};

} // namespace mini_llm::model
