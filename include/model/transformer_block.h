#pragma once

#include <cstddef>

#include "model/layer.h"
#include "model/mlp.h"
#include "model/model_config.h"
#include "model/model_weights.h"
#include "model/rms_norm.h"
#include "model/self-attention.h"
#include "tensor.h"

namespace mini_llm::model {

class TransformerBlock final : public Layer {
public:
    explicit TransformerBlock(std::size_t layer_id, ModelConfig config);

    void set_weights(const ModelWeights::LayerWeights& weights);

    [[nodiscard]] Tensor forward(const Tensor& hidden_states,
                                   const Tensor* additive_mask = nullptr) const override;

    [[nodiscard]] std::size_t layer_id() const override;
    [[nodiscard]] const ModelConfig& config() const override;

private:
    [[nodiscard]] static SelfAttentionConfig make_attention_config(const ModelConfig& config);

private:
    std::size_t layer_id_{0};
    ModelConfig config_{};
    RMSNorm input_norm_{};
    RMSNorm post_attn_norm_{};
    SelfAttention attention_;
    MLP mlp_;
    ModelWeights::LayerWeights weights_{};
};

} // namespace mini_llm::model
