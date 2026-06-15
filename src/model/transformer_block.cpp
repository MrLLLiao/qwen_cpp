#include "model/transformer_block.h"

#include <stdexcept>

namespace
{
[[nodiscard]] Tensor add_tensors(const Tensor& lhs, const Tensor& rhs)
{
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols())
    {
        throw std::invalid_argument("add_tensors shape mismatch");
    }

    Tensor out(lhs.rows(), lhs.cols(), 0.0f);
    for (std::size_t r = 0; r < lhs.rows(); ++r)
    {
        const float* a = lhs.row_data(r);
        const float* b = rhs.row_data(r);
        float* o = out.row_data(r);
        for (std::size_t c = 0; c < lhs.cols(); ++c)
        {
            o[c] = a[c] + b[c];
        }
    }
    return out;
}
} // namespace

namespace mini_llm::model {

TransformerBlock::TransformerBlock(std::size_t layer_id, ModelConfig config)
    : layer_id_(layer_id),
      config_(std::move(config)),
      input_norm_(RMSNormConfig{config_.rms_norm_eps}),
      post_attn_norm_(RMSNormConfig{config_.rms_norm_eps}),
      attention_(make_attention_config(config_)),
      mlp_(MLPConfig{config_.hidden_size, config_.intermediate_size})
{
    if (!config_.valid())
    {
        throw std::invalid_argument("TransformerBlock invalid model config");
    }
}

void TransformerBlock::set_weights(const ModelWeights::LayerWeights& weights)
{
    attention_.set_projection_weights(weights.attention_wq,
                                      weights.attention_wk,
                                      weights.attention_wv,
                                      weights.attention_wo);
    mlp_.set_weights(weights.mlp_gate, weights.mlp_up, weights.mlp_down);
    weights_ = weights;
}

Tensor TransformerBlock::forward(const Tensor& hidden_states,
                                   const Tensor* additive_mask) const
{
    if (hidden_states.cols() != config_.hidden_size)
    {
        throw std::invalid_argument("TransformerBlock::forward hidden size mismatch");
    }

    const Tensor normed_attn = input_norm_.forward(hidden_states, weights_.rms_attn_weight);
    const Tensor attn_out = attention_.forward(normed_attn, additive_mask);
    const Tensor after_attn = add_tensors(hidden_states, attn_out);
    const Tensor normed_ffn = post_attn_norm_.forward(after_attn, weights_.rms_ffn_weight);
    const Tensor ffn_out = mlp_.forward(normed_ffn);
    return add_tensors(after_attn, ffn_out);
}

std::size_t TransformerBlock::layer_id() const
{
    return layer_id_;
}

const ModelConfig& TransformerBlock::config() const
{
    return config_;
}

SelfAttentionConfig TransformerBlock::make_attention_config(const ModelConfig& config)
{
    return SelfAttentionConfig{
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        true,
        0,
        config.rope_theta,
        config.rope_scale
    };
}

} // namespace mini_llm::model
