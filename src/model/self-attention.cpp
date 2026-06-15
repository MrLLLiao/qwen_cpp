#include "model/self-attention.h"

#include "ops/attention.h"
#include "ops/matmul.h"

#include <stdexcept>

namespace
{
[[nodiscard]] Tensor make_identity_matrix(std::size_t dim)
{
    Tensor identity(dim, dim, 0.0f);
    for (std::size_t i = 0; i < dim; ++i)
    {
        identity(i, i) = 1.0f;
    }
    return identity;
}
} // namespace

namespace mini_llm::model {

SelfAttention::SelfAttention(const SelfAttentionConfig& config)
    : config_(config)
{
    if (!is_valid_config(config_))
    {
        throw std::invalid_argument("SelfAttention: invalid config");
    }

    attention_config_.causal = config_.causal;
    attention_config_.query_position_offset = config_.query_position_offset;
    attention_config_.rope_theta = config_.rope_theta;
    attention_config_.rope_scale = config_.rope_scale;

    wq_ = make_identity_matrix(config_.hidden_size);
    wk_ = Tensor(config_.hidden_size, config_.hidden_size * config_.num_key_value_heads / config_.num_heads, 0.0f);
    wv_ = Tensor(config_.hidden_size, config_.hidden_size * config_.num_key_value_heads / config_.num_heads, 0.0f);
    wo_ = make_identity_matrix(config_.hidden_size);
}

Tensor SelfAttention::forward(const Tensor& hidden_states,
                                const Tensor* additive_mask) const
{
    if (hidden_states.cols() != config_.hidden_size)
    {
        throw std::invalid_argument("SelfAttention::forward hidden_states.cols mismatch");
    }
    if (additive_mask != nullptr && (additive_mask->rows() != hidden_states.rows() || additive_mask->cols() != hidden_states.rows()))
    {
        throw std::invalid_argument("SelfAttention::forward additive_mask must be [seq_len, seq_len]");
    }

    const Tensor query = project_query(hidden_states);
    const Tensor key = project_key(hidden_states);
    const Tensor value = project_value(hidden_states);

    const Tensor context = grouped_query_attention(query,
                                                     key,
                                                     value,
                                                     config_.num_heads,
                                                     config_.num_key_value_heads,
                                                     additive_mask,
                                                     attention_config_);
    return project_output(context);
}

const SelfAttentionConfig& SelfAttention::config() const
{
    return config_;
}

void SelfAttention::set_projection_weights(const Tensor& wq,
                                           const Tensor& wk,
                                           const Tensor& wv,
                                           const Tensor& wo)
{
    if (!is_valid_projection_weight_shape(wq) || !is_valid_kv_projection_weight_shape(wk) ||
        !is_valid_kv_projection_weight_shape(wv) || !is_valid_projection_weight_shape(wo))
    {
        throw std::invalid_argument("SelfAttention::set_projection_weights invalid shape");
    }

    wq_ = wq;
    wk_ = wk;
    wv_ = wv;
    wo_ = wo;
}

bool SelfAttention::is_valid_config(const SelfAttentionConfig& config)
{
    if (config.hidden_size == 0 || config.num_heads == 0 || config.num_key_value_heads == 0)
    {
        return false;
    }
    if (config.hidden_size % config.num_heads != 0)
    {
        return false;
    }
    if (config.num_heads % config.num_key_value_heads != 0)
    {
        return false;
    }
    return config.rope_scale > 0.0f;
}

bool SelfAttention::is_valid_projection_weight_shape(const Tensor& weight) const
{
    return weight.rows() == config_.hidden_size && weight.cols() == config_.hidden_size;
}

bool SelfAttention::is_valid_kv_projection_weight_shape(const Tensor& weight) const
{
    return weight.rows() == config_.hidden_size
        && weight.cols() == config_.hidden_size * config_.num_key_value_heads / config_.num_heads;
}

Tensor SelfAttention::project_query(const Tensor& hidden_states) const
{
    return matmul(hidden_states, wq_);
}

Tensor SelfAttention::project_key(const Tensor& hidden_states) const
{
    return matmul(hidden_states, wk_);
}

Tensor SelfAttention::project_value(const Tensor& hidden_states) const
{
    return matmul(hidden_states, wv_);
}

Tensor SelfAttention::project_output(const Tensor& context) const
{
    return matmul(context, wo_);
}

} // namespace mini_llm::model
