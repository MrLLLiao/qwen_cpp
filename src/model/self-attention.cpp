#include "model/self-attention.h"

#include "ops/matmul.h"

#include <stdexcept>

namespace {

[[nodiscard]] Tensor2D make_identity_matrix(std::size_t dim) {
    Tensor2D identity(dim, dim, 0.0f);
    for (std::size_t i = 0; i < dim; ++i) {
        identity(i, i) = 1.0f;
    }
    return identity;
}

[[nodiscard]] Tensor2D slice_head_columns(const Tensor2D& x,
                                          std::size_t head_index,
                                          std::size_t head_dim) {
    const std::size_t seq_len = x.rows();
    const std::size_t col_offset = head_index * head_dim;

    Tensor2D head_tensor(seq_len, head_dim);
    for (std::size_t r = 0; r < seq_len; ++r) {
        for (std::size_t c = 0; c < head_dim; ++c) {
            head_tensor(r, c) = x(r, col_offset + c);
        }
    }
    return head_tensor;
}

void merge_head_columns(const Tensor2D& head_context,
                        std::size_t head_index,
                        Tensor2D& merged_context) {
    const std::size_t seq_len = merged_context.rows();
    const std::size_t head_dim = head_context.cols();
    const std::size_t col_offset = head_index * head_dim;

    for (std::size_t r = 0; r < seq_len; ++r) {
        for (std::size_t c = 0; c < head_dim; ++c) {
            merged_context(r, col_offset + c) = head_context(r, c);
        }
    }
}

} // namespace

namespace mini_llm::model {

SelfAttention::SelfAttention(const SelfAttentionConfig& config)
    : config_(config) {
    if (!is_valid_config(config_)) {
        throw std::invalid_argument(
            "SelfAttention: invalid config, require hidden_size>0, num_heads>0 and hidden_size % num_heads == 0");
    }

    attention_config_.causal = config_.causal;

    wq_ = make_identity_matrix(config_.hidden_size);
    wk_ = make_identity_matrix(config_.hidden_size);
    wv_ = make_identity_matrix(config_.hidden_size);
    wo_ = make_identity_matrix(config_.hidden_size);
}

Tensor2D SelfAttention::forward(const Tensor2D& hidden_states,
                                const Tensor2D* additive_mask) const {
    if (hidden_states.cols() != config_.hidden_size) {
        throw std::invalid_argument("SelfAttention::forward: hidden_states.cols() must equal config.hidden_size");
    }

    const std::size_t seq_len = hidden_states.rows();
    if (additive_mask != nullptr) {
        if (additive_mask->rows() != seq_len || additive_mask->cols() != seq_len) {
            throw std::invalid_argument(
                "SelfAttention::forward: additive_mask must be [seq_len, seq_len]");
        }
    }

    const Tensor2D query = project_query(hidden_states);
    const Tensor2D key = project_key(hidden_states);
    const Tensor2D value = project_value(hidden_states);

    if (query.rows() != seq_len || key.rows() != seq_len || value.rows() != seq_len ||
        query.cols() != config_.hidden_size || key.cols() != config_.hidden_size ||
        value.cols() != config_.hidden_size) {
        throw std::invalid_argument(
            "SelfAttention::forward: Q/K/V projection shape must be [seq_len, hidden_size]");
    }

    const std::size_t head_dim = config_.hidden_size / config_.num_heads;
    Attention attention(attention_config_);

    Tensor2D merged_context(seq_len, config_.hidden_size);
    for (std::size_t head_index = 0; head_index < config_.num_heads; ++head_index) {
        const Tensor2D q_head = slice_head_columns(query, head_index, head_dim);
        const Tensor2D k_head = slice_head_columns(key, head_index, head_dim);
        const Tensor2D v_head = slice_head_columns(value, head_index, head_dim);

        const Tensor2D head_context = attention.forward(q_head, k_head, v_head, additive_mask);
        merge_head_columns(head_context, head_index, merged_context);
    }

    return project_output(merged_context);
}

const SelfAttentionConfig& SelfAttention::config() const {
    return config_;
}

void SelfAttention::set_projection_weights(const Tensor2D& wq,
                                           const Tensor2D& wk,
                                           const Tensor2D& wv,
                                           const Tensor2D& wo) {
    if (!is_valid_projection_weight_shape(wq) || !is_valid_projection_weight_shape(wk) ||
        !is_valid_projection_weight_shape(wv) || !is_valid_projection_weight_shape(wo)) {
        throw std::invalid_argument(
            "SelfAttention::set_projection_weights: each weight must be [hidden_size, hidden_size]");
    }

    wq_ = wq;
    wk_ = wk;
    wv_ = wv;
    wo_ = wo;
}

bool SelfAttention::is_valid_config(const SelfAttentionConfig& config) {
    if (config.hidden_size == 0 || config.num_heads == 0) {
        return false;
    }
    return (config.hidden_size % config.num_heads) == 0;
}

bool SelfAttention::is_valid_projection_weight_shape(const Tensor2D& weight) const {
    return weight.rows() == config_.hidden_size && weight.cols() == config_.hidden_size;
}

Tensor2D SelfAttention::project_query(const Tensor2D& hidden_states) const {
    if (hidden_states.cols() != config_.hidden_size) {
        throw std::invalid_argument("SelfAttention::project_query: hidden_states.cols() must equal config.hidden_size");
    }

    return matmul(hidden_states, wq_);
}

Tensor2D SelfAttention::project_key(const Tensor2D& hidden_states) const {
    if (hidden_states.cols() != config_.hidden_size) {
        throw std::invalid_argument("SelfAttention::project_key: hidden_states.cols() must equal config.hidden_size");
    }

    return matmul(hidden_states, wk_);
}

Tensor2D SelfAttention::project_value(const Tensor2D& hidden_states) const {
    if (hidden_states.cols() != config_.hidden_size) {
        throw std::invalid_argument("SelfAttention::project_value: hidden_states.cols() must equal config.hidden_size");
    }

    return matmul(hidden_states, wv_);
}

Tensor2D SelfAttention::project_output(const Tensor2D& context) const {
    if (context.cols() != config_.hidden_size) {
        throw std::invalid_argument("SelfAttention::project_output: context.cols() must equal config.hidden_size");
    }

    return matmul(context, wo_);
}

} // namespace mini_llm::model
