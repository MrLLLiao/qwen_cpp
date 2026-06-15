#include "ops/attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
[[nodiscard]] float resolve_scale(const TensorConstView& key, const AttentionConfig& config)
{
    if (config.manual_scale == 0.0f)
    {
        throw std::invalid_argument("Attention manual_scale must be > 0 when provided");
    }

    if (config.manual_scale > 0.0f)
    {
        return config.manual_scale;
    }

    if (!config.enable_scaling)
    {
        return 1.0f;
    }

    return 1.0f / static_cast<float>(std::sqrt(static_cast<double>(key.cols)));
}

void validate_attention_inputs(const TensorConstView& query,
                               const TensorConstView& key,
                               const TensorConstView& value,
                               const Tensor* additive_mask,
                               const AttentionConfig& config)
{
    if (query.cols != key.cols)
    {
        throw std::invalid_argument("Attention dimension mismatch: query.cols() must equal key.cols()");
    }

    if (key.rows != value.rows)
    {
        throw std::invalid_argument("Attention dimension mismatch: key.rows() must equal value.rows()");
    }

    if (query.empty() || key.empty() || value.empty())
    {
        throw std::invalid_argument("Attention input tensors must be non-empty");
    }

    if (additive_mask != nullptr)
    {
        if (additive_mask->rows() != query.rows || additive_mask->cols() != key.rows)
        {
            throw std::invalid_argument("Attention mask shape mismatch: mask must be [query.rows(), key.rows()]");
        }
    }

    if (config.softmax_epsilon < 0.0f)
    {
        throw std::invalid_argument("Attention softmax_epsilon must be non-negative");
    }

    if (config.rope_scale <= 0.0f)
    {
        throw std::invalid_argument("Attention rope_scale must be greater than 0");
    }

    if (config.causal && query.rows != key.rows && config.query_position_offset == 0)
    {
        throw std::invalid_argument("Causal attention with seq_q != seq_k requires query_position_offset > 0");
    }
}

[[nodiscard]] float rotated_value(const float* row,
                                  size_t dim_index,
                                  size_t cols,
                                  size_t position,
                                  float rope_theta,
                                  float rope_scale)
{
    if (rope_theta <= 0.0f || cols < 2)
    {
        return row[dim_index];
    }

    const size_t pair_base = dim_index & ~static_cast<size_t>(1);
    if (pair_base + 1 >= cols)
    {
        return row[dim_index];
    }

    const double pair_index = static_cast<double>(pair_base / 2);
    const double half_dim = static_cast<double>(cols / 2);
    const double freq = 1.0 / std::pow(static_cast<double>(rope_theta), pair_index / half_dim);
    const double angle = (static_cast<double>(position) / static_cast<double>(rope_scale)) * freq;
    const float cs = static_cast<float>(std::cos(angle));
    const float sn = static_cast<float>(std::sin(angle));

    const float x0 = row[pair_base];
    const float x1 = row[pair_base + 1];
    if ((dim_index & 1U) == 0U)
    {
        return x0 * cs - x1 * sn;
    }
    return x0 * sn + x1 * cs;
}

[[nodiscard]] float dot_with_optional_rope(const float* lhs,
                                           const float* rhs,
                                           size_t cols,
                                           size_t query_position,
                                           size_t key_position,
                                           const AttentionConfig& config)
{
    float sum = 0.0f;
    for (size_t c = 0; c < cols; ++c)
    {
        const float q = rotated_value(lhs, c, cols, query_position, config.rope_theta, config.rope_scale);
        const float k = rotated_value(rhs, c, cols, key_position, config.rope_theta, config.rope_scale);
        sum += q * k;
    }
    return sum;
}

void compute_attention_row(const TensorConstView& query,
                           const TensorConstView& key,
                           const TensorConstView& value,
                           const Tensor* additive_mask,
                           const AttentionConfig& config,
                           size_t q_row,
                           float scale,
                           std::vector<float>& scores,
                           Tensor& output)
{
    const size_t absolute_query_pos = config.query_position_offset + q_row;
    const float* q = query.row_data(q_row);

    float row_max = -std::numeric_limits<float>::infinity();
    for (size_t k_row = 0; k_row < key.rows; ++k_row)
    {
        float score = dot_with_optional_rope(q,
                                             key.row_data(k_row),
                                             query.cols,
                                             absolute_query_pos,
                                             k_row,
                                             config) * scale;

        if (additive_mask != nullptr)
        {
            score += (*additive_mask)(q_row, k_row);
        }

        if (config.causal && k_row > absolute_query_pos)
        {
            score = -std::numeric_limits<float>::infinity();
        }

        scores[k_row] = score;
        row_max = std::max(row_max, score);
    }

    double sum_exp = 0.0;
    for (size_t k_row = 0; k_row < key.rows; ++k_row)
    {
        const double exp_value = std::isinf(scores[k_row]) && scores[k_row] < 0.0f
                                     ? 0.0
                                     : std::exp(static_cast<double>(scores[k_row] - row_max));
        scores[k_row] = static_cast<float>(exp_value);
        sum_exp += exp_value;
    }

    sum_exp += static_cast<double>(config.softmax_epsilon);

    float* out_row = output.row_data(q_row);
    std::fill(out_row, out_row + value.cols, 0.0f);

    if (sum_exp == 0.0)
    {
        return;
    }

    for (size_t k_row = 0; k_row < key.rows; ++k_row)
    {
        const float weight = static_cast<float>(static_cast<double>(scores[k_row]) / sum_exp);
        const float* v = value.row_data(k_row);
        for (size_t c = 0; c < value.cols; ++c)
        {
            out_row[c] += weight * v[c];
        }
    }
}
} // namespace

Attention::Attention(AttentionConfig config) : config_(config) {}

Tensor Attention::forward(const Tensor& query,
                            const Tensor& key,
                            const Tensor& value,
                            const Tensor* additive_mask) const
{
    return scaled_dot_product_attention_view(make_tensor_view(query),
                                             make_tensor_view(key),
                                             make_tensor_view(value),
                                             additive_mask,
                                             config_);
}

const AttentionConfig& Attention::config() const
{
    return config_;
}

Tensor scaled_dot_product_attention(const Tensor& query,
                                      const Tensor& key,
                                      const Tensor& value,
                                      const Tensor* additive_mask,
                                      AttentionConfig config)
{
    return scaled_dot_product_attention_view(make_tensor_view(query),
                                             make_tensor_view(key),
                                             make_tensor_view(value),
                                             additive_mask,
                                             config);
}

Tensor scaled_dot_product_attention_view(TensorConstView query,
                                           TensorConstView key,
                                           TensorConstView value,
                                           const Tensor* additive_mask,
                                           AttentionConfig config)
{
    validate_attention_inputs(query, key, value, additive_mask, config);

    Tensor output(query.rows, value.cols, 0.0f);
    std::vector<float> scores(key.rows, 0.0f);
    const float scale = resolve_scale(key, config);

    for (size_t q_row = 0; q_row < query.rows; ++q_row)
    {
        compute_attention_row(query, key, value, additive_mask, config, q_row, scale, scores, output);
    }

    return output;
}

Tensor grouped_query_attention(const Tensor& query,
                                 const Tensor& key,
                                 const Tensor& value,
                                 size_t num_query_heads,
                                 size_t num_kv_heads,
                                 const Tensor* additive_mask,
                                 AttentionConfig config)
{
    if (num_query_heads == 0 || num_kv_heads == 0)
    {
        throw std::invalid_argument("grouped_query_attention heads must be > 0");
    }
    if ((num_query_heads % num_kv_heads) != 0)
    {
        throw std::invalid_argument("grouped_query_attention num_query_heads must be divisible by num_kv_heads");
    }
    if ((query.cols() % num_query_heads) != 0)
    {
        throw std::invalid_argument("grouped_query_attention query cols must be divisible by num_query_heads");
    }

    const size_t head_dim = query.cols() / num_query_heads;
    if (key.cols() != num_kv_heads * head_dim || value.cols() != num_kv_heads * head_dim)
    {
        throw std::invalid_argument("grouped_query_attention key/value cols must be num_kv_heads * head_dim");
    }

    const size_t group_size = num_query_heads / num_kv_heads;
    Tensor output(query.rows(), query.cols(), 0.0f);

    for (size_t q_head = 0; q_head < num_query_heads; ++q_head)
    {
        const size_t kv_head = q_head / group_size;
        const TensorConstView q_view = make_tensor_column_view(query, q_head * head_dim, head_dim);
        const TensorConstView k_view = make_tensor_column_view(key, kv_head * head_dim, head_dim);
        const TensorConstView v_view = make_tensor_column_view(value, kv_head * head_dim, head_dim);
        const Tensor head_out = scaled_dot_product_attention_view(q_view, k_view, v_view, additive_mask, config);

        for (size_t r = 0; r < output.rows(); ++r)
        {
            float* out_row = output.row_data(r);
            const float* head_row = head_out.row_data(r);
            std::copy(head_row, head_row + head_dim, out_row + q_head * head_dim);
        }
    }

    return output;
}
