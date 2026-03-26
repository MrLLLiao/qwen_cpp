#include "ops/attention.h"

#include "ops/matmul.h"
#include "ops/softmax.h"

#include <cmath>
#include <limits>
#include <stdexcept>

Attention::Attention(AttentionConfig config) : config_(config) {}

Tensor2D Attention::forward(const Tensor2D& query,
                            const Tensor2D& key,
                            const Tensor2D& value,
                            const Tensor2D* additive_mask) const
{
    if (query.cols() != key.cols())
    {
        throw std::invalid_argument("Attention dimension mismatch: query.cols() must equal key.cols()");
    }

    if (key.rows() != value.rows())
    {
        throw std::invalid_argument("Attention dimension mismatch: key.rows() must equal value.rows()");
    }

    if (additive_mask != nullptr)
    {
        if (additive_mask->rows() != query.rows() || additive_mask->cols() != key.rows())
        {
            throw std::invalid_argument("Attention mask shape mismatch: mask must be [query.rows(), key.rows()]");
        }
    }

    if (config_.softmax_epsilon < 0.0f)
    {
        throw std::invalid_argument("Attention softmax_epsilon must be non-negative");
    }

    if (config_.manual_scale == 0.0f)
    {
        throw std::invalid_argument("Attention manual_scale must be > 0 when provided");
    }

    if (config_.causal && query.rows() != key.rows())
    {
        throw std::invalid_argument("Causal attention requires query.rows() == key.rows()");
    }

    const size_t seq_q = query.rows();
    const size_t seq_k = key.rows();
    const size_t d_k = key.cols();

    Tensor2D key_transpose = key;
    key_transpose.transpose();

    Tensor2D scores = matmul(query, key_transpose);

    float scale = 1.0f;
    if (config_.manual_scale > 0.0f)
    {
        scale = config_.manual_scale;
    }
    else if (config_.enable_scaling)
    {
        scale = 1.0f / static_cast<float>(std::sqrt(static_cast<double>(d_k)));
    }

    if (scale != 1.0f)
    {
        for (size_t r = 0; r < scores.rows(); ++r)
        {
            for (size_t c = 0; c < scores.cols(); ++c)
            {
                scores(r, c) *= scale;
            }
        }
    }

    if (additive_mask != nullptr)
    {
        for (size_t r = 0; r < scores.rows(); ++r)
        {
            for (size_t c = 0; c < scores.cols(); ++c)
            {
                scores(r, c) += (*additive_mask)(r, c);
            }
        }
    }

    if (config_.causal)
    {
        constexpr float neg_inf = -std::numeric_limits<float>::infinity();
        for (size_t r = 0; r < seq_q; ++r)
        {
            for (size_t c = r + 1; c < seq_k; ++c)
            {
                scores(r, c) = neg_inf;
            }
        }
    }

    const Tensor2D probs = softmax(scores, SoftmaxAxis::Row, config_.softmax_epsilon, 1.0f);
    return matmul(probs, value);
}

const AttentionConfig& Attention::config() const
{
    return config_;
}

Tensor2D scaled_dot_product_attention(const Tensor2D& query,
                                      const Tensor2D& key,
                                      const Tensor2D& value,
                                      const Tensor2D* additive_mask,
                                      AttentionConfig config)
{
    return Attention(config).forward(query, key, value, additive_mask);
}
