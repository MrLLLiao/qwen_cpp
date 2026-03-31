#include "cache/KVCache.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

bool KVCache::Tensor2DView::empty() const
{
    return tensor == nullptr || row_count == 0;
}

size_t KVCache::Tensor2DView::rows() const
{
    return row_count;
}

size_t KVCache::Tensor2DView::cols() const
{
    return tensor == nullptr ? 0 : tensor->cols();
}

const float& KVCache::Tensor2DView::at(size_t r, size_t c) const
{
    if (tensor == nullptr)
    {
        throw std::runtime_error("KVCache::Tensor2DView is empty");
    }
    if (r >= row_count)
    {
        throw std::out_of_range("KVCache::Tensor2DView row index out of range");
    }
    return tensor->at(row_offset + r, c);
}

const float& KVCache::Tensor2DView::operator()(size_t r, size_t c) const
{
    return at(r, c);
}

KVCache::KVCache() = default;

KVCache::KVCache(const Config& config)
{
    reset(config);
}

void KVCache::reset()
{
    config_ = {};
    keys_.clear();
    values_.clear();
    token_counts_.clear();
    initialized_ = false;
}

void KVCache::reset(const Config& config)
{
    config_ = config;
    keys_.assign(config_.num_layers, Tensor2D{});
    values_.assign(config_.num_layers, Tensor2D{});
    token_counts_.assign(config_.num_layers, 0);
    initialized_ = true;
}

bool KVCache::initialized() const
{
    return initialized_;
}

const KVCache::Config& KVCache::config() const
{
    return config_;
}

bool KVCache::has_layer(size_t layer_idx) const
{
    return initialized_ && layer_idx < keys_.size();
}

void KVCache::append(size_t layer_idx, const Tensor2D& key, const Tensor2D& value)
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::append layer index out of range");
    }

    if (key.rows() == 0 || key.cols() == 0 || value.rows() == 0 || value.cols() == 0)
    {
        throw std::invalid_argument("KVCache::append key/value must be non-empty");
    }
    if (key.rows() != value.rows() || key.cols() != value.cols())
    {
        throw std::invalid_argument("KVCache::append key/value shape mismatch");
    }

    const size_t expected_cols = config_.num_heads * config_.head_dim;
    if (key.cols() != expected_cols)
    {
        throw std::invalid_argument("KVCache::append invalid hidden size (cols)");
    }

    const size_t incoming_tokens = key.rows();
    const size_t current_tokens = token_counts_[layer_idx];

    if (current_tokens + incoming_tokens > config_.max_tokens)
    {
        throw std::runtime_error("KVCache::append exceed max_tokens");
    }

    const Tensor2D& old_k = keys_[layer_idx];
    const Tensor2D& old_v = values_[layer_idx];

    if (old_k.rows() == 0 && old_k.cols() == 0)
    {
        keys_[layer_idx] = key;
        values_[layer_idx] = value;
        token_counts_[layer_idx] = incoming_tokens;
        return;
    }

    if (old_k.cols() != key.cols() || old_v.cols() != value.cols())
    {
        throw std::invalid_argument("KVCache::append cached shape mismatch");
    }

    const size_t new_rows = old_k.rows() + incoming_tokens;
    const size_t cols = key.cols();

    Tensor2D merged_k(new_rows, cols);
    Tensor2D merged_v(new_rows, cols);

    for (size_t r = 0; r < old_k.rows(); ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            merged_k(r, c) = old_k(r, c);
            merged_v(r, c) = old_v(r, c);
        }
    }

    for (size_t r = 0; r < incoming_tokens; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            merged_k(old_k.rows() + r, c) = key(r, c);
            merged_v(old_v.rows() + r, c) = value(r, c);
        }
    }

    keys_[layer_idx] = merged_k;
    values_[layer_idx] = merged_v;
    token_counts_[layer_idx] = new_rows;
}

const Tensor2D& KVCache::key(size_t layer_idx) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::key layer index out of range");
    }
    return keys_[layer_idx];
}

const Tensor2D& KVCache::value(size_t layer_idx) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::value layer index out of range");
    }
    return values_[layer_idx];
}

KVCache::Tensor2DView KVCache::key_view(size_t layer_idx,
                                        size_t row_offset,
                                        size_t row_count) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::key_view layer index out of range");
    }

    const Tensor2D& tensor = keys_[layer_idx];
    if (row_offset > tensor.rows())
    {
        throw std::out_of_range("KVCache::key_view row offset out of range");
    }

    const size_t available = tensor.rows() - row_offset;
    const size_t actual_rows = (row_count == kAllRows) ? available : std::min(row_count, available);

    return Tensor2DView{&tensor, row_offset, actual_rows};
}

KVCache::Tensor2DView KVCache::value_view(size_t layer_idx,
                                          size_t row_offset,
                                          size_t row_count) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::value_view layer index out of range");
    }

    const Tensor2D& tensor = values_[layer_idx];
    if (row_offset > tensor.rows())
    {
        throw std::out_of_range("KVCache::value_view row offset out of range");
    }

    const size_t available = tensor.rows() - row_offset;
    const size_t actual_rows = (row_count == kAllRows) ? available : std::min(row_count, available);

    return Tensor2DView{&tensor, row_offset, actual_rows};
}

size_t KVCache::token_count(size_t layer_idx) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::token_count layer index out of range");
    }
    return token_counts_[layer_idx];
}

size_t KVCache::total_token_count() const
{
    return std::accumulate(token_counts_.begin(), token_counts_.end(), static_cast<size_t>(0));
}