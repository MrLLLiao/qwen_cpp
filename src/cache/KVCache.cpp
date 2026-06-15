#include "cache/KVCache.h"

#include <algorithm>
#include <stdexcept>

bool KVCache::TensorView::empty() const
{
    return tensor == nullptr || row_count == 0;
}

size_t KVCache::TensorView::rows() const
{
    return row_count;
}

size_t KVCache::TensorView::cols() const
{
    return tensor == nullptr ? 0 : tensor->cols();
}

const float& KVCache::TensorView::at(size_t r, size_t c) const
{
    if (tensor == nullptr)
    {
        throw std::runtime_error("KVCache::TensorView is empty");
    }
    if (r >= row_count)
    {
        throw std::out_of_range("KVCache::TensorView row index out of range");
    }
    return tensor->at(row_offset + r, c);
}

const float& KVCache::TensorView::operator()(size_t r, size_t c) const
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
    reserved_tokens_.clear();
    initialized_ = false;
}

void KVCache::reset(const Config& config)
{
    if (config.num_layers == 0)
    {
        throw std::invalid_argument("KVCache::reset num_layers must be > 0");
    }
    if (config.num_heads == 0)
    {
        throw std::invalid_argument("KVCache::reset num_heads must be > 0");
    }
    if (config.head_dim == 0)
    {
        throw std::invalid_argument("KVCache::reset head_dim must be > 0");
    }
    if (config.max_tokens == 0)
    {
        throw std::invalid_argument("KVCache::reset max_tokens must be > 0");
    }

    config_ = config;
    keys_.assign(config_.num_layers, Tensor{});
    values_.assign(config_.num_layers, Tensor{});
    token_counts_.assign(config_.num_layers, 0);
    reserved_tokens_.assign(config_.num_layers, 0);
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

size_t KVCache::capacity() const
{
    return config_.max_tokens;
}

size_t KVCache::total_capacity() const
{
    return config_.max_tokens;
}

size_t KVCache::used_tokens(size_t layer_idx) const
{
    return token_count(layer_idx);
}

size_t KVCache::utilization(size_t layer_idx) const
{
    if (config_.max_tokens == 0)
    {
        return 0;
    }
    return (token_count(layer_idx) * 100U) / config_.max_tokens;
}

void KVCache::ensure_capacity_for(size_t layer_idx, size_t incoming_tokens)
{
    const size_t current_tokens = token_counts_[layer_idx];
    if (current_tokens + incoming_tokens > config_.max_tokens)
    {
        throw std::runtime_error("KVCache::append exceed max_tokens");
    }

    if (keys_[layer_idx].rows() == 0 && keys_[layer_idx].cols() == 0)
    {
        return;
    }

    const size_t required = current_tokens + incoming_tokens;
    const size_t current_capacity = keys_[layer_idx].capacity() / keys_[layer_idx].cols();
    if (required <= current_capacity)
    {
        reserved_tokens_[layer_idx] = current_capacity;
        return;
    }

    const size_t preferred = config_.reserve_tokens == 0 ? config_.max_tokens : config_.reserve_tokens;
    const size_t doubled = current_capacity == 0 ? preferred : current_capacity * 2U;
    const size_t new_capacity = std::min(std::max(required, doubled), config_.max_tokens);

    keys_[layer_idx].reserve(new_capacity * keys_[layer_idx].cols());
    values_[layer_idx].reserve(new_capacity * values_[layer_idx].cols());
    reserved_tokens_[layer_idx] = new_capacity;
}

void KVCache::append_in_place(size_t layer_idx, const Tensor& key, const Tensor& value)
{
    if (keys_[layer_idx].rows() == 0 && keys_[layer_idx].cols() == 0)
    {
        keys_[layer_idx] = key;
        values_[layer_idx] = value;
        const size_t preferred = config_.reserve_tokens == 0 ? config_.max_tokens : config_.reserve_tokens;
        const size_t reserve_tokens = std::min(std::max(preferred, key.rows()), config_.max_tokens);
        keys_[layer_idx].reserve(reserve_tokens * key.cols());
        values_[layer_idx].reserve(reserve_tokens * value.cols());
        reserved_tokens_[layer_idx] = keys_[layer_idx].capacity() / key.cols();
        token_counts_[layer_idx] = key.rows();
        return;
    }

    keys_[layer_idx].append_rows(key);
    values_[layer_idx].append_rows(value);
    reserved_tokens_[layer_idx] = keys_[layer_idx].capacity() / keys_[layer_idx].cols();
    token_counts_[layer_idx] = keys_[layer_idx].rows();
}

void KVCache::append(size_t layer_idx, const Tensor& key, const Tensor& value)
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

    const Tensor& old_k = keys_[layer_idx];
    const Tensor& old_v = values_[layer_idx];

    if (old_k.rows() != 0 && (old_k.cols() != key.cols() || old_v.cols() != value.cols()))
    {
        throw std::invalid_argument("KVCache::append cached shape mismatch");
    }

    ensure_capacity_for(layer_idx, incoming_tokens);
    append_in_place(layer_idx, key, value);
}

const Tensor& KVCache::key(size_t layer_idx) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::key layer index out of range");
    }
    return keys_[layer_idx];
}

const Tensor& KVCache::value(size_t layer_idx) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::value layer index out of range");
    }
    return values_[layer_idx];
}

KVCache::TensorView KVCache::key_view(size_t layer_idx,
                                      size_t row_offset,
                                      size_t row_count) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::key_view layer index out of range");
    }

    const Tensor& tensor = keys_[layer_idx];
    if (row_offset > tensor.rows())
    {
        throw std::out_of_range("KVCache::key_view row offset out of range");
    }

    const size_t available = tensor.rows() - row_offset;
    const size_t actual_rows = (row_count == kAllRows) ? available : std::min(row_count, available);

    return TensorView{&tensor, row_offset, actual_rows};
}

KVCache::TensorView KVCache::value_view(size_t layer_idx,
                                        size_t row_offset,
                                        size_t row_count) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::value_view layer index out of range");
    }

    const Tensor& tensor = values_[layer_idx];
    if (row_offset > tensor.rows())
    {
        throw std::out_of_range("KVCache::value_view row offset out of range");
    }

    const size_t available = tensor.rows() - row_offset;
    const size_t actual_rows = (row_count == kAllRows) ? available : std::min(row_count, available);

    return TensorView{&tensor, row_offset, actual_rows};
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
    if (token_counts_.empty())
    {
        return 0;
    }

    const size_t expected = token_counts_.front();
    for (const size_t count : token_counts_)
    {
        if (count != expected)
        {
            throw std::runtime_error("KVCache::total_token_count inconsistent per-layer token counts");
        }
    }

    return expected;
}

KVCache::TensorView KVCache::key_slot_view(size_t layer_idx, size_t row_index) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::key_slot_view layer index out of range");
    }
    if (row_index >= token_counts_[layer_idx])
    {
        throw std::out_of_range("KVCache::key_slot_view row index out of range");
    }
    return TensorView{&keys_[layer_idx], row_index, 1};
}

KVCache::TensorView KVCache::value_slot_view(size_t layer_idx, size_t row_index) const
{
    if (!has_layer(layer_idx))
    {
        throw std::out_of_range("KVCache::value_slot_view layer index out of range");
    }
    if (row_index >= token_counts_[layer_idx])
    {
        throw std::out_of_range("KVCache::value_slot_view row index out of range");
    }
    return TensorView{&values_[layer_idx], row_index, 1};
}
