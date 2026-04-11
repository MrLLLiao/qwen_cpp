#include "engine/prefill.h"

#include <cassert>
#include <stdexcept>
#include <unordered_set>

#include "cache/CacheManager.h"
#include "cache/KVCache.h"

namespace
{
[[nodiscard]] size_t infer_appended_tokens(const std::vector<PrefillLayerKV>& layer_kv)
{
    size_t appended_tokens = 0;
    bool has_valid_layer = false;
    std::unordered_set<size_t> seen_layers{};

    for (const auto& [layer_idx, key, value] : layer_kv)
    {
        if ((key == nullptr) != (value == nullptr))
        {
            throw std::invalid_argument("append_prefill_kv key/value must be both null or both non-null");
        }

        if (key == nullptr)
        {
            continue;
        }

        if (!seen_layers.insert(layer_idx).second)
        {
            throw std::invalid_argument("append_prefill_kv duplicate layer index in request");
        }

        if (key->rows() == 0 || key->cols() == 0 || value->rows() == 0 || value->cols() == 0)
        {
            throw std::invalid_argument("append_prefill_kv key/value must be non-empty");
        }

        if (key->rows() != value->rows() || key->cols() != value->cols())
        {
            throw std::invalid_argument("append_prefill_kv key/value shape mismatch");
        }

        if (!has_valid_layer)
        {
            appended_tokens = key->rows();
            has_valid_layer = true;
        }
        else if (key->rows() != appended_tokens)
        {
            throw std::invalid_argument("append_prefill_kv inconsistent token count across layers");
        }
    }

    return has_valid_layer ? appended_tokens : 0;
}

void validate_prefill_against_cache(const KVCache& cache,
                                    const std::vector<PrefillLayerKV>& layer_kv,
                                    const size_t appended_tokens)
{
    const auto& config = cache.config();
    const size_t expected_cols = config.num_heads * config.head_dim;

    for (const auto& [layer_idx, key, value] : layer_kv)
    {
        if (key == nullptr && value == nullptr)
        {
            continue;
        }

        if (!cache.has_layer(layer_idx))
        {
            throw std::out_of_range("PrefillEngine::run layer index out of range");
        }

        assert(key != nullptr);
        if (key->cols() != expected_cols || value->cols() != expected_cols)
        {
            throw std::invalid_argument("PrefillEngine::run invalid hidden size for cache config");
        }

        const size_t current_tokens = cache.token_count(layer_idx);
        if (current_tokens + appended_tokens > config.max_tokens)
        {
            throw std::runtime_error("PrefillEngine::run exceed cache max_tokens");
        }
    }
}
} // namespace

void append_prefill_kv(KVCache& cache, const std::vector<PrefillLayerKV>& layer_kv)
{
    for (const auto& [layer_idx, key, value] : layer_kv)
    {
        if (key == nullptr && value == nullptr)
        {
            continue;
        }

        if (key == nullptr || value == nullptr)
        {
            throw std::invalid_argument("append_prefill_kv key/value must be both null or both non-null");
        }

        cache.append(layer_idx, *key, *value);
    }
}

PrefillEngine::PrefillEngine(CacheManager& cache_manager) : cache_manager_(&cache_manager)
{
}

PrefillResult PrefillEngine::run(const PrefillRequest& request) const
{
    PrefillResult result{};

    if (cache_manager_ == nullptr || request.layer_kv.empty())
    {
        return result;
    }

    if (request.cache_id.empty() || !cache_manager_->has_cache(request.cache_id))
    {
        return result;
    }

    KVCache& cache = cache_manager_->cache(request.cache_id);

    const size_t appended_tokens = infer_appended_tokens(request.layer_kv);
    if (appended_tokens == 0)
    {
        return result;
    }

    validate_prefill_against_cache(cache, request.layer_kv, appended_tokens);
    append_prefill_kv(cache, request.layer_kv);

    result.appended_tokens = appended_tokens;
    return result;
}
