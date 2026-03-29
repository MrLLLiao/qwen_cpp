#include "cache/CacheManager.h"

#include <stdexcept>

CacheManager::CacheManager() = default;

CacheManager::CacheManager(const ManagerConfig& config)
{
    configure(config);
}

void CacheManager::configure(const ManagerConfig& config)
{
    // TODO: 增加配置合法性校验与热更新策略。
    config_ = config;
    allocator_.configure(config_.allocator_max_buffers);
}

void CacheManager::clear()
{
    // TODO: 考虑并发访问场景下的安全清理。
    caches_.clear();
    allocator_.reset();
}

bool CacheManager::has_cache(const CacheId& cache_id) const
{
    return caches_.find(cache_id) != caches_.end();
}

size_t CacheManager::active_cache_count() const
{
    return caches_.size();
}

KVCache& CacheManager::create_cache(const CacheId& cache_id)
{
    if (has_cache(cache_id))
    {
        throw std::invalid_argument("CacheManager::create_cache cache already exists");
    }

    if (config_.max_active_caches > 0 && caches_.size() >= config_.max_active_caches)
    {
        throw std::runtime_error("CacheManager::create_cache reached max active caches");
    }

    auto [it, inserted] = caches_.emplace(cache_id, KVCache(config_.kv_config));
    if (!inserted)
    {
        throw std::runtime_error("CacheManager::create_cache failed to insert cache");
    }
    return it->second;
}

void CacheManager::remove_cache(const CacheId& cache_id)
{
    const auto erased = caches_.erase(cache_id);
    if (erased == 0)
    {
        throw std::out_of_range("CacheManager::remove_cache cache id not found");
    }
    // TODO: 与 allocator 联动释放该 cache 占用资源。
}

KVCache& CacheManager::cache(const CacheId& cache_id)
{
    auto it = caches_.find(cache_id);
    if (it == caches_.end())
    {
        throw std::out_of_range("CacheManager::cache cache id not found");
    }
    return it->second;
}

const KVCache& CacheManager::cache(const CacheId& cache_id) const
{
    auto it = caches_.find(cache_id);
    if (it == caches_.end())
    {
        throw std::out_of_range("CacheManager::cache cache id not found");
    }
    return it->second;
}

const CacheManager::ManagerConfig& CacheManager::config() const
{
    return config_;
}
CacheAllocator& CacheManager::allocator()
{
    return allocator_;
}

const CacheAllocator& CacheManager::allocator() const
{
    return allocator_;
}
