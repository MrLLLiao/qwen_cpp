//
// Created by killua on 2026/3/26.
//

#ifndef QWEN_CPP_CACHEMANAGER_H
#define QWEN_CPP_CACHEMANAGER_H

#include <cstddef>
#include <string>
#include <unordered_map>

#include "cache/CacheAllocator.h"
#include "cache/KVCache.h"

/**
 * @class CacheManager
 * @brief Transformer模型KV缓存统一管理器
 * 
 * 该类提供了对多个KV缓存实例的生命周期管理、内存分配、
 * 以及缓存配置的集中管理。通过CacheId标识不同的缓存实例，
 * 支持创建、删除和查询缓存，并管理底层内存分配器。
 */
class CacheManager final
{
public:
    using CacheId = std::string;  ///< 缓存唯一标识符类型

    /**
     * @struct ManagerConfig
     * @brief 缓存管理器配置结构体
     */
    struct ManagerConfig
    {
        KVCache::Config kv_config{};           ///< KV缓存的配置参数
        size_t max_active_caches{0};           ///< 最多同时活跃的缓存实例数
        size_t allocator_max_buffers{0};       ///< 内存分配器的最大缓冲区数
    };

public:
    /**
     * @brief 默认构造函数
     * 初始化管理器，配置为空状态
     */
    CacheManager();

    /**
     * @brief 带参构造函数
     * @param config 缓存管理器配置
     */
    explicit CacheManager(const ManagerConfig& config);

    /**
     * @brief 配置管理器
     * @param config 新的管理器配置
     * @note 该操作会清空所有现有的缓存实例
     */
    void configure(const ManagerConfig& config);

    /**
     * @brief 清空所有缓存
     * 删除所有活跃的缓存实例，重置管理器状态
     */
    void clear();

    /**
     * @brief 检查指定ID的缓存是否存在
     * @param cache_id 缓存唯一标识符
     * @return 若缓存存在返回true，否则返回false
     */
    [[nodiscard]] bool has_cache(const CacheId& cache_id) const;

    /**
     * @brief 获取当前活跃的缓存实例数
     * @return 活跃缓存实例总数
     */
    [[nodiscard]] size_t active_cache_count() const;

    /**
     * @brief 创建新的KV缓存实例
     * @param cache_id 新缓存的唯一标识符
     * @return 新创建的KVCache引用
     * @throws std::logic_error 当管理器尚未 configure 时抛出异常
     * @throws std::invalid_argument 当ID已存在时抛出异常
     * @throws std::runtime_error 当超过max_active_caches限制时抛出异常
     */
    KVCache& create_cache(const CacheId& cache_id);

    /**
     * @brief 移除指定ID的缓存实例
     * @param cache_id 待移除的缓存唯一标识符
     * @throws std::out_of_range 当缓存不存在时抛出异常
     */
    void remove_cache(const CacheId& cache_id);

    /**
     * @brief 获取指定ID的缓存实例（可修改）
     * @param cache_id 缓存唯一标识符
     * @return 对应的KVCache引用
     * @throws std::out_of_range 当缓存不存在时抛出异常
     */
    KVCache& cache(const CacheId& cache_id);

    /**
     * @brief 获取指定ID的缓存实例（只读）
     * @param cache_id 缓存唯一标识符
     * @return 对应的KVCache常引用
     * @throws std::out_of_range 当缓存不存在时抛出异常
     */
    [[nodiscard]] const KVCache& cache(const CacheId& cache_id) const;

    /**
     * @brief 获取当前配置
     * @return 管理器配置的常引用
     */
    [[nodiscard]] const ManagerConfig& config() const;

    /**
     * @brief 获取底层内存分配器（可修改）
     * @return 分配器的引用
     */
    [[nodiscard]] CacheAllocator& allocator();

    /**
     * @brief 获取底层内存分配器（只读）
     * @return 分配器的常引用
     */
    [[nodiscard]] const CacheAllocator& allocator() const;

private:
    ManagerConfig config_{};                          ///< 管理器配置
    CacheAllocator allocator_{};                      ///< 内存分配器实例
    std::unordered_map<CacheId, KVCache> caches_{};  ///< 缓存实例映射表
};

#endif //QWEN_CPP_CACHEMANAGER_H