//
// Created by killua on 2026/3/26.
//

#ifndef QWEN_CPP_CACHEALLOCATOR_H
#define QWEN_CPP_CACHEALLOCATOR_H

#include <cstddef>
#include <vector>

#include "tensor.h"

/**
 * @class CacheAllocator
 * @brief 高效的KV缓存内存分配管理器
 * 
 * 该类负责管理Transformer模型中KV缓存的内存分配和释放，
 * 采用池化分配策略提高性能和内存利用率。支持单个和批量分配。
 */
class CacheAllocator final
{
public:
    /**
     * @struct AllocationSpec
     * @brief 缓存分配规格说明
     */
    struct AllocationSpec
    {
        size_t rows{0};      ///< 缓存张量的行数（通常为序列长度）
        size_t cols{0};      ///< 缓存张量的列数（通常为隐藏维度）
        size_t count{0};     ///< 需要分配的缓存张量总数
    };

public:
    /**
     * @brief 默认构造函数
     * 初始化分配器，设置最大缓冲区数为0
     */
    CacheAllocator();

    /**
     * @brief 带参构造函数
     * @param max_buffers 最大缓冲区数量
     */
    explicit CacheAllocator(size_t max_buffers);

    /**
     * @brief 配置分配器的最大缓冲区容量
     * @param max_buffers 最大缓冲区数量
     * @note 该操作会重置分配器状态
     */
    void configure(size_t max_buffers);

    /**
     * @brief 重置分配器
     * 将所有缓冲区标记为可用，清空已分配计数
     */
    void reset();

    /**
     * @brief 获取最大缓冲区数量
     * @return 配置的最大缓冲区数
     */
    [[nodiscard]] size_t max_buffers() const;

    /**
     * @brief 获取已分配的缓冲区数
     * @return 当前已使用的缓冲区数
     */
    [[nodiscard]] size_t used_buffers() const;

    /**
     * @brief 获取可用的缓冲区数
     * @return 剩余可分配的缓冲区数
     */
    [[nodiscard]] size_t free_buffers() const;

    /**
     * @brief 分配单个缓存张量
     * @param spec 分配规格说明（行数、列数、数量）
     * @return 分配的2D张量
     * @throws std::bad_alloc 当超过最大缓冲区限制时抛出异常
     */
    [[nodiscard]] Tensor2D allocate(const AllocationSpec& spec);

    /**
     * @brief 批量分配缓存张量
     * @param spec 分配规格说明，其中count字段指定需要分配的张量数量
     * @return 分配的2D张量向量
     * @throws std::bad_alloc 当超过最大缓冲区限制时抛出异常
     */
    [[nodiscard]] std::vector<Tensor2D> allocate_batch(const AllocationSpec& spec);

    /**
     * @brief 释放缓存张量
     * @param buffer 待释放的张量，将被标记为可用
     */
    void release(const Tensor2D& buffer);

private:
    size_t max_buffers_{0};     ///< 最大缓冲区容量限制
    size_t used_buffers_{0};    ///< 当前已使用的缓冲区数
};

#endif //QWEN_CPP_CACHEALLOCATOR_H