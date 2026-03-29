//
// Created by killua on 2026/3/26.
//

#ifndef QWEN_CPP_KVCACHE_H
#define QWEN_CPP_KVCACHE_H

#include <cstddef>
#include <vector>

#include "tensor.h"

/**
 * @class KVCache
 * @brief Transformer多头注意力机制的Key-Value缓存
 * 
 * 该类实现了Transformer推理过程中的KV缓存存储与管理。
 * 支持多层多头结构，每层独立维护Key和Value张量，
 * 并记录每层已缓存的token数量。用于优化推理性能，
 * 避免重复计算历史序列的注意力。
 */
class KVCache final
{
public:
    /**
     * @struct Config
     * @brief KV缓存配置结构体
     */
    struct Config
    {
        size_t num_layers{0};    ///< Transformer层数
        size_t num_heads{0};     ///< 多头注意力的头数
        size_t head_dim{0};      ///< 每个注意力头的维度
        size_t max_tokens{0};    ///< 最大支持的缓存token数
    };

public:
    /**
     * @brief 默认构造函数
     * 初始化未初始化状态的KV缓存
     */
    KVCache();

    /**
     * @brief 带参构造函数
     * @param config KV缓存配置
     * @note 构造函数会根据配置自动分配内存空间
     */
    explicit KVCache(const Config& config);

    /**
     * @brief 重置缓存为未初始化状态
     * 清空所有Key、Value张量和token计数
     */
    void reset();

    /**
     * @brief 重置并重新配置缓存
     * @param config 新的缓存配置
     * @note 该操作会根据新配置重新分配内存空间
     */
    void reset(const Config& config);

    /**
     * @brief 检查缓存是否已初始化
     * @return 若已初始化返回true，否则返回false
     */
    [[nodiscard]] bool initialized() const;

    /**
     * @brief 获取缓存配置
     * @return 配置的常引用
     */
    [[nodiscard]] const Config& config() const;

    /**
     * @brief 检查指定层是否存在有效数据
     * @param layer_idx 层索引（从0开始）
     * @return 若该层存在Key/Value张量返回true，否则返回false
     * @throws std::out_of_range 当layer_idx超出范围时抛出异常
     */
    [[nodiscard]] bool has_layer(size_t layer_idx) const;

    /**
     * @brief 向指定层追加Key和Value张量
     * 
     * 该方法用于在推理过程中逐步增长KV缓存。在prefill阶段追加初始token，
     * 在decode阶段追加每个新生成的token的KV值。
     * 
     * @param layer_idx 目标层索引（从0开始）
     * @param key 该层新的Key张量
     * @param value 该层新的Value张量
     * @throws std::out_of_range 当layer_idx超出范围时抛出异常
     * @throws std::runtime_error 当追加后超过max_tokens限制时抛出异常
     */
    void append(size_t layer_idx, const Tensor2D& key, const Tensor2D& value);

    /**
     * @brief 获取指定层的Key张量
     * @param layer_idx 层索引（从0开始）
     * @return Key张量的常引用
     * @throws std::out_of_range 当layer_idx超出范围时抛出异常
     */
    [[nodiscard]] const Tensor2D& key(size_t layer_idx) const;

    /**
     * @brief 获取指定层的Value张量
     * @param layer_idx 层索引（从0开始）
     * @return Value张量的常引用
     * @throws std::out_of_range 当layer_idx超出范围时抛出异常
     */
    [[nodiscard]] const Tensor2D& value(size_t layer_idx) const;

    /**
     * @brief 获取指定层已缓存的token数量
     * @param layer_idx 层索引（从0开始）
     * @return 该层已缓存的token总数
     * @throws std::out_of_range 当layer_idx超出范围时抛出异常
     */
    [[nodiscard]] size_t token_count(size_t layer_idx) const;

    /**
     * @brief 获取所有层已缓存的平均token数量
     * 
     * 在正常情况下，所有层的token数应该相同。
     * 该方法返回总的token数或第一层的token数。
     * 
     * @return 缓存中的token总数
     */
    [[nodiscard]] size_t total_token_count() const;

private:
    Config config_{};                          ///< 缓存配置参数
    std::vector<Tensor2D> keys_{};             ///< 各层Key张量存储（size = num_layers）
    std::vector<Tensor2D> values_{};           ///< 各层Value张量存储（size = num_layers）
    std::vector<size_t> token_counts_{};       ///< 各层已缓存的token数量（size = num_layers）
    bool initialized_{false};                  ///< 初始化标记
};

#endif //QWEN_CPP_KVCACHE_H