//
// Created by killua on 2026/4/7.
//

#ifndef QWEN_CPP_PREFILL_H
#define QWEN_CPP_PREFILL_H

#include <cstddef>
#include <string>
#include <vector>

#include "tensor.h"

class Tensor2D;
class KVCache;
class CacheManager;

/**
 * @file prefill.h
 * @brief Prefill 阶段接口定义。
 *
 * prefill 负责将 prompt 对应的多层 Key/Value 张量一次性写入 KVCache，
 * 为后续 decode 阶段提供上下文。
 */

/**
 * @brief prefill 阶段的单层 KV 输入描述。
 *
 * 语义约束：
 * - layer_idx 必须是有效层索引（[0, num_layers)）。
 * - key/value 必须同时为空或同时非空。
 * - key/value 非空时，shape 必须一致。
 * - key.rows() 表示该层本次追加的 token 数。
 */
struct PrefillLayerKV
{
    size_t layer_idx{0};          ///< 目标层索引
    const Tensor2D* key{nullptr}; ///< 本层 Key 张量（可为空）
    const Tensor2D* value{nullptr}; ///< 本层 Value 张量（可为空）
};

/**
 * @brief 一次 prefill 调用的请求体。
 *
 * 使用说明：
 * - cache_id 指向已存在的会话缓存。
 * - layer_kv 通常覆盖所有层；若部分层为空，具体行为由实现决定。
 */
struct PrefillRequest
{
    std::string cache_id{};                ///< 目标缓存 ID
    std::vector<PrefillLayerKV> layer_kv{}; ///< 待写入的多层 KV 数据
};

/**
 * @brief prefill 执行结果。
 */
struct PrefillResult
{
    size_t appended_tokens{0}; ///< 本次成功追加的 token 数
};

/**
 * @brief 将请求中的多层 KV 逐层追加到指定缓存。
 *
 * @param cache 目标 KVCache。
 * @param layer_kv 多层 KV 输入。
 *
 * @throws std::invalid_argument 当 key/value 配对或 shape 不合法时抛出。
 * @throws std::out_of_range 当 layer_idx 非法时抛出。
 * @throws std::runtime_error 当缓存容量不足等运行时错误发生时抛出。
 */
void append_prefill_kv(KVCache& cache, const std::vector<PrefillLayerKV>& layer_kv);

/**
 * @brief prefill 阶段流程编排器。
 *
 * 负责：
 * 1) 校验请求与缓存状态；
 * 2) 调用 append 动作将 KV 写入缓存；
 * 3) 返回追加 token 统计。
 */
class PrefillEngine final
{
public:
    /**
     * @brief 构造 prefill 引擎。
     * @param cache_manager 缓存管理器（由调用方保证生命周期）。
     */
    explicit PrefillEngine(CacheManager& cache_manager);

    /**
     * @brief 执行一次 prefill。
     *
     * @param request prefill 请求。
     * @return PrefillResult 执行结果（含 appended_tokens）。
     *
     * @note 当请求为空或目标 cache 不存在时，通常返回 appended_tokens=0。
     */
    [[nodiscard]] PrefillResult run(const PrefillRequest& request) const;

private:
    CacheManager* cache_manager_{nullptr}; ///< 非拥有指针，生命周期由外部管理
};

#endif //QWEN_CPP_PREFILL_H

