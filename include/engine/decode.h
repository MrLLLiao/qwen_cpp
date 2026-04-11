//
// Created by killua on 2026/4/7.
//

#ifndef QWEN_CPP_DECODE_H
#define QWEN_CPP_DECODE_H

#include <cstddef>
#include <string>
#include <vector>

#include "tensor.h"

class Tensor2D;
class KVCache;
class CacheManager;

/**
 * @file decode.h
 * @brief Decode 阶段接口定义。
 *
 * decode 负责在已有上下文缓存的基础上，
 * 将“当前步新产生的单 token KV”追加到 KVCache，
 * 并推进会话的缓存长度。
 */

/**
 * @brief decode 阶段的单层 KV 输入描述。
 *
 * 语义约束（建议实现遵守）：
 * - layer_idx 必须是有效层索引（[0, num_layers)）。
 * - key/value 必须同时为空或同时非空。
 * - key/value 非空时 shape 必须一致。
 * - decode 单步通常要求 key.rows() == value.rows() == 1。
 */
struct DecodeLayerKV
{
    size_t layer_idx{0};            ///< 目标层索引
    const Tensor2D* key{nullptr};   ///< 本层当前步 Key 张量（通常为单 token）
    const Tensor2D* value{nullptr}; ///< 本层当前步 Value 张量（通常为单 token）
};

/**
 * @brief 一次 decode 调用的请求体。
 *
 * 使用说明：
 * - cache_id 指向已存在的会话缓存。
 * - layer_kv 通常覆盖所有层；若部分层为空，具体行为由实现决定。
 * - 当前骨架聚焦“KV 增量写入”；后续可扩展 logits、采样结果、step metadata 等字段。
 */
struct DecodeRequest
{
    std::string cache_id{};               ///< 目标缓存 ID
    std::vector<DecodeLayerKV> layer_kv{}; ///< 当前步待追加的多层 KV 数据
};

/**
 * @brief decode 执行结果。
 */
struct DecodeResult
{
    size_t appended_tokens{0}; ///< 本次成功追加的 token 数（单步 decode 通常为 1）
    size_t total_tokens{0};    ///< 追加完成后缓存中的总 token 数
};

/**
 * @brief 将请求中的多层单步 KV 逐层追加到指定缓存。
 *
 * @param cache 目标 KVCache。
 * @param layer_kv 多层 KV 输入。
 *
 * @throws std::invalid_argument 当 key/value 配对、shape 或单步 token 数不合法时抛出。
 * @throws std::out_of_range 当 layer_idx 非法时抛出。
 * @throws std::runtime_error 当缓存容量不足等运行时错误发生时抛出。
 */
void append_decode_kv(KVCache& cache, const std::vector<DecodeLayerKV>& layer_kv);

/**
 * @brief decode 阶段流程编排器。
 *
 * 负责：
 * 1) 校验请求与缓存状态；
 * 2) 调用 append 动作将当前步 KV 写入缓存；
 * 3) 返回本次追加量与追加后的总 token 数。
 */
class DecodeEngine final
{
public:
    /**
     * @brief 构造 decode 引擎。
     * @param cache_manager 缓存管理器（由调用方保证生命周期）。
     */
    explicit DecodeEngine(CacheManager& cache_manager);

    /**
     * @brief 执行一次 decode。
     *
     * @param request decode 请求。
     * @return DecodeResult 执行结果。
     *
     * @note 当请求为空或目标 cache 不存在时，通常返回 appended_tokens=0。
     */
    [[nodiscard]] DecodeResult run(const DecodeRequest& request) const;

private:
    CacheManager* cache_manager_{nullptr}; ///< 非拥有指针，生命周期由外部管理
};

#endif //QWEN_CPP_DECODE_H
