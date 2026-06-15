//
// Created by killua on 2026/3/21.
//

#ifndef QWEN_CPP_OPS_ATTENTION_H
#define QWEN_CPP_OPS_ATTENTION_H

#include <cstddef>

#include "tensor.h"

/**
 * @brief Attention 算子配置（单头、二维 Tensor 版本）。
 */
struct AttentionConfig
{
    /**
     * @brief 是否启用缩放因子 1 / sqrt(d_k)。
     */
    bool enable_scaling{true};

    /**
     * @brief 手动指定缩放因子。
     *
     * 当值 > 0 时优先使用该值；否则使用默认缩放规则。
     */
    float manual_scale{-1.0f};

    /**
     * @brief softmax 数值稳定项。
     */
    float softmax_epsilon{1e-12f};

    /**
     * @brief 是否启用 causal mask（自回归掩码）。
     */
    bool causal{false};

    /**
     * @brief query 第0行对应的绝对 token 位置，用于带 KV cache 的增量 causal mask。
     */
    std::size_t query_position_offset{0};

    /**
     * @brief RoPE theta；<=0 表示关闭 RoPE。
     */
    float rope_theta{0.0f};

    /**
     * @brief RoPE 缩放因子，Qwen 长上下文扩展时可调。
     */
    float rope_scale{1.0f};
};

/**
 * @brief Scaled Dot-Product Attention（无状态算子）。
 *
 * 输入约束：
 * - query: [seq_q, d_k]
 * - key:   [seq_k, d_k]
 * - value: [seq_k, d_v]
 * - additive_mask（可选）: [seq_q, seq_k]
 *
 * mask 规则（固定契约）：
 * - additive_mask 逐元素加到 attention score（可用 0 / -inf 或大负数）。
 * - 当 causal=true 时，再额外屏蔽上三角（禁止看未来 token）。
 * - 若同时提供 additive_mask + causal，则两者叠加生效。
 *
 * 输出：
 * - out:   [seq_q, d_v]
 */
class Attention final
{
public:
    Attention() = default;
    explicit Attention(AttentionConfig config);

    /**
     * @brief 前向计算，返回新的输出 Tensor。
     */
    [[nodiscard]] Tensor forward(const Tensor& query,
                                   const Tensor& key,
                                   const Tensor& value,
                                   const Tensor* additive_mask = nullptr) const;

    /**
     * @brief 获取当前配置。
     */
    [[nodiscard]] const AttentionConfig& config() const;

private:
    AttentionConfig config_{};
};

/**
 * @brief 函数式 API：执行 scaled dot-product attention。
 */
[[nodiscard]] Tensor scaled_dot_product_attention(const Tensor& query,
                                                    const Tensor& key,
                                                    const Tensor& value,
                                                    const Tensor* additive_mask = nullptr,
                                                    AttentionConfig config = {});

[[nodiscard]] Tensor scaled_dot_product_attention_view(TensorConstView query,
                                                         TensorConstView key,
                                                         TensorConstView value,
                                                         const Tensor* additive_mask = nullptr,
                                                         AttentionConfig config = {});

/**
 * @brief Multi-head/GQA attention 输入布局为 [seq, heads * head_dim]。
 *
 * Qwen3.x 使用 GQA 时 num_query_heads 可能大于 num_kv_heads，此函数按
 * query_head / (num_query_heads / num_kv_heads) 映射到 KV head。
 */
[[nodiscard]] Tensor grouped_query_attention(const Tensor& query,
                                               const Tensor& key,
                                               const Tensor& value,
                                               std::size_t num_query_heads,
                                               std::size_t num_kv_heads,
                                               const Tensor* additive_mask = nullptr,
                                               AttentionConfig config = {});

#endif // QWEN_CPP_OPS_ATTENTION_H
