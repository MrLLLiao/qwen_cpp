#pragma once

#include <cstddef>

#include "ops/attention.h"
#include "tensor.h"

namespace mini_llm::model {

/**
 * @brief Self-Attention 模块配置。
 */
struct SelfAttentionConfig {
    std::size_t hidden_size{0};
    std::size_t num_heads{0};
    bool causal{true};
};

/**
 * @brief Self-Attention 骨架。
 *
 * 当前阶段提供：
 * - 配置校验
 * - 多头前向接口
 * - Q/K/V/O 线性投影（支持注入权重）
 */
class SelfAttention {
public:
    explicit SelfAttention(const SelfAttentionConfig& config);

    /**
     * @brief 执行 self-attention 前向。
     *
     * @param hidden_states 输入张量（约定形状: [seq_len, hidden_size]）。
     * @param additive_mask 可选加性 mask（约定形状: [seq_len, seq_len]）。
     */
    [[nodiscard]] Tensor2D forward(const Tensor2D& hidden_states,
                                   const Tensor2D* additive_mask = nullptr) const;

    [[nodiscard]] const SelfAttentionConfig& config() const;

    /**
     * @brief 注入 Q/K/V/O 投影权重。
     *
     * 权重形状要求均为 [hidden_size, hidden_size]。
     */
    void set_projection_weights(const Tensor2D& wq,
                                const Tensor2D& wk,
                                const Tensor2D& wv,
                                const Tensor2D& wo);

private:
    [[nodiscard]] static bool is_valid_config(const SelfAttentionConfig& config);
    [[nodiscard]] bool is_valid_projection_weight_shape(const Tensor2D& weight) const;

    [[nodiscard]] Tensor2D project_query(const Tensor2D& hidden_states) const;
    [[nodiscard]] Tensor2D project_key(const Tensor2D& hidden_states) const;
    [[nodiscard]] Tensor2D project_value(const Tensor2D& hidden_states) const;
    [[nodiscard]] Tensor2D project_output(const Tensor2D& context) const;

private:
    SelfAttentionConfig config_{};
    AttentionConfig attention_config_{};

    Tensor2D wq_{};
    Tensor2D wk_{};
    Tensor2D wv_{};
    Tensor2D wo_{};
};

} // namespace mini_llm::model
