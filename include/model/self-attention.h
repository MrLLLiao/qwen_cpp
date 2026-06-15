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
    std::size_t num_key_value_heads{0};
    bool causal{true};
    std::size_t query_position_offset{0};
    float rope_theta{1000000.0f};
    float rope_scale{1.0f};
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
    [[nodiscard]] Tensor forward(const Tensor& hidden_states,
                                   const Tensor* additive_mask = nullptr) const;

    [[nodiscard]] const SelfAttentionConfig& config() const;

    /**
     * @brief 注入 Q/K/V/O 投影权重。
     *
     * 权重形状要求均为 [hidden_size, hidden_size]。
     */
    void set_projection_weights(const Tensor& wq,
                                const Tensor& wk,
                                const Tensor& wv,
                                const Tensor& wo);

private:
    [[nodiscard]] static bool is_valid_config(const SelfAttentionConfig& config);
    [[nodiscard]] bool is_valid_projection_weight_shape(const Tensor& weight) const;
    [[nodiscard]] bool is_valid_kv_projection_weight_shape(const Tensor& weight) const;

    [[nodiscard]] Tensor project_query(const Tensor& hidden_states) const;
    [[nodiscard]] Tensor project_key(const Tensor& hidden_states) const;
    [[nodiscard]] Tensor project_value(const Tensor& hidden_states) const;
    [[nodiscard]] Tensor project_output(const Tensor& context) const;

private:
    SelfAttentionConfig config_{};
    AttentionConfig attention_config_{};

    Tensor wq_{};
    Tensor wk_{};
    Tensor wv_{};
    Tensor wo_{};
};

} // namespace mini_llm::model
