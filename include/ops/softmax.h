//
// Created by killua on 2026/3/21.
//

#ifndef QWEN_CPP_SOFTMAX_H
#define QWEN_CPP_SOFTMAX_H

#include "tensor.h"

/**
 * @brief Softmax 归一化维度。
 *
 * Tensor2D 仅支持二维：
 * - Row: 对每一行做 softmax（常用于每个 token 的 logits）
 * - Col: 对每一列做 softmax
 */
enum class SoftmaxAxis
{
    Row,
    Col
};

/**
 * @brief Softmax 算子配置。
 */
struct SoftmaxConfig
{
    SoftmaxAxis axis{SoftmaxAxis::Row};

    /**
     * @brief 数值稳定项，防止分母接近 0。
     */
    float epsilon{1e-12f};

    /**
     * @brief 温度参数，默认 1.0。
     *
     * 输出为 softmax(x / temperature)，temperature 必须 > 0。
     */
    float temperature{1.0f};
};

/**
 * @brief Softmax 算子（无状态）。
 */
class Softmax final
{
public:
    Softmax() = default;
    explicit Softmax(SoftmaxConfig config);

    /**
     * @brief 返回 softmax 后的新 Tensor。
     */
    [[nodiscard]] Tensor2D forward(const Tensor2D& input) const;

    /**
     * @brief 对输入 Tensor 原地执行 softmax。
     */
    void forward_inplace(Tensor2D& input) const;

    /**
     * @brief 读取当前配置。
     */
    [[nodiscard]] const SoftmaxConfig& config() const;

private:
    SoftmaxConfig config_{};
};

/**
 * @brief 函数式 API：按指定 axis 执行 softmax。
 */
[[nodiscard]] Tensor2D softmax(const Tensor2D& input,
                               SoftmaxAxis axis = SoftmaxAxis::Row,
                               float epsilon = 1e-12f,
                               float temperature = 1.0f);

#endif //QWEN_CPP_SOFTMAX_H