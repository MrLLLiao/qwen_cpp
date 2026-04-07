#pragma once
#include <cstddef>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <vector>

#ifndef QWEN_CPP_TENSOR_H
#define QWEN_CPP_TENSOR_H

class Tensor2D
{
private:
    size_t rows_{0};
    size_t cols_{0};
    std::vector<float> data_;
private:
    [[nodiscard]] size_t index(size_t r, size_t c) const;
public:
    /**
     * @brief 默认构造得到空张量（rows=0, cols=0, size=0）。
     */
    Tensor2D();
    Tensor2D(size_t rows, size_t cols);
    Tensor2D(size_t rows, size_t cols, float init_value);
    Tensor2D(const Tensor2D& other) = default;

    [[nodiscard]] size_t rows() const;
    [[nodiscard]] size_t cols() const;
    [[nodiscard]] size_t size() const;

    /**
     * @brief 返回张量最大值。
     * @throws std::runtime_error 当张量为空（size==0）时抛出。
     */
    [[nodiscard]] double max_value() const;

    void transpose();

    /**
     * @brief 边界检查访问。
     * @throws std::out_of_range 当 r>=rows 或 c>=cols。
     */
    float& at(size_t r, size_t c);
    [[nodiscard]] const float& at(size_t r, size_t c) const;

    /**
     * @brief 等价于 at(r, c)，同样执行边界检查并抛出 out_of_range。
     */
    float& operator()(size_t r, size_t c);
    const float& operator()(size_t r, size_t c) const;

    Tensor2D& operator=(const Tensor2D& other);

    void fill(float value);
    void print() const;
};

#endif //QWEN_CPP_TENSOR_H