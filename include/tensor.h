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
    Tensor2D();
    Tensor2D(size_t rows, size_t cols);
    Tensor2D(size_t rows, size_t cols, float init_value);

    [[nodiscard]] size_t rows() const;
    [[nodiscard]] size_t cols() const;
    [[nodiscard]] size_t size() const;
    [[nodiscard]] double max_value() const;

    float& at(size_t r, size_t c);
    const float& at(size_t r, size_t c) const;

    float& operator()(size_t r, size_t c);
    const float& operator()(size_t r, size_t c) const;

    void fill(float value);
    void print() const;
};

#endif //QWEN_CPP_TENSOR_H