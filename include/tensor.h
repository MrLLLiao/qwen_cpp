#pragma once

#ifndef QWEN_CPP_TENSOR_H
#define QWEN_CPP_TENSOR_H

#include <cstddef>
#include <initializer_list>
#include <span>
#include <type_traits>
#include <vector>

class Tensor
{
public:
    using Shape = std::vector<std::size_t>;
    using Strides = std::vector<std::size_t>;

    Tensor();
    explicit Tensor(Shape shape);
    Tensor(Shape shape, float init_value);
    Tensor(std::size_t rows, std::size_t cols);
    Tensor(std::size_t rows, std::size_t cols, float init_value);

    [[nodiscard]] std::size_t rank() const;
    [[nodiscard]] const Shape& shape() const;
    [[nodiscard]] const Strides& strides() const;
    [[nodiscard]] std::size_t dim(std::size_t axis) const;
    [[nodiscard]] std::size_t rows() const;
    [[nodiscard]] std::size_t cols() const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t capacity() const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] bool contiguous() const;

    [[nodiscard]] float* data();
    [[nodiscard]] const float* data() const;
    [[nodiscard]] std::span<float> span();
    [[nodiscard]] std::span<const float> span() const;
    [[nodiscard]] float* row_data(std::size_t r);
    [[nodiscard]] const float* row_data(std::size_t r) const;

    void reserve(std::size_t elements);
    void reserve_rows(std::size_t rows);
    void resize(Shape shape, float init_value = 0.0f);
    void resize(std::size_t rows, std::size_t cols, float init_value = 0.0f);
    void reshape(Shape shape);
    void append_axis0(const Tensor& tail);
    void append_rows(const Tensor& rows);
    void transpose();
    void fill(float value);
    void print() const;

    [[nodiscard]] std::size_t offset(std::span<const std::size_t> indices) const;
    [[nodiscard]] std::size_t offset(std::initializer_list<std::size_t> indices) const;

    float& at(std::span<const std::size_t> indices);
    [[nodiscard]] const float& at(std::span<const std::size_t> indices) const;
    float& at(std::initializer_list<std::size_t> indices);
    [[nodiscard]] const float& at(std::initializer_list<std::size_t> indices) const;
    float& at(std::size_t r, std::size_t c);
    [[nodiscard]] const float& at(std::size_t r, std::size_t c) const;

    template <typename... Indices>
        requires(sizeof...(Indices) > 0 && (std::is_integral_v<Indices> && ...))
    float& operator()(Indices... indices)
    {
        const std::size_t idx[]{static_cast<std::size_t>(indices)...};
        return at(std::span<const std::size_t>(idx, sizeof...(Indices)));
    }

    template <typename... Indices>
        requires(sizeof...(Indices) > 0 && (std::is_integral_v<Indices> && ...))
    const float& operator()(Indices... indices) const
    {
        const std::size_t idx[]{static_cast<std::size_t>(indices)...};
        return at(std::span<const std::size_t>(idx, sizeof...(Indices)));
    }

    [[nodiscard]] double max_value() const;

private:
    [[nodiscard]] static Strides make_contiguous_strides(const Shape& shape);
    [[nodiscard]] static std::size_t checked_num_elements(const Shape& shape);
    void require_matrix() const;
    [[nodiscard]] std::size_t matrix_index(std::size_t r, std::size_t c) const;

private:
    Shape shape_{};
    Strides strides_{};
    std::vector<float> data_{};
};

struct TensorConstView
{
    const float* data{nullptr};
    std::size_t rows{0};
    std::size_t cols{0};
    std::size_t stride{0};

    [[nodiscard]] bool empty() const;
    [[nodiscard]] const float* row_data(std::size_t r) const;
    [[nodiscard]] const float& at(std::size_t r, std::size_t c) const;
    [[nodiscard]] const float& operator()(std::size_t r, std::size_t c) const;
};

[[nodiscard]] TensorConstView make_tensor_view(const Tensor& tensor);
[[nodiscard]] TensorConstView make_tensor_column_view(const Tensor& tensor,
                                                      std::size_t col_offset,
                                                      std::size_t col_count,
                                                      std::size_t row_count = static_cast<std::size_t>(-1));

#endif // QWEN_CPP_TENSOR_H
