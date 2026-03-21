#include "tensor.h"

Tensor2D::Tensor2D() = default;

Tensor2D::Tensor2D(size_t rows, size_t cols) :
    rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

Tensor2D::Tensor2D(size_t rows, size_t cols, float init_value) :
 rows_(rows), cols_(cols), data_(rows * cols, init_value) {}

size_t Tensor2D::index(size_t r, size_t c) const
{
    if (r >= rows_ || c >= cols_)
    {
        throw std::out_of_range("Tensor2D out of range");
    }

    return r * cols_ + c;
}

[[nodiscard]] size_t Tensor2D::rows() const
{
    return rows_;
}

[[nodiscard]] size_t Tensor2D::cols() const
{
    return cols_;
}

[[nodiscard]] size_t Tensor2D::size() const
{
    return data_.size();
}

float& Tensor2D::at(size_t r, size_t c)
{
    return data_[index(r, c)];
}

const float& Tensor2D::at(size_t r, size_t c) const
{
    return data_[index(r, c)];
}

float& Tensor2D::operator()(size_t r, size_t c)
{
    return at(r, c);
}

const float& Tensor2D::operator()(size_t r, size_t c) const
{
    return at(r, c);
}

void Tensor2D::fill(float value)
{
    for (auto&x : data_)
    {
        x = value;
    }
}

void Tensor2D::print() const
{
    for (size_t r = 0; r < rows_; ++r)
    {
        for (size_t c = 0; c < cols_; ++c)
        {
            std::cout << std::setw(8) << at(r, c) << " ";
        }
        std::cout << '\n';
    }
}