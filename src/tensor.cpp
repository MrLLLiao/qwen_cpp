#include "tensor.h"
#include <algorithm>

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

[[nodiscard]] double Tensor2D::max_value() const
{
    if (data_.empty())
    {
        throw std::runtime_error("Tensor2D is empty");
    }

    return *std::ranges::max_element(data_);
}

void Tensor2D::transpose()
{
    if (rows_ == cols_)
    {
        for (size_t row = 0; row < rows_; ++row)
        {
            for (size_t col = row + 1; col < cols_; ++col)
            {
                std::swap(data_[row * cols_ + col], data_[col * cols_ + row]);
            }
        }
        return;
    }

    std::vector<float> transposed(data_.size());
    for (size_t row = 0; row < rows_; ++row)
    {
        for (size_t col = 0; col < cols_; ++col)
        {
            transposed[col * rows_ + row] = data_[row * cols_ + col];
        }
    }

    data_.swap(transposed);
    const size_t old_rows = rows_;
    rows_ = cols_;
    cols_ = old_rows;
}

float& Tensor2D::at(size_t r, size_t c)
{
    return data_[index(r, c)];
}

[[nodiscard]] const float& Tensor2D::at(size_t r, size_t c) const
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

Tensor2D& Tensor2D::operator=(const Tensor2D& other)
{
    if (this == &other)
    {
        return *this;
    }

    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = other.data_;
    return *this;
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