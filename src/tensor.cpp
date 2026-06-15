#include "tensor.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

Tensor::Tensor() = default;

Tensor::Tensor(Shape shape)
    : Tensor(std::move(shape), 0.0f)
{
}

Tensor::Tensor(Shape shape, float init_value)
{
    resize(std::move(shape), init_value);
}

Tensor::Tensor(std::size_t rows, std::size_t cols)
    : Tensor(Shape{rows, cols}, 0.0f)
{
}

Tensor::Tensor(std::size_t rows, std::size_t cols, float init_value)
    : Tensor(Shape{rows, cols}, init_value)
{
}

std::size_t Tensor::rank() const
{
    return shape_.size();
}

const Tensor::Shape& Tensor::shape() const
{
    return shape_;
}

const Tensor::Strides& Tensor::strides() const
{
    return strides_;
}

std::size_t Tensor::dim(std::size_t axis) const
{
    if (axis >= shape_.size())
    {
        throw std::out_of_range("Tensor::dim axis out of range");
    }
    return shape_[axis];
}

std::size_t Tensor::rows() const
{
    return rank() == 0 ? 0 : dim(0);
}

std::size_t Tensor::cols() const
{
    return rank() == 0 ? 0 : dim(1);
}

std::size_t Tensor::size() const
{
    return data_.size();
}

std::size_t Tensor::capacity() const
{
    return data_.capacity();
}

bool Tensor::empty() const
{
    return data_.empty();
}

bool Tensor::contiguous() const
{
    return strides_ == make_contiguous_strides(shape_);
}

float* Tensor::data()
{
    return data_.data();
}

const float* Tensor::data() const
{
    return data_.data();
}

std::span<float> Tensor::span()
{
    return std::span<float>(data_.data(), data_.size());
}

std::span<const float> Tensor::span() const
{
    return std::span<const float>(data_.data(), data_.size());
}

float* Tensor::row_data(std::size_t r)
{
    require_matrix();
    if (r >= rows())
    {
        throw std::out_of_range("Tensor row out of range");
    }
    return data_.data() + r * cols();
}

const float* Tensor::row_data(std::size_t r) const
{
    require_matrix();
    if (r >= rows())
    {
        throw std::out_of_range("Tensor row out of range");
    }
    return data_.data() + r * cols();
}

void Tensor::reserve(std::size_t elements)
{
    data_.reserve(elements);
}

void Tensor::reserve_rows(std::size_t rows)
{
    require_matrix();
    if (cols() == 0 && rows > 0)
    {
        throw std::logic_error("Tensor::reserve_rows requires cols() > 0");
    }
    data_.reserve(rows * cols());
}

void Tensor::resize(Shape shape, float init_value)
{
    const std::size_t elements = checked_num_elements(shape);
    shape_ = std::move(shape);
    strides_ = make_contiguous_strides(shape_);
    data_.resize(elements, init_value);
}

void Tensor::resize(std::size_t rows, std::size_t cols, float init_value)
{
    resize(Shape{rows, cols}, init_value);
}

void Tensor::reshape(Shape shape)
{
    if (checked_num_elements(shape) != data_.size())
    {
        throw std::invalid_argument("Tensor::reshape element count mismatch");
    }
    shape_ = std::move(shape);
    strides_ = make_contiguous_strides(shape_);
}

void Tensor::append_axis0(const Tensor& tail)
{
    if (this == &tail)
    {
        const Tensor copy(tail);
        append_axis0(copy);
        return;
    }

    if (tail.empty())
    {
        return;
    }

    if (empty() && shape_.empty())
    {
        *this = tail;
        return;
    }

    if (shape_.empty() || tail.rank() != rank())
    {
        throw std::invalid_argument("Tensor::append_axis0 rank mismatch");
    }

    for (std::size_t axis = 1; axis < rank(); ++axis)
    {
        if (shape_[axis] != tail.shape_[axis])
        {
            throw std::invalid_argument("Tensor::append_axis0 shape mismatch");
        }
    }

    if (tail.shape_[0] == 0)
    {
        return;
    }

    data_.insert(data_.end(), tail.data_.begin(), tail.data_.end());
    shape_[0] += tail.shape_[0];
    strides_ = make_contiguous_strides(shape_);
}

void Tensor::append_rows(const Tensor& rows)
{
    require_matrix();
    if (rows.rows() == 0)
    {
        return;
    }

    if (this->rows() == 0 && cols() == 0)
    {
        *this = rows;
        return;
    }

    if (rows.rank() != 2 || rows.cols() != cols())
    {
        throw std::invalid_argument("Tensor::append_rows column mismatch");
    }

    append_axis0(rows);
}

void Tensor::transpose()
{
    require_matrix();
    if (rows() == cols())
    {
        for (std::size_t row = 0; row < rows(); ++row)
        {
            for (std::size_t col = row + 1; col < cols(); ++col)
            {
                std::swap(data_[row * cols() + col], data_[col * cols() + row]);
            }
        }
        return;
    }

    std::vector<float> transposed(data_.size());
    const std::size_t old_rows = rows();
    const std::size_t old_cols = cols();
    for (std::size_t row = 0; row < old_rows; ++row)
    {
        for (std::size_t col = 0; col < old_cols; ++col)
        {
            transposed[col * old_rows + row] = data_[row * old_cols + col];
        }
    }

    resize(Shape{old_cols, old_rows}, 0.0f);
    std::copy(transposed.begin(), transposed.end(), data_.begin());
}

void Tensor::fill(float value)
{
    std::ranges::fill(data_, value);
}

void Tensor::print() const
{
    require_matrix();
    for (std::size_t r = 0; r < rows(); ++r)
    {
        for (std::size_t c = 0; c < cols(); ++c)
        {
            std::cout << std::setw(8) << at(r, c) << " ";
        }
        std::cout << '\n';
    }
}

std::size_t Tensor::offset(std::span<const std::size_t> indices) const
{
    if (indices.size() != shape_.size())
    {
        throw std::out_of_range("Tensor index rank mismatch");
    }

    std::size_t flat = 0;
    for (std::size_t axis = 0; axis < indices.size(); ++axis)
    {
        if (indices[axis] >= shape_[axis])
        {
            throw std::out_of_range("Tensor index out of range");
        }
        flat += indices[axis] * strides_[axis];
    }
    return flat;
}

std::size_t Tensor::offset(std::initializer_list<std::size_t> indices) const
{
    return offset(std::span<const std::size_t>(indices.begin(), indices.size()));
}

float& Tensor::at(std::span<const std::size_t> indices)
{
    return data_[offset(indices)];
}

const float& Tensor::at(std::span<const std::size_t> indices) const
{
    return data_[offset(indices)];
}

float& Tensor::at(std::initializer_list<std::size_t> indices)
{
    return data_[offset(indices)];
}

const float& Tensor::at(std::initializer_list<std::size_t> indices) const
{
    return data_[offset(indices)];
}

float& Tensor::at(std::size_t r, std::size_t c)
{
    return data_[matrix_index(r, c)];
}

const float& Tensor::at(std::size_t r, std::size_t c) const
{
    return data_[matrix_index(r, c)];
}

double Tensor::max_value() const
{
    if (data_.empty())
    {
        throw std::runtime_error("Tensor is empty");
    }
    return *std::ranges::max_element(data_);
}

Tensor::Strides Tensor::make_contiguous_strides(const Shape& shape)
{
    Strides strides(shape.size(), 1);
    if (shape.empty())
    {
        return strides;
    }

    for (std::size_t axis = shape.size() - 1; axis > 0; --axis)
    {
        strides[axis - 1] = strides[axis] * shape[axis];
    }
    return strides;
}

std::size_t Tensor::checked_num_elements(const Shape& shape)
{
    if (shape.empty())
    {
        return 0;
    }

    std::size_t elements = 1;
    for (const std::size_t dim : shape)
    {
        if (dim == 0)
        {
            return 0;
        }
        if (elements > std::numeric_limits<std::size_t>::max() / dim)
        {
            throw std::overflow_error("Tensor shape element count overflow");
        }
        elements *= dim;
    }
    return elements;
}

void Tensor::require_matrix() const
{
    if (rank() == 0)
    {
        return;
    }
    if (rank() != 2 || !contiguous())
    {
        throw std::logic_error("Tensor operation requires contiguous rank-2 tensor");
    }
}

std::size_t Tensor::matrix_index(std::size_t r, std::size_t c) const
{
    require_matrix();
    return offset({r, c});
}

bool TensorConstView::empty() const
{
    return data == nullptr || rows == 0 || cols == 0;
}

const float* TensorConstView::row_data(std::size_t r) const
{
    if (data == nullptr)
    {
        throw std::runtime_error("TensorConstView is empty");
    }
    if (r >= rows)
    {
        throw std::out_of_range("TensorConstView row out of range");
    }
    return data + r * stride;
}

const float& TensorConstView::at(std::size_t r, std::size_t c) const
{
    if (c >= cols)
    {
        throw std::out_of_range("TensorConstView col out of range");
    }
    return row_data(r)[c];
}

const float& TensorConstView::operator()(std::size_t r, std::size_t c) const
{
    return at(r, c);
}

TensorConstView make_tensor_view(const Tensor& tensor)
{
    return TensorConstView{tensor.data(), tensor.rows(), tensor.cols(), tensor.cols()};
}

TensorConstView make_tensor_column_view(const Tensor& tensor,
                                        std::size_t col_offset,
                                        std::size_t col_count,
                                        std::size_t row_count)
{
    if (col_offset > tensor.cols() || col_count > tensor.cols() - col_offset)
    {
        throw std::out_of_range("make_tensor_column_view column range out of range");
    }

    const std::size_t actual_rows = row_count == static_cast<std::size_t>(-1) ? tensor.rows() : row_count;
    if (actual_rows > tensor.rows())
    {
        throw std::out_of_range("make_tensor_column_view row_count out of range");
    }

    const float* ptr = tensor.data();
    if (ptr != nullptr)
    {
        ptr += col_offset;
    }
    return TensorConstView{ptr, actual_rows, col_count, tensor.cols()};
}
