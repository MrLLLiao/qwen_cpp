#include "tensor.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
constexpr float kEps = 1e-6F;

bool nearly_equal(const float a, const float b)
{
    return std::fabs(a - b) < kEps;
}

void test_basic_properties()
{
    Tensor2D tensor(2, 3, 5.0F);

    assert(tensor.rows() == 2);
    assert(tensor.cols() == 3);
    assert(tensor.size() == 6);

    for (size_t r = 0; r < tensor.rows(); ++r)
    {
        for (size_t c = 0; c < tensor.cols(); ++c)
        {
            assert(nearly_equal(tensor(r, c), 5.0F));
        }
    }
}

void test_element_access_and_fill()
{
    Tensor2D tensor(2, 2, 1.0F);

    tensor.at(1, 1) = 10.0F;
    assert(nearly_equal(tensor(1, 1), 10.0F));

    tensor(0, 1) = -3.5F;
    assert(nearly_equal(tensor.at(0, 1), -3.5F));

    tensor.fill(7.0F);
    for (size_t r = 0; r < tensor.rows(); ++r)
    {
        for (size_t c = 0; c < tensor.cols(); ++c)
        {
            assert(nearly_equal(tensor(r, c), 7.0F));
        }
    }
}

void test_copy_and_assignment()
{
    Tensor2D a(2, 2, 2.0F);
    a(0, 1) = 9.0F;

    const Tensor2D b = a; // copy ctor
    assert(b.rows() == 2 && b.cols() == 2);
    assert(nearly_equal(b(0, 1), 9.0F));

    Tensor2D c;
    c = a; // assignment
    assert(c.rows() == 2 && c.cols() == 2);
    assert(nearly_equal(c(0, 1), 9.0F));

    c(0, 1) = 100.0F;
    assert(nearly_equal(a(0, 1), 9.0F)); // deep copy check
}

void test_transpose_square()
{
    Tensor2D tensor(2, 2, 0.0F);
    tensor(0, 0) = 1.0F;
    tensor(0, 1) = 2.0F;
    tensor(1, 0) = 3.0F;
    tensor(1, 1) = 4.0F;

    tensor.transpose();

    assert(tensor.rows() == 2 && tensor.cols() == 2);
    assert(nearly_equal(tensor(0, 0), 1.0F));
    assert(nearly_equal(tensor(0, 1), 3.0F));
    assert(nearly_equal(tensor(1, 0), 2.0F));
    assert(nearly_equal(tensor(1, 1), 4.0F));
}

void test_transpose_rectangular()
{
    Tensor2D tensor(2, 3, 0.0F);
    float value = 1.0F;
    for (size_t r = 0; r < 2; ++r)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            tensor(r, c) = value++;
        }
    }

    tensor.transpose();

    assert(tensor.rows() == 3);
    assert(tensor.cols() == 2);

    assert(nearly_equal(tensor(0, 0), 1.0F));
    assert(nearly_equal(tensor(0, 1), 4.0F));
    assert(nearly_equal(tensor(1, 0), 2.0F));
    assert(nearly_equal(tensor(1, 1), 5.0F));
    assert(nearly_equal(tensor(2, 0), 3.0F));
    assert(nearly_equal(tensor(2, 1), 6.0F));
}

void test_max_value()
{
    Tensor2D tensor(2, 3, -1.0F);
    tensor(1, 2) = 42.0F;
    assert(nearly_equal(static_cast<float>(tensor.max_value()), 42.0F));
}

void test_exceptions()
{
    Tensor2D tensor(2, 2, 0.0F);

    bool out_of_range_thrown = false;
    try
    {
        static_cast<void>(tensor.at(2, 0));
    }
    catch (const std::out_of_range&)
    {
        out_of_range_thrown = true;
    }
    assert(out_of_range_thrown);

    bool empty_max_thrown = false;
    try
    {
        const Tensor2D empty;
        static_cast<void>(empty.max_value());
    }
    catch (const std::runtime_error&)
    {
        empty_max_thrown = true;
    }
    assert(empty_max_thrown);
}
} // namespace

int main()
{
    test_basic_properties();
    test_element_access_and_fill();
    test_copy_and_assignment();
    test_transpose_square();
    test_transpose_rectangular();
    test_max_value();
    test_exceptions();

    std::cout << "[PASS] Tensor2D tests passed.\n";
    return 0;
}
