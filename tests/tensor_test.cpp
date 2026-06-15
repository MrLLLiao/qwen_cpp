#include "tensor.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace
{
constexpr float kEps = 1e-6F;

bool nearly_equal(const float a, const float b)
{
    return std::fabs(a - b) < kEps;
}

void expect_true(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "[FAIL] " << message << '\n';
        std::exit(1);
    }
}

void test_basic_properties()
{
    Tensor tensor(2, 3, 5.0F);

    expect_true(tensor.rows() == 2, "rows should be 2");
    expect_true(tensor.cols() == 3, "cols should be 3");
    expect_true(tensor.size() == 6, "size should be 6");
    expect_true(tensor.shape().size() == 2, "Tensor shape rank should be 2");
    expect_true(tensor.shape()[0] == 2 && tensor.shape()[1] == 3, "Tensor shape mismatch");
    expect_true(tensor.strides().size() == 2, "Tensor strides rank should be 2");
    expect_true(tensor.strides()[0] == 3 && tensor.strides()[1] == 1, "Tensor row-major strides mismatch");

    for (size_t r = 0; r < tensor.rows(); ++r)
    {
        for (size_t c = 0; c < tensor.cols(); ++c)
        {
            expect_true(nearly_equal(tensor(r, c), 5.0F), "Tensor init value mismatch");
        }
    }
}

void test_nd_tensor_shape_stride_and_access()
{
    Tensor tensor({2, 3, 4}, 0.0F);
    expect_true(tensor.rank() == 3, "Tensor rank should be 3");
    expect_true(tensor.shape()[0] == 2, "Tensor shape[0] mismatch");
    expect_true(tensor.shape()[1] == 3, "Tensor shape[1] mismatch");
    expect_true(tensor.shape()[2] == 4, "Tensor shape[2] mismatch");
    expect_true(tensor.strides()[0] == 12, "Tensor stride[0] mismatch");
    expect_true(tensor.strides()[1] == 4, "Tensor stride[1] mismatch");
    expect_true(tensor.strides()[2] == 1, "Tensor stride[2] mismatch");
    expect_true(tensor.size() == 24, "Tensor size should be 24");
    expect_true(!tensor.empty(), "Tensor should not be empty");
    expect_true(tensor.contiguous(), "Tensor should be contiguous");

    tensor(1, 2, 3) = 42.0F;
    expect_true(nearly_equal(tensor.at({1, 2, 3}), 42.0F), "Tensor variadic access mismatch");
    expect_true(tensor.offset({1, 2, 3}) == 23, "Tensor flat offset mismatch");

    tensor.reshape({4, 6});
    expect_true(tensor.rank() == 2, "reshaped Tensor rank should be 2");
    expect_true(tensor.shape()[0] == 4 && tensor.shape()[1] == 6, "reshaped Tensor shape mismatch");
    expect_true(tensor.strides()[0] == 6 && tensor.strides()[1] == 1, "reshaped Tensor strides mismatch");
    expect_true(nearly_equal(tensor(3, 5), 42.0F), "reshaped Tensor data mismatch");
}

void test_nd_tensor_exceptions()
{
    Tensor tensor({2, 2, 2}, 1.0F);

    bool rank_thrown = false;
    try
    {
        static_cast<void>(tensor.at({0, 1}));
    }
    catch (const std::out_of_range&)
    {
        rank_thrown = true;
    }
    expect_true(rank_thrown, "Tensor rank mismatch should throw");

    bool bounds_thrown = false;
    try
    {
        static_cast<void>(tensor.at({0, 2, 0}));
    }
    catch (const std::out_of_range&)
    {
        bounds_thrown = true;
    }
    expect_true(bounds_thrown, "Tensor bounds mismatch should throw");

    bool reshape_thrown = false;
    try
    {
        tensor.reshape({3, 3});
    }
    catch (const std::invalid_argument&)
    {
        reshape_thrown = true;
    }
    expect_true(reshape_thrown, "Tensor reshape element mismatch should throw");
}

void test_nd_tensor_append_axis0()
{
    Tensor base({1, 2, 2}, 1.0F);
    Tensor tail({2, 2, 2}, 3.0F);

    const auto old_capacity = base.capacity();
    base.reserve(12);
    expect_true(base.capacity() >= old_capacity, "Tensor reserve should not shrink capacity");

    base.append_axis0(tail);
    expect_true(base.shape()[0] == 3 && base.shape()[1] == 2 && base.shape()[2] == 2,
                "Tensor append_axis0 shape mismatch");
    expect_true(base.size() == 12, "Tensor append_axis0 size mismatch");
    expect_true(nearly_equal(base(0, 0, 0), 1.0F), "Tensor append_axis0 old data mismatch");
    expect_true(nearly_equal(base(1, 0, 0), 3.0F), "Tensor append_axis0 appended data mismatch");
    expect_true(nearly_equal(base(2, 1, 1), 3.0F), "Tensor append_axis0 tail data mismatch");

    Tensor self_append({1, 2}, 4.0F);
    self_append.append_axis0(self_append);
    expect_true(self_append.shape()[0] == 2 && self_append.shape()[1] == 2,
                "Tensor append_axis0 self-append shape mismatch");
    expect_true(nearly_equal(self_append(1, 1), 4.0F), "Tensor append_axis0 self-append data mismatch");
}

void test_element_access_and_fill()
{
    Tensor tensor(2, 2, 1.0F);

    tensor.at(1, 1) = 10.0F;
    expect_true(nearly_equal(tensor(1, 1), 10.0F), "Tensor at write mismatch");

    tensor(0, 1) = -3.5F;
    expect_true(nearly_equal(tensor.at(0, 1), -3.5F), "Tensor operator write mismatch");

    tensor.fill(7.0F);
    for (size_t r = 0; r < tensor.rows(); ++r)
    {
        for (size_t c = 0; c < tensor.cols(); ++c)
        {
            expect_true(nearly_equal(tensor(r, c), 7.0F), "Tensor fill mismatch");
        }
    }
}

void test_copy_and_assignment()
{
    Tensor a(2, 2, 2.0F);
    a(0, 1) = 9.0F;

    const Tensor b = a; // copy ctor
    expect_true(b.rows() == 2 && b.cols() == 2, "Tensor copy shape mismatch");
    expect_true(nearly_equal(b(0, 1), 9.0F), "Tensor copy data mismatch");

    Tensor c;
    c = a; // assignment
    expect_true(c.rows() == 2 && c.cols() == 2, "Tensor assignment shape mismatch");
    expect_true(nearly_equal(c(0, 1), 9.0F), "Tensor assignment data mismatch");

    c(0, 1) = 100.0F;
    expect_true(nearly_equal(a(0, 1), 9.0F), "Tensor copy should be deep");
}

void test_tensor_matrix_access_rejects_high_rank()
{
    Tensor base({2, 3}, 0.0F);
    base(1, 2) = 9.0F;

    Tensor copied(base);
    expect_true(copied.rows() == 2 && copied.cols() == 3, "Tensor copy shape mismatch");
    expect_true(nearly_equal(copied(1, 2), 9.0F), "Tensor copy data mismatch");

    bool invalid_rank_thrown = false;
    try
    {
        Tensor high_rank({2, 2, 2}, 0.0F);
        static_cast<void>(high_rank.at(0, 0));
    }
    catch (const std::logic_error&)
    {
        invalid_rank_thrown = true;
    }
    expect_true(invalid_rank_thrown, "Tensor matrix access should reject rank != 2");
}

void test_transpose_square()
{
    Tensor tensor(2, 2, 0.0F);
    tensor(0, 0) = 1.0F;
    tensor(0, 1) = 2.0F;
    tensor(1, 0) = 3.0F;
    tensor(1, 1) = 4.0F;

    tensor.transpose();

    expect_true(tensor.rows() == 2 && tensor.cols() == 2, "square transpose shape mismatch");
    expect_true(nearly_equal(tensor(0, 0), 1.0F), "square transpose [0,0] mismatch");
    expect_true(nearly_equal(tensor(0, 1), 3.0F), "square transpose [0,1] mismatch");
    expect_true(nearly_equal(tensor(1, 0), 2.0F), "square transpose [1,0] mismatch");
    expect_true(nearly_equal(tensor(1, 1), 4.0F), "square transpose [1,1] mismatch");
}

void test_transpose_rectangular()
{
    Tensor tensor(2, 3, 0.0F);
    float value = 1.0F;
    for (size_t r = 0; r < 2; ++r)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            tensor(r, c) = value++;
        }
    }

    tensor.transpose();

    expect_true(tensor.rows() == 3, "rect transpose rows mismatch");
    expect_true(tensor.cols() == 2, "rect transpose cols mismatch");

    expect_true(nearly_equal(tensor(0, 0), 1.0F), "rect transpose [0,0] mismatch");
    expect_true(nearly_equal(tensor(0, 1), 4.0F), "rect transpose [0,1] mismatch");
    expect_true(nearly_equal(tensor(1, 0), 2.0F), "rect transpose [1,0] mismatch");
    expect_true(nearly_equal(tensor(1, 1), 5.0F), "rect transpose [1,1] mismatch");
    expect_true(nearly_equal(tensor(2, 0), 3.0F), "rect transpose [2,0] mismatch");
    expect_true(nearly_equal(tensor(2, 1), 6.0F), "rect transpose [2,1] mismatch");
}

void test_max_value()
{
    Tensor tensor(2, 3, -1.0F);
    tensor(1, 2) = 42.0F;
    expect_true(nearly_equal(static_cast<float>(tensor.max_value()), 42.0F), "Tensor max_value mismatch");
}

void test_capacity_data_span_and_append_rows()
{
    Tensor tensor(1, 3, 1.0F);
    tensor.reserve_rows(4);
    expect_true(tensor.capacity() >= 12, "Tensor reserve_rows capacity mismatch");
    expect_true(tensor.data() != nullptr, "Tensor data should not be null");
    expect_true(tensor.span().size() == 3, "Tensor span size mismatch");

    Tensor extra(2, 3, 2.0F);
    tensor.append_rows(extra);
    expect_true(tensor.rows() == 3 && tensor.cols() == 3, "Tensor append_rows shape mismatch");
    expect_true(tensor.size() == 9, "Tensor append_rows size mismatch");
    expect_true(nearly_equal(tensor(0, 0), 1.0F), "Tensor append_rows old data mismatch");
    expect_true(nearly_equal(tensor(1, 0), 2.0F), "Tensor append_rows new row mismatch");
    expect_true(nearly_equal(tensor(2, 2), 2.0F), "Tensor append_rows tail mismatch");
    expect_true(tensor.row_data(2) == tensor.data() + 6, "Tensor row_data pointer mismatch");
}

void test_empty_tensor_contract()
{
    Tensor empty;
    expect_true(empty.rows() == 0, "empty Tensor rows should be 0");
    expect_true(empty.cols() == 0, "empty Tensor cols should be 0");
    expect_true(empty.size() == 0, "empty Tensor size should be 0");

    empty.transpose();
    expect_true(empty.rows() == 0, "empty transpose rows should remain 0");
    expect_true(empty.cols() == 0, "empty transpose cols should remain 0");

    bool thrown = false;
    try
    {
        static_cast<void>(empty(0, 0));
    }
    catch (const std::out_of_range&)
    {
        thrown = true;
    }
    expect_true(thrown, "empty Tensor access should throw");
}

void test_exceptions()
{
    Tensor tensor(2, 2, 0.0F);

    bool out_of_range_thrown = false;
    try
    {
        static_cast<void>(tensor.at(2, 0));
    }
    catch (const std::out_of_range&)
    {
        out_of_range_thrown = true;
    }
    expect_true(out_of_range_thrown, "Tensor out of range should throw");

    bool empty_max_thrown = false;
    try
    {
        const Tensor empty;
        static_cast<void>(empty.max_value());
    }
    catch (const std::runtime_error&)
    {
        empty_max_thrown = true;
    }
    expect_true(empty_max_thrown, "empty Tensor max_value should throw");
}
} // namespace

int main()
{
    test_basic_properties();
    test_nd_tensor_shape_stride_and_access();
    test_nd_tensor_exceptions();
    test_nd_tensor_append_axis0();
    test_element_access_and_fill();
    test_copy_and_assignment();
    test_tensor_matrix_access_rejects_high_rank();
    test_transpose_square();
    test_transpose_rectangular();
    test_max_value();
    test_capacity_data_span_and_append_rows();
    test_empty_tensor_contract();
    test_exceptions();

    std::cout << "[PASS] Tensor tests passed.\n";
    return 0;
}
