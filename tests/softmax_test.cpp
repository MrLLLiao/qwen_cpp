#include "ops/softmax.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace
{
    bool nearly_equal(float a, float b, float eps = 1e-5f)
    {
        return std::fabs(a - b) <= eps;
    }

    void expect_true(bool condition, const char* message)
    {
        if (!condition)
        {
            std::cerr << "[FAIL] " << message << '\n';
            std::exit(1);
        }
    }

    float row_sum(const Tensor2D& t, size_t row)
    {
        float sum = 0.0f;
        for (size_t col = 0; col < t.cols(); ++col)
        {
            sum += t(row, col);
        }
        return sum;
    }

    float col_sum(const Tensor2D& t, size_t col)
    {
        float sum = 0.0f;
        for (size_t row = 0; row < t.rows(); ++row)
        {
            sum += t(row, col);
        }
        return sum;
    }
}

int main()
{
    {
        Tensor2D logits(2, 3, 0.0f);
        logits(0, 0) = 1.0f; logits(0, 1) = 2.0f; logits(0, 2) = 3.0f;
        logits(1, 0) = 1.0f; logits(1, 1) = 1.0f; logits(1, 2) = 1.0f;

        const Tensor2D probs = softmax(logits, SoftmaxAxis::Row);

        expect_true(probs.rows() == 2 && probs.cols() == 3, "row softmax shape should be 2x3");
        expect_true(nearly_equal(row_sum(probs, 0), 1.0f), "row 0 sum should be 1");
        expect_true(nearly_equal(row_sum(probs, 1), 1.0f), "row 1 sum should be 1");

        expect_true(nearly_equal(probs(0, 0), 0.09003057f), "row softmax value (0,0) mismatch");
        expect_true(nearly_equal(probs(0, 1), 0.24472847f), "row softmax value (0,1) mismatch");
        expect_true(nearly_equal(probs(0, 2), 0.66524096f), "row softmax value (0,2) mismatch");

        expect_true(nearly_equal(probs(1, 0), 1.0f / 3.0f), "uniform row value (1,0) mismatch");
        expect_true(nearly_equal(probs(1, 1), 1.0f / 3.0f), "uniform row value (1,1) mismatch");
        expect_true(nearly_equal(probs(1, 2), 1.0f / 3.0f), "uniform row value (1,2) mismatch");
    }

    {
        Tensor2D logits(3, 2, 0.0f);
        logits(0, 0) = 1.0f; logits(0, 1) = 2.0f;
        logits(1, 0) = 3.0f; logits(1, 1) = 4.0f;
        logits(2, 0) = 5.0f; logits(2, 1) = 6.0f;

        const Tensor2D probs = softmax(logits, SoftmaxAxis::Col);

        expect_true(nearly_equal(col_sum(probs, 0), 1.0f), "col 0 sum should be 1");
        expect_true(nearly_equal(col_sum(probs, 1), 1.0f), "col 1 sum should be 1");

        expect_true(nearly_equal(probs(0, 0), 0.01587624f), "col softmax value (0,0) mismatch");
        expect_true(nearly_equal(probs(1, 0), 0.11731043f), "col softmax value (1,0) mismatch");
        expect_true(nearly_equal(probs(2, 0), 0.86681333f), "col softmax value (2,0) mismatch");

        expect_true(nearly_equal(probs(0, 1), 0.01587624f), "col softmax value (0,1) mismatch");
        expect_true(nearly_equal(probs(1, 1), 0.11731043f), "col softmax value (1,1) mismatch");
        expect_true(nearly_equal(probs(2, 1), 0.86681333f), "col softmax value (2,1) mismatch");
    }

    {
        Tensor2D logits(1, 3, 0.0f);
        logits(0, 0) = 2.0f;
        logits(0, 1) = 4.0f;
        logits(0, 2) = 6.0f;

        Softmax op(SoftmaxConfig{SoftmaxAxis::Row, 1e-12f, 1.0f});
        op.forward_inplace(logits);

        expect_true(nearly_equal(row_sum(logits, 0), 1.0f), "inplace softmax row sum should be 1");
    }

{
        bool thrown = false;
        try
        {
            Tensor2D logits(1, 2, 0.0f);
            static_cast<void>(softmax(logits, SoftmaxAxis::Row, 1e-12f, 0.0f));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "softmax should throw when temperature <= 0");

        thrown = false;
        try
        {
            Tensor2D logits(1, 2, 0.0f);
            static_cast<void>(softmax(logits, SoftmaxAxis::Row, -1.0f, 1.0f));
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        expect_true(thrown, "softmax should throw when epsilon < 0");
    }

    std::cout << "[PASS] softmax tests passed" << '\n';
    return 0;
}