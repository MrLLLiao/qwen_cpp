#include "ops/matmul.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

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
}

int main()
{
    Tensor2D A(2, 3, 0.0f);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

    Tensor2D B(3, 2, 0.0f);
    B(0, 0) = 7.0f;  B(0, 1) = 8.0f;
    B(1, 0) = 9.0f;  B(1, 1) = 10.0f;
    B(2, 0) = 11.0f; B(2, 1) = 12.0f;

    const Tensor2D C = matmul(A, B);

    expect_true(C.rows() == 2, "C rows should be 2");
    expect_true(C.cols() == 2, "C cols should be 2");

    expect_true(nearly_equal(C(0, 0), 58.0f), "C(0,0) should be 58");
    expect_true(nearly_equal(C(0, 1), 64.0f), "C(0,1) should be 64");
    expect_true(nearly_equal(C(1, 0), 139.0f), "C(1,0) should be 139");
    expect_true(nearly_equal(C(1, 1), 154.0f), "C(1,1) should be 154");

    bool thrown = false;
    try
    {
        const Tensor2D badB(4, 1, 1.0f);
        static_cast<void>(matmul(A, badB));
    }
    catch (const std::invalid_argument&)
    {
        thrown = true;
    }

    expect_true(thrown, "matmul should throw on dimension mismatch");

    std::cout << "[PASS] matmul tests passed" << '\n';
    return 0;
}