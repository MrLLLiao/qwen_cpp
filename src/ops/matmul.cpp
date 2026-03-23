#include "ops/matmul.h"

#include <stdexcept>

Tensor2D matmul(const Tensor2D& A, const Tensor2D& B)
{
    if (A.cols() != B.rows())
    {
        throw std::invalid_argument("matmul dimension mismatch: A.cols() must equal B.rows()");
    }

    Tensor2D C(A.rows(), B.cols(), 0.0f);

    for (size_t i = 0; i < A.rows(); ++i)
    {
        for (size_t k = 0; k < A.cols(); ++k)
        {
            const float a_ik = A(i, k);
            for (size_t j = 0; j < B.cols(); ++j)
            {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }

    return C;
}
