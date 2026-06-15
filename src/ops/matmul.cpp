#include "ops/matmul.h"

#include <algorithm>
#include <stdexcept>

void matmul_into(const Tensor& A, const Tensor& B, Tensor& out)
{
    if (A.cols() != B.rows())
    {
        throw std::invalid_argument("matmul dimension mismatch: A.cols() must equal B.rows()");
    }

    out.resize(A.rows(), B.cols(), 0.0f);

    for (size_t i = 0; i < A.rows(); ++i)
    {
        float* out_row = out.row_data(i);
        for (size_t k = 0; k < A.cols(); ++k)
        {
            const float a_ik = A.row_data(i)[k];
            const float* b_row = B.row_data(k);
            for (size_t j = 0; j < B.cols(); ++j)
            {
                out_row[j] += a_ik * b_row[j];
            }
        }
    }
}

Tensor matmul(const Tensor& A, const Tensor& B)
{
    Tensor C;
    matmul_into(A, B, C);

    return C;
}
