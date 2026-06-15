//
// Created by killua on 2026/3/21.
//

#ifndef QWEN_CPP_MATMUL_H
#define QWEN_CPP_MATMUL_H
#include "tensor.h"

Tensor matmul(const Tensor& A, const Tensor& B);
void matmul_into(const Tensor& A, const Tensor& B, Tensor& out);

#endif //QWEN_CPP_MATMUL_H
