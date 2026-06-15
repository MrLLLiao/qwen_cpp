//
// Created by killua on 2026/4/14.
//

#ifndef QWEN_CPP_MLP_H
#define QWEN_CPP_MLP_H

#include <cstddef>

#include "tensor.h"

namespace mini_llm::model {

struct MLPConfig {
    std::size_t hidden_size{0};
    std::size_t intermediate_size{0};
};

class MLP final {
public:
    explicit MLP(MLPConfig config);

    void set_weights(const Tensor& gate, const Tensor& up, const Tensor& down);

    [[nodiscard]] Tensor forward(const Tensor& input) const;
    [[nodiscard]] const MLPConfig& config() const;

private:
    [[nodiscard]] bool valid_weights() const;

private:
    MLPConfig config_{};
    Tensor gate_{};
    Tensor up_{};
    Tensor down_{};
};

} // namespace mini_llm::model

#endif //QWEN_CPP_MLP_H
