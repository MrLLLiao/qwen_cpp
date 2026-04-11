#pragma once

#include <cstddef>

namespace mini_llm::model {

class TransformerBlock {
public:
    // TODO: 补全 block 初始化参数与依赖注入（attention/mlp/norm）
    explicit TransformerBlock(std::size_t layer_id);

    // TODO: 定义 block 前向接口（prefill/decode 两种路径）
    void forward();

private:
    std::size_t layer_id_;
};

} // namespace mini_llm::model
