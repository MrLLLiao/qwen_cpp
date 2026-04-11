#include "model/transformer_block.h"

namespace mini_llm::model {

TransformerBlock::TransformerBlock(std::size_t layer_id)
    : layer_id_(layer_id) {
    // TODO: 初始化 attention/mlp/norm 子模块
}

void TransformerBlock::forward() {
    // TODO: 实现 block 前向（输入校验、计算流程、输出写回）
}

} // namespace mini_llm::model
