//
// Created by killua on 2026/3/21.
//

#ifndef QWEN_CPP_ATTENTION_H
#define QWEN_CPP_ATTENTION_H

#include "model/self-attention.h"

namespace mini_llm::model {

using AttentionLayer = SelfAttention;
using AttentionLayerConfig = SelfAttentionConfig;

} // namespace mini_llm::model

#endif //QWEN_CPP_ATTENTION_H
