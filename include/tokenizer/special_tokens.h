#pragma once

namespace mini_llm::tokenizer {

struct SpecialTokens {
    // TODO: 根据模型家族补全 special token id 映射
    int bos = -1;
    int eos = -1;
    int pad = -1;
};

} // namespace mini_llm::tokenizer
