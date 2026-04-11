#pragma once

#include <string>
#include <vector>

namespace mini_llm::tokenizer {

class Tokenizer {
public:
    // TODO: 实现 tokenizer 词表加载与版本校验
    bool load(const std::string& tokenizer_path);

    // TODO: 实现 encode/decode 与 special tokens 管理
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
};

} // namespace mini_llm::tokenizer
