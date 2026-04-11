#include "tokenizer/tokenizer.h"

namespace mini_llm::tokenizer {

bool Tokenizer::load(const std::string& tokenizer_path) {
    // TODO: 从文件加载 tokenizer 并校验完整性
    (void)tokenizer_path;
    return false;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // TODO: 实现文本到 token ids 的转换
    (void)text;
    return {};
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    // TODO: 实现 token ids 到文本的还原
    (void)tokens;
    return {};
}

} // namespace mini_llm::tokenizer
