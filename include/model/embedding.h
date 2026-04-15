//
// Created by killua on 2026/4/14.
//

#ifndef QWEN_CPP_EMBEDDING_H
#define QWEN_CPP_EMBEDDING_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mini_llm::model {

/**
 * @brief 词表映射骨架。
 *
 * 当前职责：
 * 1) 管理 token <-> id 的双向映射；
 * 2) 提供特殊 token（unk/pad/bos/eos）的配置与查询；
 * 3) 通过 simdjson 加载 vocab.json，供后续 tokenizer/runner 对接。
 */
class Embedding {
public:
    /// 统一 token id 类型，便于后续和模型输入类型对齐。
    using TokenId = std::int32_t;

    Embedding() = default;

    /**
     * @brief 从 vocab.json 加载词表（基于 simdjson）。
     * @param vocab_path 词表文件路径。
     * @return 加载成功返回 true，否则返回 false。
     */
    bool load_vocab(const std::string& vocab_path);

    // ==================== 基本查询接口 ====================

    /// @brief 词表是否为空。
    [[nodiscard]] bool empty() const;

    /// @brief 词表大小（token 数量）。
    [[nodiscard]] std::size_t size() const;

    /// @brief 判断 token 是否存在。
    [[nodiscard]] bool contains_token(const std::string& token) const;

    /// @brief 判断 id 是否在合法范围内。
    [[nodiscard]] bool contains_id(TokenId id) const;

    // ==================== token <-> id 映射 ====================

    /**
     * @brief token 转 id。
     * @return 若 token 不存在，通常返回 unk_token 对应 id（取决于实现）。
     */
    [[nodiscard]] TokenId token_to_id(const std::string& token) const;

    /**
     * @brief id 转 token。
     * @return 若 id 非法，返回内部空字符串引用（避免悬空引用）。
     */
    [[nodiscard]] const std::string& id_to_token(TokenId id) const;

    // ==================== 特殊 token 配置 ====================

    /// @brief 设置未知词 token（UNK）。
    void set_unk_token(const std::string& token);

    /// @brief 设置填充 token（PAD）。
    void set_pad_token(const std::string& token);

    /// @brief 设置句首 token（BOS）。
    void set_bos_token(const std::string& token);

    /// @brief 设置句尾 token（EOS）。
    void set_eos_token(const std::string& token);

    /// @brief 获取未知词 token 文本。
    [[nodiscard]] const std::string& unk_token() const;

    /// @brief 获取填充 token 文本。
    [[nodiscard]] const std::string& pad_token() const;

    /// @brief 获取句首 token 文本。
    [[nodiscard]] const std::string& bos_token() const;

    /// @brief 获取句尾 token 文本。
    [[nodiscard]] const std::string& eos_token() const;

    /// @brief 获取未知词 token 的 id。
    [[nodiscard]] TokenId unk_token_id() const;

    /// @brief 获取填充 token 的 id。
    [[nodiscard]] TokenId pad_token_id() const;

    /// @brief 获取句首 token 的 id。
    [[nodiscard]] TokenId bos_token_id() const;

    /// @brief 获取句尾 token 的 id。
    [[nodiscard]] TokenId eos_token_id() const;

    /// @brief 暴露完整词表（按 id 顺序存储）。
    [[nodiscard]] const std::vector<std::string>& vocabulary() const;

private:
    /// @brief 统一返回的静态空字符串，避免返回临时对象。
    static const std::string& empty_token();

    /// @brief 查询 token id，查不到时返回默认 id（通常是 UNK）。
    [[nodiscard]] TokenId find_token_id_or_default(const std::string& token) const;

    /// @brief 校验 id 是否落在 id_to_token_table_ 的范围内。
    [[nodiscard]] bool is_valid_id(TokenId id) const;

private:
    /// id -> token 映射表，下标即 token id。
    std::vector<std::string> id_to_token_table_;

    /// token -> id 映射表。
    std::unordered_map<std::string, TokenId> token_to_id_table_;

    /// 特殊 token 文本（是否存在于词表由实现阶段决定）。
    std::string unk_token_;
    std::string pad_token_;
    std::string bos_token_;
    std::string eos_token_;
};

} // namespace mini_llm::model

#endif //QWEN_CPP_EMBEDDING_H
