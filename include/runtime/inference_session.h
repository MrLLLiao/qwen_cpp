#pragma once

#include <string>

namespace mini_llm::runtime {

class InferenceSession {
public:
    // TODO: 绑定 session 与 backend context 生命周期
    explicit InferenceSession(std::string session_id);

    // TODO: 增加上下文复用与清理策略
    const std::string& id() const;

private:
    std::string session_id_;
};

} // namespace mini_llm::runtime
