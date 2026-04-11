#include "runtime/inference_session.h"

namespace mini_llm::runtime {

InferenceSession::InferenceSession(std::string session_id)
    : session_id_(std::move(session_id)) {
    // TODO: 初始化 session 级上下文资源
}

const std::string& InferenceSession::id() const {
    return session_id_;
}

} // namespace mini_llm::runtime
