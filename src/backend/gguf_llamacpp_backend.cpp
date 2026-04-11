#include "backend/gguf_llamacpp_backend.h"

namespace mini_llm::backend {

bool GgufLlamaCppBackend::load_model(const std::string& model_path) {
    // TODO: 调用 llama.cpp API 加载 GGUF 模型并处理错误
    (void)model_path;
    return false;
}

bool GgufLlamaCppBackend::create_context() {
    // TODO: 创建推理上下文并注入运行参数
    return false;
}

} // namespace mini_llm::backend
