#pragma once

#include <string>

#include "backend/backend_adapter.h"

namespace mini_llm::backend {

class GgufLlamaCppBackend final : public BackendAdapter {
public:
    // TODO: 注入 llama.cpp 配置对象与资源管理器
    bool load_model(const std::string& model_path) override;
    bool create_context() override;
};

} // namespace mini_llm::backend
