#pragma once

#include <string>

namespace mini_llm::backend {

class BackendAdapter {
public:
    virtual ~BackendAdapter() = default;

    // TODO: 支持多后端统一加载接口
    virtual bool load_model(const std::string& model_path) = 0;

    // TODO: 提供 prefill/decode 统一调用接口
    virtual bool create_context() = 0;
};

} // namespace mini_llm::backend
