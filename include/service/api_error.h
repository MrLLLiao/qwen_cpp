#pragma once

namespace mini_llm::service {

// TODO: 定义 API 错误码与 HTTP 映射规则
enum class ApiErrorCode {
    kInvalidArgument,
    kModelNotLoaded,
    kContextOverflow,
    kInternal,
};

} // namespace mini_llm::service
