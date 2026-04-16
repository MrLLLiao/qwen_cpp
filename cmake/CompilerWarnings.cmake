# 统一管理警告策略，避免使用全局 CMAKE_CXX_FLAGS
add_library(qwen_cpp_project_warnings INTERFACE)
add_library(qwen_cpp::project_warnings ALIAS qwen_cpp_project_warnings)

target_compile_options(qwen_cpp_project_warnings INTERFACE
    # MSVC
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
    # GCC/Clang
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)