# 工程级通用选项（不污染全局变量）
add_library(qwen_cpp_project_options INTERFACE)
add_library(qwen_cpp::project_options ALIAS qwen_cpp_project_options)

# 以 target 为中心声明 C++ 标准要求
target_compile_features(qwen_cpp_project_options INTERFACE cxx_std_23)

# 可选：导出 compile_commands.json，便于 clangd / 静态分析工具
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json" FORCE)