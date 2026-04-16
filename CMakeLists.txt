cmake_minimum_required(VERSION 3.20)

project(qwen_cpp
    VERSION 0.1.0
    DESCRIPTION "qwen_cpp core libraries and tests"
    LANGUAGES CXX
)

# 标准 CTest 开关：BUILD_TESTING
include(CTest)

# 构建行为开关（便于 CI / 本地按需裁剪）
option(QWEN_CPP_USE_BUNDLED_SIMDJSON "Use bundled simdjson from third_party/simdjson" ON)

# 工程级配置模块
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectOptions)      # qwen_cpp::project_options
include(CompilerWarnings)    # qwen_cpp::project_warnings

# 第三方依赖：默认使用子模块，保留切换到系统包的能力
if(QWEN_CPP_USE_BUNDLED_SIMDJSON)
    add_subdirectory(third_party/simdjson EXCLUDE_FROM_ALL)
else()
    find_package(simdjson CONFIG REQUIRED)
endif()

# 项目源码目标
add_subdirectory(src)

# 测试（受 BUILD_TESTING 控制）
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()