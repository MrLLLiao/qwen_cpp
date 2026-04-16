# -----------------------------
# ops_core
# -----------------------------
add_library(ops_core
    tensor.cpp
    ops/matmul.cpp
    ops/softmax.cpp
    ops/attention.cpp
)

target_include_directories(ops_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/ops>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/model>
)

target_link_libraries(ops_core
    PRIVATE
        qwen_cpp::project_options
        qwen_cpp::project_warnings
)

# -----------------------------
# cache_core
# -----------------------------
add_library(cache_core
    cache/KVCache.cpp
    cache/CacheAllocator.cpp
    cache/CacheManager.cpp
)

target_include_directories(cache_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/ops>
)

target_link_libraries(cache_core
    PUBLIC
        ops_core
    PRIVATE
        qwen_cpp::project_options
        qwen_cpp::project_warnings
)

# -----------------------------
# engine_core
# -----------------------------
add_library(engine_core
    engine/prefill.cpp
    engine/decode.cpp
)

target_include_directories(engine_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/engine>
)

target_link_libraries(engine_core
    PUBLIC
        ops_core
        cache_core
    PRIVATE
        qwen_cpp::project_options
        qwen_cpp::project_warnings
)

# -----------------------------
# model_core
# -----------------------------
add_library(model_core
    model/attention.cpp
    model/embedding.cpp
    model/mlp.cpp
    model/model_config.cpp
    model/model_weights.cpp
    model/model.cpp
    model/rms_norm.cpp
    model/transformer_block.cpp
)

target_include_directories(model_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/model>
)

target_link_libraries(model_core
    PUBLIC
        ops_core
        cache_core
        engine_core
        simdjson::simdjson
    PRIVATE
        qwen_cpp::project_options
        qwen_cpp::project_warnings
)