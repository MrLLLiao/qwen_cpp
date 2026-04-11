cmake_minimum_required(VERSION 3.20)
project(qwen_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

add_library(ops_core
        src/tensor.cpp
        src/ops/matmul.cpp
        src/ops/softmax.cpp
        src/ops/attention.cpp
)
target_include_directories(ops_core PUBLIC
        include
        include/ops
        include/model)

add_library(cache_core
        src/cache/KVCache.cpp
        src/cache/CacheAllocator.cpp
        src/cache/CacheManager.cpp
)
target_include_directories(cache_core PUBLIC
        include
        include/ops
)
target_link_libraries(cache_core PUBLIC ops_core)

add_library(engine_core
        src/engine/prefill.cpp
        src/engine/decode.cpp
)
target_include_directories(engine_core PUBLIC
        include
        include/engine
)
target_link_libraries(engine_core PUBLIC cache_core)

enable_testing()

function(add_unit_test TEST_NAME TEST_SOURCE TEST_LIB)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE tests)
    target_link_libraries(${TEST_NAME} PRIVATE ${TEST_LIB})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

add_unit_test(tensor-test tests/tensor_test.cpp ops_core)
add_unit_test(matmul-test tests/matmul_test.cpp ops_core)
add_unit_test(softmax-test tests/softmax_test.cpp ops_core)
add_unit_test(attention-test tests/attention_test.cpp ops_core)
add_unit_test(kvcache-test tests/KVCache_test.cpp cache_core)
add_unit_test(cache-allocator-test tests/cache_allocator_test.cpp cache_core)
add_unit_test(cache-manager-test tests/cache_manager_test.cpp cache_core)
add_unit_test(prefill-test tests/prefill_test.cpp engine_core)
