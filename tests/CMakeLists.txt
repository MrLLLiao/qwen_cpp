function(add_unit_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
    target_link_libraries(${TEST_NAME}
        PRIVATE
            qwen_cpp::project_options
            qwen_cpp::project_warnings
            ${ARGN}
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

add_unit_test(tensor-test tensor_test.cpp ops_core)
add_unit_test(matmul-test matmul_test.cpp ops_core)
add_unit_test(softmax-test softmax_test.cpp ops_core)
add_unit_test(attention-test attention_test.cpp ops_core)

add_unit_test(kvcache-test KVCache_test.cpp cache_core)
add_unit_test(cache-allocator-test cache_allocator_test.cpp cache_core)
add_unit_test(cache-manager-test cache_manager_test.cpp cache_core)

add_unit_test(prefill-test prefill_test.cpp engine_core)
add_unit_test(decode-test decode_test.cpp engine_core)

add_unit_test(embedding-test embedding_test.cpp model_core)
