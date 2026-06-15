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
    get_filename_component(QWEN_CPP_CXX_RUNTIME_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${QWEN_CPP_CXX_RUNTIME_DIR}"
    )
endfunction()

add_unit_test(tensor-test tensor_test.cpp ops_core)
set_tests_properties(tensor-test PROPERTIES LABELS "unit;tensor")
add_unit_test(matmul-test matmul_test.cpp ops_core)
set_tests_properties(matmul-test PROPERTIES LABELS "unit;ops")
add_unit_test(softmax-test softmax_test.cpp ops_core)
set_tests_properties(softmax-test PROPERTIES LABELS "unit;ops")
add_unit_test(attention-test attention_test.cpp ops_core)
set_tests_properties(attention-test PROPERTIES LABELS "unit;ops")

add_unit_test(kvcache-test KVCache_test.cpp cache_core)
set_tests_properties(kvcache-test PROPERTIES LABELS "unit;cache")
add_unit_test(cache-allocator-test cache_allocator_test.cpp cache_core)
set_tests_properties(cache-allocator-test PROPERTIES LABELS "unit;cache")
add_unit_test(cache-manager-test cache_manager_test.cpp cache_core)
set_tests_properties(cache-manager-test PROPERTIES LABELS "unit;cache")

add_unit_test(prefill-test prefill_test.cpp engine_core)
set_tests_properties(prefill-test PROPERTIES LABELS "integration;engine;cache")
add_unit_test(decode-test decode_test.cpp engine_core)
set_tests_properties(decode-test PROPERTIES LABELS "integration;engine;cache")

add_unit_test(embedding-test embedding_test.cpp model_core)
set_tests_properties(embedding-test PROPERTIES LABELS "unit;model")
add_unit_test(model-test model_test.cpp model_core)
set_tests_properties(model-test PROPERTIES LABELS "unit;model")
