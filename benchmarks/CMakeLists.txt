add_executable(qwen_decode_bench
    qwen_decode_bench.cpp
)

target_link_libraries(qwen_decode_bench
    PRIVATE
        qwen_cpp::project_options
        qwen_cpp::project_warnings
        ops_core
        cache_core
)

if(MINGW)
    get_filename_component(QWEN_CPP_CXX_RUNTIME_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    foreach(QWEN_CPP_RUNTIME_DLL libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll)
        if(EXISTS "${QWEN_CPP_CXX_RUNTIME_DIR}/${QWEN_CPP_RUNTIME_DLL}")
            add_custom_command(TARGET qwen_decode_bench POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${QWEN_CPP_CXX_RUNTIME_DIR}/${QWEN_CPP_RUNTIME_DLL}"
                    "$<TARGET_FILE_DIR:qwen_decode_bench>/${QWEN_CPP_RUNTIME_DLL}"
            )
        endif()
    endforeach()
endif()
