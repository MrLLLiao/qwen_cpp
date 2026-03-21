cmake_minimum_required(VERSION 4.1)
project(qwen_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

add_library(ops_core
        src/tensor.cpp
        src/ops/matmul.cpp
)
target_include_directories(ops_core PUBLIC include)

add_executable(tensor-test tests/tensor_test.cpp)
target_include_directories(tensor-test PRIVATE tests)
target_link_libraries(tensor-test PRIVATE ops_core)

add_executable(matmul-test tests/matmul_test.cpp)
target_include_directories(matmul-test PRIVATE test)
target_link_libraries(matmul-test PRIVATE ops_core)

