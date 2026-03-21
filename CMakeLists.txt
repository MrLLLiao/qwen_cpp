cmake_minimum_required(VERSION 4.1)
project(qwen_cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
add_executable(inference main.cpp)
add_executable(inference-test tests/matmul_test.cpp)
target_include_directories(inference PRIVATE include)
target_include_directories(inference-test PRIVATE include tests)
