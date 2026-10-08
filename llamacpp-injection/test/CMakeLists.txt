cmake_minimum_required(VERSION 3.14)
project(llm_injection_tests CXX)

# Unit tests for the ggml-free core of the llm-injection engine.
#   cmake -S llamacpp-injection/test -B build/engine-tests
#   cmake --build build/engine-tests --config Release
#   ctest --test-dir build/engine-tests -C Release --output-on-failure

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/../../vendor/llama.cpp" CACHE PATH "llama.cpp checkout (headers only)")

add_executable(test-llm-injection
    test-llm-injection.cpp
    ../src/llm-injection.cpp)

target_include_directories(test-llm-injection PRIVATE
    ../src
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include)

if (MSVC)
    target_compile_definitions(test-llm-injection PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

enable_testing()
add_test(NAME test-llm-injection COMMAND test-llm-injection)
