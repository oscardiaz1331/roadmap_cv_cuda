cmake_minimum_required(VERSION 3.18)

project(vector_add LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(vector_add
    "CUDA/Fundamentals/Architecture and Memory Model/vector_add.cu"
)

target_compile_options(vector_add
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /W4>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)


set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)

find_package(CUDAToolkit REQUIRED)