cmake_minimum_required(VERSION 3.28)
project(cpp CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cpp main.cu)
set(CUDA_Toolkit_ROOT "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4")
set_target_properties(cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
