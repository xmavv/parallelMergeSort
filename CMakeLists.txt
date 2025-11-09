cmake_minimum_required(VERSION 3.27)
project(untitled5)

#find_package(CUDAToolkit REQUIRED)
#toolchains in clion -> visualstudio for compile .cu files

set(CMAKE_CXX_STANDARD 17)

add_executable(untitled5 main.cpp
        src/CPUMergeSort.h
        src/CPUMergeSort.cpp
        src/CPUMergeSort.h
        src/parallelGPU/RecursiveGPUMergeSort.cu
        src/parallelCPU/MaximizedParallelCPUMergeSort.h
        src/parallelCPU/MaximizedParallelCPUMergeSort.cpp
        src/parallelCPU/FullParallelCPUMergeSort.cpp
        src/parallelCPU/FullParallelCPUMergeSort.h
        src/parallelGPU/IterativeGPUMergeSort.cu
)

#target_compile_options(GPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
#target_link_libraries(GPU PRIVATE
#        CUDA::cudart
#        CUDA::cublas)
