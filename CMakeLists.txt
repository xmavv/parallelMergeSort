cmake_minimum_required(VERSION 3.27)
project(untitled5)

set(CMAKE_CXX_STANDARD 17)

add_executable(untitled5 main.cpp
        src/CPUMergeSort.h
        src/CPUMergeSort.cpp
        src/sampling/ParallelGPUMergeSort.h
        src/CPUMergeSort.h
        src/sampling/ParallelGPUMergeSort.cpp
        src/sampling/ParallelCPU/MaximizedParallelCPUMergeSort.h
        src/sampling/ParallelCPU/MaximizedParallelCPUMergeSort.cpp
        src/sampling/ParallelCPU/FullParallelCPUMergeSort.cpp
        src/sampling/ParallelCPU/FullParallelCPUMergeSort.h)
