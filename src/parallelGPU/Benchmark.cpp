//this is how it run on collab (change -arch yo yours)
//nvcc -O3 -arch=sm_75 -c mergesort_mergepath.cu -o mergesort_mergepath.o
//g++ -O3 -fopenmp -std=c++17 -c mergesort_cpu_parallel.cpp -o mergesort_cpu_parallel.o
//g++ -O3 -fopenmp -std=c++17 -c mergesort_cpu_parallel_recursive.cpp -o mergesort_cpu_parallel_recursive.o
//g++ -O3 -fopenmp -std=c++17 -c mergesort_cpu_parallel_recursive_omp.cpp -o mergesort_cpu_parallel_recursive_omp.o
//
//nvcc -O3 -arch=sm_75 benchmark.cpp \
//      mergesort_mergepath.o \
//      mergesort_cpu_parallel.o \
//      mergesort_cpu_parallel_recursive.o \
//      mergesort_cpu_parallel_recursive_omp.o \
//      -o benchmark -Xcompiler -fopenmp

//./benchmark

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

// ⬅ deklaracje WSZYSTKICH funkcji (muszą być zgodne z nazwami w plikach!)
void parallelMergeSortIterative(std::vector<int>& data);   // CPU iterative OpenMP
void parallelMergeSortRecCpp(std::vector<int>& data);      // CPU recursive C++ threads
void parallelMergeSortRecOmp(std::vector<int>& data);      // CPU recursive OpenMP
void parallelMergeSortMergePath(std::vector<int>& data);   // GPU merge-path

void stdSortWrapper(std::vector<int>& data) { std::sort(data.begin(), data.end()); }

// generator
std::vector<int> generate_list_benchmark(int n) {
    std::vector<int> list(n);
    std::mt19937 gen(123);  // deterministyczne
    std::uniform_int_distribution<int> dist(0, n);
    for (int &x : list) x = dist(gen);
    return list;
}

int main() {
    const int N = 10000000;
    std::vector<int> original = generate_list_benchmark(N);

    // warmup GPU
    {
        std::vector<int> warm = {1,2,3};
        parallelMergeSortMergePath(warm);
        cudaDeviceSynchronize();
    }

    // kopie danych
    std::vector<int> vecGPU       = original;
    std::vector<int> vecIterative = original;
    std::vector<int> vecRecCpp    = original;
    std::vector<int> vecRecOmp    = original;
    std::vector<int> vecStd       = original;

    // GPU
    auto g1 = std::chrono::high_resolution_clock::now();
    parallelMergeSortMergePath(vecGPU);
    cudaDeviceSynchronize();
    auto g2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU merge-path: "
              << std::chrono::duration<double, std::milli>(g2-g1).count() << " ms\n";

    // CPU iterative OMP
    auto i1 = std::chrono::high_resolution_clock::now();
    parallelMergeSortIterative(vecIterative);
    auto i2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU iterative OMP: "
              << std::chrono::duration<double, std::milli>(i2-i1).count() << " ms\n";

    // CPU recursive C++ threads
    auto c1 = std::chrono::high_resolution_clock::now();
    parallelMergeSortRecCpp(vecRecCpp);
    auto c2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU recursive C++ threads: "
              << std::chrono::duration<double, std::milli>(c2-c1).count() << " ms\n";

    // CPU recursive OMP
    auto o1 = std::chrono::high_resolution_clock::now();
    parallelMergeSortRecOmp(vecRecOmp);
    auto o2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU recursive OMP: "
              << std::chrono::duration<double, std::milli>(o2-o1).count() << " ms\n";

    // std::sort
    auto s1 = std::chrono::high_resolution_clock::now();
    stdSortWrapper(vecStd);
    auto s2 = std::chrono::high_resolution_clock::now();
    std::cout << "std::sort: "
              << std::chrono::duration<double, std::milli>(s2-s1).count() << " ms\n";

    // correctness
    bool ok = (vecGPU == vecIterative &&
               vecIterative == vecRecCpp &&
               vecRecCpp == vecRecOmp &&
               vecRecOmp == vecStd);

    std::cout << (ok ? "All sorted identically ✅\n" : "Mismatch ❌\n");
}
