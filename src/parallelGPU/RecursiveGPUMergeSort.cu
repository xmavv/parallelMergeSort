//might add cudaMemAdvice, cudaMemPrefetchAsync for better performance jumping between cpu and gpu
//shared memory would also be a performance boost
#include <iostream>
#include <cuda_runtime.h> //runtime api like cudaMalloc, cudaMemcpy, etc
//cudaMalloc - reserve memory in DRAM GPU, VRAM memory global for all of the threads
//cudaMemcpy - copies data from RAM CPU to global GPU
//global memory - for all blocks and threads
//there is also shared memory - only for threads in the same block
//local - one thread
//shared memory is used in thrust::sort
#include <chrono>
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <random>

//kernel does not have access to the class context e.g. this
//this kernel do mostly the same work as cpu recursive implementation
__global__ void mergeKernel(int* arr, int* temp, long long left, long long mid, long long right) {
    long long i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (long long i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

void parallelMergeSort(int* arr, int* temp, long long  left, long long  right) {
    long long  mid;
    if (left < right) {
        mid = left + (right - left) / 2;
        parallelMergeSort(arr, temp, left, mid);
        parallelMergeSort(arr, temp, mid + 1, right);

        //single thread here to do whoole merge - not efficient
        //<<<1 is number of thread blocks
        //1>>> is number of threads per thread block
        mergeKernel<<<1, 1>>>(arr, temp, left, mid, right);
        //<<<numBlocks, blockSize>>>
        //total threads are numBlocks * blockSize, can be arranged in 1D, 2D or 3D grid
        cudaDeviceSynchronize();
    }
}

std::vector<int> generate_list(int n) {
    std::vector<int> list(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, n);

    std::generate(list.begin(), list.end(), [&]() {
        return dist(gen);
    });

    return list;
}

int main() {
    int n = 200000;
    std::vector<int> vec = generate_list(n);

    std::cout << "przed sortowaniem:\n";
    for (int v : vec) std::cout << v << " ";
    std::cout << "\n";

    //needs to be there in case of recursive implementation
    int *d_data, *d_temp;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));
    cudaMemcpy(d_data, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    parallelMergeSort(d_data, d_temp, 0, n - 1);;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(vec.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "po sortowaniu:\n";
    for (int v : vec) std::cout << v << " ";
    std::cout << "\n";
    std::cout << "czas: " << milliseconds << " ms\n";

    if (std::is_sorted(vec.begin(), vec.end()))
        std::cout << "gitara\n";
    else
        std::cout << "no nie posortowane\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_temp);
    return 0;
}