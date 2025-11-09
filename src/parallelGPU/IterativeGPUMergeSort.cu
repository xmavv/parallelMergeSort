#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

//iterative approach is more cuda way
//recursion is not really parallelism

//kernel does not have access to the class context e.g. this
__global__ void mergeKernel(int* d_data, int* d_temp, int width, int size) {
    //calculate index based on previously set number of blocks and threads per block
    //in our case blockDim.x = 128
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //we merge two blocks with "width" size
    int start = index * (2 * width);
    int mid = start + width;
    int end = min(start + 2 * width, size);

    if (mid >= end) return;

    int i = start, j = mid, k = start;

    while (i < mid && j < end) {
        if (d_data[i] < d_data[j])
            d_temp[k++] = d_data[i++];
        else
            d_temp[k++] = d_data[j++];
    }

    while (i < mid) d_temp[k++] = d_data[i++];
    while (j < end) d_temp[k++] = d_data[j++];

    for (int t = start; t < end; t++)
        d_data[t] = d_temp[t];
}

void parallelMergeSort(std::vector<int>& data) {
    int size = data.size();
    int* d_data;
    int* d_temp;

    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMalloc((void**)&d_temp, size * sizeof(int));

    cudaMemcpy(d_data, data.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;

    for (int width = 1; width < size; width *= 2) {
        //ceil division
        int numPairs = (size + 2 * width - 1) / (2 * width);
        int numBlocks = (numPairs + threadsPerBlock - 1) / threadsPerBlock;
        mergeKernel<<<numBlocks, threadsPerBlock>>>(d_data, d_temp, width, size);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(data.data(), d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_temp);

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
    std::vector<int> vec = generate_list(200000);

    std::cout << "przed sortowaniem:\n";
    for (int v : vec) std::cout << v << " ";
    std::cout << "\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    parallelMergeSort(vec);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

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
    return 0;
}
