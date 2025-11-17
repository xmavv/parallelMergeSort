#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << static_cast<int>(err)                    \
                      << " \"" << cudaGetErrorString(err) << "\"\n";          \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ---------- warmup kernel ----------
__global__ void warmupKernel() { /* no-op */ }

// ---------- shared-memory merge kernel ----------
extern __shared__ int sdata[]; // dynamic shared memory

__global__ void mergeSharedKernel(const int* d_src, int* d_dst, int width, int size) {
    int tidInGrid = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = static_cast<long long>(tidInGrid) * (2LL * width);
    if (start >= size) return;

    int s = static_cast<int>(start);
    int mid = s + width;
    int end = s + 2 * width;
    if (mid >= size) {
        for (int t = s; t < size; ++t) d_dst[t] = d_src[t];
        return;
    }
    if (end > size) end = size;

    int elems = end - s; // <= 2*width

    // load segment into shared memory
    for (int i = threadIdx.x; i < elems; i += blockDim.x) {
        sdata[i] = d_src[s + i];
    }
    __syncthreads();

    int left = 0;
    int right = (mid - s);

    int perThread = (elems + blockDim.x - 1) / blockDim.x;
    int outStart = threadIdx.x * perThread;
    int outEnd = outStart + perThread;
    if (outStart >= elems) return;
    if (outEnd > elems) outEnd = elems;

    // Advance pointers i,j to outStart
    int i = left;
    int j = right;
    int k = 0;
    while (k < outStart && i < right && j < elems) {
        if (sdata[i] <= sdata[j]) ++i; else ++j;
        ++k;
    }
    while (k < outStart && i < right) { ++i; ++k; }
    while (k < outStart && j < elems) { ++j; ++k; }

    int writeIdx = s + outStart;
    for (int pos = outStart; pos < outEnd; ++pos) {
        int val;
        if (i < right && (j >= elems || sdata[i] <= sdata[j])) { val = sdata[i++]; }
        else if (j < elems) { val = sdata[j++]; }
        else break;
        d_dst[writeIdx++] = val;
    }
    // done
}

// ---------- global-memory merge kernel (fallback) ----------
__global__ void mergeGlobalKernel(const int* d_src, int* d_dst, int width, int size) {
    int tidInGrid = blockIdx.x * blockDim.x + threadIdx.x;
    long long startLL = static_cast<long long>(tidInGrid) * (2LL * width);
    if (startLL >= size) return;
    int start = static_cast<int>(startLL);

    int mid = start + width;
    if (mid >= size) {
        for (int t = start; t < size; ++t) d_dst[t] = d_src[t];
        return;
    }
    int end = start + 2 * width;
    if (end > size) end = size;

    int i = start, j = mid, k = start;
    while (i < mid && j < end) {
        int a = d_src[i];
        int b = d_src[j];
        if (a <= b) d_dst[k++] = a, ++i;
        else d_dst[k++] = b, ++j;
    }
    while (i < mid) d_dst[k++] = d_src[i++];
    while (j < end) d_dst[k++] = d_src[j++];
}

// ---------- host: parallel merge sort z ping-pong buffer i bezpiecznym shared ----------
void parallelMergeSortOptimized(std::vector<int>& data) {
    int n = static_cast<int>(data.size());
    if (n <= 1) return;

    // device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int maxThreadsPerBlock = prop.maxThreadsPerBlock; // e.g. 1024
    size_t maxSharedBytesPerBlock = prop.sharedMemPerBlock; // bytes

    // choose threads per block (safe)
    int threadsPerBlock = 128;
    if (threadsPerBlock > maxThreadsPerBlock) threadsPerBlock = maxThreadsPerBlock;

    // compute safe max shared ints
    int MAX_SHARED_INTS = static_cast<int>(maxSharedBytesPerBlock / sizeof(int));
    if (MAX_SHARED_INTS < 1) MAX_SHARED_INTS = 1;

    int *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, data.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // warm up
    warmupKernel<<<1,1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_src = d_a, *d_dst = d_b;

    for (int width = 1; width < n; width *= 2) {
        long long pairSizeLL = 2LL * width;
        if (pairSizeLL <= 0) break; // overflow guard

        int pairSize = (pairSizeLL > INT_MAX) ? INT_MAX : static_cast<int>(pairSizeLL);
        int numPairs = static_cast<int>((n + pairSize - 1) / pairSize); // ceil
        long long totalThreadsLL = static_cast<long long>(numPairs) * threadsPerBlock;
        if (totalThreadsLL <= 0) break; // overflow guard

        // compute numBlocks safely
        long long numBlocksLL = (totalThreadsLL + threadsPerBlock - 1) / threadsPerBlock;
        if (numBlocksLL > static_cast<long long>(prop.maxGridSize[0])) {
            // In practice this is extremely unlikely for real sizes (maxGridSize[0] is large),
            // but check to avoid overflow: reduce threadsPerBlock if necessary.
            long long neededThreads = totalThreadsLL;
            long long maxBlocks = prop.maxGridSize[0];
            long long maxThreadsTotal = maxBlocks * threadsPerBlock;
            if (neededThreads > maxBlocks * static_cast<long long>(maxThreadsPerBlock)) {
                std::cerr << "Requested too many threads/blocks for device limits.\n";
                CUDA_CHECK(cudaFree(d_a));
                CUDA_CHECK(cudaFree(d_b));
                std::exit(EXIT_FAILURE);
            }
            // else cap numBlocks to maxGridSize[0] (should not happen for moderate n)
            numBlocksLL = maxBlocks;
        }
        int numBlocks = static_cast<int>(numBlocksLL);

        // Decide whether to use shared memory: check against device limit
        size_t sharedElems = 0;
        size_t sharedBytes = 0;
        bool useShared = false;
        // We only consider using shared when pairSize fits in device shared memory
        if (pairSize <= MAX_SHARED_INTS) {
            sharedElems = static_cast<size_t>(pairSize);
            sharedBytes = sharedElems * sizeof(int);
            if (sharedBytes <= maxSharedBytesPerBlock) {
                useShared = true;
            }
        }

        if (useShared) {
            // Launch shared kernel with sharedBytes (per-block dynamic shared memory)
            mergeSharedKernel<<<numBlocks, threadsPerBlock, sharedBytes>>>(d_src, d_dst, width, n);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Kernel launch (shared) failed: " << cudaGetErrorString(err) << "\n";
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaFree(d_a));
                CUDA_CHECK(cudaFree(d_b));
                std::exit(EXIT_FAILURE);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            // Launch global-memory fallback (no dynamic shared mem)
            mergeGlobalKernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, n);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Kernel launch (global) failed: " << cudaGetErrorString(err) << "\n";
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaFree(d_a));
                CUDA_CHECK(cudaFree(d_b));
                std::exit(EXIT_FAILURE);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // swap buffers
        std::swap(d_src, d_dst);
    }

    // copy result back
    CUDA_CHECK(cudaMemcpy(data.data(), d_src, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

// ---------- helper: generate random list ----------
std::vector<int> generate_list(int n) {
    std::vector<int> list(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, n);
    std::generate(list.begin(), list.end(), [&]() { return dist(gen); });
    return list;
}

// ---------- main ----------
int main() {
    const int N = 200000;
    std::vector<int> vec = generate_list(N);
    std::vector<int> vec_cpu = vec;

    std::cout << "Rozmiar: " << N << " elementów\n";

    // GPU timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent));
    parallelMergeSortOptimized(vec);
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float gpuMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, startEvent, stopEvent));
    std::cout << "GPU time (merge, optimized safe): " << gpuMs << " ms\n";

    // CPU timing (std::sort)
    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::sort(vec_cpu.begin(), vec_cpu.end());
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "CPU time (std::sort): " << cpuMs << " ms\n";

    // correctness
    bool ok = (vec == vec_cpu);
    std::cout << "Sorted identical? " << (ok ? "YES" : "NO") << "\n";

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    return ok ? 0 : 1;
}

