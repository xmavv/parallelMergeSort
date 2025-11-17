%%writefile mergesort_cpu_parallel_recursive.cpp
#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>

static void mergeRecCpp(std::vector<int>& data,
                        std::vector<int>& temp,
                        int L, int M, int R)
{
    int i=L, j=M, k=L;
    while (i<M && j<R) temp[k++] = (data[i] <= data[j] ? data[i++] : data[j++]);
    while (i<M) temp[k++] = data[i++];
    while (j<R) temp[k++] = data[j++];
    for (int t=L; t<R; t++) data[t] = temp[t];
}

static void mergeSortRecCpp(std::vector<int>& data,
                            std::vector<int>& temp,
                            int L, int R,
                            int depth)
{
    if (R-L <= 1) return;
    int M = (L+R)/2;

    if (depth > 0) {
        std::thread t1(mergeSortRecCpp, std::ref(data), std::ref(temp), L, M, depth-1);
        std::thread t2(mergeSortRecCpp, std::ref(data), std::ref(temp), M, R, depth-1);
        t1.join(); t2.join();
    } else {
        mergeSortRecCpp(data, temp, L, M, 0);
        mergeSortRecCpp(data, temp, M, R, 0);
    }

    mergeRecCpp(data, temp, L, M, R);
}

void parallelMergeSortRecCpp(std::vector<int>& data)
{
    if (data.size() <= 1) return;
    std::vector<int> temp(data.size());
    int depth = std::log2(std::thread::hardware_concurrency());
    mergeSortRecCpp(data, temp, 0, data.size(), depth);
}
