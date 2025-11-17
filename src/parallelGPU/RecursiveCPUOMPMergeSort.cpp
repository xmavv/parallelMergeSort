%%writefile mergesort_cpu_parallel_recursive_omp.cpp
#include <vector>
#include <algorithm>
#include <omp.h>

static void mergeOmp(std::vector<int>& data,
                     std::vector<int>& temp,
                     int L, int M, int R)
{
    int i=L, j=M, k=L;
    while (i<M && j<R) temp[k++] = (data[i] <= data[j] ? data[i++] : data[j++]);
    while (i<M) temp[k++] = data[i++];
    while (j<R) temp[k++] = data[j++];
    for (int t=L; t<R; t++) data[t] = temp[t];
}

static void mergeSortRecOmp_impl(std::vector<int>& data,
                                 std::vector<int>& temp,
                                 int L, int R,
                                 int depth)
{
    if (R-L <= 1) return;
    int M = (L+R)/2;

    if (depth > 0) {
#pragma omp parallel sections
        {
#pragma omp section
            mergeSortRecOmp_impl(data, temp, L, M, depth-1);

#pragma omp section
            mergeSortRecOmp_impl(data, temp, M, R, depth-1);
        }
    } else {
        mergeSortRecOmp_impl(data, temp, L, M, 0);
        mergeSortRecOmp_impl(data, temp, M, R, 0);
    }

    mergeOmp(data, temp, L, M, R);
}

void parallelMergeSortRecOmp(std::vector<int>& data)
{
    if (data.size() <= 1) return;
    std::vector<int> temp(data.size());
    int depth = 2;   // bezpieczne na OMP
    mergeSortRecOmp_impl(data, temp, 0, data.size(), depth);
}
