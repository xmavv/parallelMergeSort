#include <vector>
#include <algorithm>
#include <omp.h>

void parallelMergeSortIterative(std::vector<int>& data)
{
    int n = data.size();
    if (n <= 1) return;

    std::vector<int> temp(n);

    for (int width = 1; width < n; width *= 2) {
        int pairs = (n + 2*width - 1) / (2*width);

#pragma omp parallel for schedule(static)
        for (int p = 0; p < pairs; p++) {
            int L = p * 2 * width;
            int M = std::min(L + width, n);
            int R = std::min(L + 2 * width, n);

            int i=L, j=M, k=L;
            while (i<M && j<R) temp[k++] = (data[i] <= data[j] ? data[i++] : data[j++]);
            while (i<M) temp[k++] = data[i++];
            while (j<R) temp[k++] = data[j++];

            for (int t=L; t<R; t++) data[t] = temp[t];
        }
    }
}
