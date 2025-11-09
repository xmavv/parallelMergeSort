#ifndef UNTITLED5_PARALLELCPUMERGESORT_H
#define UNTITLED5_PARALLELCPUMERGESORT_H

#include <vector>

using namespace std;

class MaximizedParallelCPUMergeSort
{
private:

    vector<int> list;
    int max_depth;
    int parallel_threshold;

    void sort_sequential(int left, int right);
    void merge(int left, int mid, int right);

public:

    MaximizedParallelCPUMergeSort(vector<int>& list, int parallel_threshold, int max_depth);

    void set_list(vector<int>& list);
    void set_parallel_threshold(int parallel_threshold);
    void set_max_depth(int max_depth);

    vector<int> sort(int left, int right, int depth);
};


#endif //UNTITLED5_PARALLELCPUMERGESORT_H
