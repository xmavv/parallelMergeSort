#include "LeftSideParallelCPUMergeSort.h"

#include <thread>

LeftSideParallelCPUMergeSort::LeftSideParallelCPUMergeSort(vector<int>& list, int parallel_threshold, int max_depth)
{
    set_list(list);
    set_parallel_threshold(parallel_threshold);
    set_max_depth(max_depth);
}

void LeftSideParallelCPUMergeSort::set_list(vector<int>& list)
{
    this->list = list;
}

void LeftSideParallelCPUMergeSort::set_parallel_threshold(int parallel_threshold)
{
    this->parallel_threshold = parallel_threshold;
}

void LeftSideParallelCPUMergeSort::set_max_depth(int max_depth)
{
    this->max_depth = max_depth;
}

vector<int> LeftSideParallelCPUMergeSort::run(int left, int right, int depth)
{
    sort_parallel(left, right, depth);
    return this->list;
}

void LeftSideParallelCPUMergeSort::sort_parallel(int left, int right, int depth)
{
    if (left >= right) return;

    if ((right - left) < this->parallel_threshold || depth >= this->max_depth)
    {
        sort_sequential(left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    thread left_thread([=, this]() {
        this->sort_parallel(left, mid, depth + 1);
    });

    sort_parallel(mid + 1, right, depth + 1);
    left_thread.join();

    merge(left, mid, right);
}

void LeftSideParallelCPUMergeSort::sort_sequential(int left, int right)
{
    if (left >= right) return;

    int mid = left + (right - left) / 2;

    sort_sequential(left, mid);
    sort_sequential(mid + 1, right);
    merge(left, mid, right);
}

void LeftSideParallelCPUMergeSort::merge(int left, int mid, int right)
{
    int left_list_len = mid - left + 1;
    int right_list_len = right - mid;

    vector<int> left_list(left_list_len);
    vector<int> right_list(right_list_len);

    for (int i = 0; i < left_list_len; i++)
        left_list[i] = this->list[left + i];

    for (int j = 0; j < right_list_len; j++)
        right_list[j] = this->list[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < left_list_len && j < right_list_len)
    {
        if (left_list[i] <= right_list[j])
        {
            this->list[k++] = left_list[i++];
        }
        else
        {
            this->list[k++] = right_list[j++];
        }
    }

    while (i < left_list_len)
        this->list[k++] = left_list[i++];

    while (j < right_list_len)
        this->list[k++] = right_list[j++];
}
