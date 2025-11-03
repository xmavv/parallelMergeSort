#include "FullParallelCPUMergeSort.h"

#include <thread>

FullParallelCPUMergeSort::FullParallelCPUMergeSort(vector<int>& list, int max_depth)
{
    set_list(list);
    set_max_depth(max_depth);
}

void FullParallelCPUMergeSort::set_list(vector<int>& list)
{
    this->list = list;
}

void FullParallelCPUMergeSort::set_max_depth(int max_depth)
{
    this->max_depth = max_depth;
}

vector<int> FullParallelCPUMergeSort::sort(int left, int right, int depth)
{
    if (left >= right) return {};

    int mid = left + (right - left) / 2;

    // Spawn new thread for one half if recursion depth allows
    if (depth < this->max_depth)
    {
        thread left_thread([=, this]() {
            this->sort(left, mid, depth + 1);
        });

        sort(mid + 1, right, depth + 1);
        left_thread.join();
    }
    else
    {
        sort(left, mid, depth + 1);
        sort(mid + 1, right, depth + 1);
    }

    merge(left, mid, right);

    return list;
}

void FullParallelCPUMergeSort::merge(int left, int mid, int right)
{
    int left_list_len = mid - left + 1;
    int right_list_len = right - mid;

    vector<int> left_list(left_list_len);
    vector<int> right_list(right_list_len);

    for (int i = 0; i < left_list_len; i++)
        left_list[i] = list[left + i];

    for (int j = 0; j < right_list_len; j++)
        right_list[j] = list[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < left_list_len && j < right_list_len)
    {
        if (left_list[i] <= right_list[j])
        {
            list[k++] = left_list[i++];
        }
        else
        {
            list[k++] = right_list[j++];
        }
    }

    while (i < left_list_len)
        list[k++] = left_list[i++];

    while (j < right_list_len)
        list[k++] = right_list[j++];
}

