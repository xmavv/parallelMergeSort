#include "CPUMergeSort.h"

CPUMergeSort::CPUMergeSort(std::vector<int>& list) {
    set_list(list);
}

std::vector<int> CPUMergeSort::sort(int left, int right) {
    if (left >= right)
        return {};

    int mid = left + (right - left) / 2;
    sort(left, mid);
    sort(mid + 1, right);
    merge(left, mid, right);

    return list;
}

void CPUMergeSort::set_list(std::vector<int>& list) {
    this->list = list;
}

void CPUMergeSort::merge(int left, int mid, int right){
    int left_list_len = mid - left + 1;
    int right_list_len = right - mid;
    std::vector<int> left_list(left_list_len);
    std::vector<int> right_list(right_list_len);

    //copy -> this is the additional capacity of the algorightm
    for (int i = 0; i < left_list_len; i++)
        left_list[i] = list[left + i];
    for (int j = 0; j < right_list_len; j++)
        right_list[j] = list[mid + 1 + j];

    int i = 0, j = 0;
    int k = left;
    //merging into list[left..right]
    while (i < left_list_len && j < right_list_len) {
        if (left_list[i] <= right_list[j]) {
            list[k] = left_list[i];
            i++;
        }
        else {
            list[k] = right_list[j];
            j++;
        }
        k++;
    }

    //copy the remaining elements of left_list, this is the case when one of (i, j) indexes is out of one of merging lists
    //this means that we can just take elements from other list and copy them, they are already sorted
    while (i < left_list_len) {
        list[k] = left_list[i];
        i++;
        k++;
    }
    //same as above, but for right_list
    while (j < right_list_len) {
        list[k] = right_list[j];
        j++;
        k++;
    }
}