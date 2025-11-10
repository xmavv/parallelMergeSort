#include "CPUMergeSort.h"

CPUMergeSort::CPUMergeSort(vector<int>& list) {
    set_list(list);
}

void CPUMergeSort::set_list(vector<int>& list)
{
    this->list = list;
}

vector<int> CPUMergeSort::run(int left, int right)
{
    sort(left, right);
    return this->list;
}

void CPUMergeSort::sort(int left, int right)
{
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;
    sort(left, mid);
    sort(mid + 1, right);
    merge(left, mid, right);
}

void CPUMergeSort::merge(int left, int mid, int right){
    int left_list_len = mid - left + 1;
    int right_list_len = right - mid;
    vector<int> left_list(left_list_len);
    vector<int> right_list(right_list_len);

    //copy -> this is the additional capacity of the algorightm
    for (int i = 0; i < left_list_len; i++)
        left_list[i] = this->list[left + i];
    for (int j = 0; j < right_list_len; j++)
        right_list[j] = this->list[mid + 1 + j];

    int i = 0, j = 0;
    int k = left;
    //merging into list[left..right]
    while (i < left_list_len && j < right_list_len) {
        if (left_list[i] <= right_list[j]) {
            this->list[k] = left_list[i];
            i++;
        }
        else {
            this->list[k] = right_list[j];
            j++;
        }
        k++;
    }

    //copy the remaining elements of left_list, this is the case when one of (i, j) indexes is out of one of merging lists
    //this means that we can just take elements from other list and copy them, they are already sorted
    while (i < left_list_len) {
        this->list[k] = left_list[i];
        i++;
        k++;
    }
    //same as above, but for right_list
    while (j < right_list_len) {
        this->list[k] = right_list[j];
        j++;
        k++;
    }
}
