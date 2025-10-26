#ifndef UNTITLED5_CPUMERGESORT_H
#define UNTITLED5_CPUMERGESORT_H

#include <vector>

class CPUMergeSort {
private:
    std::vector<int> list;
    void merge(int left, int mid, int right);

public:
    CPUMergeSort(std::vector<int>& list);
    void set_list(std::vector<int>& list);
    std::vector<int> sort(int left, int right);
};


#endif //UNTITLED5_CPUMERGESORT_H
