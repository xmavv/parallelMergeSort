#ifndef UNTITLED5_CPUMERGESORT_H
#define UNTITLED5_CPUMERGESORT_H

#include <vector>

using namespace std;

class CPUMergeSort {
private:
    vector<int> list;
    void merge(int left, int mid, int right);
    void sort(int left, int right);

public:
    CPUMergeSort(vector<int>& list);
    void set_list(vector<int>& list);
    vector<int> run(int left, int right);
};


#endif //UNTITLED5_CPUMERGESORT_H
