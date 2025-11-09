#ifndef UNTITLED5_FULLPRALLELCPUMERGESORT_H
#define UNTITLED5_FULLPRALLELCPUMERGESORT_H

#include <vector>

using namespace std;

class FullParallelCPUMergeSort
{
private:

    vector<int> list;
    int max_depth;

    void merge(int left, int mid, int right);

public:

    FullParallelCPUMergeSort(vector<int>& list, int max_depth);

    void set_list(vector<int>& list);
    void set_max_depth(int max_depth);

    vector<int> sort(int left, int right, int depth);
};


#endif //UNTITLED5_FULLPRALLELCPUMERGESORT_H