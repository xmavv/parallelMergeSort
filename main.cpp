#include <iostream>
#include <vector>
#include "src/CPUMergeSort.h"

int main() {
    std::vector<int> list = { 52, 1, 62, 612, 6, 6231, 21, 2156, 261 };
    CPUMergeSort merge_sort(list);
    int len = list.size();
    std::vector<int> sorted_list = merge_sort.sort(0, len-1);

    for (int x: sorted_list) {
        std::cout << x << " ";
    }

    return 0;
}
