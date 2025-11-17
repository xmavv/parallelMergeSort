#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include "src/CPUMergeSort.h"
#include "src/parallelCPU/BothSidesParallelCPUMergeSort.h"
#include "src/ParallelCPU/LeftSideParallelCPUMergeSort.h"

using namespace std;

vector<int> generate_list(int n);
void sort_base_cpu(vector<int>& list, int len);
void sort_left_parallel_cpu(vector<int>& list, int len);
void sort_both_parallel_cpu(vector<int>& list, int len);

int main()
{
    vector<int> list = generate_list(200000);
    vector<int> list_copy = list;
    vector<int> list_copy2 = list;

    int len = list.size();

    sort_base_cpu(list, len);
//    sort_left_parallel_cpu(list_copy, len);
    sort_both_parallel_cpu(list_copy2, len);

    return 0;
}

vector<int> generate_list(int n)
{
    vector<int> list(n);

    mt19937 random(random_device{}());
    uniform_int_distribution<int> dist(0, n);

    generate(list.begin(), list.end(), [&]()
    {
        return dist(random);
    });

    return list;
}

void is_array_sorted(vector<int>& list)
{
    bool is_sorted_correctly = is_sorted(list.begin(), list.end());
    cout << "Sorted correctly: " << (is_sorted_correctly ? "Yes" : "No") << endl;
}

void sort_base_cpu(vector<int>& list, int len)
{
    CPUMergeSort merge_sort(list);

    auto start = chrono::high_resolution_clock::now();

    vector<int> sorted_list = merge_sort.run(0, len-1);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "[Base CPU Merge Sort] Elapsed time: " << duration << " ms" << endl;

    is_array_sorted(sorted_list);
    cout<<endl;
}


void sort_left_parallel_cpu(vector<int>& list, int len)
{
    LeftSideParallelCPUMergeSort merge_sort(list, 10000, 8);

    auto start = chrono::high_resolution_clock::now();

    vector<int> sorted_list = merge_sort.run(0, len-1, 0);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "[Left Side Parallel CPU Merge Sort] Elapsed time: " << duration << " ms" << endl;

    is_array_sorted(sorted_list);
    cout<<endl;
}

void sort_both_parallel_cpu(vector<int>& list, int len)
{
    BothSidesParallelCPUMergeSort merge_sort(list, 10000, 3);

    auto start = chrono::high_resolution_clock::now();

    vector<int> sorted_list = merge_sort.run(0, len-1, 0);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "[Both Sides Parallel CPU Merge Sort] Elapsed time: " << duration << " ms" << endl;

    is_array_sorted(sorted_list);
    cout<<endl;
}