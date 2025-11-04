#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include "src/CPUMergeSort.h"
#include "src/sampling/ParallelCPU/MaximizedParallelCPUMergeSort.h"
#include "src/sampling/ParallelCPU/FullParallelCPUMergeSort.h"

using namespace std;

vector<int> generate_list(int n);
void sort_base_cpu(vector<int> list, int len);
void sort_max_parallel_cpu(vector<int> list, int len);
void sort_full_parallel_cpu(vector<int> list, int len);

int main()
{
    vector<int> list = generate_list(200'000);
    int len = list.size();

    sort_base_cpu(list, len);
    sort_max_parallel_cpu(list, len);
    sort_full_parallel_cpu(list, len);

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

void sort_base_cpu(vector<int> list, int len)
{
    CPUMergeSort merge_sort(list);

    auto start = chrono::high_resolution_clock::now();

    vector<int> sorted_list = merge_sort.sort(0, len-1);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "[Base CPU Merge Sort] Elapsed time: " << duration << " ms" << endl;

    is_array_sorted(sorted_list);
    cout<<endl;
}


void sort_max_parallel_cpu(vector<int> list, int len)
{
    MaximizedParallelCPUMergeSort merge_sort(list, 10000, 4);

    auto start = chrono::high_resolution_clock::now();

    vector<int> sorted_list = merge_sort.sort(0, len-1, 0);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "[Maximized Parallel CPU Merge Sort] Elapsed time: " << duration << " ms" << endl;

    is_array_sorted(sorted_list);
    cout<<endl;
}

void sort_full_parallel_cpu(vector<int> list, int len)
{
    FullParallelCPUMergeSort merge_sort(list, 4);

    auto start = chrono::high_resolution_clock::now();

    vector<int> sorted_list = merge_sort.sort(0, len-1, 0);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "[Full Parallel CPU Merge Sort] Elapsed time: " << duration << " ms" << endl;

    is_array_sorted(sorted_list);
    cout<<endl;
}
