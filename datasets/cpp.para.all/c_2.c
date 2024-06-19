#include <stdio.h>

void get_ev(double *old_arr, double *new_arr, int size) {
    int tid;
    for (tid = 0; tid < size; tid++) {
        new_arr[tid] = old_arr[tid];
    }
}

int main() {
    // 示例用法
    double old_array[] = {1.5, 2.3, 3.7, 4.2, 5.8};
    int numElements = sizeof(old_array) / sizeof(old_array[0]);

    printf("原始数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%.2f ", old_array[i]);
    }

    // 新数组初始化
    double new_array[numElements];

    // 调用函数
    get_ev(old_array, new_array, numElements);

    printf("\n复制后的数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%.2f ", new_array[i]);
    }

    return 0;
}
