#include <stdio.h>

void test_cpu(float *input, const int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (tid == 0) {
            input[tid] = 0;
        }
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    float array[arraySize];

    printf("原始数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", array[i]);
    }

    // 调用函数
    test_cpu(array, arraySize);

    printf("\n测试后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", array[i]);
    }

    return 0;
}
