#include <stdio.h>

void incKernel(int *g_out, int *g_in, int N, int inner_reps) {
    for (int idx = 0; idx < N; idx++) {
        for (int i = 0; i < inner_reps; ++i) {
            g_out[idx] = g_in[idx] + 1;
        }
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    int inputArray[] = {1, 2, 3, 4, 5};
    int outputArray[arraySize];

    printf("输入数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", inputArray[i]);
    }

    // 调用函数
    incKernel(outputArray, inputArray, arraySize, 3);

    printf("\n计算后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", outputArray[i]);
    }

    return 0;
}
