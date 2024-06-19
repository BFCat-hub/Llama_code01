#include <stdio.h>

void PSIfill_cpu(float *array, int conv_length, int n) {
    for (int i = 0; i < n; i++) {
        array[i] = array[i % conv_length];
    }
}

int main() {
    // 示例用法
    int arraySize = 8;
    float inputArray[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
    int convLength = 3;

    printf("原始数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", inputArray[i]);
    }

    // 调用函数
    PSIfill_cpu(inputArray, convLength, arraySize);

    printf("\nPSI 填充后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", inputArray[i]);
    }

    return 0;
}
