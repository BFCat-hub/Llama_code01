#include <stdio.h>

void logistic_cpu(unsigned int n, float a, float *x, float *z) {
    for (unsigned int myId = 0; myId < n; myId++) {
        z[myId] = a * x[myId] * (1 - x[myId]);
    }
}

int main() {
    // 示例用法
    unsigned int arraySize = 5;
    float inputArray[] = {0.2, 0.5, 0.7, 0.3, 0.8};
    float resultArray[arraySize];
    float a = 2.0;

    printf("原始数组：");
    for (unsigned int i = 0; i < arraySize; i++) {
        printf("%.2f ", inputArray[i]);
    }

    // 调用函数
    logistic_cpu(arraySize, a, inputArray, resultArray);

    printf("\n计算后的数组：");
    for (unsigned int i = 0; i < arraySize; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
