#include <stdio.h>

void iKernel_cpu(float *A, float *B, float *C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // 示例用法
    int arraySize = 6;
    float arrayA[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    float arrayB[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
    float resultArray[arraySize];

    printf("数组 A：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayA[i]);
    }

    printf("\n数组 B：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayB[i]);
    }

    // 调用函数
    iKernel_cpu(arrayA, arrayB, resultArray, arraySize);

    printf("\n数组 C：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
