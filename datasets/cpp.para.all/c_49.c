#include <stdio.h>

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY) {
    for (int i = 0; i < N; ++i) {
        Y[i * INCY] = X[i * INCX];
    }
}

int main() {
    // 示例用法
    int arraySize = 4;
    float sourceArray[] = {1.1, 2.2, 3.3, 4.4};
    float destinationArray[arraySize];
    int INCX = 1;
    int INCY = 2;

    printf("源数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", sourceArray[i]);
    }

    // 调用函数
    copy_cpu(arraySize, sourceArray, INCX, destinationArray, INCY);

    printf("\n复制后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", destinationArray[i]);
    }

    return 0;
}
