#include <stdio.h>
#include <math.h>

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    for (int i = 0; i < N; ++i) {
        Y[i * INCY] = pow(X[i * INCX], ALPHA);
    }
}

int main() {
    // 示例用法
    int arraySize = 4;
    float inputArray[] = {2.0, 3.0, 4.0, 5.0};
    float outputArray[arraySize];
    float alphaValue = 3.0;

    printf("输入数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", inputArray[i]);
    }

    // 调用函数
    pow_cpu(arraySize, alphaValue, inputArray, 1, outputArray, 1);

    printf("\n计算后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", outputArray[i]);
    }

    return 0;
}
