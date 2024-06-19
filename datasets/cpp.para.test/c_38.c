#include <stdio.h>

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY) {
    for (int i = 0; i < N; ++i) {
        Y[i * INCY] *= X[i * INCX];
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    float arrayX[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    float arrayY[] = {0.5, 1.5, 2.5, 3.5, 4.5};

    printf("数组 X：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayX[i]);
    }

    printf("\n数组 Y：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayY[i]);
    }

    // 调用函数
    mul_cpu(arraySize, arrayX, 1, arrayY, 1);

    printf("\n数组 Y（乘法后）：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayY[i]);
    }

    return 0;
}
