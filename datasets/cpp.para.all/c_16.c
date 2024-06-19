#include <stdio.h>

void scal_cpu(int N, float ALPHA, float *X, int INCX) {
    int i;
    for (i = 0; i < N; ++i) {
        X[i * INCX] *= ALPHA;
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    float arrayX[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    float alpha = 2.0;
    int incX = 2;

    printf("原始数组 X：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayX[i]);
    }

    // 调用函数
    scal_cpu(arraySize, alpha, arrayX, incX);

    printf("\n缩放后的数组 X：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayX[i]);
    }

    return 0;
}
