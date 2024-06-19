#include <stdio.h>

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);

int main() {
    // 在这里可以创建测试用的数据，并调用 dot_cpu 函数
    // 例如：
    int N = 5;
    int INCX = 1;
    int INCY = 1;

    // 假设 X 和 Y 是相应大小的数组
    float X[N * INCX];
    float Y[N * INCY];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < N; i++) {
        X[i * INCX] = i + 1;
        Y[i * INCY] = i + 2;
    }

    // 调用函数
    float result = dot_cpu(N, X, INCX, Y, INCY);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    printf("Dot Product: %f\n", result);

    return 0;
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY) {
    int i;
    float dot = 0;

    for (i = 0; i < N; ++i) {
        dot += X[i * INCX] * Y[i * INCY];
    }

    return dot;
}
