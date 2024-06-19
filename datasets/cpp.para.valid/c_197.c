#include <stdio.h>

// 函数声明
void const_cpu(int N, float ALPHA, float *X, int INCX);

int main() {
    // 示例数据
    int N = 5;
    float ALPHA = 2.0;
    float X[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int INCX = 1;

    // 调用函数
    const_cpu(N, ALPHA, X, INCX);

    // 输出结果
    printf("Array after setting each element to ALPHA:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", X[i]);
    }

    return 0;
}

// 函数定义
void const_cpu(int N, float ALPHA, float *X, int INCX) {
    for (int i = 0; i < N; ++i) {
        X[i * INCX] = ALPHA;
    }
}
 
