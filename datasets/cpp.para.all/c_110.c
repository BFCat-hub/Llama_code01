#include <stdio.h>
#include <math.h>

void fabsf_clamp_cpu(int N, float *X, int INCX, float clamp_min, float clamp_max) {
    for (int i = 0; i < N; ++i) {
        if (X[i * INCX] >= 0) {
            X[i * INCX] = fmin(clamp_max, fmax(clamp_min, X[i * INCX]));
        } else {
            X[i * INCX] = fmin(-clamp_min, fmax(-clamp_max, X[i * INCX]));
        }
    }
}

int main() {
    // 示例数据
    const int N = 5;
    const float clamp_min = -1.0;
    const float clamp_max = 1.0;
    float X[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

    // 调用 fabsf_clamp_cpu 函数
    fabsf_clamp_cpu(N, X, 1, clamp_min, clamp_max);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array X:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", X[i]);
    }
    printf("\n");

    return 0;
}
