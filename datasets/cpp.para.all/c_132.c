#include <stdio.h>

void residual(double *out, double *x, double *b, double *cotans, int *neighbors, double *diag, int meshStride, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = diag[i] * x[i] - b[i];
        for (int iN = 0; iN < meshStride; ++iN) {
            int neighbor = neighbors[i * meshStride + iN];
            double weight = cotans[i * meshStride + iN];
            out[i] -= weight * x[neighbor];
        }
    }
}

int main() {
    // 示例数据
    const int n = 5;
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    double cotans[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    int neighbors[] = {1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 3, 2};
    double diag[] = {2.0, 3.0, 4.0, 5.0, 6.0};
    double out[n];

    // 调用 residual 函数
    residual(out, x, b, cotans, neighbors, diag, 3, n);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    for (int i = 0; i < n; i++) {
        printf("Resultant out[%d]: %f\n", i, out[i]);
    }

    return 0;
}
