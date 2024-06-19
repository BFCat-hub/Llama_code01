#include <stdio.h>

void compute_b_minus_Rx(double *out, double *x, double *b, double *cotans, int *neighbors, int meshStride, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = b[i];
        for (int iN = 0; iN < meshStride; ++iN) {
            int neighbor = neighbors[i * meshStride + iN];
            double weight = cotans[i * meshStride + iN];
            out[i] += weight * x[neighbor];
        }
    }
}

int main() {
    // 示例数据
    const int n = 3;
    const int meshStride = 2;
    double x[] = {1.0, 2.0, 3.0};
    double b[] = {4.0, 5.0, 6.0};
    double cotans[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    int neighbors[] = {1, 2, 0, 2, 0, 1};
    double out[n];

    // 调用 compute_b_minus_Rx 函数
    compute_b_minus_Rx(out, x, b, cotans, neighbors, meshStride, n);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array out:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", out[i]);
    }

    return 0;
}
