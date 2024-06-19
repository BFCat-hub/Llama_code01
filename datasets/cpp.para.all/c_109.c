#include <stdio.h>

void mxm_1d_cpu(double *a, const int m, double *b, const int n, double *c, const int p) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < p; k++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) {
                s += a[j * m + i] * b[k * n + j];
            }
            c[k * m + i] = s;
        }
    }
}

int main() {
    // 示例数据
    const int m = 2;
    const int n = 3;
    const int p = 4;
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double b[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double c[p * m];

    // 调用 mxm_1d_cpu 函数
    mxm_1d_cpu(a, m, b, n, c, p);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant matrix c:\n");
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < m; j++) {
            printf("%f ", c[i * m + j]);
        }
        printf("\n");
    }

    return 0;
}
