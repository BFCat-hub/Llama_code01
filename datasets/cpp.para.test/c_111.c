#include <stdio.h>

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main() {
    // 示例数据
    const int m = 2;
    const int n = 3;
    const int k = 2;
    int h_a[] = {1, 2, 3, 4, 5, 6};
    int h_b[] = {2, 3, 4, 5, 6, 7};
    int h_result[m * k];

    // 调用 cpu_matrix_mult 函数
    cpu_matrix_mult(h_a, h_b, h_result, m, n, k);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant matrix h_result:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            printf("%d ", h_result[i * k + j]);
        }
        printf("\n");
    }

    return 0;
}
