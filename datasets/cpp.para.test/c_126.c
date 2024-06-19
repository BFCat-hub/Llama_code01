#include <stdio.h>

void matrixMultiplication_cpu(int *host_a, int *host_b, int *host_c, int row_a, int col_a, int col_b) {
    for (int i = 0; i < row_a; ++i) {
        for (int j = 0; j < col_b; ++j) {
            int tmp = 0;
            for (int k = 0; k < col_a; ++k) {
                tmp += host_a[i * col_a + k] * host_b[k * col_b + j];
            }
            host_c[i * col_b + j] = tmp;
        }
    }
}

int main() {
    // 示例数据
    const int row_a = 2;
    const int col_a = 3;
    const int col_b = 4;
    int host_a[] = {1, 2, 3, 4, 5, 6};
    int host_b[] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    int host_c[row_a * col_b];

    // 调用 matrixMultiplication_cpu 函数
    matrixMultiplication_cpu(host_a, host_b, host_c, row_a, col_a, col_b);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array host_c:\n");
    for (int i = 0; i < row_a; i++) {
        for (int j = 0; j < col_b; j++) {
            printf("%d ", host_c[i * col_b + j]);
        }
        printf("\n");
    }

    return 0;
}
