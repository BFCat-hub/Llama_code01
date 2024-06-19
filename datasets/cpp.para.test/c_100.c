#include <stdio.h>

void cpu_matrix_mul(int *a, int *b, int *c, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int sum = 0;
            for (int i = 0; i < N; i++) {
                sum += a[row * N + i] * b[i * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

int main() {
    // 示例数据
    const int N = 3;
    int a[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int b[N * N] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    int c[N * N];

    // 调用 cpu_matrix_mul 函数
    cpu_matrix_mul(a, b, c, N);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant matrix c:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}
