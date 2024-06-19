#include <stdio.h>

// 函数声明
void add_matrix_cpu(double *a, double *b, double *c, int N);

int main() {
    // 示例数据
    int N = 9;
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double b[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double c[9];

    // 调用函数
    add_matrix_cpu(a, b, c, N);

    // 输出结果
    printf("Resultant matrix after addition:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", c[i]);
        if ((i + 1) % 3 == 0) {
            printf("\n");
        }
    }

    return 0;
}

// 函数定义
void add_matrix_cpu(double *a, double *b, double *c, int N) {
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] + b[idx];
    }
}
 
