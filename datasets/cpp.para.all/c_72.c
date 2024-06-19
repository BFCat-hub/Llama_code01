#include <stdio.h>

void matPerRowDivInplace_cpu(double *mat, const double *alphas, int m, int n) {
    for (int index = 0; index < m * n; index++) {
        int i = index / n;
        int j = index % n;
        mat[i * n + j] /= (alphas[i] + 10 * 3);
    }
}

int main() {
    // 示例用法
    int m = 3;
    int n = 4;
    double mat[] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    double alphas[] = {2.0, 3.0, 4.0};

    printf("输入矩阵 mat：\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", mat[i * n + j]);
        }
        printf("\n");
    }

    printf("\n输入 alphas 数组：\n");
    for (int i = 0; i < m; i++) {
        printf("%.2f ", alphas[i]);
    }

    // 调用函数
    matPerRowDivInplace_cpu(mat, alphas, m, n);

    printf("\n每行除以 alphas 后的矩阵 mat：\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", mat[i * n + j]);
        }
        printf("\n");
    }

    return 0;
}
