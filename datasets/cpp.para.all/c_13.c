#include <stdio.h>

void matDiagAddInplace_cpu(double *mat, double alpha, int dim) {
    for (int i = 0; i < dim; i++) {
        mat[i * dim + i] += alpha;
    }
}

int main() {
    // 示例用法
    int matrixDim = 3;
    double matrix[3][3] = {{1.1, 2.2, 3.3},
                           {4.4, 5.5, 6.6},
                           {7.7, 8.8, 9.9}};
    double alpha = 10.0;

    printf("原始矩阵：\n");
    for (int i = 0; i < matrixDim; i++) {
        for (int j = 0; j < matrixDim; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }

    // 调用函数
    matDiagAddInplace_cpu((double *)matrix, alpha, matrixDim);

    printf("\n对角线元素加上常数后的矩阵：\n");
    for (int i = 0; i < matrixDim; i++) {
        for (int j = 0; j < matrixDim; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}
