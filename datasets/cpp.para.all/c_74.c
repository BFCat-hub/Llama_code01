#include <stdio.h>
#include <stdlib.h>

void copy_array_d2d(double **src, double **dst, int m, int n) {
    int i, j;
    for (i = 1; i < m + 1; i++)
        for (j = 1; j < n + 1; j++)
            dst[i][j] = src[i][j];
}

int main() {
    // 示例用法
    int m = 3;
    int n = 4;

    // 分配内存并初始化源数组 src
    double **src = (double **)malloc((m + 2) * sizeof(double *));
    for (int i = 0; i < m + 2; i++) {
        src[i] = (double *)malloc((n + 2) * sizeof(double));
        for (int j = 0; j < n + 2; j++) {
            src[i][j] = i * (n + 2) + j; // 假设初始化为一些值
        }
    }

    // 分配内存给目标数组 dst
    double **dst = (double **)malloc((m + 2) * sizeof(double *));
    for (int i = 0; i < m + 2; i++) {
        dst[i] = (double *)malloc((n + 2) * sizeof(double));
    }

    // 调用函数
    copy_array_d2d(src, dst, m, n);

    // 打印结果
    printf("源数组 src：\n");
    for (int i = 0; i < m + 2; i++) {
        for (int j = 0; j < n + 2; j++) {
            printf("%.2f ", src[i][j]);
        }
        printf("\n");
    }

    printf("\n目标数组 dst（复制后）：\n");
    for (int i = 0; i < m + 2; i++) {
        for (int j = 0; j < n + 2; j++) {
            printf("%.2f ", dst[i][j]);
        }
        printf("\n");
    }

    // 释放内存
    for (int i = 0; i < m + 2; i++) {
        free(src[i]);
        free(dst[i]);
    }
    free(src);
    free(dst);

    return 0;
}
