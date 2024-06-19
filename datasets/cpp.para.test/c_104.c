#include <stdio.h>

void SparseMatmul_forward(float *a, float *b, float *c, int *indptr, int *indices, int p, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
            int j = indices[jj];
            for (int k = 0; k < p; k++) {
                c[i * p + k] += a[jj] * b[j * p + k];
            }
        }
    }
}

int main() {
    // 示例数据
    const int size = 3;
    const int p = 2;
    float a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float b[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0};
    float c[size * p] = {0.0};

    int indptr[] = {0, 2, 4};   // Example indptr array
    int indices[] = {1, 2, 0, 1};   // Example indices array

    // 调用 SparseMatmul_forward 函数
    SparseMatmul_forward(a, b, c, indptr, indices, p, size);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant matrix c:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < p; j++) {
            printf("%f ", c[i * p + j]);
        }
        printf("\n");
    }

    return 0;
}
