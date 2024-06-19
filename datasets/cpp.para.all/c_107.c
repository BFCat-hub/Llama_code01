#include <stdio.h>

void SparseMatmul_backward(float *a, float *b_grad, float *c_grad, int *indptr, int *indices, int p, int size, float *grad) {
    for (int i = 0; i < size - 1; i++) {
        for (int jj = indptr[i]; jj < indptr[i + 1]; jj++) {
            int j = indices[jj];
            for (int k = 0; k < p; k++) {
                b_grad[j * p + k] += c_grad[i * p + k] * a[jj];
            }
        }
    }
}

int main() {
    // 示例数据
    const int size = 3;
    const int p = 2;
    float a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float b_grad[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float c_grad[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float grad[size * p] = {0.0};

    int indptr[] = {0, 2, 4};   // Example indptr array
    int indices[] = {1, 2, 0, 1};   // Example indices array

    // 调用 SparseMatmul_backward 函数
    SparseMatmul_backward(a, b_grad, c_grad, indptr, indices, p, size, grad);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant gradient b_grad:\n");
    for (int i = 0; i < size * p; i++) {
        printf("%f ", b_grad[i]);
    }
    printf("\n");

    return 0;
}
