#include <stdio.h>
#include <math.h>

void GraphSum_backward(float *in_grad, float *out_grad, int *indptr, int *indices, int size, int dim);

int main() {
    // 在这里可以创建测试用的数据，并调用 GraphSum_backward 函数
    // 例如：
    int size = 5;  // 你的图的大小
    int dim = 3;   // 维度

    // 假设 in_grad、out_grad、indptr 和 indices 是相应大小的数组
    float in_grad[size * dim];
    float out_grad[size * dim];
    int indptr[size + 1];
    int indices[10] = {0, 1, 1, 2, 0, 2, 3, 4, 4, 3};  // 你的边信息

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < size * dim; i++) {
        in_grad[i] = 0;
        out_grad[i] = i + 1;
    }

    for (int i = 0; i < size + 1; i++) {
        indptr[i] = i * 2;  // 假设每个节点有两个边
    }

    // 调用函数
    GraphSum_backward(in_grad, out_grad, indptr, indices, size, dim);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%f ", in_grad[i * dim + j]);
        }
        printf("\n");
    }

    return 0;
}

void GraphSum_backward(float *in_grad, float *out_grad, int *indptr, int *indices, int size, int dim) {
    for (int src = 0; src < size - 1; src++) {
        for (int i = indptr[src]; i < indptr[src + 1]; i++) {
            int dst = indices[i];
            float coef = 1.0 / sqrtf((indptr[src + 1] - indptr[src]) * (indptr[dst + 1] - indptr[dst]));
            for (int j = 0; j < dim; j++) {
                in_grad[src * dim + j] += coef * out_grad[dst * dim + j];
            }
        }
    }
}
