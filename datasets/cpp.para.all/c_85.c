#include <stdio.h>
#include <stdlib.h>

void gpu_matrix_transpose(int *mat_in, int *mat_out, unsigned int rows, unsigned int cols) {
    unsigned int idx;
    unsigned int idy;

    for (idx = 0; idx < cols; idx++) {
        for (idy = 0; idy < rows; idy++) {
            unsigned int pos = idy * cols + idx;
            unsigned int trans_pos = idx * rows + idy;
            mat_out[trans_pos] = mat_in[pos];
        }
    }
}

int main() {
    // 设置示例的行和列
    unsigned int rows = 3;
    unsigned int cols = 4;

    // 分配内存
    int *mat_in = (int *)malloc(rows * cols * sizeof(int));
    int *mat_out = (int *)malloc(rows * cols * sizeof(int));

    // 填充示例输入数据，这里只是一个例子，实际应用中需要根据具体情况进行初始化
    for (unsigned int i = 0; i < rows * cols; i++) {
        mat_in[i] = i;
    }

    // 调用函数进行转置
    gpu_matrix_transpose(mat_in, mat_out, rows, cols);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    for (unsigned int i = 0; i < rows * cols; i++) {
        printf("%d ", mat_out[i]);
    }

    // 释放内存
    free(mat_in);
    free(mat_out);

    return 0;
}
