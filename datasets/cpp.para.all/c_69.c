#include <stdio.h>

void sum_backward(float *db, float *dout, int r, int c) {
    for (int j = 0; j < c; j++) {
        for (int i = 0; i < r; i++) {
            db[j] += dout[i * c + j];
        }
    }
}

int main() {
    // 示例用法
    int rows = 3;
    int cols = 4;
    float dout[] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    float db[cols];

    printf("输入 dout 数组：\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", dout[i * cols + j]);
        }
        printf("\n");
    }

    // 初始化 db 数组
    for (int i = 0; i < cols; i++) {
        db[i] = 0.0;
    }

    // 调用函数
    sum_backward(db, dout, rows, cols);

    printf("\n计算后的 db 数组：\n");
    for (int i = 0; i < cols; i++) {
        printf("%.2f ", db[i]);
    }

    return 0;
}
