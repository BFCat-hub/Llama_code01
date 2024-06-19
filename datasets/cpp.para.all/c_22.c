#include <stdio.h>

void saxpy_serial(const int dim, float a, float *x, float *y) {
    for (int i = 0; i < dim; i++) {
        y[i] += a * x[i];
    }
}

int main() {
    // 示例用法
    int vectorSize = 5;
    float vectorX[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    float vectorY[] = {0.5, 1.5, 2.5, 3.5, 4.5};
    float alpha = 2.0;

    printf("向量 X：");
    for (int i = 0; i < vectorSize; i++) {
        printf("%.2f ", vectorX[i]);
    }

    printf("\n向量 Y：");
    for (int i = 0; i < vectorSize; i++) {
        printf("%.2f ", vectorY[i]);
    }

    // 调用函数
    saxpy_serial(vectorSize, alpha, vectorX, vectorY);

    printf("\nSAXPY 后的向量 Y：");
    for (int i = 0; i < vectorSize; i++) {
        printf("%.2f ", vectorY[i]);
    }

    return 0;
}
