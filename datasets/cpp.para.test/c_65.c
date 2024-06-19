#include <stdio.h>

void MMDOuterProdComputeWithSum(float *x_average, int size_x, float *x_outer_prod) {
    for (int i = 0; i < size_x; i++) {
        x_outer_prod[i] = x_average[i] * x_average[i];
    }
}

int main() {
    // 示例用法
    int size = 4;
    float averageValues[] = {1.5, 2.0, 3.5, 4.0};
    float outerProdResult[size];

    printf("平均值数组：");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", averageValues[i]);
    }

    // 调用函数
    MMDOuterProdComputeWithSum(averageValues, size, outerProdResult);

    printf("\n外积结果数组：");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", outerProdResult[i]);
    }

    return 0;
}
