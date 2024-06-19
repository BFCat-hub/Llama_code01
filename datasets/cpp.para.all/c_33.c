#include <stdio.h>

void doubleArrayScalarDivide_cpu(double *d_in, int *d_out, int length, double scalar) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in[idx] / scalar;
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    double doubleArray[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    int resultArray[arraySize];
    double scalar = 2.0;

    printf("原始双精度数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", doubleArray[i]);
    }

    // 调用函数
    doubleArrayScalarDivide_cpu(doubleArray, resultArray, arraySize, scalar);

    printf("\n除法后的整数数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", resultArray[i]);
    }

    return 0;
}
