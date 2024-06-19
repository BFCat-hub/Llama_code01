#include <stdio.h>

void add_arrays(int n, float *x, float *y, float *z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    float arrayX[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    float arrayY[] = {0.5, 1.5, 2.5, 3.5, 4.5};
    float resultArray[arraySize];

    printf("数组 X：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayX[i]);
    }

    printf("\n数组 Y：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayY[i]);
    }

    // 调用函数
    add_arrays(arraySize, arrayX, arrayY, resultArray);

    printf("\n数组 Z（和）：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", resultArray[i]);
    }

    return 0;
}
