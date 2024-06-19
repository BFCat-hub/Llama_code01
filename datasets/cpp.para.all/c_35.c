#include <stdio.h>

void activate_array_leaky_cpu(float *x, int n) {
    for (int index = 0; index < n; index++) {
        float val = x[index];
        x[index] = (val > 0) ? val : val / 10;
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    float inputArray[] = {2.0, -3.0, 4.0, -5.0, 6.0};

    printf("原始数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", inputArray[i]);
    }

    // 调用函数
    activate_array_leaky_cpu(inputArray, arraySize);

    printf("\n激活后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", inputArray[i]);
    }

    return 0;
}
