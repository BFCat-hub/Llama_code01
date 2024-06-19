#include <stdio.h>

void compute_array_square(float *array, float *outArray, int size) {
    for (int i = 0; i < size; i++) {
        outArray[i] = array[i] * array[i];
    }
}

int main() {
    // 示例用法
    int arraySize = 4;
    float inputArray[] = {2.0, 3.0, 4.0, 5.0};
    float outputArray[arraySize];

    printf("输入数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", inputArray[i]);
    }

    // 调用函数
    compute_array_square(inputArray, outputArray, arraySize);

    printf("\n计算后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", outputArray[i]);
    }

    return 0;
}
