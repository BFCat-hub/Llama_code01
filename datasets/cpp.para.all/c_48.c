#include <stdio.h>

void evenoddincrement_cpu(float *g_data, int even_inc, int odd_inc, int size) {
    for (int tx = 0; tx < size; tx++) {
        if ((tx % 2) == 0) {
            g_data[tx] += even_inc;
        } else {
            g_data[tx] += odd_inc;
        }
    }
}

int main() {
    // 示例用法
    int arraySize = 6;
    float dataArray[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int evenIncrement = 2;
    int oddIncrement = 1;

    printf("原始数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", dataArray[i]);
    }

    // 调用函数
    evenoddincrement_cpu(dataArray, evenIncrement, oddIncrement, arraySize);

    printf("\n增量后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", dataArray[i]);
    }

    return 0;
}
