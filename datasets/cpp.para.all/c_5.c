#include <stdio.h>

void scale_host(float *array, float scale, int N) {
    for (int idx = 0; idx < N; idx++) {
        array[idx] *= scale;
    }
}

int main() {
    // 示例用法
    int numElements = 5;
    float array[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    float scale_factor = 2.0;

    printf("原始数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%.2f ", array[i]);
    }

    // 调用函数
    scale_host(array, scale_factor, numElements);

    printf("\n缩放后的数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%.2f ", array[i]);
    }

    return 0;
}
