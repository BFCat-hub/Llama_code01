#include <stdio.h>
#include <math.h>

void kComputeActs(const float *d_nets, float *d_acts, int size) {
    for (int un_idx = 0; un_idx < size; un_idx++) {
        float tact = 1.0f / (1.0f + expf(-d_nets[un_idx]));
        d_acts[un_idx] = tact;
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    float netsArray[] = {0.5, -1.0, 1.5, -2.0, 2.5};
    float actsArray[arraySize];

    printf("输入数组（nets）：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", netsArray[i]);
    }

    // 调用函数
    kComputeActs(netsArray, actsArray, arraySize);

    printf("\n计算后的数组（acts）：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.4f ", actsArray[i]);
    }

    return 0;
}
