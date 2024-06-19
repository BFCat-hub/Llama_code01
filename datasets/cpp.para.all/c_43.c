#include <stdio.h>

void operacionCPU(float *u, float *lu, float u_m, float u_d, int n) {
    int idx = 0;
    while (idx < n) {
        lu[idx] = (u[idx] - u_m) / u_d;
        idx += 1;
    }
}

int main() {
    // 示例用法
    int arraySize = 4;
    float uArray[] = {2.0, 3.0, 4.0, 5.0};
    float luArray[arraySize];
    float u_m = 3.0;
    float u_d = 2.0;

    printf("数组 u：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", uArray[i]);
    }

    // 调用函数
    operacionCPU(uArray, luArray, u_m, u_d, arraySize);

    printf("\n计算后的数组 lu：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", luArray[i]);
    }

    return 0;
}
