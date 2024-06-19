#include <stdio.h>

void cpuAddCorrAndCorrection(float *L, float *r, int N) {
    for (int u = 0; u < N; u++) {
        L[u] -= r[u];
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    float arrayL[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    float arrayR[] = {0.5, 1.5, 2.5, 3.5, 4.5};

    printf("数组 L：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayL[i]);
    }

    printf("\n数组 R：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayR[i]);
    }

    // 调用函数
    cpuAddCorrAndCorrection(arrayL, arrayR, arraySize);

    printf("\n相加后的数组 L：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", arrayL[i]);
    }

    return 0;
}
