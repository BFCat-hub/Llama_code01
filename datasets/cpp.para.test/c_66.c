#include <stdio.h>

void saxpy_cpu(float *vecY, float *vecX, float alpha, int n) {
    for (int i = 0; i < n; i++) {
        vecY[i] = alpha * vecX[i] + vecY[i];
    }
}

int main() {
    // 示例用法
    int size = 5;
    float vecY[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float vecX[] = {0.5, 1.0, 1.5, 2.0, 2.5};
    float alpha = 2.0;

    printf("输入向量 vecY：");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", vecY[i]);
    }

    printf("\n输入向量 vecX：");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", vecX[i]);
    }

    // 调用函数
    saxpy_cpu(vecY, vecX, alpha, size);

    printf("\n执行 saxpy 后的向量 vecY：");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", vecY[i]);
    }

    return 0;
}
