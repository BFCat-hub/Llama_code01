#include <stdio.h>

void dot_cpu(float *c, float *a, float *b, int size) {
    int t_id;
    for (t_id = 0; t_id < size; t_id++) {
        c[t_id] = a[t_id] * b[t_id];
    }
}

int main() {
    // 示例用法
    int vectorSize = 5;
    float vectorA[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    float vectorB[] = {0.5, 1.5, 2.5, 3.5, 4.5};
    float resultVector[vectorSize];

    printf("向量 A：");
    for (int i = 0; i < vectorSize; i++) {
        printf("%.2f ", vectorA[i]);
    }

    printf("\n向量 B：");
    for (int i = 0; i < vectorSize; i++) {
        printf("%.2f ", vectorB[i]);
    }

    // 调用函数
    dot_cpu(resultVector, vectorA, vectorB, vectorSize);

    printf("\n点乘后的向量 C：");
    for (int i = 0; i < vectorSize; i++) {
        printf("%.2f ", resultVector[i]);
    }

    return 0;
}
