#include <stdio.h>

void add_vector_cpu(float *a, float *b, float *c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
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
    add_vector_cpu(vectorA, vectorB, resultVector, vectorSize);

    printf("\n相加后的向量 C：");
    for (int i = 0; i < vectorSize; i++) {
        printf("%.2f ", resultVector[i]);
    }

    return 0;
}
