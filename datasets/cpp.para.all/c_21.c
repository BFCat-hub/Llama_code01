#include <stdio.h>

void zeroIndices_cpu(long *vec_out, const long N) {
    for (int idx = 0; idx < N; idx++) {
        vec_out[idx] = vec_out[idx] - vec_out[0];
    }
}

int main() {
    // 示例用法
    int arraySize = 6;
    long vector[arraySize] = {10, 20, 30, 40, 50, 60};

    printf("原始向量：");
    for (int i = 0; i < arraySize; i++) {
        printf("%ld ", vector[i]);
    }

    // 调用函数
    zeroIndices_cpu(vector, arraySize);

    printf("\n零化后的向量：");
    for (int i = 0; i < arraySize; i++) {
        printf("%ld ", vector[i]);
    }

    return 0;
}
