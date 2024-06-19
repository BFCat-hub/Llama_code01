#include <stdio.h>
#include <math.h>

void binarize_weights(float *weights, int n, int size, float *binary) {
    int i, f;
    for (f = 0; f < n; ++f) {
        float mean = 0;
        for (i = 0; i < size; ++i) {
            mean += fabs(weights[f * size + i]);
        }
        mean = mean / size;
        for (i = 0; i < size; ++i) {
            binary[f * size + i] = (weights[f * size + i] > 0) ? mean : -mean;
        }
    }
}

int main() {
    // 示例数据
    const int n = 2;
    const int size = 3;
    float weights[] = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
    float binary[n * size];

    // 调用 binarize_weights 函数
    binarize_weights(weights, n, size, binary);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array binary:\n");
    for (int f = 0; f < n; f++) {
        for (int i = 0; i < size; i++) {
            printf("%f ", binary[f * size + i]);
        }
        printf("\n");
    }

    return 0;
}
