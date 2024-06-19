#include <stdio.h>

int matrixMulHost(float *h_M, float *h_N, float *h_P, int width) {
    int Pvalue;
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            Pvalue = 0;
            for (int k = 0; k < width; ++k) {
                Pvalue += h_M[row * width + k] * h_N[k * width + col];
            }
            h_P[row * width + col] = Pvalue;
        }
    }
    return 0;
}

int main() {
    // 示例数据
    const int width = 2;
    float h_M[] = {1.0, 2.0, 3.0, 4.0};
    float h_N[] = {5.0, 6.0, 7.0, 8.0};
    float h_P[width * width];

    // 调用 matrixMulHost 函数
    matrixMulHost(h_M, h_N, h_P, width);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant matrix h_P:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", h_P[i * width + j]);
        }
        printf("\n");
    }

    return 0;
}
