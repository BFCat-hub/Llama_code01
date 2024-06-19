#include <stdio.h>

void vectorMatrixMult(long int totalPixels, int availablePixels, int outPixelOffset, float *matrix, float *vector, float *out) {
    for (long int i = 0; i < availablePixels; i++) {
        float sum = 0.0;
        for (long int j = 0; j < totalPixels; j++) {
            sum += matrix[i * totalPixels + j] * vector[j];
        }
        out[i + outPixelOffset] = sum;
    }
}

int main() {
    // 示例数据
    const long int totalPixels = 4;
    const int availablePixels = 2;
    const int outPixelOffset = 1;
    float matrix[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float vector[] = {2.0, 1.0, 3.0, 4.0};
    float out[availablePixels + outPixelOffset];

    // 调用 vectorMatrixMult 函数
    vectorMatrixMult(totalPixels, availablePixels, outPixelOffset, matrix, vector, out);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant vector out:\n");
    for (int i = 0; i < availablePixels + outPixelOffset; i++) {
        printf("%f ", out[i]);
    }
    printf("\n");

    return 0;
}
