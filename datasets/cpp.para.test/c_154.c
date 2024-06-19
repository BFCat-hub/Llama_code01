#include <stdio.h>
#include <math.h>

void distanceMatCalc(long int totalPixels, int availablePixels, int outPixelOffset, int patchSize, float *distMat, float *data, float filtSig);

int main() {
    // 在这里可以创建测试用的数据，并调用 distanceMatCalc 函数
    // 例如：
    long int totalPixels = 3;
    int availablePixels = 2;
    int outPixelOffset = 1;
    int patchSize = 2;

    // 假设 distMat 和 data 是相应大小的数组
    float distMat[availablePixels * totalPixels];
    float data[totalPixels * patchSize * patchSize];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (long int i = 0; i < totalPixels * patchSize * patchSize; i++) {
        data[i] = i + 1;
    }

    // 调用函数
    distanceMatCalc(totalPixels, availablePixels, outPixelOffset, patchSize, distMat, data, 1.0);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (long int i = 0; i < availablePixels * totalPixels; i++) {
        printf("%f ", distMat[i]);
    }
    printf("\n");

    return 0;
}

void distanceMatCalc(long int totalPixels, int availablePixels, int outPixelOffset, int patchSize, float *distMat, float *data, float filtSig) {
    for (long int i = 0; i < availablePixels * totalPixels; i++) {
        int data_i = i / totalPixels + outPixelOffset;
        int data_j = i % totalPixels;
        float tmp = 0.0;

        if (data_i != data_j) {
            for (int elem = 0; elem < patchSize * patchSize; elem++) {
                float diff = data[data_i * patchSize * patchSize + elem] - data[data_j * patchSize * patchSize + elem];
                tmp += diff * diff;
            }
            tmp = exp(-tmp / filtSig);
        }

        distMat[i] = tmp;
    }
}
