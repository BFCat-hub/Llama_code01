#include <stdio.h>

void subtractMean_cpu(double *images, const double *meanImage, int imageNum, int pixelNum) {
    for (int col = 0; col < pixelNum; col++) {
        for (int row = 0; row < imageNum; ++row) {
            images[row * pixelNum + col] -= meanImage[col];
            if (images[row * pixelNum + col] < 0.0) {
                images[row * pixelNum + col] = 0.0;
            }
        }
    }
}

int main() {
    // 示例数据
    const int imageNum = 2;
    const int pixelNum = 3;
    double images[imageNum * pixelNum] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double meanImage[pixelNum] = {2.0, 3.0, 4.0};

    // 调用 subtractMean_cpu 函数
    subtractMean_cpu(images, meanImage, imageNum, pixelNum);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant images after subtracting mean:\n");
    for (int i = 0; i < imageNum; i++) {
        for (int j = 0; j < pixelNum; j++) {
            printf("%f ", images[i * pixelNum + j]);
        }
        printf("\n");
    }

    return 0;
}
