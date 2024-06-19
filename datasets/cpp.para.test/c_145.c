#include <stdio.h>
#include <math.h>

void CDFfunction(float *median, float *stdvLogNormalFrame, float *MeanLogNormalFrame, unsigned char *currentFrame, int pixelsPerFrame);

int main() {
    // 在这里可以创建测试用的数据，并调用 CDFfunction 函数
    // 例如：
    int pixelsPerFrame = 100;  // 你的每帧像素数

    // 假设 median、stdvLogNormalFrame、MeanLogNormalFrame 和 currentFrame 是相应大小的数组
    float median[pixelsPerFrame];
    float stdvLogNormalFrame[pixelsPerFrame];
    float MeanLogNormalFrame[pixelsPerFrame];
    unsigned char currentFrame[pixelsPerFrame];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < pixelsPerFrame; i++) {
        median[i] = 10.0;
        stdvLogNormalFrame[i] = 2.0;
        MeanLogNormalFrame[i] = 5.0;
        currentFrame[i] = i % 256;  // 假设像素值在 0 到 255 之间
    }

    // 调用函数
    CDFfunction(median, stdvLogNormalFrame, MeanLogNormalFrame, currentFrame, pixelsPerFrame);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < pixelsPerFrame; i++) {
        printf("%u ", currentFrame[i]);
    }

    return 0;
}

void CDFfunction(float *median, float *stdvLogNormalFrame, float *MeanLogNormalFrame, unsigned char *currentFrame, int pixelsPerFrame) {
    int pixel;
    for (pixel = 0; pixel < pixelsPerFrame; pixel++) {
        float newvalue;
        float x = currentFrame[pixel];
        newvalue = -((log(x) - median[pixel]) - MeanLogNormalFrame[pixel]) / (sqrt(2.0) * stdvLogNormalFrame[pixel]);
        float summ = 0.5f + 0.5f * erf(newvalue);
        if (summ >= 0.3) {
            currentFrame[pixel] = (unsigned char)255;
        } else {
            currentFrame[pixel] = (unsigned char)0;
        }
    }
}
