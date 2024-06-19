#include <stdio.h>

void convertKinectDisparityToRegularDisparity_cpu(float *d_regularDisparity, int d_regularDisparityPitch, const float *d_KinectDisparity, int d_KinectDisparityPitch, int width, int height);

int main() {
    // 在这里可以创建测试用的数据，并调用 convertKinectDisparityToRegularDisparity_cpu 函数
    // 例如：
    int width = 5;
    int height = 3;

    // 假设 d_KinectDisparity 和 d_regularDisparity 是相应大小的数组
    float d_KinectDisparity[height][width];
    float d_regularDisparity[height][width];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            d_KinectDisparity[y][x] = x + y + 1;
        }
    }

    // 调用函数
    convertKinectDisparityToRegularDisparity_cpu((float *)d_regularDisparity, width * sizeof(float), (float *)d_KinectDisparity, width * sizeof(float), width, height);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%f ", d_regularDisparity[y][x]);
        }
        printf("\n");
    }

    return 0;
}

void convertKinectDisparityToRegularDisparity_cpu(float *d_regularDisparity, int d_regularDisparityPitch, const float *d_KinectDisparity, int d_KinectDisparityPitch, int width, int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            float d_in = *((float *)((char *)d_KinectDisparity + y * d_KinectDisparityPitch) + x);
            float d_out = (d_in == 0.0f) ? 1 : -d_in;
            *((float *)((char *)d_regularDisparity + y * d_regularDisparityPitch) + x) = d_out;
        }
    }
}
