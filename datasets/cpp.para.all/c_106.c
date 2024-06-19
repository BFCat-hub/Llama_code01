#include <stdio.h>

void convertKinectDisparityInPlace_cpu(float *d_disparity, int pitch, int width, int height, float depth_scale) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            float *d_in = (float *)((char *)d_disparity + y * pitch) + x;
            *d_in = (*d_in == 0.0f) ? 1 : (-depth_scale / *d_in);
        }
    }
}

int main() {
    // 示例数据
    const int width = 3;
    const int height = 2;
    const int pitch = width * sizeof(float);
    float d_disparity[width * height] = {0.0, 2.0, 0.0, 4.0, 0.0, 6.0};
    float depth_scale = 2.0;

    // 调用 convertKinectDisparityInPlace_cpu 函数
    convertKinectDisparityInPlace_cpu(d_disparity, pitch, width, height, depth_scale);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant disparity values:\n");
    for (int i = 0; i < width * height; i++) {
        printf("%f ", d_disparity[i]);
    }
    printf("\n");

    return 0;
}
