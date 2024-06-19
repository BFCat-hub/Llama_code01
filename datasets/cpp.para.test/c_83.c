#include <stdio.h>

void convertFloatToRGBA_cpu(char *out_image, const float *in_image, int width, int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int IND = (y * width + x) * 4;  // 每个像素有四个分量
            float val = in_image[y * width + x];

            // 将浮点值映射到范围 [0, 255]，并存储为 char 类型
            char temp = static_cast<char>(val * 255.0f);

            // 设置 RGBA 分量
            out_image[IND] = temp;     // Red 分量
            out_image[IND + 1] = temp; // Green 分量
            out_image[IND + 2] = temp; // Blue 分量
            out_image[IND + 3] = 255;  // Alpha 分量，这里设为 255 表示完全不透明
        }
    }
}

int main() {
    // 示例用法
    int width = 3;   // 图像宽度
    int height = 2;  // 图像高度
    float *in_image = new float[width * height];  // 输入浮点数图像
    char *out_image = new char[width * height * 4]; // 输出 RGBA 图像

    // 假设 in_image 数组已经被赋值

    // 调用函数
    convertFloatToRGBA_cpu(out_image, in_image, width, height);

    // 打印结果（这里简化为显示 RGBA 分量）
    for (int i = 0; i < width * height * 4; i++) {
        printf("%d ", out_image[i]);
    }

    // 释放内存
    delete[] in_image;
    delete[] out_image;

    return 0;
}
