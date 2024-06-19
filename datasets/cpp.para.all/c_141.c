#include <stdio.h>

void apply_grayscale(const unsigned char *image, unsigned char *grayimg, int width, int height);

int main() {
    // 你可以在这里创建测试用的图像数据，并调用 apply_grayscale 函数进行灰度处理
    // 例如：
    int width = 100;  // 你的图像宽度
    int height = 100; // 你的图像高度

    // 假设 image 是一个 width * height * 3 大小的数组，存储图像的 RGB 数据
    unsigned char image[width * height * 3];

    // 假设 grayimg 是一个 width * height 大小的数组，存储灰度图像数据
    unsigned char grayimg[width * height];

    // 调用灰度处理函数
    apply_grayscale(image, grayimg, width, height);

    // 在这里可以添加打印灰度图像或其他操作
    // 例如：
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%u ", grayimg[y * width + x]);
        }
        printf("\n");
    }

    return 0;
}

void apply_grayscale(const unsigned char *image, unsigned char *grayimg, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const unsigned char R = image[(y * width + x) * 3 + 0];
            const unsigned char G = image[(y * width + x) * 3 + 1];
            const unsigned char B = image[(y * width + x) * 3 + 2];
            unsigned char gray = (307 * R + 604 * G + 113 * B) >> 10;
            grayimg[y * width + x] = gray;
        }
    }
}
