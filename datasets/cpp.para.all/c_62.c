#include <stdio.h>

void Blend_CPU(unsigned char *aImg1, unsigned char *aImg2, unsigned char *aRS, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        aRS[i] = (unsigned char)(0.5 * aImg1[i] + 0.5 * aImg2[i]);
    }
}

int main() {
    // 示例用法
    int width = 2;
    int height = 2;
    unsigned char img1[] = {100, 150, 200, 255};
    unsigned char img2[] = {50, 75, 100, 255};
    unsigned char result[width * height];

    printf("输入图像1：");
    for (int i = 0; i < width * height; i++) {
        printf("%d ", img1[i]);
    }

    printf("\n输入图像2：");
    for (int i = 0; i < width * height; i++) {
        printf("%d ", img2[i]);
    }

    // 调用函数
    Blend_CPU(img1, img2, result, width, height);

    printf("\n混合后的图像：");
    for (int i = 0; i < width * height; i++) {
        printf("%d ", result[i]);
    }

    return 0;
}
