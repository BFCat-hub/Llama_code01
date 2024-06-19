#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void normalize_img(double *image, long int image_size, int bands);

int main() {
    // 在这里可以创建测试用的数据，并调用 normalize_img 函数
    // 例如：
    long int image_size = 10;  // 你的图像大小
    int bands = 3;             // 波段数

    // 假设 image 是相应大小的数组
    double *image = (double *)malloc(image_size * bands * sizeof(double));

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (long int i = 0; i < image_size * bands; i++) {
        image[i] = i + 1;
    }

    // 调用函数
    normalize_img(image, image_size, bands);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < bands; i++) {
        for (long int j = 0; j < image_size; j++) {
            printf("%f ", image[i * image_size + j]);
        }
        printf("\n");
    }

    free(image);

    return 0;
}

void normalize_img(double *image, long int image_size, int bands) {
    long int i, j;
    long int row;
    double *D = (double *)calloc(image_size, sizeof(double));

    for (i = 0; i < image_size * bands; i++) {
        D[i % image_size] += image[i];
    }

    for (i = 0; i < image_size; i++) {
        D[i] = pow(D[i] + 1.0e-16, -1);
    }

    for (i = 0; i < bands; i++) {
        row = i * image_size;
        for (j = 0; j < image_size; j++) {
            image[row + j] = image[row + j] * D[j];
        }
    }

    free(D);
}
