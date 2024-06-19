#include <stdio.h>

void rgb2yuv_kernel(int img_size, unsigned char *gpu_img_in_r, unsigned char *gpu_img_in_g, unsigned char *gpu_img_in_b,
                    unsigned char *gpu_img_out_y, unsigned char *gpu_img_out_u, unsigned char *gpu_img_out_v);

int main() {
    // 在这里可以创建测试用的数据，并调用 rgb2yuv_kernel 函数
    // 例如：
    int img_size = 5;

    // 假设 gpu_img_in_r, gpu_img_in_g, gpu_img_in_b, gpu_img_out_y, gpu_img_out_u, 和 gpu_img_out_v 是相应大小的数组
    unsigned char gpu_img_in_r[img_size];
    unsigned char gpu_img_in_g[img_size];
    unsigned char gpu_img_in_b[img_size];
    unsigned char gpu_img_out_y[img_size];
    unsigned char gpu_img_out_u[img_size];
    unsigned char gpu_img_out_v[img_size];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < img_size; i++) {
        gpu_img_in_r[i] = i + 1;
        gpu_img_in_g[i] = i + 2;
        gpu_img_in_b[i] = i + 3;
    }

    // 调用函数
    rgb2yuv_kernel(img_size, gpu_img_in_r, gpu_img_in_g, gpu_img_in_b, gpu_img_out_y, gpu_img_out_u, gpu_img_out_v);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    printf("Y: ");
    for (int i = 0; i < img_size; i++) {
        printf("%u ", gpu_img_out_y[i]);
    }
    printf("\n");

    printf("U: ");
    for (int i = 0; i < img_size; i++) {
        printf("%u ", gpu_img_out_u[i]);
    }
    printf("\n");

    printf("V: ");
    for (int i = 0; i < img_size; i++) {
        printf("%u ", gpu_img_out_v[i]);
    }
    printf("\n");

    return 0;
}

void rgb2yuv_kernel(int img_size, unsigned char *gpu_img_in_r, unsigned char *gpu_img_in_g, unsigned char *gpu_img_in_b,
                    unsigned char *gpu_img_out_y, unsigned char *gpu_img_out_u, unsigned char *gpu_img_out_v) {
    unsigned char r, g, b;

    for (int index = 0; index < img_size; index++) {
        r = gpu_img_in_r[index];
        g = gpu_img_in_g[index];
        b = gpu_img_in_b[index];

        gpu_img_out_y[index] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        gpu_img_out_u[index] = (unsigned char)(-0.169 * r - 0.331 * g + 0.499 * b + 128);
        gpu_img_out_v[index] = (unsigned char)(0.499 * r - 0.418 * g - 0.0813 * b + 128);
    }
}
