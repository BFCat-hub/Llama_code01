#include <stdio.h>

void init_image_array_CPU(unsigned long long int *image, int pixels_per_image) {
    for (int my_pixel = 0; my_pixel < pixels_per_image; my_pixel++) {
        image[my_pixel] = (unsigned long long int)(0);
        my_pixel += pixels_per_image;
        image[my_pixel] = (unsigned long long int)(0);
        my_pixel += pixels_per_image;
        image[my_pixel] = (unsigned long long int)(0);
        my_pixel += pixels_per_image;
        image[my_pixel] = (unsigned long long int)(0);
    }
}

int main() {
    // 示例数据
    const int pixels_per_image = 4;
    unsigned long long int image[pixels_per_image * 4]; // Assuming 4 images

    // 调用 init_image_array_CPU 函数
    init_image_array_CPU(image, pixels_per_image);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array image:\n");
    for (int i = 0; i < pixels_per_image * 4; i++) {
        printf("%llu ", image[i]);
    }

    return 0;
}
