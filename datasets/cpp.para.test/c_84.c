#include <stdio.h>
#include <stdlib.h>

void convertEdgeMaskToFloatCpu(float *d_output, unsigned char *d_input, unsigned int width, unsigned int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            d_output[y * width + x] = fminf(d_input[y * width + x], d_input[width * height + y * width + x]);
        }
    }
}

int main() {
    // 设置示例的宽和高
    unsigned int width = 5;
    unsigned int height = 5;

    // 分配内存
    unsigned char *d_input = (unsigned char *)malloc(width * height * 2 * sizeof(unsigned char));  // 假设输入是两倍的宽高
    float *d_output = (float *)malloc(width * height * sizeof(float));

    // 填充示例输入数据，这里只是一个例子，实际应用中需要根据具体情况进行初始化
    for (int i = 0; i < width * height * 2; i++) {
        d_input[i] = i % 256;  // 以简单的方式初始化输入数据
    }

    // 调用函数进行转换
    convertEdgeMaskToFloatCpu(d_output, d_input, width, height);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    for (int i = 0; i < width * height; i++) {
        printf("%f ", d_output[i]);
    }

    // 释放内存
    free(d_input);
    free(d_output);

    return 0;
}
