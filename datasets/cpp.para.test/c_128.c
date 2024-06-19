#include <stdio.h>

void convolution_cpu_1d(float *input, const float *mask, float *output, int array_size, int mask_size) {
    int MASK_RADIUS = mask_size / 2;
    float temp = 0.0f;
    int ELEMENT_INDEX = 0;

    for (int i = 0; i < array_size; i++) {
        temp = 0;

        for (int j = 0; j < mask_size; j++) {
            ELEMENT_INDEX = i - MASK_RADIUS + j;

            if (!(ELEMENT_INDEX < 0 || ELEMENT_INDEX > (array_size - 1))) {
                temp += input[ELEMENT_INDEX] * mask[j];
            }
        }

        output[i] = temp;
    }
}

int main() {
    // 示例数据
    const int array_size = 10;
    const int mask_size = 3;
    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float mask[] = {0.1, 0.2, 0.1};
    float output[array_size];

    // 调用 convolution_cpu_1d 函数
    convolution_cpu_1d(input, mask, output, array_size, mask_size);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array output:\n");
    for (int i = 0; i < array_size; i++) {
        printf("%f ", output[i]);
    }

    return 0;
}
