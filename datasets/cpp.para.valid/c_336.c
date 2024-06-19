#include <stdio.h>
#include <stdlib.h>

void conv1x1_cpu(int input_channels, int input_size, int n, float *input_im, float *filter_weight, float *filter_bias, float *output_im) {
    for (int filter_index = 0; filter_index < n; filter_index++) {
        filter_weight += filter_index * input_channels;
        float bias = filter_bias[filter_index];
        output_im += filter_index * input_size * input_size;

        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < input_size; j++) {
                float tmp = bias;
                for (int k = 0; k < input_channels; k++) {
                    tmp += input_im[k * input_size * input_size + i * input_size + j] * filter_weight[k];
                }
                output_im[i * input_size + j] = (tmp > 0.0) ? tmp : 0.0;
            }
        }
    }
}

int main() {
    // 定义输入数据和滤波器参数
    int input_channels = 3;
    int input_size = 4;
    int n = 2;

    // 初始化输入数据和滤波器参数（示例数据）
    float input_im[input_channels * input_size * input_size];
    float filter_weight[input_channels * n];
    float filter_bias[n];
    float output_im[input_size * input_size * n];

    // 填充示例数据
    for (int i = 0; i < input_channels * input_size * input_size; i++) {
        input_im[i] = i + 1;  // 示例：填充输入数据
    }

    for (int i = 0; i < input_channels * n; i++) {
        filter_weight[i] = 0.5;  // 示例：填充滤波器权重
    }

    for (int i = 0; i < n; i++) {
        filter_bias[i] = 0.1;  // 示例：填充滤波器偏置
    }

    // 调用conv1x1_cpu函数
    conv1x1_cpu(input_channels, input_size, n, input_im, filter_weight, filter_bias, output_im);

    // 输出结果
    printf("Output Image:\n");
    for (int i = 0; i < n; i++) {
        printf("Channel %d:\n", i + 1);
        for (int j = 0; j < input_size; j++) {
            for (int k = 0; k < input_size; k++) {
                printf("%f ", output_im[i * input_size * input_size + j * input_size + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}
 
