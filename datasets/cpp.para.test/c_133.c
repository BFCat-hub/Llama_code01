#include <stdio.h>

void forward_avgpool_layer(int batch, int c, int h, int w, float *input, float *output) {
    int b, i, k;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            int out_index = k + b * c;
            output[out_index] = 0;
            for (i = 0; i < h * w; ++i) {
                int in_index = i + h * w * (k + b * c);
                output[out_index] += input[in_index];
            }
            output[out_index] /= h * w;
        }
    }
}

int main() {
    // 示例数据
    const int batch = 2;
    const int channels = 3;
    const int height = 4;
    const int width = 4;

    float input[batch * channels * height * width];
    float output[batch * channels];

    // 假设 input 数组已经被正确初始化

    // 调用 forward_avgpool_layer 函数
    forward_avgpool_layer(batch, channels, height, width, input, output);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            printf("Resultant output[%d][%d]: %f\n", b, c, output[c + b * channels]);
        }
    }

    return 0;
}
