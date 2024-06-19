#include <stdio.h>

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

int main() {
    // 在这里可以创建测试用的数据，并调用 upsample_cpu 函数
    // 例如：
    int w = 2;
    int h = 2;
    int c = 3;
    int batch = 1;
    int stride = 2;
    int forward = 1; // Set to 1 for forward pass, 0 for backward pass
    float scale = 2.0;

    // 假设 in 和 out 是相应大小的数组
    float in[batch * w * h * c];
    float out[batch * w * h * c * stride * stride];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < batch * w * h * c; i++) {
        in[i] = i + 1;
    }

    // 调用函数
    upsample_cpu(in, w, h, c, batch, stride, forward, scale, out);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < batch * w * h * c * stride * stride; i++) {
        printf("%f ", out[i]);
    }
    printf("\n");

    return 0;
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out) {
    int i, j, k, b;

    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h * stride; ++j) {
                for (i = 0; i < w * stride; ++i) {
                    int in_index = b * w * h * c + k * w * h + (j / stride) * w + i / stride;
                    int out_index = b * w * h * c * stride * stride + k * w * h * stride * stride + j * w * stride + i;

                    if (forward)
                        out[out_index] = scale * in[in_index];
                    else
                        in[in_index] += scale * out[out_index];
                }
            }
        }
    }
}
