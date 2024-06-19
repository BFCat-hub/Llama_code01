#include <stdio.h>

void shortcut_kernel_cpu(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);

int main() {
    // 在这里可以创建测试用的数据，并调用 shortcut_kernel_cpu 函数
    // 例如：
    int size = 3;
    int minw = 2;
    int minh = 2;
    int minc = 2;
    int stride = 2;
    int sample = 3;
    int batch = 4;
    int w1 = 1;
    int h1 = 1;
    int c1 = 1;
    int w2 = 2;
    int h2 = 2;
    int c2 = 2;

    // 假设 add 和 out 是相应大小的数组
    float add[size * stride * w1 * h1 * c1 * batch];
    float out[sample * w2 * h2 * c2 * batch];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < size * stride * w1 * h1 * c1 * batch; i++) {
        add[i] = i + 1;
    }

    for (int i = 0; i < sample * w2 * h2 * c2 * batch; i++) {
        out[i] = i + 2;
    }

    // 调用函数
    shortcut_kernel_cpu(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, out);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < sample * w2 * h2 * c2 * batch; i++) {
        printf("%f ", out[i]);
    }
    printf("\n");

    return 0;
}

void shortcut_kernel_cpu(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out) {
    for (int id = 0; id < size; id++) {
        int i = id % minw;
        id /= minw;
        int j = id % minh;
        id /= minh;
        int k = id % minc;
        id /= minc;
        int b = id % batch;

        int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
        int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));

        out[out_index] += add[add_index];
    }
}
