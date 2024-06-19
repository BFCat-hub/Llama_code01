#include <stdio.h>
#include <assert.h>

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);

int main() {
    // 在这里可以创建测试用的数据，并调用 shortcut_cpu 函数
    // 例如：
    int batch = 2;
    int w1 = 4, h1 = 4, c1 = 3;
    int w2 = 2, h2 = 2, c2 = 2;
    float s1 = 0.5, s2 = 0.7;

    // 假设 add 和 out 是相应大小的数组
    float add[w1 * h1 * c1 * batch];
    float out[w2 * h2 * c2 * batch];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < w1 * h1 * c1 * batch; i++) {
        add[i] = i + 1;
    }

    // 调用函数
    shortcut_cpu(batch, w1, h1, c1, add, w2, h2, c2, s1, s2, out);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < w2 * h2 * c2 * batch; i++) {
        printf("%f ", out[i]);
    }
    printf("\n");

    return 0;
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out) {
    int stride = w1 / w2;
    int sample = w2 / w1;

    assert(stride == h1 / h2);
    assert(sample == h2 / h1);

    if (stride < 1)
        stride = 1;
    if (sample < 1)
        sample = 1;

    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i, j, k, b;

    for (b = 0; b < batch; ++b) {
        for (k = 0; k < minc; ++k) {
            for (j = 0; j < minh; ++j) {
                for (i = 0; i < minw; ++i) {
                    int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                    int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                    out[out_index] = s1 * out[out_index] + s2 * add[add_index];
                }
            }
        }
    }
}
