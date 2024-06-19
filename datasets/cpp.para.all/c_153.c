#include <stdio.h>
#include <math.h>

void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial);

int main() {
    // 在这里可以创建测试用的数据，并调用 l2normalize_cpu 函数
    // 例如：
    int batch = 2;
    int filters = 3;
    int spatial = 4;

    // 假设 x 和 dx 是相应大小的数组
    float x[batch * filters * spatial];
    float dx[batch * filters * spatial];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < batch * filters * spatial; i++) {
        x[i] = i + 1;
    }

    // 调用函数
    l2normalize_cpu(x, dx, batch, filters, spatial);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    printf("Normalized x: ");
    for (int i = 0; i < batch * filters * spatial; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");

    printf("dx: ");
    for (int i = 0; i < batch * filters * spatial; i++) {
        printf("%f ", dx[i]);
    }
    printf("\n");

    return 0;
}

void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial) {
    int b, f, i;

    for (b = 0; b < batch; ++b) {
        for (i = 0; i < spatial; ++i) {
            float sum = 0;

            for (f = 0; f < filters; ++f) {
                int index = b * filters * spatial + f * spatial + i;
                sum += powf(x[index], 2);
            }

            sum = sqrtf(sum);

            for (f = 0; f < filters; ++f) {
                int index = b * filters * spatial + f * spatial + i;
                x[index] /= sum;
                dx[index] = (1 - x[index]) / sum;
            }
        }
    }
}
