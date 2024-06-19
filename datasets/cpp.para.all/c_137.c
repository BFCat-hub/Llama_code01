#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance) {
    float scale = 1. / (batch * spatial - 1);
    int i, j, k;

    for (i = 0; i < filters; ++i) {
        variance[i] = 0;

        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }

        variance[i] *= scale;
    }
}

int main() {
    // 示例数据
    const int batch = 2;
    const int filters = 3;
    const int spatial = 4;

    float *x = (float *)malloc(batch * filters * spatial * sizeof(float));
    float *mean = (float *)malloc(filters * sizeof(float));
    float *variance = (float *)malloc(filters * sizeof(float));

    // 初始化示例数据（这里只是一个例子，实际应用中需要根据需要初始化数据）
    for (int i = 0; i < batch * filters * spatial; ++i) {
        x[i] = i + 1.0;
    }

    for (int i = 0; i < filters; ++i) {
        mean[i] = i + 0.5;
    }

    // 调用 variance_cpu 函数
    variance_cpu(x, mean, batch, filters, spatial, variance);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    for (int i = 0; i < filters; ++i) {
        printf("Resultant variance[%d]: %f\n", i, variance[i]);
    }

    // 释放动态分配的内存
    free(x);
    free(mean);
    free(variance);

    return 0;
}
