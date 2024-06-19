#include <stdio.h>

void filterFFT_cpu(float *FFT, float *filter, int nxprj2, int nviews, float scale) {
    for (int i = 0; i < nviews; i++) {
        for (int j = 0; j < nxprj2; j++) {
            FFT[i * nxprj2 + j] *= filter[i * nxprj2 + j] * scale;
        }
    }
}

int main() {
    // 示例用法
    int nxprj2 = 3;    // nxprj2 的值
    int nviews = 2;    // nviews 的值
    float scale = 0.5; // scale 的值
    float *FFT = new float[nxprj2 * nviews];
    float *filter = new float[nxprj2 * nviews];

    // 假设 FFT 和 filter 数组已经被赋值

    // 调用函数
    filterFFT_cpu(FFT, filter, nxprj2, nviews, scale);

    // 打印结果
    printf("处理后的 FFT 数组：\n");
    for (int i = 0; i < nviews; i++) {
        for (int j = 0; j < nxprj2; j++) {
            printf("%.2f ", FFT[i * nxprj2 + j]);
        }
        printf("\n");
    }

    // 释放内存
    delete[] FFT;
    delete[] filter;

    return 0;
}
