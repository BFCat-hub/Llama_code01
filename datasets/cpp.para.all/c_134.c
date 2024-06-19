#include <stdio.h>

void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter, int imageW, int imageH, int filterR) {
    int x, y, k;
    for (y = 0; y < imageH; y++) {
        for (x = 0; x < imageW; x++) {
            float sum = 0;
            for (k = -filterR; k <= filterR; k++) {
                int d = y + k;
                if (d >= 0 && d < imageH) {
                    sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
                }
            }
            h_Dst[y * imageW + x] = sum;
        }
    }
}

int main() {
    // 示例数据
    const int imageW = 4;
    const int imageH = 4;
    const int filterR = 1;

    float h_Src[imageH * imageW];
    float h_Filter[2 * filterR + 1];
    float h_Dst[imageH * imageW];

    // 假设 h_Src 和 h_Filter 数组已经被正确初始化

    // 调用 convolutionColumnCPU 函数
    convolutionColumnCPU(h_Dst, h_Src, h_Filter, imageW, imageH, filterR);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    for (int y = 0; y < imageH; ++y) {
        for (int x = 0; x < imageW; ++x) {
            printf("Resultant h_Dst[%d][%d]: %f\n", y, x, h_Dst[y * imageW + x]);
        }
    }

    return 0;
}
