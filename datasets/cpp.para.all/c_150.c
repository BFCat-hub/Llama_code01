#include <stdio.h>
#include <math.h>

void cpuSimpleCorrelator(float *xi, float *xq, float *sr, float *si, int sLength, float *L, int uLength);

int main() {
    // 在这里可以创建测试用的数据，并调用 cpuSimpleCorrelator 函数
    // 例如：
    int sLength = 5;   // 你的信号长度
    int uLength = 8;   // 你的输出长度

    // 假设 xi、xq、sr、si 和 L 是相应大小的数组
    float xi[uLength + sLength];
    float xq[uLength + sLength];
    float sr[uLength];
    float si[uLength];
    float L[uLength];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < uLength + sLength; i++) {
        xi[i] = i + 1;
        xq[i] = i + 2;
    }

    for (int i = 0; i < uLength; i++) {
        sr[i] = i + 3;
        si[i] = i + 4;
    }

    // 调用函数
    cpuSimpleCorrelator(xi, xq, sr, si, sLength, L, uLength);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < uLength; i++) {
        printf("%f ", L[i]);
    }

    return 0;
}

void cpuSimpleCorrelator(float *xi, float *xq, float *sr, float *si, int sLength, float *L, int uLength) {
    for (int u = 0; u < uLength; u++) {
        float real = 0;
        float imag = 0;
        float a, b, c, d;

        for (int n = u; n < u + sLength; n++) {
            a = xi[n];
            b = xq[n];
            c = sr[n - u];
            d = si[n - u] * (-1);
            real += (a * c) - (b * d);
            imag += (a * d) + (b * c);
        }

        L[u] = sqrt(real * real + imag * imag);
    }
}
