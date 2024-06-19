#include <stdio.h>
#include <math.h>


void calculateOuterSumsNew_cpu(float *innerSums, float *L, int uLength) {
    for (int u = 0; u < uLength; u++) {
        float real, imag, u_sum;
        int realIdx = 2 * u;
        int imagIdx = realIdx + 1;

        real = innerSums[realIdx];
        imag = innerSums[imagIdx];
        u_sum = (real * real) + (imag * imag);

        realIdx += 64;
        imagIdx += 64;
        real = innerSums[realIdx];
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag);

        realIdx += 64;
        imagIdx += 64;
        real = innerSums[realIdx];
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag);

        realIdx += 64;
        imagIdx += 64;
        real = innerSums[realIdx];
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag);

        realIdx += 64;
        imagIdx += 64;
        real = innerSums[realIdx];
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag);

        realIdx += 64;
        imagIdx += 64;
        real = innerSums[realIdx];
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag);

        realIdx += 64;
        imagIdx += 64;
        real = innerSums[realIdx];
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag);

        realIdx += 64;
        imagIdx += 64;
        real = innerSums[realIdx];
        imag = innerSums[imagIdx];
        u_sum += (real * real) + (imag * imag);

        L[u] = u_sum;
    }
}

// 主函数
int main() {
    const int uLength = 10;  // 请替换为实际的长度
    float innerSums[uLength * 2];  // 请替换为实际的数组

    // 初始化 innerSums 数组，这里仅为示例
    for (int i = 0; i < uLength * 2; ++i) {
        innerSums[i] = (float)i;
    }

    float L[uLength];

    // 调用您的函数
    calculateOuterSumsNew_cpu(innerSums, L, uLength);

    // 打印结果
    printf("Result: ");
    for (int u = 0; u < uLength; ++u) {
        printf("%f ", L[u]);
    }
    printf("\n");

    return 0;
}
 
