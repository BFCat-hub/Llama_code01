#include <stdio.h>

void cudaKernel_estimateSnr_cpu(const float *corrSum, const int *corrValidCount, const float *maxval, float *snrValue, const int size) {
    for (int idx = 0; idx < size; idx++) {
        float mean = (corrSum[idx] - maxval[idx] * maxval[idx]) / (corrValidCount[idx] - 1);
        snrValue[idx] = maxval[idx] * maxval[idx] / mean;
    }
}

int main() {
    // 设置示例数据
    const int size = 5;
    float corrSum[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    int corrValidCount[] = {2, 3, 4, 5, 6};
    float maxval[] = {2.0, 4.0, 6.0, 8.0, 10.0};
    float snrValue[size];

    // 调用函数进行 SNR 估算
    cudaKernel_estimateSnr_cpu(corrSum, corrValidCount, maxval, snrValue, size);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("SNR Values:\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", snrValue[i]);
    }

    return 0;
}
