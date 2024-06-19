#include <stdio.h>

void runFilterCpu(float *I, float *Q, int samplesLength, float *filter, int filterLength, float *filtered_I, float *filtered_Q, int convLength);

int main() {
    // 在这里可以创建测试用的数据，并调用 runFilterCpu 函数
    // 例如：
    int samplesLength = 10;
    int filterLength = 3;
    int convLength = samplesLength - filterLength + 1;

    // 假设 I、Q、filter、filtered_I 和 filtered_Q 是相应大小的数组
    float I[samplesLength];
    float Q[samplesLength];
    float filter[filterLength];
    float filtered_I[convLength];
    float filtered_Q[convLength];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < samplesLength; i++) {
        I[i] = i + 1;
        Q[i] = i + 2;
    }

    for (int i = 0; i < filterLength; i++) {
        filter[i] = i + 1;
    }

    // 调用函数
    runFilterCpu(I, Q, samplesLength, filter, filterLength, filtered_I, filtered_Q, convLength);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    printf("Filtered I: ");
    for (int i = 0; i < convLength; i++) {
        printf("%f ", filtered_I[i]);
    }
    printf("\n");

    printf("Filtered Q: ");
    for (int i = 0; i < convLength; i++) {
        printf("%f ", filtered_Q[i]);
    }
    printf("\n");

    return 0;
}

void runFilterCpu(float *I, float *Q, int samplesLength, float *filter, int filterLength, float *filtered_I, float *filtered_Q, int convLength) {
    for (int sampleIndex = 0; sampleIndex < convLength; sampleIndex++) {
        int index;
        float sumI, sumQ;
        sumI = 0;
        sumQ = 0;

        for (int j = sampleIndex - filterLength + 1; j <= sampleIndex; j++) {
            index = sampleIndex - j;

            if ((j < samplesLength) && (j >= 0)) {
                sumI += filter[index] * I[j];
                sumQ += filter[index] * Q[j];
            }
        }

        filtered_I[sampleIndex] = sumI;
        filtered_Q[sampleIndex] = sumQ;
    }
}
