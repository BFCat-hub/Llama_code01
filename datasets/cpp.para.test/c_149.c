#include <stdio.h>

void permuteData_cpu(const float *input, float *output, int num, int devideNum, int featureSize, int priorNum, int batchSize);

int main() {
    // 在这里可以创建测试用的数据，并调用 permuteData_cpu 函数
    // 例如：
    int num = 2;           // 你的 num 数
    int devideNum = 3;     // 你的 devideNum 数
    int featureSize = 4;   // 特征大小
    int priorNum = 2;      // 你的 priorNum 数
    int batchSize = 2;     // 批处理大小

    // 假设 input 和 output 是相应大小的数组
    float *input = (float *)malloc(batchSize * num * devideNum * priorNum * featureSize * sizeof(float));
    float *output = (float *)malloc(batchSize * num * devideNum * priorNum * sizeof(float));

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < batchSize * num * devideNum * priorNum * featureSize; i++) {
        input[i] = i + 1;
    }

    // 调用函数
    permuteData_cpu(input, output, num, devideNum, featureSize, priorNum, batchSize);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < num * devideNum * priorNum; j++) {
            printf("%f ", output[i * num * devideNum * priorNum + j]);
        }
        printf("\n");
    }

    free(input);
    free(output);

    return 0;
}

void permuteData_cpu(const float *input, float *output, int num, int devideNum, int featureSize, int priorNum, int batchSize) {
    for (int tid = 0; tid < num; tid++) {
        int numPerbatch = num * devideNum * priorNum;
        for (int s = 0; s < batchSize; s++) {
            for (int i = 0; i < priorNum; i++) {
                for (int j = 0; j < devideNum; j++) {
                    output[s * numPerbatch + tid * priorNum * devideNum + i * devideNum + j] =
                        input[s * numPerbatch + (i * devideNum * featureSize) + (j * featureSize) + tid];
                }
            }
        }
    }
}
