#include <stdio.h>

void getTopkNum(const float *inputScore, const int *inputIndex, float *outputScore, int *outputIndex, float threshold, const int dims, int *anchorIndex, int *classIndex, const int classNum, int batchSize, int totalScoreNum);

int main() {
    // 在这里可以创建测试用的数据，并调用 getTopkNum 函数
    // 例如：
    int dims = 10;
    int batchSize = 3;
    int totalScoreNum = 5;
    int classNum = 2;
    float threshold = 0.5;

    // 假设 inputScore、inputIndex、outputScore、outputIndex、anchorIndex、classIndex 是相应大小的数组
    float inputScore[batchSize * totalScoreNum];
    int inputIndex[batchSize * totalScoreNum];
    float outputScore[batchSize * dims];
    int outputIndex[batchSize * dims];
    int anchorIndex[batchSize * dims];
    int classIndex[batchSize * dims];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < batchSize * totalScoreNum; i++) {
        inputScore[i] = 0.1 * i;
        inputIndex[i] = i;
    }

    // 调用函数
    getTopkNum(inputScore, inputIndex, outputScore, outputIndex, threshold, dims, anchorIndex, classIndex, classNum, batchSize, totalScoreNum);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < batchSize * dims; i++) {
        printf("outputScore[%d] = %f, outputIndex[%d] = %d, anchorIndex[%d] = %d, classIndex[%d] = %d\n", i, outputScore[i], i, outputIndex[i], i, anchorIndex[i], i, classIndex[i]);
    }

    return 0;
}

void getTopkNum(const float *inputScore, const int *inputIndex, float *outputScore, int *outputIndex, float threshold, const int dims, int *anchorIndex, int *classIndex, const int classNum, int batchSize, int totalScoreNum) {
    for (int tid = 0; tid < dims; tid++) {
        for (int i = 0; i < batchSize; i++) {
            if (inputScore[i * totalScoreNum + tid] >= threshold) {
                outputScore[i * dims + tid] = inputScore[i * totalScoreNum + tid];
                outputIndex[i * dims + tid] = inputIndex[i * totalScoreNum + tid];
                anchorIndex[i * dims + tid] = outputIndex[i * dims + tid] / classNum;
                classIndex[i * dims + tid] = outputIndex[i * dims + tid] % classNum;
            } else {
                outputScore[i * dims + tid] = 0.0f;
                outputIndex[i * dims + tid] = -1;
                anchorIndex[i * dims + tid] = -1;
                classIndex[i * dims + tid] = -1;
            }
        }
    }
}
