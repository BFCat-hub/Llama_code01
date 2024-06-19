#include <stdio.h>

void getOffsetBox_cpu(const int *clsIndex, const float *max_coordinate, float *offset, int dims, int batchSize, const float *before_nms_boxes);

int main() {
    // 在这里可以创建测试用的数据，并调用 getOffsetBox_cpu 函数
    // 例如：
    int dims = 10;         // 你的维度
    int batchSize = 5;     // 你的批处理大小

    // 假设 clsIndex、max_coordinate 和 before_nms_boxes 是相应大小的数组
    int clsIndex[batchSize * dims];
    float max_coordinate[batchSize * dims * 4];
    float offset[batchSize * dims];
    float before_nms_boxes[batchSize * dims * 4];

    // 调用函数
    getOffsetBox_cpu(clsIndex, max_coordinate, offset, dims, batchSize, before_nms_boxes);

    // 在这里可以添加打印 offset 或其他操作
    // 例如：
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < dims; j++) {
            printf("%f ", offset[i * dims + j]);
        }
        printf("\n");
    }

    return 0;
}

void getOffsetBox_cpu(const int *clsIndex, const float *max_coordinate, float *offset, int dims, int batchSize, const float *before_nms_boxes) {
    for (int tid = 0; tid < dims; tid++) {
        int numPerbatch = dims;
        for (int i = 0; i < batchSize; i++) {
            if (before_nms_boxes[i * dims * 4 + tid * 4] == -1) {
                offset[i * numPerbatch + tid] = 0;
            } else {
                offset[i * numPerbatch + tid] = clsIndex[i * numPerbatch + tid] * (max_coordinate[i * dims * 4] + 1);
            }
        }
    }
}
