#include <stdio.h>

void get_before_nms_data_cpu(const float *boxes, const float *scores, const int *labels, const int *index, float *boxes_out, float *scores_out, int *labels_out, int dims);

int main() {
    // 在这里可以创建测试用的数据，并调用 get_before_nms_data_cpu 函数
    // 例如：
    int dims = 5;

    // 假设 boxes, scores, labels, index, boxes_out, scores_out, 和 labels_out 是相应大小的数组
    float boxes[dims * 4];
    float scores[dims];
    int labels[dims];
    int index[dims];
    float boxes_out[dims * 4];
    float scores_out[dims];
    int labels_out[dims];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < dims * 4; i++) {
        boxes[i] = i + 1;
    }

    for (int i = 0; i < dims; i++) {
        scores[i] = i + 0.1;
        labels[i] = i;
        index[i] = i % 2; // Alternating 0 and 1 for testing
    }

    // 调用函数
    get_before_nms_data_cpu(boxes, scores, labels, index, boxes_out, scores_out, labels_out, dims);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < dims; i++) {
        printf("Box %d: (%f, %f, %f, %f) - Score: %f - Label: %d\n", i,
               boxes_out[i * 4], boxes_out[i * 4 + 1], boxes_out[i * 4 + 2], boxes_out[i * 4 + 3],
               scores_out[i], labels_out[i]);
    }

    return 0;
}

void get_before_nms_data_cpu(const float *boxes, const float *scores, const int *labels, const int *index, float *boxes_out, float *scores_out, int *labels_out, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (index[tid] == 0) {
            boxes_out[tid * 4 + 0] = -1;
            boxes_out[tid * 4 + 1] = -1;
            boxes_out[tid * 4 + 2] = -1;
            boxes_out[tid * 4 + 3] = -1;
            scores_out[tid] = -1;
            labels_out[tid] = -1;
        } else {
            boxes_out[tid * 4 + 0] = boxes[tid * 4 + 0];
            boxes_out[tid * 4 + 1] = boxes[tid * 4 + 1];
            boxes_out[tid * 4 + 2] = boxes[tid * 4 + 2];
            boxes_out[tid * 4 + 3] = boxes[tid * 4 + 3];
            scores_out[tid] = scores[tid];
            labels_out[tid] = labels[tid];
        }
    }
}
