#include <stdio.h>

void set_valid_mask_cpu(const float *score, float score_thr, int *valid_mask, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (score[tid] > score_thr) {
            valid_mask[tid] = 1;
        } else {
            valid_mask[tid] = 0;
        }
    }
}

int main() {
    // 示例用法
    int size = 6;
    float scores[] = {0.8, 0.5, 0.9, 0.3, 0.7, 0.6};
    float threshold = 0.6;
    int validMask[size];

    printf("输入分数数组：");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", scores[i]);
    }

    // 调用函数
    set_valid_mask_cpu(scores, threshold, validMask, size);

    printf("\n有效掩码数组：");
    for (int i = 0; i < size; i++) {
        printf("%d ", validMask[i]);
    }

    return 0;
}
