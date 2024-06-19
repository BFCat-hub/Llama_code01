#include <stdio.h>

void resizedClsScore_cpu(const float *score, const float *score_factors, float *output, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (score[tid] == (-1)) {
            output[tid] = -1;
        } else {
            output[tid] = score[tid] * score_factors[tid];
        }
    }
}

int main() {
    // 示例用法
    int dims = 5;   // 你的 dims 值
    float *score = new float[dims];
    float *score_factors = new float[dims];
    float *output = new float[dims];

    // 假设 score 和 score_factors 已经被赋值

    // 调用函数
    resizedClsScore_cpu(score, score_factors, output, dims);

    // 打印结果
    printf("处理后的 output 数组：\n");
    for (int tid = 0; tid < dims; tid++) {
        printf("%.2f ", output[tid]);
    }

    // 释放内存
    delete[] score;
    delete[] score_factors;
    delete[] output;

    return 0;
}
