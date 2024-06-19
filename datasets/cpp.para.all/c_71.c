#include <stdio.h>

void kmeans_average(int *means, int *counts, int BID, int DIM) {
    for (int bid = 0; bid < BID; bid++) {
        for (int tid = 0; tid < DIM; tid++) {
            if (counts[bid] == 0) {
                means[bid * DIM + tid] = 0;
            } else {
                means[bid * DIM + tid] /= counts[bid];
            }
        }
    }
}

int main() {
    // 示例用法
    int BID = 3;
    int DIM = 4;
    int means[BID * DIM];
    int counts[BID];

    // 初始化 means 数组和 counts 数组
    for (int i = 0; i < BID; i++) {
        counts[i] = i + 1;  // 假设 counts 从 1 递增
        for (int j = 0; j < DIM; j++) {
            means[i * DIM + j] = i * DIM + j + 1;  // 假设 means 按照一定规律初始化
        }
    }

    printf("输入 means 数组和 counts 数组：\n");
    for (int i = 0; i < BID; i++) {
        for (int j = 0; j < DIM; j++) {
            printf("%d ", means[i * DIM + j]);
        }
        printf(" | Count: %d\n", counts[i]);
    }

    // 调用函数
    kmeans_average(means, counts, BID, DIM);

    printf("\n平均化后的 means 数组：\n");
    for (int i = 0; i < BID; i++) {
        for (int j = 0; j < DIM; j++) {
            printf("%d ", means[i * DIM + j]);
        }
        printf("\n");
    }

    return 0;
}
