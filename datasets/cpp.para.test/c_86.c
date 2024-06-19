#include <stdio.h>
#include <stdlib.h>

void LreluBackward(float *srcDiff, float *dstDiff, float *srcData, int data_size, float alpha) {
    for (int i = 0; i < data_size; i++) {
        dstDiff[i] = (srcData[i] > 0) ? srcDiff[i] * 1.0 : srcDiff[i] * alpha;
    }
}

int main() {
    // 设置示例数据大小
    int data_size = 5;

    // 分配内存
    float *srcDiff = (float *)malloc(data_size * sizeof(float));
    float *dstDiff = (float *)malloc(data_size * sizeof(float));
    float *srcData = (float *)malloc(data_size * sizeof(float));

    // 填充示例输入数据，这里只是一个例子，实际应用中需要根据具体情况进行初始化
    for (int i = 0; i < data_size; i++) {
        srcDiff[i] = i;
        srcData[i] = i - 2.0;
    }

    // 设置 alpha 值
    float alpha = 0.1;

    // 调用函数进行反向传播
    LreluBackward(srcDiff, dstDiff, srcData, data_size, alpha);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("srcDiff: ");
    for (int i = 0; i < data_size; i++) {
        printf("%f ", srcDiff[i]);
    }

    printf("\nsrcData: ");
    for (int i = 0; i < data_size; i++) {
        printf("%f ", srcData[i]);
    }

    printf("\ndstDiff: ");
    for (int i = 0; i < data_size; i++) {
        printf("%f ", dstDiff[i]);
    }

    // 释放内存
    free(srcDiff);
    free(dstDiff);
    free(srcData);

    return 0;
}
