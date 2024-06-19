#include <stdio.h>

void LreluForward(float *srcData, float *dstData, int data_size, float alpha) {
    for (int i = 0; i < data_size; i++) {
        dstData[i] = srcData[i] > 0 ? srcData[i] : srcData[i] * alpha;
    }
}

int main() {
    // 示例用法
    int data_size = 5;   // 数组的大小
    float alpha = 0.01;   // LReLU 的 alpha 值
    float *srcData = new float[data_size];
    float *dstData = new float[data_size];

    // 假设 srcData 数组已经被赋值

    // 调用函数
    LreluForward(srcData, dstData, data_size, alpha);

    // 打印结果
    printf("处理后的 dstData 数组：\n");
    for (int i = 0; i < data_size; i++) {
        printf("%.2f ", dstData[i]);
    }

    // 释放内存
    delete[] srcData;
    delete[] dstData;

    return 0;
}
