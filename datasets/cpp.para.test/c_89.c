#include <stdio.h>
#include <stdlib.h>

void cpuConvertToBits(int *bit_decisions, unsigned short *bit_stream, int dec_size) {
    for (int dec_index = 0; dec_index < dec_size; dec_index++) {
        int bit_index = dec_index * 2;
        int curr_decision = bit_decisions[dec_index];
        bit_stream[bit_index] = ((curr_decision & 2) >> 1);
        bit_stream[bit_index + 1] = (curr_decision & 1);
    }
}

int main() {
    // 设置示例数据大小
    int dec_size = 5;

    // 分配内存
    int *bit_decisions = (int *)malloc(dec_size * sizeof(int));
    unsigned short *bit_stream = (unsigned short *)malloc(dec_size * 2 * sizeof(unsigned short));

    // 填充示例输入数据，这里只是一个例子，实际应用中需要根据具体情况进行初始化
    for (int i = 0; i < dec_size; i++) {
        bit_decisions[i] = i % 4;  // 假设某些决策值
    }

    // 调用函数进行转换
    cpuConvertToBits(bit_decisions, bit_stream, dec_size);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("bit_decisions: ");
    for (int i = 0; i < dec_size; i++) {
        printf("%d ", bit_decisions[i]);
    }

    printf("\nbit_stream: ");
    for (int i = 0; i < dec_size * 2; i++) {
        printf("%hu ", bit_stream[i]);
    }

    // 释放内存
    free(bit_decisions);
    free(bit_stream);

    return 0;
}
