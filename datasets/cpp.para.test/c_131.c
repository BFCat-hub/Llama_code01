#include <stdio.h>

void bitPrune_cpu(unsigned char *out, float *in, int frontPrune, int outputlength, int inputLength, int n) {
    for (int i = 0; i < n; i++) {
        int batch = i / outputlength;
        int indexInBatch = i % outputlength;
        int batchInJump = batch * inputLength;
        int indexOutBatch = i % outputlength;
        int batchOutJump = batch * outputlength;
        int frontJump = frontPrune;
        out[batchOutJump + indexOutBatch] = (char)(in[batchInJump + frontJump + indexInBatch] > 0);
    }
}

int main() {
    // 示例数据
    const int outputlength = 4;
    const int inputLength = 6;
    const int n = 8;
    float in[] = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0};
    unsigned char out[n];

    // 调用 bitPrune_cpu 函数
    bitPrune_cpu(out, in, 1, outputlength, inputLength, n);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    for (int i = 0; i < n; i++) {
        printf("Resultant out[%d]: %d\n", i, out[i]);
    }

    return 0;
}
