#include <stdio.h>

void forward_dropout_layer(int batch, int inputs, float *input, float probability, float *rand, float scale) {
    for (int i = 0; i < batch * inputs; ++i) {
        if (rand[i] < probability) {
            input[i] = 0;
        } else {
            input[i] *= scale;
        }
    }
}

int main() {
    // 示例用法
    int batchSize = 2;
    int inputSize = 3;
    float inputArray[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float randArray[] = {0.2, 0.8, 0.4, 0.1, 0.9, 0.5};
    float probability = 0.5;
    float scale = 2.0;

    printf("输入数组：");
    for (int i = 0; i < batchSize * inputSize; i++) {
        printf("%.2f ", inputArray[i]);
    }

    // 调用函数
    forward_dropout_layer(batchSize, inputSize, inputArray, probability, randArray, scale);

    printf("\n处理后的数组：");
    for (int i = 0; i < batchSize * inputSize; i++) {
        printf("%.2f ", inputArray[i]);
    }

    return 0;
}
