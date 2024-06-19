#include <stdio.h>
#include <math.h>

void squareSerial(float *d_in, float *d_out, int N) {
    for (unsigned int i = 0; i < N; ++i) {
        d_out[i] = pow(d_in[i] / (d_in[i] - 2.3), 3);
    }
}

int main() {
    // 示例用法
    int arraySize = 3;
    float inputArray[] = {1.0, 3.0, 5.0};
    float outputArray[arraySize];

    printf("输入数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", inputArray[i]);
    }

    // 调用函数
    squareSerial(inputArray, outputArray, arraySize);

    printf("\n计算后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", outputArray[i]);
    }

    return 0;
}
