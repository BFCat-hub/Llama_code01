#include <stdio.h>

void Function_update_sgd_cpu(float lr, float *parameter, float *gradient, int size) {
    for (int i = 0; i < size; i++) {
        parameter[i] -= lr * gradient[i];
    }
}

int main() {
    // 示例用法
    int arraySize = 3;
    float learningRate = 0.1;
    float parameterArray[] = {1.0, 2.0, 3.0};
    float gradientArray[] = {0.5, 1.0, 1.5};

    printf("参数数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", parameterArray[i]);
    }

    printf("\n梯度数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", gradientArray[i]);
    }

    // 调用函数
    Function_update_sgd_cpu(learningRate, parameterArray, gradientArray, arraySize);

    printf("\n更新后的参数数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%.2f ", parameterArray[i]);
    }

    return 0;
}
