#include <stdio.h>
#include <math.h>

// 函数声明
void sigmoid_kernel(float *input, float *output, int n);

int main() {
    // 示例数据
    int n = 5;
    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float output[5];

    // 调用函数
    sigmoid_kernel(input, output, n);

    // 输出结果
    printf("Array after applying sigmoid function:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", output[i]);
    }

    return 0;
}

// 函数定义
void sigmoid_kernel(float *input, float *output, int n) {
    for (int tid = 0; tid < n; tid++) {
        output[tid] = 1.0 / (1.0 + expf(-input[tid]));
    }
}
 
