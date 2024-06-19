#include <stdio.h>

// 函数声明
void test1_cpu(float *input, int dims);

int main() {
    // 示例数据
    int dims = 5;
    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 10.0};

    // 调用函数
    test1_cpu(input, dims);

    // 输出结果
    printf("Resultant array after test1_cpu:\n");
    for (int i = 0; i < dims * 4; i++) {
        printf("%f ", input[i]);
    }

    return 0;
}

// 函数定义
void test1_cpu(float *input, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (input[tid * 4] != 0) {
            input[tid * 4] = 0;
        }
    }
}
 
