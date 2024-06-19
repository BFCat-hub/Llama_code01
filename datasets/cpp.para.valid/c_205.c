#include <stdio.h>

// 函数声明
void vecAdd_cpu(float *in1, float *in2, float *out, int len);

int main() {
    // 示例数据
    int len = 5;
    float in1[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float in2[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    float out[5];

    // 调用函数
    vecAdd_cpu(in1, in2, out, len);

    // 输出结果
    printf("Resultant vector after addition:\n");
    for (int i = 0; i < len; i++) {
        printf("%f ", out[i]);
    }

    return 0;
}

// 函数定义
void vecAdd_cpu(float *in1, float *in2, float *out, int len) {
    for (int i = 0; i < len; i++) {
        out[i] = in1[i] + in2[i];
    }
}
 
