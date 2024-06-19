#include <stdio.h>

// 函数声明
void k_vec_divide(float *vec1, float *vec2, int max_size);

int main() {
    // 示例数据
    const int max_size = 5;
    float vec1[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    float vec2[] = {2.0, 4.0, 5.0, 8.0, 10.0};

    // 调用函数
    k_vec_divide(vec1, vec2, max_size);

    // 输出结果
    printf("Resultant vector after elementwise division:\n");
    for (int i = 0; i < max_size; i++) {
        printf("%f ", vec1[i]);
    }

    return 0;
}

// 函数定义
void k_vec_divide(float *vec1, float *vec2, int max_size) {
    for (int i = 0; i < max_size; i++) {
        vec1[i] = vec1[i] / vec2[i];
    }
}
 
