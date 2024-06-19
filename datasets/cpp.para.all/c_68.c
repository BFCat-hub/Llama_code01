#include <stdio.h>

void copy_swap(float *f_in, float *f_target, const int L_x) {
    for (int k_x = 0; k_x < L_x; k_x++) {
        float tempval = 0.0f;
        tempval = f_in[k_x];
        f_in[k_x] = f_target[k_x];
        f_target[k_x] = tempval;
    }
}

int main() {
    // 示例用法
    int size = 5;
    float array1[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float array2[] = {10.0, 20.0, 30.0, 40.0, 50.0};

    printf("数组1：");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", array1[i]);
    }

    printf("\n数组2：");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", array2[i]);
    }

    // 调用函数
    copy_swap(array1, array2, size);

    printf("\n交换后的数组1：");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", array1[i]);
    }

    printf("\n交换后的数组2：");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", array2[i]);
    }

    return 0;
}
