#include <stdio.h>

// 函数声明
void Copy_List_cpu(const int element_numbers, const float *origin_list, float *list);

int main() {
    // 示例数据
    const int element_numbers = 5;
    float origin_list[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float list[5];

    // 调用函数
    Copy_List_cpu(element_numbers, origin_list, list);

    // 输出结果
    printf("Copied list:\n");
    for (int i = 0; i < element_numbers; i++) {
        printf("%f ", list[i]);
    }

    return 0;
}

// 函数定义
void Copy_List_cpu(const int element_numbers, const float *origin_list, float *list) {
    for (int i = 0; i < element_numbers; i++) {
        list[i] = origin_list[i];
    }
}
 
