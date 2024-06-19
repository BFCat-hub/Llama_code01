#include <stdio.h>

void grayscale(unsigned char *input, unsigned char *output, int size) {
    unsigned char r, g, b;
    
    for (int i = 0; i < size; i++) {
        r = input[3 * i];
        g = input[3 * i + 1];
        b = input[3 * i + 2];
        output[i] = (unsigned char)(0.21 * (float)r + 0.71 * (float)g + 0.07 * (float)b);
    }
}

int main() {
    // 示例数据
    const int size = 3;
    unsigned char input[size * 3] = {255, 0, 0, 0, 255, 0, 0, 0, 255};
    unsigned char output[size];

    // 调用 grayscale 函数
    grayscale(input, output, size);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant grayscale values:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    return 0;
}
