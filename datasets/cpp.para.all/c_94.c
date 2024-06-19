#include <stdio.h>

void matmul(int a[100][100], int b[100][100], int c[100][100]) {
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            c[i][j] = 0;
            for (int k = 0; k < 100; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    // 设置示例数据
    int a[100][100];
    int b[100][100];
    int c[100][100];

    // 填充示例输入数据，这里只是一个例子，实际应用中需要根据具体情况进行初始化
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    // 调用函数进行矩阵乘法
    matmul(a, b, c);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant matrix c:\n");
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    return 0;
}
