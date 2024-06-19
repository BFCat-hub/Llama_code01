#include <stdio.h>

void colorConvert(unsigned char *grayImage, unsigned char *colorImage, int rows, int columns) {
    for (int column = 0; column < columns; column++) {
        for (int row = 0; row < rows; row++) {
            int offset = column + (columns * row);
            unsigned char grayValue = 0.07 * colorImage[offset * 3] + 0.71 * colorImage[offset * 3 + 1] + 0.21 * colorImage[offset * 3 + 2];
            grayImage[offset] = grayValue;
        }
    }
}

int main() {
    // 示例数据
    const int rows = 2;
    const int columns = 2;
    unsigned char colorImage[] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128};
    unsigned char grayImage[rows * columns];

    // 调用 colorConvert 函数
    colorConvert(grayImage, colorImage, rows, columns);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array grayImage:\n");
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            int offset = column + (columns * row);
            printf("%u ", grayImage[offset]);
        }
        printf("\n");
    }

    return 0;
}
