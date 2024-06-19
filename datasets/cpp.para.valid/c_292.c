#include <stdio.h>

// Function prototype
void histogrammPrimitive(unsigned int *histogramVector, unsigned char *grayImage, int rows, int columns);

int main() {
    // Example data
    int rows = 3;
    int columns = 3;
    unsigned char grayImage[] = {1, 2, 3,
                                 1, 2, 3,
                                 1, 2, 3};
    unsigned int histogramVector[256] = {0};

    // Call the function
    histogramPrimitive(histogramVector, grayImage, rows, columns);

    // Display the results
    printf("Histogram Vector:\n");
    for (int i = 0; i < 256; i++) {
        printf("%u ", histogramVector[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void histogramPrimitive(unsigned int *histogramVector, unsigned char *grayImage, int rows, int columns) {
    int column;
    int row;
    for (column = 0; column < columns; column++) {
        for (row = 0; row < rows; row++) {
            int offset = (column) + (columns * row);
            unsigned char grayValue = grayImage[offset];
            histogramVector[grayValue]++;
        }
    }
}
 
