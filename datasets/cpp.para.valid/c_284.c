#include <stdio.h>

// Function prototype
void getMeanImage_cpu(const double *images, double *meanImage, int imageNum, int pixelNum);

int main() {
    // Example data
    int imageNum = 3;
    int pixelNum = 4;
    double images[] = {1.0, 2.0, 3.0, 4.0,
                       5.0, 6.0, 7.0, 8.0,
                       9.0, 10.0, 11.0, 12.0};
    double meanImage[pixelNum];

    // Call the function
    getMeanImage_cpu(images, meanImage, imageNum, pixelNum);

    // Display the results
    printf("Mean Image:\n");
    for (int col = 0; col < pixelNum; col++) {
        printf("%.2f ", meanImage[col]);
    }
    printf("\n");

    return 0;
}

// Function definition
void getMeanImage_cpu(const double *images, double *meanImage, int imageNum, int pixelNum) {
    for (int col = 0; col < pixelNum; col++) {
        meanImage[col] = 0.0;
        for (int row = 0; row < imageNum; ++row) {
            meanImage[col] += images[row * pixelNum + col];
        }
        meanImage[col] /= imageNum;
    }
}
 
