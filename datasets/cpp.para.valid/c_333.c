#include <stdio.h>

void distanceMatFinal(long int totalPixels, int availablePixels, int outPixelOffset, float *distMat);

int main() {
    // Example dimensions
    long int totalPixels = 4;
    int availablePixels = 3;
    int outPixelOffset = 1;

    // Example input data (distMat)
    float distMat[12] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    };

    // Applying distanceMatFinal
    distanceMatFinal(totalPixels, availablePixels, outPixelOffset, distMat);

    // Print the result
    printf("Output distMat:\n");
    for (long int i = 0; i < availablePixels; ++i) {
        for (long int j = 0; j < totalPixels; ++j) {
            printf("%8.4f ", distMat[i * totalPixels + j]);
        }
        printf("\n");
    }

    return 0;
}

void distanceMatFinal(long int totalPixels, int availablePixels, int outPixelOffset, float *distMat) {
    for (long int i = 0; i < availablePixels; i++) {
        float sum = 0.0;
        float max = 0.0;

        // Find the maximum element in the current row
        for (long int j = 0; j < totalPixels; j++) {
            float element = distMat[i * totalPixels + j];
            if (element > max) max = element;
            sum += element;
        }

        // Update the matrix using max and sum
        sum += max;

        for (long int j = 0; j < totalPixels; j++) {
            if ((i + outPixelOffset) == j)
                distMat[i * totalPixels + j] = max / sum;
            else
                distMat[i * totalPixels + j] /= sum;
        }
    }
}
 
