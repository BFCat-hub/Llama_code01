#include <stdio.h>
#include <stdlib.h>

// Function prototype
void equalization(float *cdf, float *mincdf, unsigned char *ucharImage, int imageWidth, int imageHeight, int channels, int pixelSize);

int main() {
    // Example data
    int imageWidth = 100;
    int imageHeight = 100;
    int channels = 3;
    int pixelSize = imageWidth * imageHeight * channels;

    float *cdf = (float *)malloc(256 * sizeof(float));
    float *mincdf = (float *)malloc(1 * sizeof(float));
    unsigned char *ucharImage = (unsigned char *)malloc(pixelSize * sizeof(unsigned char));

    // Initialize input data (for example)
    for (int i = 0; i < 256; i++) {
        cdf[i] = i / 255.0; // Replace with your data
    }

    for (int i = 0; i < 1; i++) {
        mincdf[i] = 0.1; // Replace with your data
    }

    for (int i = 0; i < pixelSize; i++) {
        ucharImage[i] = i % 256; // Replace with your data
    }

    // Call the function
    equalization(cdf, mincdf, ucharImage, imageWidth, imageHeight, channels, pixelSize);

    // Display the results
    printf("Equalized Image:\n");
    for (int i = 0; i < pixelSize; i++) {
        printf("%u ", ucharImage[i]);
        if ((i + 1) % channels == 0) {
            printf("\n");
        }
    }

    // Free allocated memory
    free(cdf);
    free(mincdf);
    free(ucharImage);

    return 0;
}

// Function definition
void equalization(float *cdf, float *mincdf, unsigned char *ucharImage, int imageWidth, int imageHeight, int channels, int pixelSize) {
    int idx;

    for (idx = 0; idx < pixelSize; idx++) {
        unsigned char val = ucharImage[idx];
        float data = 255 * (cdf[val] - mincdf[0]) / (1 - mincdf[0]);

        if (data < 0.0f) {
            data = 0.0f;
        } else if (data > 255.0f) {
            data = 255.0f;
        }

        ucharImage[idx] = (unsigned char)data;
    }
}
 
