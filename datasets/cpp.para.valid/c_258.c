#include <stdio.h>

// Function declaration
void castImageToUchar(float *deviceInputImageData, unsigned char *ucharImage, int imageWidth, int imageHeight, int channels, int pixelSize);

int main() {
    // Example data
    const int imageWidth = 2;
    const int imageHeight = 2;
    const int channels = 3;
    const int pixelSize = channels;
    float deviceInputImageData[pixelSize] = {0.1f, 0.5f, 0.9f, 0.3f, 0.7f, 1.0f};
    unsigned char ucharImage[pixelSize];

    // Function call
    castImageToUchar(deviceInputImageData, ucharImage, imageWidth, imageHeight, channels, pixelSize);

    // Output result
    printf("Resultant ucharImage:\n");
    for (int i = 0; i < pixelSize; ++i) {
        printf("%d ", ucharImage[i]);
    }

    return 0;
}

// Function definition
void castImageToUchar(float *deviceInputImageData, unsigned char *ucharImage, int imageWidth, int imageHeight, int channels, int pixelSize) {
    int w;
    for (w = 0; w < pixelSize; w++) {
        ucharImage[w] = (unsigned char)(255 * deviceInputImageData[w]);
    }
}
 
