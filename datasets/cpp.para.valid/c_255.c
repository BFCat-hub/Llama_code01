#include <stdio.h>

// Function declaration
void castImageTofloat(float *deviceOutputImageData, unsigned char *ucharImage,
                       int imageWidth, int imageHeight, int channels, int pixelSize);

int main() {
    // Example data
    const int imageWidth = 2;
    const int imageHeight = 2;
    const int channels = 3;
    const int pixelSize = channels; // Assuming 1 channel = 1 byte
    unsigned char ucharImage[] = {255, 128, 0, 200, 100, 50, 0, 0, 255, 128, 255, 0};

    // Assuming each pixel is represented by a float for each channel
    const int floatPixelSize = channels; // Adjust accordingly
    float deviceOutputImageData[floatPixelSize];

    // Function call
    castImageTofloat(deviceOutputImageData, ucharImage, imageWidth, imageHeight, channels, pixelSize);

    // Output result
    printf("Resultant array after casting to float:\n");
    for (int i = 0; i < floatPixelSize; i++) {
        printf("%f ", deviceOutputImageData[i]);
    }

    return 0;
}

// Function definition
void castImageTofloat(float *deviceOutputImageData, unsigned char *ucharImage,
                       int imageWidth, int imageHeight, int channels, int pixelSize) {
    for (int w = 0; w < pixelSize; w++) {
        deviceOutputImageData[w] = (float)(ucharImage[w] / 255.0);
    }
}
 
