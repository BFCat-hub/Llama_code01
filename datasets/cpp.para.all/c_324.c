#include <stdio.h>
#include <stdlib.h>

void castImageToGrayScale(unsigned char *ucharImage, unsigned char *grayImage, int imageWidth, int imageHeight, int channels) {
    int w, h;
    for (w = 0; w < imageWidth; w++) {
        for (h = 0; h < imageHeight; h++) {
            int idx = imageWidth * h + w;
            unsigned char r = ucharImage[idx * channels];
            unsigned char g = ucharImage[idx * channels + 1];
            unsigned char b = ucharImage[idx * channels + 2];
            grayImage[idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
        }
    }
}

int main() {
    // Define your image parameters
    int imageWidth = 5;
    int imageHeight = 5;
    int channels = 3; // Assuming RGB image

    // Allocate memory for ucharImage and grayImage
    unsigned char *ucharImage = (unsigned char *)malloc(imageWidth * imageHeight * channels * sizeof(unsigned char));
    unsigned char *grayImage = (unsigned char *)malloc(imageWidth * imageHeight * sizeof(unsigned char));

    // Initialize ucharImage (example: filling with random values)
    for (int i = 0; i < imageWidth * imageHeight * channels; i++) {
        ucharImage[i] = rand() % 256; // Replace with your initialization logic
    }

    // Call the castImageToGrayScale function
    castImageToGrayScale(ucharImage, grayImage, imageWidth, imageHeight, channels);

    // Display the result (for demonstration purposes)
    printf("Original Image:\n");
    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            printf("(%3u, %3u, %3u)\t", ucharImage[(imageWidth * i + j) * channels], ucharImage[(imageWidth * i + j) * channels + 1], ucharImage[(imageWidth * i + j) * channels + 2]);
        }
        printf("\n");
    }

    printf("\nGray Image:\n");
    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            printf("%3u\t", grayImage[imageWidth * i + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(ucharImage);
    free(grayImage);

    return 0;
}
 
