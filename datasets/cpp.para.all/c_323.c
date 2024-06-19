#include <stdio.h>
#include <stdlib.h>

void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, int imageW, int imageH, int filterR) {
    int x, y, k;
    for (y = 0; y < imageH; y++) {
        for (x = 0; x < imageW; x++) {
            float sum = 0;
            for (k = -filterR; k <= filterR; k++) {
                int d = x + k;
                if (d >= 0 && d < imageW) {
                    sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
                }
            }
            h_Dst[y * imageW + x] = sum;
        }
    }
}

int main() {
    // Define your image and filter parameters
    int imageW = 5;
    int imageH = 5;
    int filterR = 1;

    // Allocate memory for source image, filter, and destination image
    float *h_Src = (float *)malloc(imageW * imageH * sizeof(float));
    float *h_Filter = (float *)malloc((2 * filterR + 1) * sizeof(float));
    float *h_Dst = (float *)malloc(imageW * imageH * sizeof(float));

    // Initialize your source image and filter (example: filling with random values)
    for (int i = 0; i < imageW * imageH; i++) {
        h_Src[i] = rand() % 10;  // Replace with your initialization logic
    }

    for (int i = 0; i < 2 * filterR + 1; i++) {
        h_Filter[i] = 0.1;  // Replace with your filter values
    }

    // Call the convolutionRowCPU function
    convolutionRowCPU(h_Dst, h_Src, h_Filter, imageW, imageH, filterR);

    // Display the result (for demonstration purposes)
    printf("Result:\n");
    for (int i = 0; i < imageH; i++) {
        for (int j = 0; j < imageW; j++) {
            printf("%.2f\t", h_Dst[i * imageW + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(h_Src);
    free(h_Filter);
    free(h_Dst);

    return 0;
}
 
