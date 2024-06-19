#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ENDCOM

void convolutionCPU(float *host_outputMatrix, float *host_inputMatrix, float *host_filter, int imageRows, int imageColumns, int filterSize) {
    #pragma omp parallel for ENDCOM
    for (int eachRowOfImage = 0; eachRowOfImage < (int)imageRows; ++eachRowOfImage) {
        for (int eachColumnOfImage = 0; eachColumnOfImage < (int)imageColumns; ++eachColumnOfImage) {
            float convolvedValue = 0.f;

            for (int eachRowOfFilter = -filterSize / 2; eachRowOfFilter <= filterSize / 2; ++eachRowOfFilter) {
                for (int eachColumnOfFilter = -filterSize / 2; eachColumnOfFilter <= filterSize / 2; ++eachColumnOfFilter) {
                    int imageRow = eachRowOfImage + eachRowOfFilter;
                    int imageColumn = eachColumnOfImage + eachColumnOfFilter;

                    float pixelValue = (imageRow >= 0 && imageRow < imageRows && imageColumn >= 0 && imageColumn < imageColumns)
                        ? host_inputMatrix[imageRow * imageColumns + imageColumn]
                        : 0.f;

                    float filterValue = host_filter[(eachRowOfFilter + filterSize / 2) * filterSize + eachColumnOfFilter + filterSize / 2];

                    convolvedValue += pixelValue * filterValue;
                }
            }

            host_outputMatrix[eachRowOfImage * imageColumns + eachColumnOfImage] = convolvedValue;
        }
    }
}

int main() {
    // Test convolutionCPU function with a simple example
    int imageRows = 4;
    int imageColumns = 4;
    int filterSize = 3;

    float *host_outputMatrix = (float *)malloc(imageRows * imageColumns * sizeof(float));
    float *host_inputMatrix = (float *)malloc(imageRows * imageColumns * sizeof(float));
    float *host_filter = (float *)malloc(filterSize * filterSize * sizeof(float));

    // Initialize data (you may modify this part based on your actual data)
    for (int i = 0; i < imageRows * imageColumns; i++) {
        host_inputMatrix[i] = (float)i;
    }

    for (int i = 0; i < filterSize * filterSize; i++) {
        host_filter[i] = 1.0f;
    }

    // Call the convolutionCPU function
    convolutionCPU(host_outputMatrix, host_inputMatrix, host_filter, imageRows, imageColumns, filterSize);

    // Display the results (you may modify this part based on your actual data)
    printf("Results after convolutionCPU function:\n");
    for (int i = 0; i < imageRows * imageColumns; i++) {
        printf("%f ", host_outputMatrix[i]);
    }

    free(host_outputMatrix);
    free(host_inputMatrix);
    free(host_filter);

    return 0;
}
 
