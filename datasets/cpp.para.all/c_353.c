#include <stdio.h>
#include <stdlib.h>

void convoluteCPU(float *dData, float *hData, int height, int width, float *mask, int masksize) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int S = (masksize - 1) / 2;
            float sum = 0;
            int pixPos = row * width + col;
            dData[pixPos] = 0.0;

            for (int maskrow = -S; maskrow <= S; maskrow++) {
                for (int maskcol = -S; maskcol <= S; maskcol++) {
                    int pixP = (row + maskrow) * width + (col + maskcol);
                    int maskP = (maskrow + S) * masksize + (maskcol + S);

                    if (pixP < height * width && pixP > 0 && maskP < masksize * masksize) {
                        sum += mask[maskP] * hData[pixP];
                    }
                }
            }

            dData[pixPos] = sum;

            if (dData[pixPos] < 0) {
                dData[pixPos] = 0;
            } else if (dData[pixPos] > 1) {
                dData[pixPos] = 1;
            }
        }
    }
}

int main() {
    // Test convoluteCPU function with a simple example
    int height = 3;
    int width = 3;
    int masksize = 3;

    float *dData = (float *)malloc(height * width * sizeof(float));
    float *hData = (float *)malloc(height * width * sizeof(float));
    float *mask = (float *)malloc(masksize * masksize * sizeof(float));

    // Initialize data (you may modify this part based on your actual data)
    for (int i = 0; i < height * width; i++) {
        hData[i] = (float)i;
    }

    // Initialize mask (you may modify this part based on your actual data)
    for (int i = 0; i < masksize * masksize; i++) {
        mask[i] = 0.1;
    }

    // Call the convoluteCPU function
    convoluteCPU(dData, hData, height, width, mask, masksize);

    // Display the results (you may modify this part based on your actual data)
    printf("Results after convoluteCPU function:\n");
    for (int i = 0; i < height * width; i++) {
        printf("%f ", dData[i]);
    }

    free(dData);
    free(hData);
    free(mask);

    return 0;
}
 
