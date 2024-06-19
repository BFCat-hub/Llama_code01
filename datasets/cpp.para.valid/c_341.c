#include <stdio.h>

void cpuRunComplexFilter(float *I, float *Q, int samplesLength, float *hr, float *hi,
                          int filterLength, float *filtered_I, float *filtered_Q, int convLength) {
    for (int sampleIndex = 0; sampleIndex < convLength; sampleIndex++) {
        int index;
        float sumI, sumQ;
        sumI = 0;
        sumQ = 0;
        for (int j = sampleIndex - filterLength + 1; j <= sampleIndex; j++) {
            index = sampleIndex - j;
            if ((j < samplesLength) && (j >= 0)) {
                sumI += (I[j] * hr[index]) - (Q[j] * hi[index]);
                sumQ += (I[j] * hi[index]) + (Q[j] * hr[index]);
            }
        }
        filtered_I[sampleIndex] = sumI;
        filtered_Q[sampleIndex] = sumQ;
    }
}

int main() {
    // Test cpuRunComplexFilter function with a simple example
    int samplesLength = 5;
    int filterLength = 3;
    int convLength = samplesLength + filterLength - 1;

    float I[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float Q[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float hr[] = {0.5, 0.3, 0.1};
    float hi[] = {0.2, 0.4, 0.6};
    float filtered_I[convLength];
    float filtered_Q[convLength];

    cpuRunComplexFilter(I, Q, samplesLength, hr, hi, filterLength, filtered_I, filtered_Q, convLength);

    printf("Filtered I:\n");
    for (int i = 0; i < convLength; i++) {
        printf("%.2f ", filtered_I[i]);
    }
    printf("\n");

    printf("Filtered Q:\n");
    for (int i = 0; i < convLength; i++) {
        printf("%.2f ", filtered_Q[i]);
    }
    printf("\n");

    return 0;
}
 
