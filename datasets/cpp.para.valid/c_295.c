#include <stdio.h>

// Function prototype
void downsampleCpu(float *I, float *Q, unsigned int numDownsampledSamples, float *downsampled_I, float *downsampled_Q, unsigned int factor);

int main() {
    // Example data
    unsigned int numDownsampledSamples = 3;
    unsigned int factor = 2;
    float I[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float Q[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    float downsampled_I[numDownsampledSamples];
    float downsampled_Q[numDownsampledSamples];

    // Call the function
    downsampleCpu(I, Q, numDownsampledSamples, downsampled_I, downsampled_Q, factor);

    // Display the results
    printf("Downsampled Data:\n");
    for (int i = 0; i < numDownsampledSamples; i++) {
        printf("I=%.2f, Q=%.2f\n", downsampled_I[i], downsampled_Q[i]);
    }

    return 0;
}

// Function definition
void downsampleCpu(float *I, float *Q, unsigned int numDownsampledSamples, float *downsampled_I, float *downsampled_Q, unsigned int factor) {
    for (int sampleIndex = 0; sampleIndex < numDownsampledSamples; sampleIndex++) {
        unsigned int absoluteIndex = sampleIndex * factor;
        downsampled_I[sampleIndex] = I[absoluteIndex];
        downsampled_Q[sampleIndex] = Q[absoluteIndex];
    }
}
 
