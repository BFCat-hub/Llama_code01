#include <stdio.h>


void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)；

int main() {
    // Example data
    int n = 3;
    float pred[] = {1.0, 2.0, 3.0};
    float truth[] = {2.0, 3.0, 5.0};
    float delta[3];
    float error[3];

    // Function call
    l2_cpu(n, pred, truth, delta, error);

    // Output results
    printf("Delta array: ");
    for (int i = 0; i < n; ++i) {
        printf("%f ", delta[i]);
    }
    printf("\nError array: ");
    for (int i = 0; i < n; ++i) {
        printf("%f ", error[i]);
    }

    return 0;
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error) {
    int i;
    for (i = 0; i < n; ++i) {
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}
 
