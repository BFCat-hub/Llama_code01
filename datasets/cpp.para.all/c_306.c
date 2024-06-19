#include <stdio.h>

// Function prototype
void fa_cpu(const float *q, const float *h, int nq, float *a, float *fa);

int main() {
    // Example data
    int nq = 4;
    float q[] = {1.0, 2.0, 3.0, 4.0};
    float h[] = {0.5, 1.0, 1.5, 2.0};
    float a[nq - 1], fa[nq - 1];

    // Call the function
    fa_cpu(q, h, nq, a, fa);

    // Display the results
    printf("Acceleration (a) Array:\n");
    for (int i = 0; i < nq - 1; ++i) {
        printf("%.2f ", a[i]);
    }

    printf("\n\n");

    printf("Result (fa) Array:\n");
    for (int i = 0; i < nq - 1; ++i) {
        printf("%.2f ", fa[i]);
    }

    return 0;
}

// Function definition
void fa_cpu(const float *q, const float *h, int nq, float *a, float *fa) {
    for (int iq = 0; iq < (nq - 1); iq++) {
        float dq = q[iq + 1] - q[iq];
        a[iq] = (h[iq + 1] * q[iq + 1] - h[iq] * q[iq]) / dq;
        fa[iq] = q[iq] * (a[iq] - h[iq]) + 1.0;
    }
}
 
