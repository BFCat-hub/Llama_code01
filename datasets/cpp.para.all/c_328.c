#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void cuda_Adam_step_kernel(float *grad, float *data, float *m, float *v,
                            short decay, float weight_decay,
                            float beta1, float beta2,
                            float eps, float step_size, int varsize) {
    for (int i = 0; i < varsize; i++) {
        float g = grad[i];
        if (decay) g += weight_decay * data[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        data[i] -= step_size * m[i] / (sqrt(v[i]) + eps);
    }
}

int main() {
    // Define your array dimensions
    int varsize = 10;  // Replace with your actual variable size

    // Allocate memory for the arrays
    float *grad = (float *)malloc(varsize * sizeof(float));
    float *data = (float *)malloc(varsize * sizeof(float));
    float *m = (float *)malloc(varsize * sizeof(float));
    float *v = (float *)malloc(varsize * sizeof(float));

    // Initialize the arrays (example: filling with random values)
    for (int i = 0; i < varsize; i++) {
        grad[i] = rand() % 100;  // Replace with your initialization logic
        data[i] = rand() % 100;  // Replace with your initialization logic
        m[i] = rand() % 100;    // Replace with your initialization logic
        v[i] = rand() % 100;    // Replace with your initialization logic
    }

    // Set additional parameters
    short decay = 1;            // Example: setting decay to true
    float weight_decay = 0.01;  // Example: setting weight_decay

    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    float step_size = 0.001;

    // Call the cuda_Adam_step_kernel function
    cuda_Adam_step_kernel(grad, data, m, v, decay, weight_decay, beta1, beta2, eps, step_size, varsize);

    // Display the result (for demonstration purposes)
    printf("Updated Data:\n");
    for (int i = 0; i < varsize; i++) {
        printf("%8.4f\t", data[i]);
    }
    printf("\n");

    // Free allocated memory
    free(grad);
    free(data);
    free(m);
    free(v);

    return 0;
}
 
