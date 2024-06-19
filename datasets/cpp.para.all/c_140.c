#include <stdio.h>
#include <math.h>

void GraphSum_forward(float *in, float *out, int *indptr, int *indices, int dim, int size) {
    for (int src = 0; src < size - 1; src++) {
        for (int i = indptr[src]; i < indptr[src + 1]; i++) {
            int dst = indices[i];
            float coef = 1.0 / sqrtf((indptr[src + 1] - indptr[src]) * (indptr[dst + 1] - indptr[dst]));

            for (int j = 0; j < dim; j++) {
                out[src * dim + j] += coef * in[dst * dim + j];
            }
        }
    }
}

int main() {
    // Example usage
    int dim = 3; // Assuming a 3-dimensional vector
    int size = 4; // Assuming a graph with 4 nodes

    // Example graph structure represented by CSR format
    int indptr[5] = {0, 2, 3, 5, 7};
    int indices[7] = {1, 2, 0, 3, 0, 2, 3};

    // Assuming your input and output arrays are flat 1D arrays
    float in[size * dim];
    float out[size * dim];

    // Initialize your arrays with some values (for example, 1.0 for simplicity)
    for (int i = 0; i < size * dim; i++) {
        in[i] = 1.0f;
        out[i] = 0.0f; // Initialize output array with zeros
    }

    // Call the function to perform the graph sum operation
    GraphSum_forward(in, out, indptr, indices, dim, size);

    // Print the result (for demonstration purposes)
    for (int i = 0; i < size * dim; i++) {
        printf("%f ", out[i]);
    }

    return 0;
}
