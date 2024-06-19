#include <stdio.h>
#include <stdlib.h>

void waterElevationToDepth_cpu(const int nx_, const int ny_, float *h_ptr_, int h_pitch_, float *Bm_ptr_, int Bm_pitch_) {
    for (int ti = 0; ti < nx_; ti++) {
        for (int tj = 0; tj < ny_; tj++) {
            float *h_row = (float *)((char *)h_ptr_ + h_pitch_ * tj);
            float *Bm_row = (float *)((char *)Bm_ptr_ + Bm_pitch_ * tj);
            h_row[ti] -= Bm_row[ti];
        }
    }
}

int main() {
    // Define your array dimensions
    int nx = 5; // Replace with your actual size
    int ny = 5; // Replace with your actual size

    // Allocate memory for h and Bm arrays
    float *h_ptr = (float *)malloc(nx * ny * sizeof(float));
    float *Bm_ptr = (float *)malloc(nx * ny * sizeof(float));

    // Initialize h and Bm arrays (example: filling with random values)
    for (int i = 0; i < nx * ny; i++) {
        h_ptr[i] = rand() % 100; // Replace with your initialization logic
        Bm_ptr[i] = rand() % 50; // Replace with your initialization logic
    }

    // Call the waterElevationToDepth_cpu function
    waterElevationToDepth_cpu(nx, ny, h_ptr, nx, Bm_ptr, nx);

    // Display the result (for demonstration purposes)
    printf("h Array:\n");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            printf("%8.2f\t", h_ptr[nx * i + j]);
        }
        printf("\n");
    }

    printf("\nBm Array:\n");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            printf("%8.2f\t", Bm_ptr[nx * i + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(h_ptr);
    free(Bm_ptr);

    return 0;
}
 
