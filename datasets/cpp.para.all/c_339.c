#include <stdio.h>

void waterDepthToElevation_cpu(const int nx_, const int ny_, float *w_ptr_, int w_pitch_,
                               float *h_ptr_, int h_pitch_, float *Bm_ptr_, int Bm_pitch_) {
    for (int ti = 0; ti < nx_; ti++) {
        for (int tj = 0; tj < ny_; tj++) {
            float *h_row = (float *)((char *)h_ptr_ + h_pitch_ * tj);
            float *Bm_row = (float *)((char *)Bm_ptr_ + Bm_pitch_ * tj);
            float *w_row = (float *)((char *)w_ptr_ + w_pitch_ * tj);
            w_row[ti] = h_row[ti] + Bm_row[ti];
        }
    }
}

int main() {
    // Test waterDepthToElevation_cpu function with a simple example
    int nx = 4;
    int ny = 4;

    float waterDepth[nx * ny];
    float elevation[nx * ny];
    float bedrockElevation[nx * ny];

    // Initialize waterDepth and bedrockElevation arrays
    for (int i = 0; i < nx * ny; i++) {
        waterDepth[i] = i + 1;  // Just an example, you can modify this based on your needs
        bedrockElevation[i] = 2 * i + 1;  // Just an example, you can modify this based on your needs
    }

    printf("Water Depth:\n");
    for (int i = 0; i < nx * ny; i++) {
        printf("%.2f ", waterDepth[i]);
    }
    printf("\n");

    printf("Bedrock Elevation:\n");
    for (int i = 0; i < nx * ny; i++) {
        printf("%.2f ", bedrockElevation[i]);
    }
    printf("\n");

    // Call waterDepthToElevation_cpu function
    waterDepthToElevation_cpu(nx, ny, waterDepth, nx, elevation, nx, bedrockElevation, nx);

    printf("Elevation:\n");
    for (int i = 0; i < nx * ny; i++) {
        printf("%.2f ", elevation[i]);
    }
    printf("\n");

    return 0;
}
 
