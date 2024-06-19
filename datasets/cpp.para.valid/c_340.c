#include <stdio.h>

void invalidateFlow_cpu(float *modFlowX, float *modFlowY, const float *constFlowX,
                         const float *constFlowY, int width, int height, float cons_thres) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int ind = y * width + x;
            float mFX = modFlowX[ind];
            float mFY = modFlowY[ind];
            float cFX = constFlowX[ind];
            float cFY = constFlowY[ind];
            float err = (mFX - cFX) * (mFX - cFX) + (mFY - cFY) * (mFY - cFY);
            if (err > cons_thres) {
                mFX = 0;
                mFY = 0;
            }
            modFlowX[ind] = mFX;
            modFlowY[ind] = mFY;
        }
    }
}

int main() {
    // Test invalidateFlow_cpu function with a simple example
    int width = 4;
    int height = 4;

    float modFlowX[width * height];
    float modFlowY[width * height];
    float constFlowX[width * height];
    float constFlowY[width * height];

    // Initialize modFlowX, modFlowY, constFlowX, and constFlowY arrays
    for (int i = 0; i < width * height; i++) {
        modFlowX[i] = i + 1;  // Just an example, you can modify this based on your needs
        modFlowY[i] = 2 * i + 1;  // Just an example, you can modify this based on your needs
        constFlowX[i] = 3 * i + 1;  // Just an example, you can modify this based on your needs
        constFlowY[i] = 4 * i + 1;  // Just an example, you can modify this based on your needs
    }

    printf("Modified Flow X:\n");
    for (int i = 0; i < width * height; i++) {
        printf("%.2f ", modFlowX[i]);
    }
    printf("\n");

    printf("Modified Flow Y:\n");
    for (int i = 0; i < width * height; i++) {
        printf("%.2f ", modFlowY[i]);
    }
    printf("\n");

    printf("Constant Flow X:\n");
    for (int i = 0; i < width * height; i++) {
        printf("%.2f ", constFlowX[i]);
    }
    printf("\n");

    printf("Constant Flow Y:\n");
    for (int i = 0; i < width * height; i++) {
        printf("%.2f ", constFlowY[i]);
    }
    printf("\n");

    // Call invalidateFlow_cpu function
    invalidateFlow_cpu(modFlowX, modFlowY, constFlowX, constFlowY, width, height, 10.0);

    printf("Invalidated Flow X:\n");
    for (int i = 0; i < width * height; i++) {
        printf("%.2f ", modFlowX[i]);
    }
    printf("\n");

    printf("Invalidated Flow Y:\n");
    for (int i = 0; i < width * height; i++) {
        printf("%.2f ", modFlowY[i]);
    }
    printf("\n");

    return 0;
}
 
