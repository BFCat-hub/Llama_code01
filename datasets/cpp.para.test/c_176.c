#include <stdio.h>

void decode_cpu(const float *anchor, const float *locData, float *predictBox, int dims, float scaleClamp, int batchSize);

int main() {
    // Example parameters
    const int dims = 4;
    const int batchSize = 2;
    const float scaleClamp = 5.0;

    // Example input data (replace with your actual data)
    float anchor[batchSize * dims * 4];
    float locData[batchSize * dims * 4];
    float predictBox[batchSize * dims * 4];

    // Initialize your input data here (replace with your actual data)
    for (int i = 0; i < batchSize * dims * 4; ++i) {
        anchor[i] = i + 1;
        locData[i] = (i + 1) * 0.1;
    }

    // Call decode_cpu function
    decode_cpu(anchor, locData, predictBox, dims, scaleClamp, batchSize);

    // Print the results or add further processing as needed
    for (int i = 0; i < batchSize * dims * 4; ++i) {
        printf("%f ", predictBox[i]);
    }

    return 0;
}

void decode_cpu(const float *anchor, const float *locData, float *predictBox, int dims, float scaleClamp, int batchSize) {
    for (int tid = 0; tid < dims; tid++) {
        for (int i = 0; i < batchSize; i++) {
            float anchorW = anchor[i * dims * 4 + tid * 4 + 2] - anchor[i * dims * 4 + tid * 4];
            float anchorH = anchor[i * dims * 4 + tid * 4 + 3] - anchor[i * dims * 4 + tid * 4 + 1];
            float anchorCx = anchor[i * dims * 4 + tid * 4] + 0.5 * anchorW;
            float anchorCy = anchor[i * dims * 4 + tid * 4 + 1] + 0.5 * anchorH;

            float dx = locData[i * dims * 4 + tid * 4];
            float dy = locData[i * dims * 4 + tid * 4 + 1];
            float dw = locData[i * dims * 4 + tid * 4 + 2];
            float dh = locData[i * dims * 4 + tid * 4 + 3];

            if (dw > scaleClamp) {
                dw = scaleClamp;
            }
            if (dh > scaleClamp) {
                dh = scaleClamp;
            }

            float preCx = dx * anchorW + anchorCx;
            float preCy = dy * anchorH + anchorCy;
            float preW = anchorW * 0.5;
            float preH = anchorH * 0.5;

            predictBox[i * dims * 4 + tid * 4] = preCx - 0.5 * preW;
            predictBox[i * dims * 4 + tid * 4 + 1] = preCy - 0.5 * preH;
            predictBox[i * dims * 4 + tid * 4 + 2] = preCx + 0.5 * preW;
            predictBox[i * dims * 4 + tid * 4 + 3] = preCy + 0.5 * preH;
        }
    }
}
