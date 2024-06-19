#include <stdio.h>
#include <math.h>

void fractal_cpu(const int width, const int frames, unsigned char *const pic);

int main() {
    // Example parameters
    const int width = 800;
    const int frames = 30;
    unsigned char pic[width * width * frames];

    // Generate fractal
    fractal_cpu(width, frames, pic);

    // Print the result (for testing purposes)
    for (int frame = 0; frame < frames; frame++) {
        printf("Frame %d:\n", frame);
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < width; col++) {
                printf("%4d ", pic[frame * width * width + row * width + col]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}

void fractal_cpu(const int width, const int frames, unsigned char *const pic) {
    for (int i = 0; i < width * width * frames; i++) {
        const float Delta = 0.00304f;
        const float xMid = -0.055846456f;
        const float yMid = -0.668311119f;
        const int frame = i / (width * width);
        float delta = Delta * powf(0.975f, frame);
        const int col = i % width;
        const float xMin = xMid - delta;
        const float yMin = yMid - delta;
        const float dw = 2.0f * delta / width;
        const int row = (i / width) % width;
        const float cy = yMin + row * dw;
        const float cx = xMin + col * dw;
        float x = cx;
        float y = cy;
        float x2, y2;
        int count = 256;

        do {
            x2 = x * x;
            y2 = y * y;
            y = 2.0 * x * y + cy;
            x = x2 - y2 + cx;
            count--;
        } while ((count > 0) && ((x2 + y2) <= 5.0));

        pic[frame * width * width + row * width + col] = (unsigned char)count;
    }
}
