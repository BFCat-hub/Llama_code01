#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void fractal_cpu(const int width, const int frames, unsigned char *const pic) {
    for (int i = 0; i < width * width * frames; i++) {
        const double Delta = 0.00304;
        const double xMid = -0.055846456;
        const double yMid = -0.668311119;

        const int frame = i / (width * width);
        double delta = Delta * pow(0.975, frame);

        const int col = i % width;
        const double xMin = xMid - delta;

        const int row = (i / width) % width;
        const double yMin = yMid - delta;

        const double dw = 2.0 * delta / width;
        const double cy = yMin + row * dw;
        const double cx = xMin + col * dw;

        double x = cx;
        double y = cy;
        double x2, y2;
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

int main() {
    // Test fractal_cpu function with a simple example
    int width = 512;
    int frames = 3;
    unsigned char *pic = (unsigned char *)malloc(width * width * frames * sizeof(unsigned char));

    // Call the fractal_cpu function
    fractal_cpu(width, frames, pic);

    // Display the results
    printf("Results after fractal_cpu function:\n");
    for (int i = 0; i < width * width * frames; i++) {
        printf("%d ", pic[i]);
    }

    free(pic);
    return 0;
}
 
