#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void fractal(const int width, const int frames, unsigned char *const pic) {
    long i = threadIdx.x + blockIdx.x * (long)blockDim.x;

    if (i >= width * width * frames) {
        return;
    }

    const double Delta = 0.00304;
    const double xMid = -0.055846456;
    const double yMid = -0.668311119;

    int frame = i / (width * width);
    double delta = Delta * pow(0.975, frame);

    int col = i % width;
    double xMin = xMid - delta;
    double yMin = yMid - delta;

    double dw = 2.0 * delta / width;
    int row = (i / width) % width;

    double cy = yMin + row * dw;
    double cx = xMin + col * dw;

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

int main() {
    const int width = 512; // Adjust the width based on your data
    const int frames = 30; // Adjust the number of frames based on your requirement

    unsigned char *pic;

    // Allocate host memory
    pic = (unsigned char *)malloc(width * width * frames * sizeof(unsigned char));

    // Allocate device memory
    unsigned char *d_pic;
    cudaMalloc((void **)&d_pic, width * width * frames * sizeof(unsigned char));

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((width * width * frames + block_size.x - 1) / block_size.x);

    // Launch the kernel
    fractal<<<grid_size, block_size>>>(width, frames, d_pic);

    // Copy data from device to host
    cudaMemcpy(pic, d_pic, width * width * frames * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_pic);
    free(pic);

    return 0;
}
 
