#include "kernel.cuh"

#define TX 32
#define TY 32
#define RAD 1

int divUp(int a, int b) {
    return (a + b - 1) / b;
}

__device__ unsigned char clip(int n) {
    return n > 255 ? 255 : (n < 0 ? 0 : n);
}

__device__ int idxClip(int idx, int idxMax) {
    return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__ int flatten(int col, int row, int width, int height) {
    return idxClip(col, width) + idxClip(row, height) * width;
}

__global__ void sharpenKernel(uchar4* d_out, const uchar4* d_in, const float* d_filter, int w, int h) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((x >= w) || (y >= h)) {
        return;
    }
    const int i = flatten(x, y, w, h);
    const int filterSize = 2 * RAD + 1;
    float rgb[3] = {0.0f, 0.0f, 0.0f};

    // apply filter to each pixel
    for (int yd = -RAD; yd <= RAD; yd++) {
        for (int xd = -RAD; xd <= RAD; xd++) {
            int imgIdx = flatten(x + xd, y + yd, w, h);
            int filterIdx = flatten(RAD + xd, RAD + yd, filterSize, filterSize);
            uchar4 color = d_in[imgIdx];
            float weight = d_filter[filterIdx];

            rgb[0] += weight * color.x;
            rgb[1] += weight * color.y;
            rgb[2] += weight * color.z;
        }
    }
    d_out[i].x = clip(rgb[0]);
    d_out[i].y = clip(rgb[1]);
    d_out[i].z = clip(rgb[2]);
}

void brighten(uchar4* arr, int w, int h) {
    // TODO:
}

void contrast(uchar4* arr, int w, int h) {
    // TODO:
}

void sharpen(uchar4 *arr, int w, int h) {
    const float filter[9] = {
        -1.0f, -1.0f, -1.0f,
        -1.0f, 9.0f, -1.0f,
        -1.0f, -1.0f, -1.0f
    };
    const int filterSize = 2 * RAD + 1;

    float* d_filter = 0;
    uchar4* d_in = 0;
    uchar4* d_out = 0;

    cudaMalloc(&d_filter, filterSize * filterSize * sizeof(float));
    cudaMalloc(&d_in, w * h * sizeof(uchar4));
    cudaMalloc(&d_out, w * h * sizeof(uchar4));

    cudaMemcpy(d_filter, filter, filterSize * filterSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, arr, w * h * sizeof(uchar4), cudaMemcpyHostToDevice);

    const dim3 blockSize(TX, TY);
    const dim3 gridSize(divUp(w, blockSize.x), divUp(h, blockSize.y));

    sharpenKernel<<<gridSize, blockSize>>>(d_out, d_in, d_filter, w, h);
    cudaMemcpy(arr, d_out, w * h* sizeof(uchar4), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_filter);
}

void saturate(uchar4* arr, int w, int h) {
    // TODO:
}

void hueShift(uchar4* arr, int w, int h) {
    // TODO:
}
