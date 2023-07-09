#include "kernel.cuh"

#define TX 32
#define TY 32
#define RAD 1

void brighten(uchar4* imgArr, int w, int h) {
    // TODO:
}

void contrast(uchar4* imgArr, int w, int h) {
    // TODO:
}

__global__ void sharpenKernel(uchar4* d_out, const uchar4* d_in, int w, int h, float amount) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = y * w + x;
        float4 blur = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float blurWeights = 0.0f;

        // apply blur to image, "unsharp masking"
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                int px = min(max(x + i, 0), w - 1);
                int py = min(max(y + j, 0), h - 1);
                int pi = py * w + px;
                
                float4 pixel = make_float4(d_in[pi].x, d_in[pi].y, d_in[pi].z, d_in[pi].w);
                float weight = (i == 0 && j == 0) ? 25.0f : 1.0f;
                
                blur.x += pixel.x * weight;
                blur.y += pixel.y * weight;
                blur.z += pixel.z * weight;
                blur.w += pixel.w * weight;

                blurWeights += weight;
            }
        }
        blur.x /= blurWeights;
        blur.y /= blurWeights;
        blur.z /= blurWeights;
        blur.w /= blurWeights;

        // compute sharpened pixel
        float4 sharp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 pixel = make_float4(d_in[idx].x, d_in[idx].y, d_in[idx].z, d_in[idx].w);
        sharp.x = pixel.x + amount * (pixel.x - blur.x);
        sharp.y = pixel.y + amount * (pixel.y - blur.y);
        sharp.z = pixel.z + amount * (pixel.z - blur.z);
        sharp.w = pixel.w;

        d_out[idx].x = min(max(static_cast<int>(sharp.x), 0), 255);
        d_out[idx].y = min(max(static_cast<int>(sharp.y), 0), 255);
        d_out[idx].z = min(max(static_cast<int>(sharp.z), 0), 255);
        d_out[idx].w = sharp.w; // preserve alpha
    }
}

void sharpen(uchar4* imgArr, int w, int h, float amount) {
    uchar4* d_in = 0;
    uchar4* d_out = 0;
    size_t imageSize = w * h * sizeof(uchar4);

    cudaMalloc(&d_in, imageSize);
    cudaMalloc(&d_out, imageSize);

    const dim3 blockSize(TX, TY);
    const dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    cudaMemcpy(d_in, imgArr, imageSize, cudaMemcpyHostToDevice);
    sharpenKernel<<<gridSize, blockSize>>>(d_out, d_in, w, h, amount);
    cudaDeviceSynchronize();
    cudaMemcpy(imgArr, d_out, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);
}

__global__ void gaussianBlurKernel(uchar4* d_out, const uchar4* d_in, int w, int h) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = y * w + x;
        float kernel[9] = {0.0625f, 0.125f, 0.1875f, 0.25f, 0.1875f, 0.125f, 0.0625f};
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int i = -3; i <= 3; i++) {
            int px = min(max(x + i, 0), w - 1);
            int pi = y * w + px;
            float4 pixel = make_float4(d_in[pi].x, d_in[pi].y, d_in[pi].z, d_in[pi].w);

            sum.x += pixel.x * kernel[i + 3];
            sum.y += pixel.y * kernel[i + 3];
            sum.z += pixel.z * kernel[i + 3];
            sum.w += pixel.w * kernel[i + 3];
        }
        d_out[idx].x = min(max(static_cast<int>(sum.x), 0), 255);
        d_out[idx].y = min(max(static_cast<int>(sum.y), 0), 255);
        d_out[idx].z = min(max(static_cast<int>(sum.z), 0), 255);
        d_out[idx].w = d_in[idx].w;
    }
}

void gaussianBlur(uchar4* imgArr, int w, int h) {
    uchar4* d_in = 0;
    uchar4* d_out = 0;
    size_t imageSize = w * h * sizeof(uchar4);

    cudaMalloc(&d_in, imageSize);
    cudaMalloc(&d_out, imageSize);

    const dim3 blockSize(TX, TY);
    const dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    cudaMemcpy(d_in, imgArr, imageSize, cudaMemcpyHostToDevice);
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_out, d_in, w, h);
    cudaDeviceSynchronize();
    cudaMemcpy(imgArr, d_out, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);
}

void saturate(uchar4* imgArr, int w, int h) {
    // TODO:
}

void hueShift(uchar4* imgArr, int w, int h) {
    // TODO:
}
