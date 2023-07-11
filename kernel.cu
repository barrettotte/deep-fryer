#include "kernel.cuh"

#define TX 32
#define TY 32

__device__ float3 rgbToHsv(float3 rgb) {
    float r = rgb.x;
    float g = rgb.y;
    float b = rgb.z;
    float maxVal = fmaxf(fmaxf(r, g), b);
    float minVal = fminf(fminf(r, g), b);

    float hue = 0.0f;
    float saturation = 0.0f;
    float val = maxVal;
    float delta = maxVal - minVal;

    if (maxVal != 0.0f) {
        saturation = delta / maxVal;
        
        if (delta != 0.0f) {
            if (maxVal == r) {
                hue = (g - b) / delta + (g < b ? 6.0f : 0.0f);
            } else if (maxVal == g) {
                hue = (b - r) / delta + 2.0f;
            } else {
                hue = (r - g) / delta + 4.0f;
            }
            hue /= 6.0f;
        }
    }
    return make_float3(hue, saturation, val);
}

__device__ float3 hsvToRgb(float3 hsv) {
    float hue = hsv.x;
    float saturation = hsv.y;
    float val = hsv.z;

    float r = val;
    float g = val;
    float b = val;

    if (saturation != 0.0f) {
        hue *= 6.0f;
        int i = static_cast<int>(hue);
        float f = hue - i;

        float p = val * (1.0f - saturation);
        float q = val * (1.0f - saturation * f);
        float t = val * (1.0f - saturation * (1.0f - f));

        switch (i) {
            case 0: return make_float3(val, t, p);
            case 1: return make_float3(q, val, p);
            case 2: return make_float3(p, val, t);
            case 3: return make_float3(p, q, val);
            case 4: return make_float3(t, p, val);
            case 5: return make_float3(val, p, q);
        }
    }
    return make_float3(r, g, b);
}

void brighten(uchar4* imgArr, int w, int h) {
    // TODO:
}

__global__ void contrastKernel(uchar4* d_out, const uchar4* d_in, int w, int h, float amt) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = y * w + x;
        uchar4 pixel = d_in[idx];
        float4 pixelFloat = make_float4(pixel.x / 255.0f, pixel.y / 255.0f, pixel.z / 255.0f, pixel.w);

        pixelFloat.x = (pixelFloat.x - 0.5f) * amt + 0.5f;
        pixelFloat.y = (pixelFloat.y - 0.5f) * amt + 0.5f;
        pixelFloat.z = (pixelFloat.z - 0.5f) * amt + 0.5f;

        pixel.x = min(max(static_cast<int>(pixelFloat.x * 255.0f), 0), 255);
        pixel.y = min(max(static_cast<int>(pixelFloat.y * 255.0f), 0), 255);
        pixel.z = min(max(static_cast<int>(pixelFloat.z * 255.0f), 0), 255);
        pixel.w = pixelFloat.w;

        d_out[idx] = pixel;
    }
}

void contrast(uchar4* imgArr, int w, int h, float amt) {
    uchar4* d_in = 0;
    uchar4* d_out = 0;
    size_t imgSize = w * h * sizeof(uchar4);

    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);

    const dim3 blockSize(TX, TY);
    const dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    cudaMemcpy(d_in, imgArr, imgSize, cudaMemcpyHostToDevice);
    contrastKernel<<<gridSize, blockSize>>>(d_out, d_in, w, h, amt);
    cudaDeviceSynchronize();
    cudaMemcpy(imgArr, d_out, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);
}

__global__ void sharpenKernel(uchar4* d_out, const uchar4* d_in, int w, int h, float amt) {
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
        sharp.x = pixel.x + amt * (pixel.x - blur.x);
        sharp.y = pixel.y + amt * (pixel.y - blur.y);
        sharp.z = pixel.z + amt * (pixel.z - blur.z);
        sharp.w = pixel.w;

        d_out[idx].x = min(max(static_cast<int>(sharp.x), 0), 255);
        d_out[idx].y = min(max(static_cast<int>(sharp.y), 0), 255);
        d_out[idx].z = min(max(static_cast<int>(sharp.z), 0), 255);
        d_out[idx].w = sharp.w; // preserve alpha
    }
}

void sharpen(uchar4* imgArr, int w, int h, float amt) {
    uchar4* d_in = 0;
    uchar4* d_out = 0;
    size_t imgSize = w * h * sizeof(uchar4);

    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);

    const dim3 blockSize(TX, TY);
    const dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    cudaMemcpy(d_in, imgArr, imgSize, cudaMemcpyHostToDevice);
    sharpenKernel<<<gridSize, blockSize>>>(d_out, d_in, w, h, amt);
    cudaDeviceSynchronize();
    cudaMemcpy(imgArr, d_out, imgSize, cudaMemcpyDeviceToHost);

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
    size_t imgSize = w * h * sizeof(uchar4);

    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);

    const dim3 blockSize(TX, TY);
    const dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    cudaMemcpy(d_in, imgArr, imgSize, cudaMemcpyHostToDevice);
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_out, d_in, w, h);
    cudaDeviceSynchronize();
    cudaMemcpy(imgArr, d_out, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);
}

__global__ void saturateKernel(uchar4* d_out, const uchar4* d_in, int w, int h, float amt) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = y * w + x;
        uchar4 pixel = d_in[idx];
        float3 pixelFloat = make_float3(pixel.x / 255.0f, pixel.y / 255.0f, pixel.z / 255.0f);
        float3 hsv = rgbToHsv(pixelFloat);

        hsv.y *= amt;
        pixelFloat = hsvToRgb(hsv);

        pixel.x = min(max(static_cast<int>(pixelFloat.x * 255.0f), 0), 255);
        pixel.y = min(max(static_cast<int>(pixelFloat.y * 255.0f), 0), 255);
        pixel.z = min(max(static_cast<int>(pixelFloat.z * 255.0f), 0), 255);

        d_out[idx].x = pixel.x;
        d_out[idx].y = pixel.y;
        d_out[idx].z = pixel.z;
        d_out[idx].w = d_in[idx].w; // preserve alpha
    }
}

void saturate(uchar4* imgArr, int w, int h, float amt) {
    uchar4* d_in = 0;
    uchar4* d_out = 0;
    size_t imgSize = w * h * sizeof(uchar4);

    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);

    const dim3 blockSize(TX, TY);
    const dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    cudaMemcpy(d_in, imgArr, imgSize, cudaMemcpyHostToDevice);
    saturateKernel<<<gridSize, blockSize>>>(d_out, d_in, w, h, amt);
    cudaDeviceSynchronize();
    cudaMemcpy(imgArr, d_out, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);
}

void hueShift(uchar4* imgArr, int w, int h) {
    // TODO:
}
