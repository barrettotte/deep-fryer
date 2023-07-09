#include "kernel.cuh"

#define cimg_display 0 // display functionality not needed
#include "CImg.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: deep-frier <IMAGE_PATH> [OUTPUT_PATH]\n");
        return -1;
    }
    string inputPath = filesystem::absolute(argv[1]).string();
    string outputPath = filesystem::absolute(argc == 3 ? argv[1] : "./out.png").string();
    // TODO: switch to argparse lib

    try {
        // TODO: refactor to function
        cimg_library::CImg<unsigned char> image(inputPath.c_str());
        const int w = image.width();
        const int h = image.height();
        cout << "Input: " << inputPath << ", w=" << w << ", h=" << h << endl;
        cout << "Output: " << outputPath << endl;

        // copy image to array
        uchar4 *imgArr = (uchar4*) malloc(w * h * sizeof(uchar4));
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                imgArr[y * w + x].x = image(x, y, 0); // red
                imgArr[y * w + x].y = image(x, y, 1); // blue
                imgArr[y * w + x].z = image(x, y, 2); // green
            }
        }

        // deep fry
        brighten(imgArr, w, h); // TODO:
        contrast(imgArr, w, h); // TODO:

        sharpen(imgArr, w, h, 160.0f);

        saturate(imgArr, w, h);  // TODO:
        hueShift(imgArr, w, h); // TODO:

        // posterization?
        // overexposure?
        // noise?
        // reddish-orange hue-shift?

        // copy results back
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < h; x++) {
                image(x, y, 0) = imgArr[y * w + x].x; // red
                image(x, y, 1) = imgArr[y * w + x].y; // blue
                image(x, y, 2) = imgArr[y * w + x].z; // green
            }
        }

        // save and cleanup
        image.save_png(outputPath.c_str());
        free(imgArr);

    } catch(cimg_library::CImgIOException& e) {
        return -2; // image failed to open, CImg prints exception
    }
    return 0;
}
