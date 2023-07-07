#include <stdio.h>
#include "kernel.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: deep-frier <IMAGE_PATH> [OUTPUT_PATH]\n");
        return 1;
    }
    // TODO: optional OUTPUT_PATH

    printf("Hello world\n");

    // open file

    // run kernel

    // write/close file

    return 0;
}
