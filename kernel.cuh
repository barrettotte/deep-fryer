#pragma once

struct uchar4;

void brighten(uchar4* img, int w, int h, float amt);
void contrast(uchar4* img, int w, int h, float amt);
void sharpen(uchar4* img, int w, int h, float amt);
void saturate(uchar4* img, int w, int h, float amt);
void hueShift(uchar4* img, int w, int h, float amt);
void redShift(uchar4* img, int w, int h, float amt);
void posterize(uchar4* img, int w, int h, float amt);
void overexpose(uchar4* img, int w, int h, float amt);
