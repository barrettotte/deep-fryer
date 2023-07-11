#pragma once

struct uchar4;


void brighten(uchar4* imgArr, int w, int h, float amt);
void contrast(uchar4* imgArr, int w, int h, float amt);
void sharpen(uchar4* imgArr, int w, int h, float amt);
void gaussianBlur(uchar4* imgArr, int w, int h);
void saturate(uchar4* imgArr, int w, int h, float amt);
void hueShift(uchar4* imgArr, int w, int h, float amt);
void redShift(uchar4* imgArr, int w, int h, float amt);
