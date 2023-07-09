#pragma once

struct uchar4;


void brighten(uchar4* imgArr, int w, int h);
void contrast(uchar4* imgArr, int w, int h);
void sharpen(uchar4* imgArr, int w, int h, float amount);
void gaussianBlur(uchar4* imgArr, int w, int h);
void saturate(uchar4* imgArr, int w, int h);
void hueShift(uchar4* imgArr, int w, int h);
