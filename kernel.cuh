#pragma once

struct uchar4;


void brighten(uchar4* arr, int w, int h);
void contrast(uchar4* arr, int w, int h);
void sharpen(uchar4* arr, int w, int h);
void saturate(uchar4* arr, int w, int h);
void hueShift(uchar4* arr, int w, int h);
