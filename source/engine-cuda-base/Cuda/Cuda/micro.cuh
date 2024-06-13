#pragma once

#ifndef MICRO_H
#define MICRO_H

#define PI  (3.141592f)
#define RPI (0.3183099f)

#define FLOAT_MAX (3.4028235E38)
#define FLOAT_MIN (-3.4028234E38)

#define COPY4(a, b) {(a).x = (b).x; (a).y = (b).y; (a).z = (b).z; (a).w = (b).w;}

#define IF_int(flag, a, b) ( ((-flag) & (a - b)) + b)

#include "micro_smid_datatype.cuh"

#ifndef WITHIN_WIDTH
#define WITHIN_WIDTH

#define within_width4(v, index4, stride, width) {\
	v.x *= ((index4    ) % stride) < width;\
	v.y *= ((index4 + 1) % stride) < width;\
	v.z *= ((index4 + 2) % stride) < width;\
	v.w *= ((index4 + 3) % stride) < width;}

#define within_width8(v, index8, stride, width) {\
	v.x0 *= ((index8    ) % stride) < width;\
	v.y0 *= ((index8 + 1) % stride) < width;\
	v.z0 *= ((index8 + 2) % stride) < width;\
	v.w0 *= ((index8 + 3) % stride) < width;\
	v.x1 *= ((index8 + 4) % stride) < width;\
	v.y1 *= ((index8 + 5) % stride) < width;\
	v.z1 *= ((index8 + 6) % stride) < width;\
	v.w1 *= ((index8 + 7) % stride) < width;}

#define within_width16(v, index16, stride, width) {\
	v.x0 *= ((index16     ) % stride) < width;\
	v.y0 *= ((index16 +  1) % stride) < width;\
	v.z0 *= ((index16 +  2) % stride) < width;\
	v.w0 *= ((index16 +  3) % stride) < width;\
	v.x1 *= ((index16 +  4) % stride) < width;\
	v.y1 *= ((index16 +  5) % stride) < width;\
	v.z1 *= ((index16 +  6) % stride) < width;\
	v.w1 *= ((index16 +  7) % stride) < width;\
	v.x2 *= ((index16 +  8) % stride) < width;\
	v.y2 *= ((index16 +  9) % stride) < width;\
	v.z2 *= ((index16 + 10) % stride) < width;\
	v.w2 *= ((index16 + 11) % stride) < width;\
	v.x3 *= ((index16 + 12) % stride) < width;\
	v.y3 *= ((index16 + 13) % stride) < width;\
	v.z3 *= ((index16 + 14) % stride) < width;\
	v.w3 *= ((index16 + 15) % stride) < width;}

#endif

#endif