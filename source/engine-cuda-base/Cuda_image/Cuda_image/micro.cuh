#pragma once

#ifndef MICRO_H
#define MICRO_H

#ifndef SIMD_DATA_TYPE
#define SIMD_DATA_TYPE

struct __device_builtin__ __align__(16) float8 {
	float x0, y0, z0, w0;
	float x1, y1, z1, w1;
};

struct __device_builtin__ __align__(16) float16 {
	float x0, y0, z0, w0;
	float x1, y1, z1, w1;
	float x2, y2, z2, w2;
	float x3, y3, z3, w3;
};


struct __device_builtin__ __align__(8) char8 {
	char x0, y0, z0, w0;
	char x1, y1, z1, w1;
};

struct __device_builtin__ __align__(16) char16 {
	char x0, y0, z0, w0;
	char x1, y1, z1, w1;
	char x2, y2, z2, w2;
	char x3, y3, z3, w3;
};


struct __device_builtin__ __align__(8) uchar8 {
	unsigned char x0, y0, z0, w0;
	unsigned char x1, y1, z1, w1;
};

struct __device_builtin__ __align__(16) uchar16 {
	unsigned char x0, y0, z0, w0;
	unsigned char x1, y1, z1, w1;
	unsigned char x2, y2, z2, w2;
	unsigned char x3, y3, z3, w3;
};


#define F32_0_4 float4{0, 0, 0, 0}
#define F32_0_8 float8{0, 0, 0, 0, 0, 0, 0, 0}
#define F32_0_16 float16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

#define UINT8_0_4 uchar4{0, 0, 0, 0}
#define UINT8_0_8 uchar8{0, 0, 0, 0, 0, 0, 0, 0}
#define UINT8_0_16 uchar16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

#endif

#define get2d(A, y, x, stride)   A[(y)*(stride) + (x)]
#define lget2d(A, y, x, lstride) A[((y)<<(lstride)) + (x)] //lstride = log2(stride)

#define get3d(A, z, y, x, Sy, Sx)    A[(((z)*(Sy)) + (y))*(Sx) + (x)]
#define lget3d(A, z, y, x, Sy, LSx)  A[((((z)*(Sy)) + y)<<(LSx)) + (x)]

#define get4d(A, w, z, y, x, Sz, Sy, Sx)    A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]
#define lget4d(A, w, z, y, x, Sz, LSy, LSx) A[(((((w)*(Sz) + (z))<<(LSy)) + (y))<<(LSx)) + (x)]

#define IF_int(flag, a, b) ( ((-flag) & (a - b)) + b)

#define COPY4(a, b) {(a).x = (b).x; (a).y = (b).y; (a).z = (b).z; (a).w = (b).w;}

#define MAX_INT(x, y) ((x)^(((x)^(y))& -((x)<(y))))
#define MIN_INT(x, y) ((y)^(((x)^(y))& -((x)<(y))))

#define in_range2D(y, x, Y, X) ((y>=0) && (y<Y) && (x>=0) && (x<X))
#define in_range3D(z, y, x, Z, Y, X) ((z>=0) && (z<Z) && (y>=0) && (y<Y) && (x>=0) && (x<X))

#define PI  (3.141592f)
#define RPI (0.3183099f)

#define FLOAT_MAX (3.4028235E38)
#define FLOAT_MIN (-3.4028234E38)


#ifndef PIXEL_CLIP_MICRO
#define PIXEL_CLIP_MICRO

//vmin = 0, vmax = 255
#define PIXEL_CLIP(x) ((x<255.0f && x>0.0f)*x + (x>=255.0f)*255.0f)

#define PIXEL_CLIP_4(a, b) {\
	a.x = PIXEL_CLIP(b.x);\
	a.y = PIXEL_CLIP(b.y);\
	a.z = PIXEL_CLIP(b.z);\
	a.w = PIXEL_CLIP(b.w);}

#define PIXEL_CLIP_8(a, b) {\
	a.x0 = PIXEL_CLIP(b.x0);\
	a.y0 = PIXEL_CLIP(b.y0);\
	a.z0 = PIXEL_CLIP(b.z0);\
	a.w0 = PIXEL_CLIP(b.w0);\
	a.x1 = PIXEL_CLIP(b.x1);\
	a.y1 = PIXEL_CLIP(b.y1);\
	a.z1 = PIXEL_CLIP(b.z1);\
	a.w1 = PIXEL_CLIP(b.w1);}

#define PIXEL_CLIP_16(a, b) {\
	a.x0 = PIXEL_CLIP(b.x0);\
	a.y0 = PIXEL_CLIP(b.y0);\
	a.z0 = PIXEL_CLIP(b.z0);\
	a.w0 = PIXEL_CLIP(b.w0);\
	a.x1 = PIXEL_CLIP(b.x1);\
	a.y1 = PIXEL_CLIP(b.y1);\
	a.z1 = PIXEL_CLIP(b.z1);\
	a.w1 = PIXEL_CLIP(b.w1);\
	a.x2 = PIXEL_CLIP(b.x2);\
	a.y2 = PIXEL_CLIP(b.y2);\
	a.z2 = PIXEL_CLIP(b.z2);\
	a.w2 = PIXEL_CLIP(b.w2);\
	a.x3 = PIXEL_CLIP(b.x3);\
	a.y3 = PIXEL_CLIP(b.y3);\
	a.z3 = PIXEL_CLIP(b.z3);\
	a.w3 = PIXEL_CLIP(b.w3);}

#endif


#ifndef WITHIN_WIDTH_MICRO
#define WITHIN_WIDTH_MICRO

//pay attention to nan caused by 0
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

//use more resource, but can zero nan caused by zero: 1, if within with
#define within_width4_zero_nan(v, index4, table, stride, width) {\
	table[1] = v;\
	v.x = table[((index4    ) % stride) < width].x;\
	v.y = table[((index4 + 1) % stride) < width].y;\
	v.z = table[((index4 + 2) % stride) < width].z;\
	v.w = table[((index4 + 3) % stride) < width].w;}
#endif


#ifndef REDUCE_MICRO
#define REDUCE_MICRO

__device__ __forceinline__ void warp_sum_4(volatile float *sdata, int index) {
	sdata[index] += sdata[index + 4];
	sdata[index] += sdata[index + 2];
	sdata[index] += sdata[index + 1];
}

__device__ __forceinline__ void warp_sum_8(volatile float *sdata, int index) {
	sdata[index] += sdata[index + 8];
	sdata[index] += sdata[index + 4];
	sdata[index] += sdata[index + 2];
	sdata[index] += sdata[index + 1];
}

#endif

#endif