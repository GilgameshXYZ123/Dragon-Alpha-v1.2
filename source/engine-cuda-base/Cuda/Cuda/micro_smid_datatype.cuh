#pragma once

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
