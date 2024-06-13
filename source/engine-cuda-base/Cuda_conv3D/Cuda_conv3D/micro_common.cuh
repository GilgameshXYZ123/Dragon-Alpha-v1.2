#pragma once

#ifndef MICRO_COMMON_H
#define MICRO_COMMON_H

#define GRID_MAX 8192

#define MAX_STREAM_SIZE 10

#define concat_float4(a, v0, v1) {\
	a[0] = v0.x; a[1] = v0.y; a[2] = v0.z; a[3] = v0.w;\
	a[4] = v1.x; a[5] = v1.y; a[6] = v1.z; a[7] = v1.w;}


__device__ float HOLE[260];
__device__ __constant__ float _ZERO = 0;

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


//XIdx = ((XIdx >> 1) << 1) + (ux & 1£©
//ux = (ux >> 1) + (XIdx & 1) * 8


//(XIdx: 4, ux: 16)                                                  |8: XIdx % 2 != 0
//(0, 0), (0, 2), (0, 4), (0, 6), (0, 8), (0, 10), (0, 12), (0, 14), |(1, 0), (1, 2), (1, 4), (1, 6), (1, 8), (1, 10), (1, 12), (1, 14) 
//(0, 1), (0, 3), (0, 5), (0, 7), (0, 9), (0, 11), (0, 13), (0, 15), |(1, 1), (1, 3), (1, 5), (1, 7), (1, 9), (1, 11), (1, 13), (1, 15)
//(2, 0), (2, 2), (2, 4), (2, 6), (2, 8), (2, 10), (2, 12), (2, 14), |(3, 0), (3, 2), (3, 4), (3, 6), (3, 8), (3, 10), (3, 12), (3, 14)
//(2, 1), (2, 3), (2, 5), (2, 7), (2, 9), (2, 11), (2, 13), (2, 15), |(3, 1), (3, 3), (3, 5), (3, 7), (3, 9), (3, 11), (3, 13), (3, 15)
__device__ __constant__ int2 Winograd_Xs_Idx[4 * 16]{//[XIdx: 4, ux: 16]
	int2{0, 0}, int2{1, 0}, int2{1, 0}, int2{0, 6}, int2{0, 8}, int2{0, 0},
	int2{0, 1}, int2{0, 3}, int2{0, 5}, int2{0, 7}, int2{0, 9},

};


#define F32_2_0 float2{ 0, 0 }
#define F32_4_0 float4{ 0, 0, 0, 0 }


#define IS_POWER2(x) ( ((x)>0) && ((x)&((x)-1))==0 )

#define get2d(A, y, x, stride)   A[(y)*(stride) + (x)]
#define lget2d(A, y, x, lstride) A[((y)<<(lstride)) + (x)] //lstride = log2(stride)

#define get3d(A, z, y, x, Sy, Sx)    A[(((z)*(Sy)) + (y))*(Sx) + (x)]
#define lget3d(A, z, y, x, Sy, LSx)  A[((((z)*(Sy)) + y)<<(LSx)) + (x)]

#define get4d(A, w, z, y, x, Sz, Sy, Sx)    A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]
#define lget4d(A, w, z, y, x, Sz, LSy, LSx) A[(((((w)*(Sz) + (z))<<(LSy)) + (y))<<(LSx)) + (x)]

//(1) if: flag == 1: -flag = -1 = 0xffffffff
//(2) if: flag == 0: -flag =  0 
#define IF_int(flag, a, b) ( ((-(flag)) & ((a) - (b))) + (b))


#define zero_float(X, flag, v) {float fv = v; int iv = -(flag) & *(int*)(&fv); X = *(float*)(&iv); }
#define zero_float4(X, flag) {if (!flag) X.x = X.y = X.z = X.w = 0;}


#define next_cudaStream(stream, streams, index, length) \
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index]; { index = (index + 1) % length; }

#endif
