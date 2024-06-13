#pragma once

#ifndef EQUAL_ABS_2D_CHAR_H
#define EQUAL_ABS_2D_CHAR_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef EQUAL_ABS_2D_CHAR_CALL
#define EQUAL_ABS_2D_CHAR_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define equal_abs2d_char_k4_small(stream, X1, X2, min, max, Y, lengthv, width, stride)\
	equal_abs2D_char_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, min, max, Y, lengthv, width, stride)

//common
#define equal_abs2d_char_k4(stream, LB, LT, X1, X2, min, max, Y, lengthv, width, stride)\
	equal_abs2D_char_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, min, max, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define equal_abs2d_char_k4_max(stream, X1, X2, min, max, Y, lengthv, width, stride)\
	equal_abs2D_char_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X1, X2, min, max, Y, lengthv, width, stride)

#endif


#ifndef EQUAL_ABS_2D_CHAR_KERNEL
#define EQUAL_ABS_2D_CHAR_KERNEL

//max >= min >= 0 
__global__ void equal_abs2D_char_kernel_4(
	const char* __restrict__ X1,
	const char* __restrict__ X2,
	char min, char max,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		char4 x1 = *(char4*)(X1 + index4);
		char4 x2 = *(char4*)(X2 + index4);

		float4 y;
		y.x = fabsf(x1.x - x2.x); y.x = (y.x >= min) && (y.x <= max);
		y.y = fabsf(x1.y - x2.y); y.y = (y.y >= min) && (y.y <= max);
		y.z = fabsf(x1.z - x2.z); y.z = (y.z >= min) && (y.z <= max);
		y.w = fabsf(x1.w - x2.w); y.w = (y.w >= min) && (y.w <= max);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __equal_abs2D_char(cudaStream_t stream,
	const char* X1, 
	const char* X2,
	char min, char max,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { equal_abs2d_char_k4_small(stream, X1, X2, min, max, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { equal_abs2d_char_k4_max(stream, X1, X2, min, max, Y, lengthv, width, stride); return; }
	equal_abs2d_char_k4(stream, 5, 2, X1, X2, min, max, Y, lengthv, width, stride);
}

#endif