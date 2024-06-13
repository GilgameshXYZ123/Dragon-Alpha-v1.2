#pragma once

#ifndef CHAR_SET_2D_H
#define CHAR_SET_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef CHAR_SET_2D_CALL
#define CHAR_SET_2D_CALL

//lengthv % 16 == 0
#define char_set2d_k16(stream, LB, LT, X, value, lengthv, width, stride) \
	char_set_2D_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, value, lengthv, width, stride)

//lengthv % 8 == 0
#define char_set2d_k8(stream, LB, LT, X, value, lengthv, width, stride) \
	char_set_2D_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, value, lengthv, width, stride)

//common
#define char_set2d_k4(stream, LB, LT, X, value, lengthv, width, stride) \
	char_set_2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, value, lengthv, width, stride)

//common
#define char_set2d_k4_small(stream, X, value, lengthv, width, stride) \
	char_set_2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, value, lengthv, width, stride)

#endif


//lengthv % 16 == 0
#ifndef CHAR_SET_2D_KERNEL_16
#define CHAR_SET_2D_KERNEL_16

__global__ void char_set_2D_kernel_16(
	char* __restrict__ X,
	const char value,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		char16 v = char16{
			value, value, value, value,
			value, value, value, value,
			value, value, value, value,
			value, value, value, value
		};

		within_width16(v, index16, stride, width);
		*(char16*)(X + index16) = v;
	}
}

#endif


//lengthv % 8 == 0
#ifndef CHAR_SET_2D_KERNEL_8
#define CHAR_SET_2D_KERNEL_8

__global__ void char_set_2D_kernel_8(
	char* __restrict__ X,
	const char value,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		char8 v = char8{ 
			value, value, value, value, 
			value, value, value, value 
		};

		within_width8(v, index8, stride, width);
		*(char8*)(X + index8) = v;
	}
}

#endif


//common
#ifndef CHAR_SET_2D_KERNEL_4
#define CHAR_SET_2D_KERNEL_4

__global__ void char_set_2D_kernel_4(
	char* __restrict__ X,
	const char value,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		char4 v = char4{ value, value, value, value };

		within_width4(v, index4, stride, width);
		*(char4*)(X + index4) = v;
	}
}

#endif


void __char_set2D(cudaStream_t stream,
	char* X, 
	const int value,
	int height, int width, int stride)
{
	int lengthv = height * stride;
	if (lengthv < 256) { char_set2d_k4_small(stream, X, value, lengthv, width, stride); return; }
	if (lengthv >= 8192) { char_set2d_k16(stream, 5, 4, X, value, lengthv, width, stride); return; }
	if (lengthv >= 4096) { char_set2d_k8(stream, 5, 3, X, value, lengthv, width, stride); return; }
	char_set2d_k4(stream, 5, 2, X, value, lengthv, width, stride);
}

#endif