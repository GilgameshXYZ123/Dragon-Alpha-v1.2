#pragma once

#ifndef CHAR_SET_1D_H
#define CHAR_SET_1D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef CHAR_SET_1D_CALL
#define CHAR_SET_1D_CALL

//length >= 8192
#define char_set1d_k16(stream, LB, LT, X, value, length) \
	char_set_1D_kernel_16\
		<<< (length>>LB>>LT), (1<<LB), 0, stream >>>\
			(X, value, length)

//length >= 4096
#define char_set1d_k8(stream, LB, LT, X, value, length) \
	char_set_1D_kernel_8\
		<<< (length>>LB>>LT), (1<<LB), 0, stream >>>\
			(X, value, length)
//common
#define char_set1d_k4(stream, LB, LT, X, value, length) \
	char_set_1D_kernel_4\
		<<< (length>>LB>>LT), (1<<LB), 0, stream >>>\
			(X, value, length)

//common
#define char_set1d_k1(stream, X, value, length)\
	char_set_1D_kernel_1\
		<<< 1, length, 0, stream >>>\
			(X, value, length)

#endif


//length >= 8192
#ifndef CHAR_SET_1D_KERNEL_16
#define CHAR_SET_1D_KERNEL_16

__global__ void char_set_1D_kernel_16(
	char* __restrict__ X,
	const char value,
	int length)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x, index16 = index << 4;
	int step = gridDim.x*blockDim.x, step16 = step << 4;

	//use char8----------------------------------------------
	int I = length / step16;
	char16 x16 = char16{ 
		value, value, value, value, 
		value, value, value, value,
		value, value, value, value,
		value, value, value, value
	};

	for (int i = 0; i < I; i++) {
		*(char16*)(X + index16) = x16;
		index16 += step16;
	}

	//use char-----------------------------------------------
	for (index += I * step16; index < length; index += step)
		X[index] = value;
}

#endif


//length >= 4096
#ifndef CHAR_SET_1D_KERNEL_8
#define CHAR_SET_1D_KERNEL_8

__global__ void char_set_1D_kernel_8(
	char* __restrict__ X,
	const char value,
	int length)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x, index8 = index << 3;
	int step = gridDim.x*blockDim.x, step8 = step << 3;

	//use char8----------------------------------------------
	int I = length / step8;
	char8 x8 = char8{ value, value, value, value, value, value, value, value };
	for (int i = 0; i < I; i++) {
		*(char8*)(X + index8) = x8;
		index8 += step8;
	}

	//use char-----------------------------------------------
	for (index += I * step8; index < length; index += step) 
		X[index] = value;
}

#endif


//common
#ifndef CHAR_SET_1D_KERNEL_4
#define CHAR_SET_1D_KERNEL_4

__global__ void char_set_1D_kernel_4(
	char* __restrict__ X, 
	const char value,
	int length)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x, index4 = index << 2;
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	
	//use char4----------------------------------------------
	int I = length / step4;
	char4 x4 = char4{ value, value, value, value };
	for (int i = 0; i < I; i++) {
		*(char4*)(X + index4) = x4;
		index4 += step4;
	}

	//use char-----------------------------------------------
	for (index += I * step4; index < length; index += step)
		X[index] = value;
}

#endif


//common
#ifndef CHAR_SET_1D_KERNEL_1
#define CHAR_SET_1D_KERNEL_1

__global__ void char_set_1D_kernel_1(
	char* __restrict__ X, 
	const char value,
	int length)
{
	int index = threadIdx.x;
	int step = blockDim.x;
	while (index < length) {
		X[index] = value;
		index += step;
	}
}

#endif


void __char_set1D(cudaStream_t stream,
	char* __restrict__ X,
	const int value,
	int length)
{
	if (length < 256) { char_set1d_k1(stream, X, value, length); return; }
	if (length >= 8192) { char_set1d_k16(stream, 5, 4, X, value, length); return; }
	if (length >= 4096) { char_set1d_k8(stream, 5, 3, X, value, length); return; }
	char_set1d_k4(stream, 5, 2, X, value, length);
}

#endif
