#pragma once

#ifndef UNIFORM_2D_H
#define UNIFORM_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef UNIFORM_2D_CALL
#define UNIFORM_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define uniform2d_k4(stream, LB, LT, X, seed, threshold, base, lengthv, width, stride)\
	uniform2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, seed, threshold, base, lengthv, width, stride)

#define uniform2d_k4_small(stream, X, seed, threshold, base, lengthv, width, stride)\
	uniform2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, seed, threshold, base, lengthv, width, stride)

#endif


#ifndef UNIFORM_2D_KERNEL
#define UNIFORM_2D_KERNEL

__global__ void uniform2D_kernel_4(
	float* __restrict__ X, 
	unsigned int seed,
	float threshold, float base,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x * blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	const int thread_mul = (THREAD_MUL * (step + index) + step) & THREAD_MUL_MOD;
	seed = (seed * (thread_mul + THREAD_ADD0) + step) & THREAD_MOD0;
	seed = (seed * (thread_mul + THREAD_ADD1) + step) & THREAD_MOD1;
	seed = (seed * (thread_mul + THREAD_ADD2) + step) & THREAD_MOD2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x; simdNextFloat4(x, seed);
		simdLinear4(x, threshold, x, base);

		within_width(x, index4, stride, width);
		*(float4*)(X + index4) = x;
	}
}

#endif


void __uniform2D(cudaStream_t stream,
	float* X,	
	int seed, 
	float vmin, float vmax,
	int lengthv, int width, int stride)
{
	float base = vmin, threshold = vmax - vmin;
	if (lengthv < 256) { uniform2d_k4_small(stream, X, seed, threshold, base, lengthv, width, stride); return; }
	if (lengthv >= 16384) { uniform2d_k4(stream, 5, 6, X, seed, threshold, base, lengthv, width, stride); return; }//8 block: 2^11 = 2048, 64 nums per thread
	if (lengthv >=  8192) { uniform2d_k4(stream, 5, 5, X, seed, threshold, base, lengthv, width, stride); return; }//8 block: 2^10 = 1024, 32 nums per thread
	if (lengthv >=  4096) { uniform2d_k4(stream, 5, 4, X, seed, threshold, base, lengthv, width, stride); return; }//8 block: 2^ 9 =  512, 16 nums per thread
	if (lengthv >=  2048) { uniform2d_k4(stream, 5, 3, X, seed, threshold, base, lengthv, width, stride); return; }//8 block: 2^ 8 =  256,  8 nums per thread
	uniform2d_k4(stream, 5, 2, X, seed, threshold, base, lengthv, width, stride);//2^7 = 128, 4 nums per thread
}

#endif