#pragma once

#ifndef GAUSSIAN_2D_H
#define GAUSSIAN_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef GAUSSIAN_2D_CALL
#define GAUSSIAN_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define gaussian2d_k4(stream, LB, LT, X, seed1, seed2, mu, sigma, lengthv, width, stride)\
	gaussian2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, seed1, seed2, mu, sigma, lengthv, width, stride)

#define gaussian2d_k4_small(stream, X, seed1, seed2, mu, sigma, lengthv, width, stride)\
	gaussian2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, seed1, seed2, mu, sigma, lengthv, width, stride)

#endif


#ifndef GAUSSIAN_2D_KERNEL
#define GAUSSIAN_2D_KERNEL

__global__ void gaussian2D_kernel_4(
	float* __restrict__ X,
	unsigned int seed1, 
	unsigned int seed2,
	float mu, float sigma,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	const int thread_mul1 = (THREAD_MUL * (step + (index)     ) + step) & THREAD_MUL_MOD;
	const int thread_mul2 = (THREAD_MUL * (step + (index << 1)) + step) & THREAD_MUL_MOD;

	seed1 = (seed1 * (thread_mul1 + THREAD_ADD0) + step) & THREAD_MOD0;
	seed1 = (seed1 * (thread_mul1 + THREAD_ADD1) + step) & THREAD_MOD1;
	seed1 = (seed1 * (thread_mul1 + THREAD_ADD2) + step) & THREAD_MOD2;

	seed2 = (seed2 * (thread_mul2 + THREAD_ADD0) + step) & THREAD_MOD0;
	seed2 = (seed2 * (thread_mul2 + THREAD_ADD1) + step) & THREAD_MOD1;
	seed2 = (seed2 * (thread_mul2 + THREAD_ADD2) + step) & THREAD_MOD2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//v1 = sqrt(-2 * log(v1))
		float2 v1; simdNextFloat2(v1, seed1);
		v1.x = v1.x + ((v1.x <= 0.0f) - (v1.x >= 1.0f))*(1e-3f);
		v1.y = v1.y + ((v1.y <= 0.0f) - (v1.y >= 1.0f))*(1e-3f);
		v1.x = sqrtf(-2.0f * logf(v1.x));
		v1.y = sqrtf(-2.0f * logf(v1.y));

		//v2 = 2pi * v2
		float2 v2; simdNextFloat2(v2, seed2);
		v2.x *= TWO_PI;
		v2.y *= TWO_PI;
		
		float4 r;//a1 = v1*cos(v2), a1 = v2*sin(v1)
		r.x = v1.x * cosf(v2.x);
		r.y = v1.x * sinf(v2.x);
		r.z = v1.y * cosf(v2.y);
		r.w = v1.y * sinf(v2.y);
		simdLinear4(r, sigma, r, mu);//N(0, 1) -> N(mu, sigma^2)

		within_width(r, index4, stride, width);
		*(float4*)(X + index4) = r;
	}
}

#endif


void __gaussian2D(cudaStream_t stream,
	float* X,
	int seed1, int seed2, 
	float mu, float sigma,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { gaussian2d_k4_small(stream, X, seed1, seed2, mu, sigma, lengthv, width, stride); return; }
	if (lengthv >= 16384) { gaussian2d_k4(stream, 5, 6, X, seed1, seed2, mu, sigma, lengthv, width, stride); return; }//8 blocks: 2^11 = 2048, 64 nums per thread
	if (lengthv >=  8192) { gaussian2d_k4(stream, 5, 5, X, seed1, seed2, mu, sigma, lengthv, width, stride); return; }//8 blocks: 2^10 = 1024, 32 nums per thread
	if (lengthv >=  4096) { gaussian2d_k4(stream, 5, 4, X, seed1, seed2, mu, sigma, lengthv, width, stride); return; }//8 blocks: 2^ 9 =  512, 16 nums per thread
	if (lengthv >=  2048) { gaussian2d_k4(stream, 5, 3, X, seed1, seed2, mu, sigma, lengthv, width, stride); return; }//8 blocks: 2^ 8 =  256,  8 nums per thread
	gaussian2d_k4(stream, 5, 2, X, seed1, seed2, mu, sigma, lengthv, width, stride);//2^7 = 128, 4 nums per thread
}

#endif