#pragma once

#ifndef LEAKY_RELU_BERNOULI_MULTIPLE_2D_H
#define LEAKY_RELU_BERNOULI_MULTIPLE_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LEAKY_RELU_BERNOULI_MULTIPLE_2D_CALL
#define LEAKY_RELU_BERNOULI_MULTIPLE_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define leakyRelu_bernouli_mul2d_k4(stream, LB, LT, X, R, Y, k, seed, p, v1, v2, lengthv, width, stride)\
	leakyRelu_bernouli_mul2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, R, Y, k, seed, p, v1, v2, lengthv, width, stride)

#define leakyRelu_bernouli_mul2d_k4_small(stream, X, R, Y, k, seed, p, v1, v2, lengthv, width, stride)\
	leakyRelu_bernouli_mul2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, R, Y, k, seed, p, v1, v2, lengthv, width, stride)

#endif


#ifndef LEAKY_RELU_BERNOULI_MULTIPLE_2D_KERNEL
#define LEAKY_RELU_BERNOULI_MULTIPLE_2D_KERNEL 

//Y = R * leaky_relu(X)
//Y = R * (X < 0 ? k : 1) * X
//let: K = (X < 0 ? k : 1)
//Y = (R * K) * X
//let: R = R * K
//we have: Y = R * X

__global__ void leakyRelu_bernouli_mul2D_kernel_4(
	const float* __restrict__ X,
	      float* __restrict__ R,
	      float* __restrict__ Y, float k,
	unsigned int seed,
	float p, float v1, float v2,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	const int thread_mul = (THREAD_MUL * (step + index) + step) & THREAD_MUL_MOD;
	seed = (seed * (thread_mul + THREAD_ADD0) + step) & THREAD_MOD0;
	seed = (seed * (thread_mul + THREAD_ADD1) + step) & THREAD_MOD1;
	seed = (seed * (thread_mul + THREAD_ADD2) + step) & THREAD_MOD2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);
		
		float4 r; simdNextFloat4(r, seed);//stage1: R = bernouli(p, v1, v2)
		r.x = BERNOULI(r.x, p, v1, v2);
		r.y = BERNOULI(r.y, p, v1, v2);
		r.z = BERNOULI(r.z, p, v1, v2);
		r.w = BERNOULI(r.w, p, v1, v2);

		char4 flag;//stage2: R = R * K
		flag.x = (x.x > 0.0f); r.x *= (flag.x + !flag.x * k);
		flag.y = (x.y > 0.0f); r.y *= (flag.y + !flag.y * k);
		flag.z = (x.z > 0.0f); r.z *= (flag.z + !flag.z * k);
		flag.w = (x.w > 0.0f); r.w *= (flag.w + !flag.w * k);

		float4 y;//stage3: Y = R * X
		y.x = x.x * r.x;
		y.y = x.y * r.y;
		y.z = x.z * r.z;
		y.w = x.w * r.w;

		within_width(r, index4, stride, width);
		within_width(y, index4, stride, width);
		*(float4*)(R + index4) = r;
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __leakyRelu_bernouli_mul2D(cudaStream_t stream,
	const float* X, float* R, float *Y,
	float k, int seed, 
	float p, float v1, float v2,
	int lengthv, int width, int stride)
{
	if (lengthv <   256) { leakyRelu_bernouli_mul2d_k4_small(stream, X, R, Y, k, seed, p, v1, v2, lengthv, width, stride); return; }
	if (lengthv >= 8192) { leakyRelu_bernouli_mul2d_k4(stream, 5, 5, X, R, Y, k, seed, p, v1, v2, lengthv, width, stride); return; }//2^10 = 1024, 32 nums per thread
	if (lengthv >= 4096) { leakyRelu_bernouli_mul2d_k4(stream, 5, 4, X, R, Y, k, seed, p, v1, v2, lengthv, width, stride); return; }//2^ 9 =  512, 16 nums per thread
	if (lengthv >= 2048) { leakyRelu_bernouli_mul2d_k4(stream, 5, 3, X, R, Y, k, seed, p, v1, v2, lengthv, width, stride); return; }//2^ 8 =  256,  8 nums per thread
	leakyRelu_bernouli_mul2d_k4(stream, 5, 2, X, R, Y, k, seed, p, v1, v2, lengthv, width, stride);//2^7 = 128, 4 nums
}

#endif

