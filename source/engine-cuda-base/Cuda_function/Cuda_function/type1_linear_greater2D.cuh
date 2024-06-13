#pragma once

#ifndef LINEAR_GREATER_2D_H
#define LINEAR_GREATER_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LINEAR_GREATER_2D_CALL
#define LINEAR_GREATER_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define linear_greater2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	linear_greater2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//common
#define linear_greater2d_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	linear_greater2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define linear_greater2d_k4_max(stream, alpha, X, beta, Y, lengthv, width, stride)\
	linear_greater2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef LINEAR_GREATER_2D_KERNEL
#define LINEAR_GREATER_2D_KERNEL

//Y = alpha*X + beta > 0

__global__ void linear_greater2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4); 

		float4 y;
		y.x = (alpha * x.x + beta) > 0.0f;
		y.y = (alpha * x.y + beta) > 0.0f;
		y.z = (alpha * x.z + beta) > 0.0f;
		y.w = (alpha * x.w + beta) > 0.0f;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __linear_greater2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear_greater2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { linear_greater2d_k4_max(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	linear_greater2d_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif