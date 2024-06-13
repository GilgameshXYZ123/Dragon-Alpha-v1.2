#pragma once

#ifndef CEILING_2D_H
#define CEILING_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef CEILING_2D_CALL
#define CEILING_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define ceil2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	ceil2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//common
#define ceil2d_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	ceil2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define ceil2d_k4_max(stream, alpha, X, beta, Y, lengthv, width, stride)\
	ceil2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef CEILING_2D_KERNEL
#define CEILING_2D_KERNEL

__global__ void ceil2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y; simdLinear4(y, alpha, x, beta);
		y.x = ceilf(y.x); 
		y.y = ceilf(y.y);
		y.z = ceilf(y.z);
		y.w = ceilf(y.w);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __ceil2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { ceil2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { ceil2d_k4_max(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	ceil2d_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif