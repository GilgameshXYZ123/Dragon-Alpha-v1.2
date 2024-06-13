#pragma once

#ifndef LINEAR_2D_FLOAT_TO_PIXEL_H
#define LINEAR_2D_FLOAT_TO_PIXEL_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LINEAR_2D_FLOAT_TO_PIXEL_CALL
#define LINEAR_2D_FLOAT_TO_PIXEL_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define linear2d_float2pixel_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	linear2D_float2pixel_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#define linear2d_float2pixel_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	linear2D_float2pixel_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef LINEAR_2D_FLOAT_TO_PIXEL_KERNEL
#define LINEAR_2D_FLOAT_TO_PIXEL_KERNEL

__global__ void linear2D_float2pixel_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 fy; uchar4 y;//fy = alpha*X + beta; y = clip(y, -128, 127)
		fy.x = alpha * x.x + beta; y.x = PIXEL_CLIP(fy.x);
		fy.y = alpha * x.y + beta; y.y = PIXEL_CLIP(fy.y);
		fy.z = alpha * x.z + beta; y.z = PIXEL_CLIP(fy.z);
		fy.w = alpha * x.w + beta; y.w = PIXEL_CLIP(fy.w);

		within_width4(y, index4, stride, width);
		*(uchar4*)(Y + index4) = y;
	}
}

#endif


void __linear2D_float2pixel(cudaStream_t stream,
	float alpha, const float* X, float beta,
	char* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear2d_float2pixel_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	linear2d_float2pixel_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif