#pragma once

#ifndef LINEAR_2D_FLOAT_TO_CHAR_H
#define LINEAR_2D_FLOAT_TO_CHAR_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LINEAR_2D_FLOAT_TO_CHAR_CALL
#define LINEAR_2D_FLOAT_TO_CHAR_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define linear2d_float2char_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	linear2D_float2char_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//common
#define linear2d_float2char_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	linear2D_float2char_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define linear2d_float2char_k4_max(stream, alpha, X, beta, Y, lengthv, width, stride)\
	linear2D_float2char_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)


#endif


#ifndef LINEAR_2D_FLOAT_TO_CHAR_KERNEL
#define LINEAR_2D_FLOAT_TO_CHAR_KERNEL

__global__ void linear2D_float2char_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		char4 y;//y = alpha*X + beta
		y.x = (char)(alpha * x.x + beta);
		y.y = (char)(alpha * x.y + beta);
		y.z = (char)(alpha * x.z + beta);
		y.w = (char)(alpha * x.w + beta);

		within_width(y, index4, stride, width);
		*(char4*)(Y + index4) = y;
	}
}

#endif


void __linear2D_float2char(cudaStream_t stream,
	float alpha, const float* X, float beta,
	char* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear2d_float2char_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { linear2d_float2char_k4_max(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	linear2d_float2char_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif