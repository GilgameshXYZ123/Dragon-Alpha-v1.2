#pragma once

#ifndef LINEAR_2D_PIXEL_TO_FLOAT_H
#define LINEAR_2D_PIXEL_TO_FLOAT_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LINEAR_2D_PIXEL_TO_FLOAT_CALL
#define LINEAR_2D_PIXEL_TO_FLOAT_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define linear2d_pixel2float_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	linear2D_pixel2float_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#define linear2d_pixel2float_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	linear2D_pixel2float_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef LINEAR_2D_PIXEL_TO_FLOAT_KERNEL
#define LINEAR_2D_PIXEL_TO_FLOAT_KERNEL

__global__ void linear2D_pixel2float_kernel_4(
	float alpha, const char* __restrict__ X, float beta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x = *(uchar4*)(X + index4);

		float4 y;//y = alpha*x + beta
		y.x = alpha * x.x + beta;
		y.y = alpha * x.y + beta;
		y.z = alpha * x.z + beta;
		y.w = alpha * x.w + beta;
		
		within_width4(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __linear2D_pixel2float(cudaStream_t stream,
	float alpha, const char* X, float beta,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear2d_pixel2float_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	linear2d_pixel2float_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif