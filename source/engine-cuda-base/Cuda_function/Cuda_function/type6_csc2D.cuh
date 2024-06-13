#pragma once

#ifndef CSC_2D_H
#define CSC_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef CSC_2D_CALL
#define CSC_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define csc2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	csc2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//common
#define csc2d_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	csc2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define csc2d_k4_max(stream, alpha, X, beta, Y, lengthv, width, stride)\
	csc2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef CSC_2D_KERNEL
#define CSC_2D_KERNEL

//<1> Y = csc(alpha*X + beta) = 1 / sin(alpha*X + beta)
//    Y = sec(alpha*X + beta) = csc(alpha*X + (beta + 0.5pi))
//<2> Y' = -alpha * cos(alpha*X + beta) / sin(alpha*X + beta)^2

__global__ void csc2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y; //Y = sin(alpha*X + beta)
		y.x = 1.0f / sinf(alpha*x.x + beta);
		y.y = 1.0f / sinf(alpha*x.y + beta);
		y.z = 1.0f / sinf(alpha*x.z + beta);
		y.w = 1.0f / sinf(alpha*x.w + beta);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __csc2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { csc2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { csc2d_k4_max(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	csc2d_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif