#pragma once

#ifndef MUL_LINEAR_DUAL_2D_H
#define MUL_LINEAR_DUAL_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef MUL_LINEAR_DUAL_2D_CALL
#define MUL_LINEAR_DUAL_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define mul_linear_dual2d_k4_small(stream, X, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)\
	mul_linear_dual2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)

//common
#define mul_linear_dual2d_k4(stream, LB, LT, X, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)\
	mul_linear_dual2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define mul_linear_dual2d_k4_max(stream, X, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)\
	mul_linear_dual2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)

#endif


#ifndef MUL_LINEAR_DUAL_2D_KERNEL
#define MUL_LINEAR_DUAL_2D_KERNEL

//Y = X * (alpha*X1 + beta*X2 + gamma),
//deltaX1 = deltaY*alpha
//deltaX2 = deltaY*beta
__global__ void mul_linear_dual2D_kernel_4(
	const float* __restrict__ X,
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float alpha, float beta, float gamma,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);
		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + index4);

		float4 y;//Y = X * (alpha*X1 + beta * X2 + gamma),
		y.x = x.x * (alpha * x1.x + beta * x2.x + gamma);
		y.y = x.y * (alpha * x1.y + beta * x2.y + gamma);
		y.z = x.z * (alpha * x1.z + beta * x2.z + gamma);
		y.w = x.w * (alpha * x1.w + beta * x2.w + gamma);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __mul_linear_dual2D(cudaStream_t stream,
	const float* X,
	const float* X1,
	const float* X2,
	float alpha, float beta, float gamma,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { mul_linear_dual2d_k4_small(stream, X, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { mul_linear_dual2d_k4_max(stream, X, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	mul_linear_dual2d_k4(stream, 5, 2, X, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride);
}

#endif