#pragma once

#ifndef LINEAR_GREATER_SWITCH_MUL_2D_H
#define LINEAR_GREATER_SWITCH_MUL_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LINEAR_GREATER_SWITCH_MUL_2D_CALL
#define LINEAR_GREATER_SWITCH_MUL_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define linear_greater_switch_mul2d_k4_small(stream, alpha, X1, beta, X2, v1, v2, Y, lengthv, width, stride)\
	linear_greater_switch_mul2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X1, beta, X2, v1, v2, Y, lengthv, width, stride)

//common
#define linear_greater_switch_mul2d_k4(stream, LB, LT, alpha, X1, beta, X2, v1, v2, Y, lengthv, width, stride)\
	linear_greater_switch_mul2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X1, beta, X2, v1, v2, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define linear_greater_switch_mul2d_k4_max(stream, alpha, X1, beta, X2, v1, v2, Y, lengthv, width, stride)\
	linear_greater_switch_mul2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(alpha, X1, beta, X2, v1, v2, Y, lengthv, width, stride)

#endif


#ifndef LINEAR_GREATER_SWITCH_MUL_2D_KERNEL
#define LINEAR_GREATER_SWITCH_MUL_2D_KERNEL

//<1> flag = (alpha*X1 + beta) > 0
//<2> y = (flag? v1 : v2) * X2

__global__ void linear_greater_switch_mul2D_kernel_4(
	float alpha, const float* __restrict__ X1, float beta,
	float* __restrict__ X2, float v1, float v2,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + index4);

		char4 flag;//<1> flag = (alpha*X1 + beta) > 0
		flag.x = (alpha * x1.x + beta) > 0.0f;
		flag.y = (alpha * x1.y + beta) > 0.0f;
		flag.z = (alpha * x1.z + beta) > 0.0f;
		flag.w = (alpha * x1.w + beta) > 0.0f;

		float4 y;//<2> y = (flag? v1 : v2) * X2
		y.x = (flag.x * v1 + (!flag.x) * v2) * x2.x;
		y.y = (flag.y * v1 + (!flag.y) * v2) * x2.y;
		y.z = (flag.z * v1 + (!flag.z) * v2) * x2.z;
		y.w = (flag.w * v1 + (!flag.w) * v2) * x2.w;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __linear_greater_switch_mul2D(cudaStream_t stream,
	float alpha, const float* X1, float beta,
	float* X2, float v1, float v2,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear_greater_switch_mul2d_k4_small(stream, alpha, X1, beta, X2, v1, v2, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { linear_greater_switch_mul2d_k4_max(stream, alpha, X1, beta, X2, v1, v2, Y, lengthv, width, stride); return; }
	linear_greater_switch_mul2d_k4(stream, 5, 2, alpha, X1, beta, X2, v1, v2, Y, lengthv, width, stride);
}

#endif