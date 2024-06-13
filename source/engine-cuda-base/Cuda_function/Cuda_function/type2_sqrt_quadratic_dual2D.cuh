#pragma once

#ifndef SQRT_QUADRATIC_DUAL_2D_H
#define SQRT_QUADRATIC_DUAL_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SQRT_QUADRATIC_DUAL_2D_CALL
#define SQRT_QUADRATIC_DUAL_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define sqrt_quadratic_dual2d_k4_small(stream, X1, X2, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)\
	sqrt_quadratic_dual2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)

//common
#define sqrt_quadratic_dual2d_k4(stream, LB, LT, X1, X2, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)\
	sqrt_quadratic_dual2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define sqrt_quadratic_dual2d_k4_max(stream, X1, X2, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)\
	sqrt_quadratic_dual2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X1, X2, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)

#endif


#ifndef SQRT_QUADRATIC_DUAL_2D_KERNEL
#define SQRT_QUADRATIC_DUAL_2D_KERNEL

//<1> forward propagation: (with fewer operation)
//Y = k11*X1^2 + k12*X1*X2 + k22*X2^2 + k1*X1 + k2*X2 + C
//Y = (k11*X1^2 + k12*X1*X2 + k1*X1) + (k22*X2^2 + k2*X2) + C
//Y = X1*(k11*X1 + k12*X2 + k1) + X2*(k22*X2 + k2) + C
//STEP:
//<1> A = X1*(k11*X1 + k12*X2 + k1)
//<2> B = X2*(k22*X2 + k2)
//<3> Y = sqrt(A + B + C)

__global__ void sqrt_quadratic_dual2D_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float k11, float k12, float k22,
	float k1, float k2,
	float C,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = float4{ 0, 0, 0, 0 };
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + index4);

		float4 A;//<1> A = X1*(k11*X1 + k12*X2 + k1)
		A.x = x1.x * (k11 * x1.x + k12 * x2.x + k1);
		A.y = x1.y * (k11 * x1.y + k12 * x2.y + k1);
		A.z = x1.z * (k11 * x1.z + k12 * x2.z + k1);
		A.w = x1.w * (k11 * x1.w + k12 * x2.w + k1);

		float4 B;//<2> B = X2*(k22*X2 + k2)
		B.x = x2.x * (k22 * x2.x + k2);
		B.y = x2.y * (k22 * x2.y + k2);
		B.z = x2.z * (k22 * x2.z + k2);
		B.w = x2.w * (k22 * x2.w + k2);

		float4 y;//<3> Y = sqrt(A + B + C)
		y.x = sqrtf(A.x + B.x + C);
		y.y = sqrtf(A.y + B.y + C);
		y.z = sqrtf(A.z + B.z + C);
		y.w = sqrtf(A.w + B.w + C);

		within_width_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __sqrt_quadratic_dual2D(cudaStream_t stream,
	const float* X1,
	const float* X2,
	float k11, float k12, float k22,
	float k1, float k2,
	float C,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sqrt_quadratic_dual2d_k4_small(stream, X1, X2, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { sqrt_quadratic_dual2d_k4_max(stream, X1, X2, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride); return; }
	sqrt_quadratic_dual2d_k4(stream, 5, 2, X1, X2, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride);
}

#endif