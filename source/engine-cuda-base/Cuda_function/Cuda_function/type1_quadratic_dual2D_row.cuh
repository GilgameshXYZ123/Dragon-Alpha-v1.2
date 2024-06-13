#pragma once

#ifndef QUADRATIC_DUAL_2D_ROW_H
#define QUADRATIC_DUAL_2D_ROW_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lengthv, row_lengthv] % stride == 0
//X2 must a 1D Tensor[row_lengthv]
//field_length * row_lengthv = X1.lengthv
#ifndef QUADRATIC_DUAL_2D_ROW_CALL
#define QUADRATIC_DUAL_2D_ROW_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define quadratic_dual2d_row_k4_small(stream, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)\
	quadratic_dual2D_row_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)

//common
#define quadratic_dual2d_row_k4(stream, LB, LT, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)\
	quadratic_dual2D_row_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define quadratic_dual2d_row_k4_max(stream, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)\
	quadratic_dual2D_row_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)

#endif


#ifndef QUADRATIC_DUAL_2D_ROW_KERNEL
#define QUADRATIC_DUAL_2D_ROW_KERNEL

//<1> forward propagation: (with fewer operation)
//Y = k11*X1^2 + k12*X1*X2 + k22*X2^2 + k1*X1 + k2*X2 + C
//Y = (k11*X1^2 + k12*X1*X2 + k1*X1) + (k22*X2^2 + k2*X2) + C
//Y = X1*(k11*X1 + k12*X2 + k1) + X2*(k22*X2 + k2) + C
//STEP:
//<1> A = X1*(k11*X1 + k12*X2 + k1)
//<2> B = X2*(k22*X2 + k2)
//<3> Y = A + B + C
//
//=========OLD fashion================================
//index4 % row_lengthv -> row_index
//x2.x = X2[(index4    ) % row_lengthv];
//x2.y = X2[(index4 + 1) % row_lengthv];
//x2.z = X2[(index4 + 2) % row_lengthv];
//x2.w = X2[(index4 + 3) % row_lengthv];
//=========OLD fashion================================

__global__ void quadratic_dual2D_row_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2, int row_lengthv,
	float k11, float k12, float k22,
	float k1, float k2,
	float C,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);

		int field_index4 = index4 % row_lengthv;//X2_lengthv % 4 == 0
		float4 x2 = *(float4*)(X2 + field_index4);

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

		float4 y;//<3> Y = A + B + C
		y.x = A.x + B.x + C;
		y.y = A.y + B.y + C;
		y.z = A.z + B.z + C;
		y.w = A.w + B.w + C;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __quadratic_dual2D_row(cudaStream_t stream,
	const float* X1,
	const float* X2, int row_lengthv,
	float k11, float k12, float k22,
	float k1, float k2,
	float C,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { quadratic_dual2d_row_k4_small(stream, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { quadratic_dual2d_row_k4_max(stream, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride); return; }
	quadratic_dual2d_row_k4(stream, 5, 2, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride);
}

#endif