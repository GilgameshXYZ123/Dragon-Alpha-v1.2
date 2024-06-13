#pragma once

#ifndef LINEAR_DUAL_2D_CENTER_H
#define LINEAR_DUAL_2D_CENTER_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//dim2 % stride == 0 
//X1[dim0, dim1, dim2]
//X2[dim0,       dim2]
// Y[dim0, dim1, dim2]
#ifndef LINEAR_DUAL_2D_CENTER_CALL
#define LINEAR_DUAL_2D_CENTER_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv <= 256
#define linear_dual2d_center_k4_small(stream, X1, X2, alpha, beta, gamma, Y, dim1, dim2, lengthv, width, stride)\
	linear_dual2D_center_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, alpha, beta, gamma, Y, (dim1*dim2), dim2, lengthv, width, stride)

//common
#define linear_dual2d_center_k4(stream, LB, LT, X1, X2, alpha, beta, gamma, Y, dim1, dim2, lengthv, width, stride)\
	linear_dual2D_center_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, alpha, beta, gamma, Y, (dim1*dim2), dim2, lengthv, width, stride)

//lengthv > lengthv_max
#define linear_dual2d_center_k4_max(stream, X1, X2, alpha, beta, gamma, Y, dim1, dim2, lengthv, width, stride)\
	linear_dual2D_center_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X1, X2, alpha, beta, gamma, Y, (dim1*dim2), dim2, lengthv, width, stride)

#endif


#ifndef LINEAR_DUAL_2D_CENTER_KERNEL
#define LINEAR_DUAL_2D_CENTER_KERNEL

//Y = alpha*X1 + beta*X2 + gamma,
//deltaX1 = deltaY*alpha
//deltaX2 = deltaY*beta

__global__ void linear_dual2D_center_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float alpha, float beta, float gamma,
	float* __restrict__ Y,
	int dim12, int dim2,//dim12 = dim1 * dim2
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		int d0 = index4 / dim12, r = index4 - dim12 * d0;//[d0, d1, d2]
		int d1 = r / dim2, d2 = r - dim2 * d1;//d2 = index4 % dim2 => {x2[0-3]}
		int x2_offset = d0 * dim2 + d2;//[d0, d2]

		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + x2_offset);

		float4 y;
		y.x = alpha * x1.x + beta * x2.x + gamma;
		y.y = alpha * x1.y + beta * x2.y + gamma;
		y.z = alpha * x1.z + beta * x2.z + gamma;
		y.w = alpha * x1.w + beta * x2.w + gamma;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __linear_dual2D_center(cudaStream_t stream,
	const float* X1, const float* X2,
	float alpha, float beta, float gamma,
	float* Y,
	int dim0, int dim1, int dim2,
	int width, int stride)
{
	int lengthv = dim0 * dim1 * dim2;//dim2 % 4 == 0
	if (lengthv < 256) { linear_dual2d_center_k4_small(stream, X1, X2, alpha, beta, gamma, Y, dim1, dim2, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { linear_dual2d_center_k4_max(stream, X1, X2, alpha, beta, gamma, Y, dim1, dim2, lengthv, width, stride); return; }
	linear_dual2d_center_k4(stream, 5, 2, X1, X2, alpha, beta, gamma, Y, dim1, dim2, lengthv, width, stride);
}

#endif