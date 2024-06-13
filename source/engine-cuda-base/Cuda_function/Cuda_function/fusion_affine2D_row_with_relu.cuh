#pragma once

#ifndef AFFINE_2D_ROW_WITH_RELU_H
#define AFFINE_2D_ROW_WITH_RELU_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef AFFINE_2D_ROW_WITH_RELU_CALL
#define AFFINE_2D_ROW_WITH_RELU_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define affine2d_row_with_relu_k4_small(stream, X, A, B, row_lengthv, Y, lengthv, width, stride)\
	affine2D_row_with_relu_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, A, B, row_lengthv, Y, lengthv, width, stride)

//common
#define affine2d_row_with_relu_k4(stream, LB, LT, X, A, B, row_lengthv, Y, lengthv, width, stride)\
	affine2D_row_with_relu_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, A, B, row_lengthv, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define affine2d_row_with_relu_k4_max(stream, X, A, B, row_lengthv, Y, lengthv, width, stride)\
	affine2D_row_with_relu_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X, A, B, row_lengthv, Y, lengthv, width, stride)

#endif


#ifndef AFFINE_2D_ROW_WITH_RELU_KERNEL
#define AFFINE_2D_ROW_WITH_RELU_KERNEL

//X2_lengthv = X_mean.lengthv = X_square_mean.lengthv
__global__ void affine2D_row_with_relu_kernel_4(
	const float* __restrict__ X,
	const float* __restrict__ A,
	const float* __restrict__ B, int row_lengthv,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const float4 x = *(float4*)(X + index4);

		const int field_index4 = index4 % row_lengthv;
		const float4 a = *(float4*)(A + field_index4);
		const float4 b = *(float4*)(B + field_index4);

		float4 y;//Y[i] = relu(A * X[i] + B)
		y.x = (a.x * x.x) + b.x; y.x = RELU(y.x);
		y.y = (a.y * x.y) + b.y; y.y = RELU(y.y);
		y.z = (a.z * x.z) + b.z; y.z = RELU(y.z);
		y.w = (a.w * x.w) + b.w; y.w = RELU(y.w);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __affine2D_row_with_relu(cudaStream_t stream,
	const float* X,
	const float* A,
	const float* B, int row_lengthv,
	      float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { affine2d_row_with_relu_k4_small(stream, X, A, B, row_lengthv, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { affine2d_row_with_relu_k4_max(stream, X, A, B, row_lengthv, Y, lengthv, width, stride); return; }
	affine2d_row_with_relu_k4(stream, 5, 2, X, A, B, row_lengthv, Y, lengthv, width, stride);
}

#endif