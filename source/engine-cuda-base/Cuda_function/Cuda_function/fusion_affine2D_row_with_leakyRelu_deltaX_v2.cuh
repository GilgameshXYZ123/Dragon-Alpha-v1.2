#pragma once

#ifndef AFFINE_2D_ROW_WITH_LEAKY_RELU_DELTAX_V2_H
#define AFFINE_2D_ROW_WITH_LEAKY_RELU_DELTAX_V2_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V2: holdX(), X is not changed
#ifndef AFFINE_2D_ROW_WITH_LEAKY_RELU_DELTAX_V2_CALL
#define AFFINE_2D_ROW_WITH_LEAKY_RELU_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define affine2d_row_with_leakyRelu_deltaX_v2_k4_small(stream, deltaX, deltaY, k, X, A, B, row_lengthv, lengthv, width, stride)\
	affine2D_row_with_leakyRelu_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, k, X, A, B, row_lengthv, lengthv, width, stride)

//common
#define affine2d_row_with_leakyRelu_deltaX_v2_k4(stream, LB, LT, deltaX, deltaY, k, X, A, B, row_lengthv, lengthv, width, stride)\
	affine2D_row_with_leakyRelu_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, k, X, A, B, row_lengthv, lengthv, width, stride)

//lengthv > lengthv_max
#define affine2d_row_with_leakyRelu_deltaX_v2_k4_max(stream, deltaX, deltaY, k, X, A, B, row_lengthv, lengthv, width, stride)\
	affine2D_row_with_leakyRelu_deltaX_v2_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX, deltaY, k, X, A, B, row_lengthv, lengthv, width, stride)

#endif


#ifndef AFFINE_2D_ROW_WITH_LEAKY_RELU_DELTAX_V2_KERNEL
#define AFFINE_2D_ROW_WITH_LEAKY_RELU_DELTAX_V2_KERNEL

//Forward propagation:
//(1) Y1 = A*X + B
//(2) Y2 = leaky_relu(Y1)
//
//Backward propagation:
//(1) flag = (A*X + B) > 0
//(1) deltaY1 = deltaY2 * (flag > 0 ? 1 : k), deltaY2 = deltaY
//(2) deltaX = A * deltaY1

__global__ void affine2D_row_with_leakyRelu_deltaX_v2_kernel_4(
	      float* __restrict__ deltaX,
	const float* __restrict__ deltaY, float k,
	const float* __restrict__ X, 
	const float* __restrict__ A,
	const float* __restrict__ B, int row_lengthv,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const float4 x = *(float4*)(X + index4);
		const float4 dy2 = *(float4*)(deltaY + index4);

		const int field_index4 = index4 % row_lengthv;
		const float4 a = *(float4*)(A + field_index4);
		const float4 b = *(float4*)(B + field_index4);

		char4 flag;//flag = (A*X + B) > 0
		flag.x = (a.x * x.x + b.x) > 0.0f;
		flag.y = (a.y * x.y + b.y) > 0.0f;
		flag.z = (a.z * x.z + b.z) > 0.0f;
		flag.w = (a.w * x.w + b.w) > 0.0f;

		float4 dy1;//deltaY1 = deltaY2 * (flag > 0 ? 1 : k)
		dy1.x = dy2.x * (flag.x + !flag.x*k);
		dy1.y = dy2.y * (flag.y + !flag.y*k);
		dy1.z = dy2.z * (flag.z + !flag.z*k);
		dy1.w = dy2.w * (flag.w + !flag.w*k);

		float4 dx;//deltaX = A * deltaY1
		dx.x = a.x * dy1.x;
		dx.y = a.y * dy1.y;
		dx.z = a.z * dy1.z;
		dx.w = a.w * dy1.w;

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __affine2D_row_with_leakyRelu_deltaX_v2(cudaStream_t stream,
	      float* deltaX,
	const float* deltaY, float k,
	const float* X,
	const float* A, const float* B,
	int row_lengthv,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { affine2d_row_with_leakyRelu_deltaX_v2_k4_small(stream, deltaX, deltaY, k, X, A, B, row_lengthv, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { affine2d_row_with_leakyRelu_deltaX_v2_k4_max(stream, deltaX, deltaY, k, X, A, B, row_lengthv, lengthv, width, stride); return; }
	affine2d_row_with_leakyRelu_deltaX_v2_k4(stream, 5, 2, deltaX, deltaY, k, X, A, B, row_lengthv, lengthv, width, stride);
}

#endif