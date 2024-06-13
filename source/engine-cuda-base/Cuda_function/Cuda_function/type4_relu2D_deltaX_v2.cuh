#pragma once

#ifndef RELU_2D_DELTAX_V2_H
#define RELU_2D_DELTAX_V2_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V2: holdX(), X is not changed
#ifndef RELU_2D_DELTAX_V2_CALL
#define RELU_2D_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define relu2d_deltaX_v2_k4_small(stream,  deltaX, deltaY, X, lengthv, width, stride)\
	relu2D_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

//common
#define relu2d_deltaX_v2_k4(stream, LB, LT, deltaX, deltaY, X, lengthv, width, stride)\
	relu2D_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

//lengthv > lengthv_max
#define relu2d_deltaX_v2_k4_max(stream,  deltaX, deltaY, X, lengthv, width, stride)\
	relu2D_deltaX_v2_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

#endif


#ifndef RELU_2D_DELTAX_V2_KERNEL
#define RELU_2D_DELTAX_V2_KERNEL

//X  > 0: Y = X, Y' = 1
//X <= 0: Y = 0, Y' = 0
//Y' = (X > 0)
__global__ void relu2D_deltaX_v2_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy = *(float4*)(deltaY + index4);
		float4 x = *(float4*)(X + index4);

		float4 dx;//deltaX = deltaY * driY
		dx.x = (x.x > 0.0f) * dy.x;
		dx.y = (x.y > 0.0f) * dy.y;
		dx.z = (x.z > 0.0f) * dy.z;
		dx.w = (x.w > 0.0f) * dy.w;
		
		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __relu2D_deltaX_v2(cudaStream_t stream,
	      float* deltaX, 
	const float* deltaY,
	const float *X,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { relu2d_deltaX_v2_k4_small(stream, deltaX, deltaY, X, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { relu2d_deltaX_v2_k4_max(stream, deltaX, deltaY, X, lengthv, width, stride); return; }
	relu2d_deltaX_v2_k4(stream, 5, 2, deltaX, deltaY, X, lengthv, width, stride);
}

#endif