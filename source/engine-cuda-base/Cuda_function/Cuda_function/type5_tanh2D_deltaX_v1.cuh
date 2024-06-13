#pragma once

#ifndef TANH_2D_DELTAX_V1_H
#define TANH_2D_DELTAX_V1_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V1: holdY(), Y is not changed
#ifndef TANH_2D_DELTAX_V1_CALL
#define TANH_2D_DELTAX_V1_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define tanh2d_deltaX_v1_k4_small(stream,  deltaX, deltaY, Y, lengthv, width, stride)\
	tanh2D_deltaX_v1_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, lengthv, width, stride)

//common
#define tanh2d_deltaX_v1_k4(stream, LB, LT, deltaX, deltaY, Y, lengthv, width, stride)\
	tanh2D_deltaX_v1_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define tanh2d_deltaX_v1_k4_max(stream,  deltaX, deltaY, Y, lengthv, width, stride)\
	tanh2D_deltaX_v1_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX, deltaY, Y, lengthv, width, stride)

#endif


#ifndef TANH_2D_DELTAX_V1_KERNEL
#define TANH_2D_DELTAX_V1_KERNEL

//Y = 1 - 2/(e^(2*X) + 1).
//Y'= 1 - Y^2

__global__ void tanh2D_deltaX_v1_kernel_4(
	      float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy = *(float4*)(deltaY + index4);
		float4 y = *(float4*)(Y + index4);

		float4 dx;//find derivative: Y'= 1 - Y^2
		dx.x = TANH_DERI(y.x);
		dx.y = TANH_DERI(y.y);
		dx.z = TANH_DERI(y.z);
		dx.w = TANH_DERI(y.w);
		
		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __tanh2D_deltaX_v1(cudaStream_t stream,
	      float* deltaX,
	const float* deltaY,
	const float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { tanh2d_deltaX_v1_k4_small(stream, deltaX, deltaY, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { tanh2d_deltaX_v1_k4_max(stream, deltaX, deltaY, Y, lengthv, width, stride); return; }
	tanh2d_deltaX_v1_k4(stream, 5, 2, deltaX, deltaY, Y, lengthv, width, stride);
}

#endif