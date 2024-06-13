#pragma once

#ifndef ARCSIN_2D_DELTAX_H
#define ARCSIN_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ARCSIN_2D_DELTAX_CALL
#define ARCSIN_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define arcsin2d_deltaX_k4_small(stream, deltaX, deltaY, Y, alpha, lengthv, width, stride)\
	arcsin2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, lengthv, width, stride)

//common
#define arcsin2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, Y, alpha, lengthv, width, stride)\
	arcsin2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, lengthv, width, stride)

//lengthv > lengthv_max
#define arcsin2d_deltaX_k4_max(stream, deltaX, deltaY, Y, alpha, lengthv, width, stride)\
	arcsin2D_deltaX_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX, deltaY, Y, alpha, lengthv, width, stride)

#endif


#ifndef ARCSIN_2D_DELTAX_KERNEL
#define ARCSIN_2D_DELTAX_KERNEL

//forward propagation:
//<1> Y = arcsin(alpha*X + beta)
//
//backward propagation:
//<1> Y' = alpha / sqrt(1 - (alpha*X + beta)^2)
//As: alpha*X + beta = sin(Y)
//<2> Y' = alpha / sqrt(1 - sin^2(Y))
//<3> Y' = alpha / fabsf(cos(Y))

__global__ void arcsin2D_deltaX_kernel_4(
	      float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	float alpha,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y  = *(float4*)(Y      + index4);
		float4 dy = *(float4*)(deltaY + index4);

		float4 dx;//Y' = alpha / fabsf(cos(Y))
		dx.x = alpha / fabsf(cosf(y.x));
		dx.y = alpha / fabsf(cosf(y.y));
		dx.z = alpha / fabsf(cosf(y.z));
		dx.w = alpha / fabsf(cosf(y.w));
		
		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __arcsin2D_deltaX(cudaStream_t stream,
	      float* deltaX,
	const float* deltaY,
	const float* Y, float alpha,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { arcsin2d_deltaX_k4_small(stream, deltaX, deltaY, Y, alpha, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { arcsin2d_deltaX_k4_max(stream, deltaX, deltaY, Y, alpha, lengthv, width, stride); return; }
	arcsin2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, Y, alpha, lengthv, width, stride);
}

#endif