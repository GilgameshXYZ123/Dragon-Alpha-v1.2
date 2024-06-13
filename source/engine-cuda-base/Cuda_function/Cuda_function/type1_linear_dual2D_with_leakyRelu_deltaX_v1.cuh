#pragma once

#ifndef LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V1_H
#define LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V1_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V1: holdY(), Y is not changed
#ifndef LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V1_CALL
#define LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V1_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define linear_dual2d_with_leakyRelu_deltaX_v1_k4_small(stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, k, lengthv)\
	linear_dual2D_with_leakyRelu_deltaX_v1_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, Y, alpha, beta, k, lengthv, width, stride)

//common
#define linear_dual2d_with_leakyRelu_deltaX_v1_k4(stream, LB, LT, deltaX1, deltaX2, deltaY, Y, alpha, beta, k, lengthv)\
	linear_dual2D_with_leakyRelu_deltaX_v1_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, Y, alpha, beta, k, lengthv, width, stride)

//lengthv > lengthv_max
#define linear_dual2d_with_leakyRelu_deltaX_v1_k4_max(stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, k, lengthv)\
	linear_dual2D_with_leakyRelu_deltaX_v1_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX1, deltaX2, deltaY, Y, alpha, beta, k, lengthv, width, stride)

#endif


#ifndef LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V1_KERNEL
#define LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V1_KERNEL

//forward propagation:
//(1) Y1 = alpha*X1 + beta*X2 + gamma
//(2) Y2 = (Y1 > 0? Y1 : k*Y1)
//backward propagation:
//(2) deltaY1 = deltaY2 * (Y2 > 0? 1 : k)
//(3) deltaX1 = deltaY1 * alpha
//(4) deltaX2 = deltaY2 * beta

__global__ void linear_dual2D_with_leakyRelu_deltaX_v1_kernel_4(
	      float* __restrict__ deltaX1,
	      float* __restrict__ deltaX2,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	float alpha, float beta, float k,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy2 = *(float4*)(deltaY + index4);
		float4 y2 = *(float4*)(Y + index4);

		char4 flag;//flag = Y2 > 0
		flag.x = y2.x > 0.0f;
		flag.y = y2.y > 0.0f;
		flag.z = y2.z > 0.0f;
		flag.w = y2.w > 0.0f;

		float4 dy1;//deltaY1 = deltaY2 * (flags ? Y1 : k)
		dy1.x = dy2.x * (flag.x + !flag.x*k);
		dy1.y = dy2.y * (flag.y + !flag.y*k);
		dy1.z = dy2.z * (flag.z + !flag.z*k);
		dy1.w = dy2.w * (flag.w + !flag.w*k);

		float4 dx1, dx2;//deltaX1 = deltaY1 * alpha; deltaX2 = deltaY2 * beta
		dx1.x = dy1.x * alpha; dx2.x = dy1.x * beta;
		dx1.y = dy1.y * alpha; dx2.y = dy1.y * beta;
		dx1.z = dy1.z * alpha; dx2.z = dy1.z * beta;
		dx1.w = dy1.w * alpha; dx2.w = dy1.w * beta;

		within_width(dx1, index4, stride, width);
		within_width(dx2, index4, stride, width);
		*(float4*)(deltaX1 + index4) = dx1;
		*(float4*)(deltaX2 + index4) = dx2;
	}
}

#endif


void __linear_dual2D_with_leakyRelu_deltaX_v1(cudaStream_t stream,
	      float* deltaX1, 
	      float* deltaX2,
	const float* deltaY,
	const float* Y,
	float alpha, float beta, float k,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear_dual2d_with_leakyRelu_deltaX_v1_k4_small(stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, k, lengthv); return; }
	if (lengthv > LENGTHV_MAX) { linear_dual2d_with_leakyRelu_deltaX_v1_k4_max(stream, deltaX1, deltaX2, deltaY, Y, alpha, beta, k, lengthv); return; }
	linear_dual2d_with_leakyRelu_deltaX_v1_k4(stream, 5, 2, deltaX1, deltaX2, deltaY, Y, alpha, beta, k, lengthv);
}

#endif