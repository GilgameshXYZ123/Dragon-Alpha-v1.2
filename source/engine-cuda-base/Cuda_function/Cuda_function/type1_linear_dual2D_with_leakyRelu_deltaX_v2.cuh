#pragma once

#ifndef LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V2_H
#define LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V2_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V2: holdX(), {X1, X2} are not changed
#ifndef LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V2_CALL
#define LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define linear_dual2d_with_leakyRelu_deltaX_v2_k4_small(stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, k, lengthv)\
	linear_dual2D_with_leakyRelu_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, k, lengthv, width, stride)

//common
#define linear_dual2d_with_leakyRelu_deltaX_v2_k4(stream, LB, LT, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, k, lengthv)\
	linear_dual2D_with_leakyRelu_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, k, lengthv, width, stride)

//lengthv > lengthv_max
#define linear_dual2d_with_leakyRelu_deltaX_v2_k4_max(stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, k, lengthv)\
	linear_dual2D_with_leakyRelu_deltaX_v2_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, k, lengthv, width, stride)

#endif


#ifndef LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V2_KERNEL
#define LINEAR_DUAL_2D_WITH_LEAKY_RELU_DELTAX_V2_KERNEL

//forward propagation:
//(1) Y1 = alpha*X1 + beta*X2 + gamma
//(2) Y2 = (Y1 > 0? Y1 : k*Y1)
//backward propagation:
//(1) Y1 = alpha*X1 + beta*X2 + gamma 
//(2) deltaY1 = deltaY2 * (Y1 > 0? 1 : k)
//(3) deltaX1 = deltaY1 * alpha
//(4) deltaX2 = deltaY2 * beta

__global__ void linear_dual2D_with_leakyRelu_deltaX_v2_kernel_4(
	      float* __restrict__ deltaX1,
	      float* __restrict__ deltaX2,
	const float* __restrict__ deltaY,
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float alpha, float beta, float gamma, float k,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy2 = *(float4*)(deltaY + index4);
		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + index4);

		char4 flag;//Y1 = alpha*X1 + beta*X2 + gamma
		flag.x = (alpha * x1.x + beta * x2.x + gamma) > 0.0f;
		flag.y = (alpha * x1.y + beta * x2.y + gamma) > 0.0f;
		flag.z = (alpha * x1.z + beta * x2.z + gamma) > 0.0f;
		flag.w = (alpha * x1.w + beta * x2.w + gamma) > 0.0f;

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


void __linear_dual2D_with_leakyRelu_deltaX_v2(cudaStream_t stream,
	float* deltaX1, float* deltaX2,
	const float* deltaY,
	const float* X1, const float* X2,
	float alpha, float beta, float gamma, float k,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear_dual2d_with_leakyRelu_deltaX_v2_k4_small(stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, k, lengthv); return; }
	if (lengthv > LENGTHV_MAX) { linear_dual2d_with_leakyRelu_deltaX_v2_k4_max(stream, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, k, lengthv); return; }
	linear_dual2d_with_leakyRelu_deltaX_v2_k4(stream, 5, 2, deltaX1, deltaX2, deltaY, X1, X2, alpha, beta, gamma, k, lengthv);
}

#endif