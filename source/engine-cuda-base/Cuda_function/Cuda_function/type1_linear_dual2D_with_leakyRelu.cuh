#pragma once

#ifndef LINEAR_DUAL_2D_WITH_LEAKY_RELU_H
#define LINEAR_DUAL_2D_WITH_LEAKY_RELU_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LINEAR_DUAL_2D_WITH_LEAKY_RELU_CALL
#define LINEAR_DUAL_2D_WITH_LEAKY_RELU_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define linear_dual2d_with_leakyRelu_k4_small(stream, X1, X2, alpha, beta, gamma, k, Y, lengthv, width, stride)\
	linear_dual2D_with_leakyRelu_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, alpha, beta, gamma, k, Y, lengthv, width, stride)

//common
#define linear_dual2d_with_leakyRelu_k4(stream, LB, LT, X1, X2, alpha, beta, gamma, k, Y, lengthv, width, stride)\
	linear_dual2D_with_leakyRelu_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, alpha, beta, gamma, k, Y, lengthv, width, stride)

//lengthv < 256
#define linear_dual2d_with_leakyRelu_k4_max(stream, X1, X2, alpha, beta, gamma, k, Y, lengthv, width, stride)\
	linear_dual2D_with_leakyRelu_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X1, X2, alpha, beta, gamma, k, Y, lengthv, width, stride)


#endif


#ifndef LINEAR_DUAL_2D_WITH_LEAKY_RELU_KERNEL
#define LINEAR_DUAL_2D_WITH_LEAKY_RELU_KERNEL

//forward propagation:
//(1) Y1 = alpha*X1 + beta*X2 + gamma
//(2) Y2 = (Y1 > 0? Y1 : k*Y1)
//backward propagation:
//(1) Y1 = alpha*X1 + beta*X2 + gamma 
//(2) deltaY1 = deltaY2 * (Y1 > 0? Y1 : k)
//(3) deltaX1 = deltaY1 * alpha
//(4) deltaX2 = deltaY2 * beta

__global__ void linear_dual2D_with_leakyRelu_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float alpha, float beta, float gamma, float k,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + index4);

		float4 y1;//Y1 = alpha*X1 + beta*X2 + gamma
		y1.x = alpha * x1.x + beta * x2.x + gamma;
		y1.y = alpha * x1.y + beta * x2.y + gamma;
		y1.z = alpha * x1.z + beta * x2.z + gamma;
		y1.w = alpha * x1.w + beta * x2.w + gamma;

		float4 y2;//Y2 = leakyRelu(Y1, k)
		y2.x = LEAKY_RELU(y1.x, k);
		y2.y = LEAKY_RELU(y1.y, k);
		y2.z = LEAKY_RELU(y1.z, k);
		y2.w = LEAKY_RELU(y1.w, k);

		within_width(y2, index4, stride, width);
		*(float4*)(Y + index4) = y2;
	}
}

#endif


void __linear_dual2D_with_leakyRelu(cudaStream_t stream,
	const float* X1,
	const float* X2,
	float alpha, float beta, float gamma, float k,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear_dual2d_with_leakyRelu_k4_small(stream, X1, X2, alpha, beta, gamma, k, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { linear_dual2d_with_leakyRelu_k4_max(stream, X1, X2, alpha, beta, gamma, k, Y, lengthv, width, stride); return; }
	linear_dual2d_with_leakyRelu_k4(stream, 5, 2, X1, X2, alpha, beta, gamma, k, Y, lengthv, width, stride);
}

#endif