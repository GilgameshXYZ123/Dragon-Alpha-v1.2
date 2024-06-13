#pragma once

#ifndef GELU_2D_H
#define GELU_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef GELU_2D_CALL
#define GELU_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define gelu2d_k4_small(stream, X, Y, lengthv, width, stride)\
	gelu2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, lengthv, width, stride)

//common
#define gelu2d_k4(stream, LB, LT, X, Y, lengthv, width, stride)\
	gelu2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define gelu2d_k4_max(stream, X, Y, lengthv, width, stride)\
	gelu2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X, Y, lengthv, width, stride)

#endif


#ifndef GELU_2D_KERNEL
#define GELU_2D_KERNEL

//[Forward]
//(1) a = sqrt(2 / pi) = 0.7978846
//(2) b = 0.044715
//(3) u = a*(x + b*x^3) = a*x * (1 + b*x^2)
//		= 0.7978846 * x * (1 + 0.044715 * x^2)
//(4) GELU(x) = 0.5*x * (1 + tanh(u))
//		= 0.5*x * (1 + 2/(1 + e^(-2*u)) - 1)
//      = x / (1 + e^(-2u))
//Step:
//(1) u = -1.5957692 * x * (1 + 0.044715 * x^2)
//(2) y = x / (1 + e^u)

__global__ void gelu2D_kernel_4(
	const float* __restrict__ X,
	      float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 u;//u = 0.7978846 * x * (1 + 0.044715 * x^2)
		u.x = -1.5957692f * x.x * (1.0f + 0.044715f * x.x * x.x);
		u.y = -1.5957692f * x.y * (1.0f + 0.044715f * x.y * x.y);
		u.z = -1.5957692f * x.z * (1.0f + 0.044715f * x.z * x.z);
		u.w = -1.5957692f * x.w * (1.0f + 0.044715f * x.w * x.w);

		float4 y;//y = x / (1 + e^u)
		y.x = x.x / (1.0f + expf(u.x));
		y.y = x.y / (1.0f + expf(u.y));
		y.z = x.z / (1.0f + expf(u.z));
		y.w = x.w / (1.0f + expf(u.w));

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __gelu2D(cudaStream_t stream,
	const float* X,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { gelu2d_k4_small(stream, X, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { gelu2d_k4_max(stream, X, Y, lengthv, width, stride); return; }
	gelu2d_k4(stream, 5, 2, X, Y, lengthv, width, stride);
}

#endif