#pragma once

#ifndef LOG_2D_H
#define LOG_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LOG_2D_CALL
#define LOG_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define log2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	log2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//common
#define log2d_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	log2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//lengthv > lengthv_max
#define log2d_k4_max(stream, alpha, X, beta, Y, lengthv, width, stride)\
	log2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef LOG_2D_KERNEL
#define LOG_2D_KERNEL

//forward propagation: Y = log(alpha*X + beta);
//backward propagation:
//<1> Y' = alpha/(alpha*X + beta) 
//<2> Y = log(alpha*X + beta) -> exp(Y) = alpha*X + beta
//<3> Y' = alpha * exp(-Y)
//<4> deltaX = deltaY * Y' = alpha * deltaY / exp(Y)

__global__ void log2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_4_0;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y;//Y = log(alpha*X + beta);
		y.x = logf(alpha*x.x + beta);
		y.y = logf(alpha*x.y + beta);
		y.z = logf(alpha*x.z + beta);
		y.w = logf(alpha*x.w + beta);

		within_width_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __log2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { log2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { log2d_k4_max(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	log2d_k4(stream, 5, 2, alpha, X,  beta, Y, lengthv, width, stride);
}

#endif