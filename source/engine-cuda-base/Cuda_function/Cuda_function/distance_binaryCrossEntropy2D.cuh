#pragma once

#ifndef BINARY_CROSS_ENTROPY_2D_H
#define BINARY_CROSS_ENTROPY_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef BINARY_CROSS_ENTROPY_2D_CALL
#define BINARY_CROSS_ENTROPY_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define binaryCrossEntropy_2d_k4_small(stream, Y, Yh, alpha, beta, L, lengthv, width, stride)\
	binaryCrossEntropy_2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, Yh, alpha, beta, L, lengthv, width, stride)

//common
#define binaryCrossEntropy_2d_k4(stream, LB, LT, Y, Yh, alpha, beta, L, lengthv, width, stride)\
	binaryCrossEntropy_2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, Yh, alpha, beta, L, lengthv, width, stride)

//lengthv > lengthv_max
#define binaryCrossEntropy_2d_k4_max(stream, Y, Yh, alpha, beta, L, lengthv, width, stride)\
	binaryCrossEntropy_2D_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(Y, Yh, alpha, beta, L, lengthv, width, stride)

#endif


#ifndef BINARY_CROSS_ENTROPY_2D_KERNEL
#define BINARY_CROSS_ENTROPY_2D_KERNEL

__global__ void binaryCrossEntropy_2D_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ Yh,
	float alpha, float beta,
	float* __restrict__ L,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	alpha = -alpha;
	float4 table[2]; table[0] = float4{ 0, 0, 0, 0 };
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 yh = *(float4*)(Yh + index4);
		float4 y = *(float4*)(Y + index4);

		float4 loss;//L = -Y*In(Yh) + (Y - 1)*ln(1 - Yh)
		loss.x = alpha * y.x * logf(yh.x) + beta * (y.x - 1.0f)*logf(1.0f - yh.x);
		loss.y = alpha * y.y * logf(yh.y) + beta * (y.y - 1.0f)*logf(1.0f - yh.y);
		loss.z = alpha * y.z * logf(yh.z) + beta * (y.z - 1.0f)*logf(1.0f - yh.z);
		loss.w = alpha * y.w * logf(yh.w) + beta * (y.w - 1.0f)*logf(1.0f - yh.w);

		within_width_zero_nan(loss, index4, table, stride, width);
		*(float4*)(L + index4) = loss;
	}
}

#endif


void __binaryCrossEntropy_2D(cudaStream_t stream,
	const float* Y, const float* Yh, 
	float alpha, float beta,
	float* L,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { binaryCrossEntropy_2d_k4_small(stream, Y, Yh, alpha, beta, L, lengthv, width, stride); return; }
	if (lengthv > LENGTHV_MAX) { binaryCrossEntropy_2d_k4_max(stream, Y, Yh, alpha, beta, L, lengthv, width, stride); return; }
	binaryCrossEntropy_2d_k4(stream, 5, 2, Y, Yh, alpha, beta, L, lengthv, width, stride);
}

#endif