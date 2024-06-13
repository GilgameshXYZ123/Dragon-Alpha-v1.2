#pragma once

#ifndef IMG_EXP_2D_H
#define IMG_EXP_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef IMG_EXP_2D_CALL
#define IMG_EXP_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv % 16 == 0
#define img_exp2d_k16(stream, LB, LT, X, Y, alpha, beta, C, lengthv, width, stride)\
	img_exp2D_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, alpha, beta, C, lengthv, width, stride)

//lengthv % 8 == 0
#define img_exp2d_k8(stream, LB, LT, X, Y, alpha, beta, C, lengthv, width, stride)\
	img_exp2D_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, alpha, beta, C, lengthv, width, stride)

//common
#define img_exp2d_k4(stream, LB, LT, X, Y, alpha, beta, C, lengthv, width, stride)\
	img_exp2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, alpha, beta, C, lengthv, width, stride)

//common
#define img_exp2d_k4_small(stream, X, Y, alpha, beta, C, lengthv, width, stride)\
	img_exp2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, alpha, beta, C, lengthv, width, stride)

#endif


//lengthv % 16 == 0
#ifndef IMG_EXP_2D_KERNEL16
#define IMG_EXP_2D_KERNEL16

//(5, 4): Size = 16, Time = 0.059 mesc, Speed = 264.831 GB/s
__global__ void img_exp2D_kernel_16(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	float alpha, float beta, float C,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		uchar16 x = *(uchar16*)(X + index16);

		float16 fy;//y = exp(alpha*x + beta) + C
		fy.x0 = expf(x.x0 * alpha + beta) + C;
		fy.y0 = expf(x.y0 * alpha + beta) + C;
		fy.z0 = expf(x.z0 * alpha + beta) + C;
		fy.w0 = expf(x.w0 * alpha + beta) + C;

		fy.x1 = expf(x.x1 * alpha + beta) + C;
		fy.y1 = expf(x.y1 * alpha + beta) + C;
		fy.z1 = expf(x.z1 * alpha + beta) + C;
		fy.w1 = expf(x.w1 * alpha + beta) + C;

		fy.x2 = expf(x.x2 * alpha + beta) + C;
		fy.y2 = expf(x.y2 * alpha + beta) + C;
		fy.z2 = expf(x.z2 * alpha + beta) + C;
		fy.w2 = expf(x.w2 * alpha + beta) + C;

		fy.x3 = expf(x.x3 * alpha + beta) + C;
		fy.y3 = expf(x.y3 * alpha + beta) + C;
		fy.z3 = expf(x.z3 * alpha + beta) + C;
		fy.w3 = expf(x.w3 * alpha + beta) + C;
		uchar16 y; PIXEL_CLIP_16(y, fy);

		within_width16(y, index16, stride, width);
		*(uchar16*)(Y + index16) = y;
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_EXP_2D_KERNEL8
#define IMG_EXP_2D_KERNEL8

//(5, 3): Size = 16, Time = 0.068 mesc, Speed = 229.779 GB/s
__global__ void img_exp2D_kernel_8(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	float alpha, float beta, float C,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		uchar8 x = *(uchar8*)(X + index8);

		float8 fy;//y = exp(alpha*x + beta) + C
		fy.x0 = expf(x.x0 * alpha + beta) + C;
		fy.y0 = expf(x.y0 * alpha + beta) + C;
		fy.z0 = expf(x.z0 * alpha + beta) + C;
		fy.w0 = expf(x.w0 * alpha + beta) + C;

		fy.x1 = expf(x.x1 * alpha + beta) + C;
		fy.y1 = expf(x.y1 * alpha + beta) + C;
		fy.z1 = expf(x.z1 * alpha + beta) + C;
		fy.w1 = expf(x.w1 * alpha + beta) + C;
		uchar8 y; PIXEL_CLIP_8(y, fy);

		within_width8(y, index8, stride, width);
		*(uchar8*)(Y + index8) = y;
	}
}

#endif


//common
#ifndef IMG_EXP_2D_KERNEL4
#define IMG_EXP_2D_KERNEL4

//(5, 3): Size = 16, Time = 0.086 mesc, Speed = 181.686 GB/s
//(5, 2): Size = 16, Time = 0.099 mesc, Speed = 157.828 GB/s
__global__ void img_exp2D_kernel_4(
	const char* __restrict__ X,
	      char* __restrict__ Y,
	float alpha, float beta, float C,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x = *(uchar4*)(X + index4);

		float4 fy;//y = exp(alpha*x + beta) + C
		fy.x = expf(x.x * alpha + beta) + C;
		fy.y = expf(x.y * alpha + beta) + C;
		fy.z = expf(x.z * alpha + beta) + C;
		fy.w = expf(x.w * alpha + beta) + C;
		uchar4 y; PIXEL_CLIP_4(y, fy);

		within_width4(y, index4, stride, width);
		*(uchar4*)(Y + index4) = y;
	}
}

#endif


void __img_exp2D(cudaStream_t stream,
	const char* X,
	char* Y,
	float alpha, float beta, float C,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { img_exp2d_k4_small(stream, X, Y, alpha, beta, C, lengthv, width, stride); return; }
	if (!(lengthv & 15) && lengthv >= 16484) { img_exp2d_k16(stream, 5, 4, X, Y, alpha, beta, C, lengthv, width, stride); return; }
	if (!(lengthv & 7) && lengthv >= 8192) { img_exp2d_k8(stream, 5, 3, X, Y, alpha, beta, C, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_exp2d_k4(stream, 5, 3, X, Y, alpha, beta, C, lengthv, width, stride); return; }
	img_exp2d_k4(stream, 5, 2, X, Y, alpha, beta, C, lengthv, width, stride); 
}

#endif