#pragma once

#ifndef IMG_LOG_2D_H
#define IMG_LOG_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef IMG_LOG_2D_CALL
#define IMG_LOG_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv % 16 == 0
#define img_log2d_k16(stream, LB, LT, X, Y, C, alpha, beta, lengthv, width, stride)\
	img_log2D_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, C, alpha, beta, lengthv, width, stride)

//lengthv % 8 == 0
#define img_log2d_k8(stream, LB, LT, X, Y, C, alpha, beta, lengthv, width, stride)\
	img_log2D_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, C, alpha, beta, lengthv, width, stride)

//common
#define img_log2d_k4(stream, LB, LT, X, Y, C, alpha, beta, lengthv, width, stride)\
	img_log2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, C, alpha, beta, lengthv, width, stride)

//common
#define img_log2d_k4_small(stream, X, Y, C, alpha, beta, lengthv, width, stride)\
	img_log2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, C, alpha, beta, lengthv, width, stride)

#endif


//lengthv % 16 == 0
#ifndef IMG_LOG_2D_KERNEL16
#define IMG_LOG_2D_KERNEL16

//(5, 4): Size = 16, Time = 0.077 mesc, Speed = 202.922 GB/s
__global__ void img_log2D_kernel_16(
	const char* __restrict__ X, 
	      char* __restrict__ Y,
	float C, float alpha, float beta,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		uchar16 x = *(uchar16*)(X + index16);

		float16 fy;//y = C * log(alpha*x + beta)
		fy.x0 = C * logf(x.x0 * alpha + beta);
		fy.y0 = C * logf(x.y0 * alpha + beta);
		fy.z0 = C * logf(x.z0 * alpha + beta);
		fy.w0 = C * logf(x.w0 * alpha + beta);

		fy.x1 = C * logf(x.x1 * alpha + beta);
		fy.y1 = C * logf(x.y1 * alpha + beta);
		fy.z1 = C * logf(x.z1 * alpha + beta);
		fy.w1 = C * logf(x.w1 * alpha + beta);

		fy.x2 = C * logf(x.x2 * alpha + beta);
		fy.y2 = C * logf(x.y2 * alpha + beta);
		fy.z2 = C * logf(x.z2 * alpha + beta);
		fy.w2 = C * logf(x.w2 * alpha + beta);

		fy.x3 = C * logf(x.x3 * alpha + beta);
		fy.y3 = C * logf(x.y3 * alpha + beta);
		fy.z3 = C * logf(x.z3 * alpha + beta);
		fy.w3 = C * logf(x.w3 * alpha + beta);
		uchar16 y; PIXEL_CLIP_16(y, fy);

		within_width16(y, index16, stride, width);
		*(uchar16*)(Y + index16) = y;
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_LOG_2D_KERNEL8
#define IMG_LOG_2D_KERNEL8

//(5, 3): Size = 16, Time = 0.081 mesc, Speed = 192.901 GB/s
__global__ void img_log2D_kernel_8(
	const char* __restrict__ X, 
	      char* __restrict__ Y,
	float C, float alpha, float beta,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		uchar8 x = *(uchar8*)(X + index8);

		float8 fy;//y = C * log(alpha*x + beta)
		fy.x0 = C * logf(x.x0 * alpha + beta);
		fy.y0 = C * logf(x.y0 * alpha + beta);
		fy.z0 = C * logf(x.z0 * alpha + beta);
		fy.w0 = C * logf(x.w0 * alpha + beta);

		fy.x1 = C * logf(x.x1 * alpha + beta);
		fy.y1 = C * logf(x.y1 * alpha + beta);
		fy.z1 = C * logf(x.z1 * alpha + beta);
		fy.w1 = C * logf(x.w1 * alpha + beta);
		uchar8 y; PIXEL_CLIP_8(y, fy);

		within_width8(y, index8, stride, width);
		*(uchar8*)(Y + index8) = y;
	}
}

#endif


//common
#ifndef IMG_LOG_2D_KERNEL4
#define IMG_LOG_2D_KERNEL4
 
//(5, 3): Size = 8, Time = 0.054 mesc, Speed = 144.676 GB/s
//(5, 2): Size = 8, Time = 0.052 mesc, Speed = 150.24 GB/s
__global__ void img_log2D_kernel_4(
	const char* __restrict__ X, 
	      char* __restrict__ Y, 
	float C, float alpha, float beta,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x = *(uchar4*)(X + index4);

		float4 fy;//y = C * log(alpha*x + beta)
		fy.x = C * logf(x.x * alpha + beta);
		fy.y = C * logf(x.y * alpha + beta);
		fy.z = C * logf(x.z * alpha + beta);
		fy.w = C * logf(x.w * alpha + beta);
		uchar4 y; PIXEL_CLIP_4(y, fy);

		within_width4(y, index4, stride, width);
		*(uchar4*)(Y + index4) = y;
	}
}

#endif


void __img_log2D(cudaStream_t stream,
	const char* X, 
	      char* Y,
	float C, float alpha, float beta,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { img_log2d_k4_small(stream, X, Y, C, alpha, beta, lengthv, width, stride); return; }
	if (!(lengthv & 15) && lengthv >= 16484) { img_log2d_k16(stream, 5, 4, X, Y, C, alpha, beta, lengthv, width, stride); return; }
	if (!(lengthv &  7) && lengthv >=  8192) { img_log2d_k8(stream, 5, 3, X, Y, C, alpha, beta, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_log2d_k4(stream, 5, 3, X, Y, C, alpha, beta, lengthv, width, stride); return; }
	img_log2d_k4(stream, 5, 2, X, Y, C, alpha, beta, lengthv, width, stride); 
}

#endif