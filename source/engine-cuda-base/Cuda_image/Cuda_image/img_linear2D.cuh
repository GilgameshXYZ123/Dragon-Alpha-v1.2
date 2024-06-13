#pragma once

#ifndef IMG_LINEAR_2D_H
#define IMG_LINEAR_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef IMG_LINEAR_2D_CALL
#define IMG_LINEAR_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv % 16 == 0
#define img_linear2d_k16(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	img_linear2D_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//lengthv % 8 == 0
#define img_linear2d_k8(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	img_linear2D_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//common
#define img_linear2d_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	img_linear2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

//common
#define img_linear2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	img_linear2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


//lengthv % 16 == 0
#ifndef IMG_LINEAR_2D_KERNEL16
#define IMG_LINEAR_2D_KERNEL16

//(5, 4): Size = 8, Time = 0.025 mesc, Speed = 312.5 GB/s
__global__ void img_linear2D_kernel_16(
	float alpha, const char* __restrict__ X, float beta,
	char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		uchar16 x = *(uchar16*)(X + index16);

		float4 fy; uchar16 y;//y = alpha*x + beta
		fy.x = x.x0 * alpha + beta; y.x0 = PIXEL_CLIP(fy.x);
		fy.y = x.y0 * alpha + beta; y.y0 = PIXEL_CLIP(fy.y);
		fy.z = x.z0 * alpha + beta; y.z0 = PIXEL_CLIP(fy.z);
		fy.w = x.w0 * alpha + beta; y.w0 = PIXEL_CLIP(fy.w);
		
		fy.x = x.x1 * alpha + beta; y.x1 = PIXEL_CLIP(fy.x);
		fy.y = x.y1 * alpha + beta; y.y1 = PIXEL_CLIP(fy.y);
		fy.z = x.z1 * alpha + beta; y.z1 = PIXEL_CLIP(fy.z);
		fy.w = x.w1 * alpha + beta; y.w1 = PIXEL_CLIP(fy.w);

		fy.x = x.x2 * alpha + beta; y.x2 = PIXEL_CLIP(fy.x);
		fy.y = x.y2 * alpha + beta; y.y2 = PIXEL_CLIP(fy.y);
		fy.z = x.z2 * alpha + beta; y.z2 = PIXEL_CLIP(fy.z);
		fy.w = x.w2 * alpha + beta; y.w2 = PIXEL_CLIP(fy.w);

		fy.x = x.x3 * alpha + beta; y.x3 = PIXEL_CLIP(fy.x);
		fy.y = x.y3 * alpha + beta; y.y3 = PIXEL_CLIP(fy.y);
		fy.z = x.z3 * alpha + beta; y.z3 = PIXEL_CLIP(fy.z);
		fy.w = x.w3 * alpha + beta; y.w3 = PIXEL_CLIP(fy.w);

		within_width16(y, index16, stride, width);
		*(uchar16*)(Y + index16) = y;
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_LINEAR_2D_KERNEL8
#define IMG_LINEAR_2D_KERNEL8

//(5, 3): Size = 8, Time = 0.032 mesc, Speed = 244.141 GB/s
__global__ void img_linear2D_kernel_8(
	float alpha, const char* __restrict__ X, float beta,
	char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		uchar8 x = *(uchar8*)(X + index8);

		float4 fy; uchar8 y;//y = alpha*x + beta

		fy.x = x.x0 * alpha + beta; y.x0 = PIXEL_CLIP(fy.x);
		fy.y = x.y0 * alpha + beta; y.y0 = PIXEL_CLIP(fy.y);
		fy.z = x.z0 * alpha + beta; y.z0 = PIXEL_CLIP(fy.z);
		fy.w = x.w0 * alpha + beta; y.w0 = PIXEL_CLIP(fy.w);

		fy.x = x.x1 * alpha + beta; y.x1 = PIXEL_CLIP(fy.x);
		fy.y = x.y1 * alpha + beta; y.y1 = PIXEL_CLIP(fy.y);
		fy.z = x.z1 * alpha + beta; y.z1 = PIXEL_CLIP(fy.z);
		fy.w = x.w1 * alpha + beta; y.w1 = PIXEL_CLIP(fy.w);

		within_width8(y, index8, stride, width);
		*(uchar8*)(Y + index8) = y;
	}
}

#endif


//common
#ifndef IMG_LINEAR_2D_KERNEL4
#define IMG_LINEAR_2D_KERNEL4

//(5, 2): Size = 8, Time = 0.047 mesc, Speed = 166.223 GB/s
//(5, 3): Size = 8, Time = 0.039 mesc, Speed = 200.321 GB/s
__global__ void img_linear2D_kernel_4(
	float alpha, const char* __restrict__ X, float beta,
	char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x = *(uchar4*)(X + index4);

		float4 fy; uchar4 y;//y = alpha*x + beta
		fy.x = x.x * alpha + beta; y.x = PIXEL_CLIP(fy.x);
		fy.y = x.y * alpha + beta; y.y = PIXEL_CLIP(fy.y);
		fy.z = x.z * alpha + beta; y.z = PIXEL_CLIP(fy.z);
		fy.w = x.w * alpha + beta; y.w = PIXEL_CLIP(fy.w);

		within_width4(y, index4, stride, width);
		*(uchar4*)(Y + index4) = y;
	}
}

#endif


void __img_linear2D(cudaStream_t stream,
	float alpha, const char* X, float beta,
	char* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { img_linear2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	if (!(lengthv & 15) && lengthv >= 16484) { img_linear2d_k16(stream, 5, 4, alpha, X, beta, Y, lengthv, width, stride); return; }
	if (!(lengthv &  7) && lengthv >=  8192) { img_linear2d_k8(stream, 5, 3, alpha, X, beta, Y, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_linear2d_k4(stream, 5, 3, alpha, X, beta, Y, lengthv, width, stride); return; }
	img_linear2d_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif