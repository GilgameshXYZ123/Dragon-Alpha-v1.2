#pragma once

#ifndef IMG_QUADRATIC_2D_H
#define IMG_QUADRATIC_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef IMG_QUADRATIC_2D_CALL
#define IMG_QUADRATIC_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv % 16 == 0
#define img_quadratic2d_k16(stream, LB, LT, X, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_quadratic2D_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha, beta, gamma, Y, lengthv, width, stride)

//lengthv % 8 == 0
#define img_quadratic2d_k8(stream, LB, LT, X, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_quadratic2D_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha, beta, gamma, Y, lengthv, width, stride)

//common
#define img_quadratic2d_k4(stream, LB, LT, X, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_quadratic2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha, beta, gamma, Y, lengthv, width, stride)

//common
#define img_quadratic2d_k4_small(stream, X, alpha, beta, gamma, Y, lengthv, width, stride)\
	img_quadratic2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, alpha, beta, gamma, Y, lengthv, width, stride)

#endif


//lengthv % 16 == 0
#ifndef IMG_QUADRATIC_2D_KERNEL16
#define IMG_QUADRATIC_2D_KERNEL16

//(5, 4): Size = 8, Time = 0.025 mesc, Speed = 312.5 GB/s
__global__ void img_quadratic2D_kernel_16(
	const char* __restrict__ X, float alpha, float beta, float gamma,
	      char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		uchar16 x = *(uchar16*)(X + index16);

		float4 fy; uchar16 y;//y = alpha*x*x + beta*x + gamma
		
		fy.x = alpha * x.x0*x.x0 + beta * x.x0 + gamma; y.x0 = PIXEL_CLIP(fy.x);
		fy.y = alpha * x.y0*x.y0 + beta * x.y0 + gamma; y.y0 = PIXEL_CLIP(fy.y);
		fy.z = alpha * x.z0*x.z0 + beta * x.z0 + gamma; y.z0 = PIXEL_CLIP(fy.z);
		fy.w = alpha * x.w0*x.w0 + beta * x.w0 + gamma; y.w0 = PIXEL_CLIP(fy.w);

		fy.x = alpha * x.x1*x.x1 + beta * x.x1 + gamma; y.x1 = PIXEL_CLIP(fy.x);
		fy.y = alpha * x.y1*x.y1 + beta * x.y1 + gamma; y.y1 = PIXEL_CLIP(fy.y);
		fy.z = alpha * x.z1*x.z1 + beta * x.z1 + gamma; y.z1 = PIXEL_CLIP(fy.z);
		fy.w = alpha * x.w1*x.w1 + beta * x.w1 + gamma; y.w1 = PIXEL_CLIP(fy.w);

		fy.x = alpha * x.x2*x.x2 + beta * x.x2 + gamma; y.x2 = PIXEL_CLIP(fy.x);
		fy.y = alpha * x.y2*x.y2 + beta * x.y2 + gamma; y.y2 = PIXEL_CLIP(fy.y);
		fy.z = alpha * x.z2*x.z2 + beta * x.z2 + gamma; y.z2 = PIXEL_CLIP(fy.z);
		fy.w = alpha * x.w2*x.w2 + beta * x.w2 + gamma; y.w2 = PIXEL_CLIP(fy.w);

		fy.x = alpha * x.x3*x.x3 + beta * x.x3 + gamma; y.x3 = PIXEL_CLIP(fy.x);
		fy.y = alpha * x.y3*x.y3 + beta * x.y3 + gamma; y.y3 = PIXEL_CLIP(fy.y);
		fy.z = alpha * x.z3*x.z3 + beta * x.z3 + gamma; y.z3 = PIXEL_CLIP(fy.z);
		fy.w = alpha * x.w3*x.w3 + beta * x.w3 + gamma; y.w3 = PIXEL_CLIP(fy.w);

		within_width16(y, index16, stride, width);
		*(uchar16*)(Y + index16) = y;
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_QUADRATIC_2D_KERNEL8
#define IMG_QUADRATIC_2D_KERNEL8

//(5, 3): Size = 8, Time = 0.032 mesc, Speed = 244.141 GB/s
__global__ void img_quadratic2D_kernel_8(
	const char* __restrict__ X, float alpha, float beta, float gamma,
	      char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		uchar8 x = *(uchar8*)(X + index8);

		float4 fy; uchar8 y;//y = alpha*x*x + beta*x + gamma

		fy.x = alpha * x.x0*x.x0 + beta * x.x0 + gamma; y.x0 = PIXEL_CLIP(fy.x);
		fy.y = alpha * x.y0*x.y0 + beta * x.y0 + gamma; y.y0 = PIXEL_CLIP(fy.y);
		fy.z = alpha * x.z0*x.z0 + beta * x.z0 + gamma; y.z0 = PIXEL_CLIP(fy.z);
		fy.w = alpha * x.w0*x.w0 + beta * x.w0 + gamma; y.w0 = PIXEL_CLIP(fy.w);

		fy.x = alpha * x.x1*x.x1 + beta * x.x1 + gamma; y.x1 = PIXEL_CLIP(fy.x);
		fy.y = alpha * x.y1*x.y1 + beta * x.y1 + gamma; y.y1 = PIXEL_CLIP(fy.y);
		fy.z = alpha * x.z1*x.z1 + beta * x.z1 + gamma; y.z1 = PIXEL_CLIP(fy.z);
		fy.w = alpha * x.w1*x.w1 + beta * x.w1 + gamma; y.w1 = PIXEL_CLIP(fy.w);

		within_width8(y, index8, stride, width);
		*(uchar8*)(Y + index8) = y;
	}
}

#endif


//common
#ifndef IMG_QUADRATIC_2D_KERNEL4
#define IMG_QUADRATIC_2D_KERNEL4

//(5, 2): Size = 8, Time = 0.047 mesc, Speed = 166.223 GB/s
//(5, 3): Size = 8, Time = 0.039 mesc, Speed = 200.321 GB/s
__global__ void img_quadratic2D_kernel_4(
	const char* __restrict__ X, float alpha, float beta, float gamma,
	      char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x = *(uchar4*)(X + index4);

		float4 fy; uchar4 y;//y = alpha*x*x + beta*x + gamma
		fy.x = alpha * x.x*x.x + beta * x.x + gamma; y.x = PIXEL_CLIP(fy.x);
		fy.y = alpha * x.y*x.y + beta * x.y + gamma; y.y = PIXEL_CLIP(fy.y);
		fy.z = alpha * x.z*x.z + beta * x.z + gamma; y.z = PIXEL_CLIP(fy.z);
		fy.w = alpha * x.w*x.w + beta * x.w + gamma; y.w = PIXEL_CLIP(fy.w);

		within_width4(y, index4, stride, width);
		*(uchar4*)(Y + index4) = y;
	}
}

#endif


void __img_quadratic2D(cudaStream_t stream,
	const char* X, float alpha, float beta, float gamma,
	      char* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { img_quadratic2d_k4_small(stream, X, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (!(lengthv & 15) && lengthv >= 16484) { img_quadratic2d_k16(stream, 5, 4, X, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (!(lengthv &  7) && lengthv >=  8192) { img_quadratic2d_k8(stream, 5, 3, X, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_quadratic2d_k4(stream, 5, 3, X, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	img_quadratic2d_k4(stream, 5, 2, X, alpha, beta, gamma, Y, lengthv, width, stride);
}

#endif