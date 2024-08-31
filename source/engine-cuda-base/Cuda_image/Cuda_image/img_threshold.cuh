#pragma once

#ifndef IMG_THRESHOLD_2D_H
#define IMG_THRESHOLD_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef IMG_THRESHOLD_2D_CALL
#define IMG_THRESHOLD_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv % 16 == 0
#define img_threshold2d_k16(stream, LB, LT, X, alpha, v, v1, v2, Y, lengthv, width, stride)\
	img_threshold2D_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha, v, v1, v2, Y, lengthv, width, stride)

//lengthv % 8 == 0
#define img_threshold2d_k8(stream, LB, LT, X, alpha, v, v1, v2, Y, lengthv, width, stride)\
	img_threshold2D_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha, v, v1, v2, Y, lengthv, width, stride)

//common
#define img_threshold2d_k4(stream, LB, LT, X, alpha, v, v1, v2, Y, lengthv, width, stride)\
	img_threshold2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha, v, v1, v2, Y, lengthv, width, stride)

//common
#define img_threshold2d_k4_small(stream, X, alpha, v, v1, v2, Y, lengthv, width, stride)\
	img_threshold2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, alpha, v, v1, v2, Y, lengthv, width, stride)

#endif


//lengthv % 16 == 0
#ifndef IMG_THRESHOLD_2D_KERNEL16
#define IMG_THRESHOLD_2D_KERNEL16

//(5, 4): Size = 8, Time = 0.025 mesc, Speed = 312.5 GB/s
__global__ void img_threshold2D_kernel_16(
	const char* __restrict__ X, float alpha, float v, char v1, char v2,
	      char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		uchar16 x = *(uchar16*)(X + index16);

		float4 fy; uchar16 y;//y = (a*x > v ?  v1  : v2)

		fy.x = x.x0 * alpha; y.x0 = (fy.x > v ? v1 : v2);
		fy.y = x.y0 * alpha; y.y0 = (fy.y > v ? v1 : v2);
		fy.z = x.z0 * alpha; y.z0 = (fy.z > v ? v1 : v2);
		fy.w = x.w0 * alpha; y.w0 = (fy.w > v ? v1 : v2);

		fy.x = x.x1 * alpha; y.x1 = (fy.x > v ? v1 : v2);
		fy.y = x.y1 * alpha; y.y1 = (fy.y > v ? v1 : v2);
		fy.z = x.z1 * alpha; y.z1 = (fy.z > v ? v1 : v2);
		fy.w = x.w1 * alpha; y.w1 = (fy.w > v ? v1 : v2);

		fy.x = x.x2 * alpha; y.x2 = (fy.x > v ? v1 : v2);
		fy.y = x.y2 * alpha; y.y2 = (fy.y > v ? v1 : v2);
		fy.z = x.z2 * alpha; y.z2 = (fy.z > v ? v1 : v2);
		fy.w = x.w2 * alpha; y.w2 = (fy.w > v ? v1 : v2);

		fy.x = x.x3 * alpha; y.x3 = (fy.x > v ? v1 : v2);
		fy.y = x.y3 * alpha; y.y3 = (fy.y > v ? v1 : v2);
		fy.z = x.z3 * alpha; y.z3 = (fy.z > v ? v1 : v2);
		fy.w = x.w3 * alpha; y.w3 = (fy.w > v ? v1 : v2);

		within_width16(y, index16, stride, width);
		*(uchar16*)(Y + index16) = y;
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_THRESHOLD_2D_KERNEL8
#define IMG_THRESHOLD_2D_KERNEL8

//(5, 3): Size = 8, Time = 0.032 mesc, Speed = 244.141 GB/s
__global__ void img_threshold2D_kernel_8(
	const char* __restrict__ X, float alpha, float v, char v1, char v2,
	      char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		uchar8 x = *(uchar8*)(X + index8);

		float4 fy; uchar8 y;//y = (a*x > v ?  v1  : v2)

		fy.x = x.x0 * alpha; y.x0 = (fy.x > v ? v1 : v2);
		fy.y = x.y0 * alpha; y.y0 = (fy.y > v ? v1 : v2);
		fy.z = x.z0 * alpha; y.z0 = (fy.z > v ? v1 : v2);
		fy.w = x.w0 * alpha; y.w0 = (fy.w > v ? v1 : v2);

		fy.x = x.x1 * alpha; y.x1 = (fy.x > v ? v1 : v2);
		fy.y = x.y1 * alpha; y.y1 = (fy.y > v ? v1 : v2);
		fy.z = x.z1 * alpha; y.z1 = (fy.z > v ? v1 : v2);
		fy.w = x.w1 * alpha; y.w1 = (fy.w > v ? v1 : v2);

		within_width8(y, index8, stride, width);
		*(uchar8*)(Y + index8) = y;
	}
}

#endif


//common
#ifndef IMG_THRESHOLD_2D_KERNEL4
#define IMG_THRESHOLD_2D_KERNEL4

//(5, 2): Size = 8, Time = 0.047 mesc, Speed = 166.223 GB/s
//(5, 3): Size = 8, Time = 0.039 mesc, Speed = 200.321 GB/s
__global__ void img_threshold2D_kernel_4(
	const char* __restrict__ X, float alpha, float v, char v1, char v2,
	      char* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x = *(uchar4*)(X + index4);

		float4 fy; uchar4 y;//y = (a*x > v ?  v1  : v2)
		fy.x = x.x * alpha; y.x = (fy.x > v ? v1 : v2);
		fy.y = x.y * alpha; y.y = (fy.y > v ? v1 : v2);
		fy.z = x.z * alpha; y.z = (fy.z > v ? v1 : v2);
		fy.w = x.w * alpha; y.w = (fy.w > v ? v1 : v2);
		within_width4(y, index4, stride, width);
		*(uchar4*)(Y + index4) = y;
	}
}

#endif


void __img_threshold2D(cudaStream_t stream,
	const char* X, float alpha, float v, char v1, char v2,
	char* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { img_threshold2d_k4_small(stream, X, alpha, v, v1, v2, Y, lengthv, width, stride); return; }
	if (!(lengthv & 15) && lengthv >= 16484) { img_threshold2d_k16(stream, 5, 4, X, alpha, v, v1, v2, Y, lengthv, width, stride); return; }
	if (!(lengthv & 7) && lengthv >= 8192) { img_threshold2d_k8(stream, 5, 3, X, alpha, v, v1, v2, Y, lengthv, width, stride); return; }
	if (lengthv >= 8192) { img_threshold2d_k4(stream, 5, 3, X, alpha, v, v1, v2, Y, lengthv, width, stride); return; }
	img_threshold2d_k4(stream, 5, 2, X, alpha, v, v1, v2, Y, lengthv, width, stride);
}

#endif