#pragma once

#ifndef IMG_AFFINE_H
#define IMG_AFFINE_H

//C(channel) % 4 == 0
//lengthv = N*OH*OW*C, so lengthv % 4 == 0
#ifndef IMG_AFFINE_CALL
#define IMG_AFFINE_CALL

//C % 16 == 0
#define img_affine_k16(stream, LB, LT, X, IH, IW, Y, OH, OW, C, lengthv)\
	img_affine_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,IH,IW, Y,(OH*OW*C),(OW*C),C, r00,r01,r02,r10,r11,r12, lengthv)

//C % 8 == 0
#define img_affine_k8(stream, LB, LT, X, IH, IW, Y, OH, OW, C, lengthv)\
	img_affine_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,IH,IW, Y,(OH*OW*C),(OW*C),C, r00,r01,r02,r10,r11,r12, lengthv)

#define img_affine_k4(stream, LB, LT, X, IH, IW, Y, OH, OW, C, lengthv)\
	img_affine_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,IH,IW, Y,(OH*OW*C),(OW*C),C, r00,r01,r02,r10,r11,r12, lengthv)

#define img_affine_k4_small(stream, X, IH, IW, Y, OH, OW, C, lengthv)\
	img_affine_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X,IH,IW, Y,(OH*OW*C),(OW*C),C, r00,r01,r02,r10,r11,r12, lengthv)

#endif


#ifndef IMG_AFFINE_KERNEL16
#define IMG_AFFINE_KERNEL16

//(5, 4): Size = 15.3906, Time = 0.046 mesc, Speed = 326.737GB/s
__global__ void img_affine_kernel_16(
	const char* __restrict__ X, int IH, int IW,
	char* __restrict__ Y, int OH_OW_C, int OW_C, int C,
	float r00, float r01, float r02,
	float r10, float r11, float r12,
	int lengthv)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		const int yindex = index16;//[n, oh, ow, c]
		int n = yindex / OH_OW_C, yr = yindex - n * OH_OW_C;
		int oh = yr / OW_C; yr -= oh * OW_C;
		int ow = yr / C, c = yr - ow * C;

		int iw = lroundf(r00*ow + r01 * oh + r02);//find the nearset pixel
		int ih = lroundf(r10*ow + r11 * oh + r12);

		const int xindex = ((n*IH + ih)*IW + iw)*C + c;//[n, ih, iw, c]
		bool loadx = in_range2D(ih, iw, IH, IW);
		*(uchar16*)(Y + yindex) = (loadx ? *(uchar16*)(X + xindex) : UINT8_0_16);
	}
}

#endif


#ifndef IMG_AFFINE_KERNEL8
#define IMG_AFFINE_KERNEL8

//(5, 3): Size = 15.3906, Time = 0.068 mesc, Speed = 221.028GB/s
__global__ void img_affine_kernel_8(
	const char* __restrict__ X, int IH, int IW,
	char* __restrict__ Y, int OH_OW_C, int OW_C, int C,
	float r00, float r01, float r02,
	float r10, float r11, float r12,
	int lengthv)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		const int yindex = index8;//[n, oh, ow, c]
		int n = yindex / OH_OW_C, yr = yindex - n * OH_OW_C;
		int oh = yr / OW_C; yr -= oh * OW_C;
		int ow = yr / C, c = yr - ow * C;

		int iw = lroundf(r00*ow + r01 * oh + r02);//find the nearset pixel
		int ih = lroundf(r10*ow + r11 * oh + r12);

		const int xindex = ((n*IH + ih)*IW + iw)*C + c;//[n, ih, iw, c]
		bool loadx = in_range2D(ih, iw, IH, IW);
		*(uchar8*)(Y + yindex) = (loadx ? *(uchar8*)(X + xindex) : UINT8_0_8);
	}
}

#endif


#ifndef IMG_AFFINE_KERNEL4
#define IMG_AFFINE_KERNEL4

//(5, 3): Size = 15.3906, Time = 0.079 mesc, Speed = 190.252GB/s
//(5, 2): Size = 15.3906, Time = 0.13  mesc, Speed = 115.615GB/s
__global__ void img_affine_kernel_4(
	const char* __restrict__ X, int IH, int IW,
	      char* __restrict__ Y, int OH_OW_C, int OW_C, int C,
	float r00, float r01, float r02,
	float r10, float r11, float r12,
	int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int yindex = index4;//[n, oh, ow, c]
		int n = yindex / OH_OW_C, yr = yindex - n*OH_OW_C;
		int oh = yr / OW_C; yr -= oh * OW_C;
		int ow = yr / C, c = yr - ow * C;

		int iw = lroundf(r00*ow + r01 * oh + r02);//find the nearset pixel
		int ih = lroundf(r10*ow + r11 * oh + r12);

		const int xindex = ((n*IH + ih)*IW + iw)*C + c;//[n, ih, iw, c]
		bool loadx = in_range2D(ih, iw, IH, IW);
		*(uchar4*)(Y + yindex) = (loadx ? *(uchar4*)(X + xindex) : UINT8_0_4);
	}
}

#endif


void __img_affine(cudaStream_t stream,
	const char* __restrict__ X, int IH, int IW,
	      char* __restrict__ Y, int OH, int OW,
	float r00, float r01, float r02,
	float r10, float r11, float r12,
	int N, int C)
{
	int lengthv = N * OH*OW*C;
	if (lengthv < 256) { img_affine_k4_small(stream, X, IH, IW, Y, OH, OW, C, lengthv); return; }
	if (!(C & 15) && lengthv >= 16384) { img_affine_k16(stream, 5, 4, X, IH, IW, Y, OH, OW, C, lengthv); return; }
	if (!(C &  7) && lengthv >=  8192) { img_affine_k8(stream, 5, 3, X, IH, IW, Y, OH, OW, C, lengthv); return; }
	if (lengthv >= 8192) { img_affine_k4(stream, 5, 3, X, IH, IW, Y, OH, OW, C, lengthv); return; }
	img_affine_k4(stream, 5, 2, X, IH, IW, Y, OH, OW, C, lengthv);
}

#endif