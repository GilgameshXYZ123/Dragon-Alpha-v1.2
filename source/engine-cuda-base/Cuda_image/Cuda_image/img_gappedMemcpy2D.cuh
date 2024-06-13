#pragma once

#ifndef IMG_GAPPED_MEMCPY_2D_H
#define IMG_GAPPED_MEMCPY_2D_H


#ifndef IMG_GAPPED_MEMCPY_CALL
#define IMG_GAPPED_MEMCPY_CALL

//char1===========================================================
#define img_gappedMemcpy_k1_small(stream, X, strideX, Y, strideY, width, length) \
	img_gappedMemcpy_kernel1\
		<<< 1, length, 0, stream >>> \
			(X, strideX, Y, strideY, width, length)

#define img_gappedMemcpy_k1(stream, LB, LT, X, strideX, Y, strideY, width, length) \
	img_gappedMemcpy_kernel1\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, strideX, Y, strideY, width, length)

//char4===========================================================
#define img_gappedMemcpy_k4_small(stream, X, strideX, Y, strideY, width, length) \
	img_gappedMemcpy_kernel4\
		<<< 1, length, 0, stream >>> \
			(X, strideX, Y, strideY, width, length)

#define img_gappedMemcpy_k4(stream, LB, LT, X, strideX, Y, strideY, width, length) \
	img_gappedMemcpy_kernel4\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, strideX, Y, strideY, width, length)

//char8 (LT >= 3)=================================================
#define img_gappedMemcpy_k8_small(stream, X, strideX, Y, strideY, width, length) \
	img_gappedMemcpy_kernel8\
		<<< 1, length, 0, stream >>> \
			(X, strideX, Y, strideY, width, length)

#define img_gappedMemcpy_k8(stream, LB, LT, X, strideX, Y, strideY, width, length) \
	img_gappedMemcpy_kernel8\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, strideX, Y, strideY, width, length)

//char16(LT >= 4)=================================================
#define img_gappedMemcpy_k16_small(stream, X, strideX, Y, strideY, width, length) \
	img_gappedMemcpy_kernel16\
		<<< 1, length, 0, stream >>> \
			(X, strideX, Y, strideY, width, length)

#define img_gappedMemcpy_k16(stream, LB, LT, X, strideX, Y, strideY, width, length) \
	img_gappedMemcpy_kernel16\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, strideX, Y, strideY, width, length)

#endif


//width % 16 == 0
#ifndef IMG_GAPPED_MEMCPY_KERNEL16
#define IMG_GAPPED_MEMCPY_KERNEL16

__global__ void img_gappedMemcpy_kernel16(
	const char* __restrict__ X, int strideX,
	char* __restrict__ Y, int strideY,
	int width, int length)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x, index16 = index << 4;
	int step = (blockDim.x * gridDim.x), step16 = step << 4;

	for (; index16 < length; index16 += step16)
	{
		//index4 = y*width + x
		int y = index16 / width;
		int x = index16 - y * width;

		int Xindex = y * strideX + x;
		int Yindex = y * strideY + x;
		*(char16*)(Y + Yindex) = *(char16*)(X + Xindex);
	}
}

#endif


//width % 8 == 0
#ifndef IMG_GAPPED_MEMCPY_KERNEL8
#define IMG_GAPPED_MEMCPY_KERNEL8

__global__ void img_gappedMemcpy_kernel8(
	const char* __restrict__ X, int strideX,
	      char* __restrict__ Y, int strideY,
	int width, int length)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x, index8 = index << 3;
	int step = (blockDim.x * gridDim.x), step8 = step << 3;

	for (; index8 < length; index8 += step8)
	{
		//index4 = y*width + x
		int y = index8 / width;
		int x = index8 - y * width;

		int Xindex = y * strideX + x;
		int Yindex = y * strideY + x;
		*(char8*)(Y + Yindex) = *(char8*)(X + Xindex);
	}
}

#endif


//width % 4 == 0
#ifndef IMG_GAPPED_MEMCPY_KERNEL4
#define IMG_GAPPED_MEMCPY_KERNEL4

__global__ void img_gappedMemcpy_kernel4(
	const char* __restrict__ X, int strideX,
	      char* __restrict__ Y, int strideY,
	int width, int length)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x, index4 = index << 2;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (; index4 < length; index4 += step4)
	{
		//index4 = y*width + x
		int y = index4 / width;
		int x = index4 - y * width;

		int Xindex = y * strideX + x;
		int Yindex = y * strideY + x;
		*(char4*)(Y + Yindex) = *(char4*)(X + Xindex);
	}
}

#endif


//common
#ifndef IMG_GAPPED_MEMCPY_KERNEL1
#define IMG_GAPPED_MEMCPY_KERNEL1

__global__ void img_gappedMemcpy_kernel1(
	const char* __restrict__ X, int strideX,
	      char* __restrict__ Y, int strideY,
	int width, int length)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x);

	for (; index < length; index += step)
	{
		//index = y*width + x
		int y = index / width;
		int x = index - y * width;

		int Xindex = y * strideX + x;
		int Yindex = y * strideY + x;
		Y[Yindex] = X[Xindex];
	}
}

#endif


//total_stride >= copy_stride
//length % copy_stride == 0
//X(src) -> Y(dst)
void __img_gappedMemcpy2D(cudaStream_t stream,
	const char* X, int Xstart, int strideX,
	      char* Y, int Ystart, int strideY,
	int width, int length)
{
	X += Xstart; Y += Ystart;
	if (length < 256) { img_gappedMemcpy_k1_small(stream, X, strideX, Y, strideY, width, length); return; }
	if (!(width & 15) && length >= 16384) { img_gappedMemcpy_k16(stream, 5, 4, X, strideX, Y, strideY, width, length); return; }
	if (!(width &  7) && length >=  8192) { img_gappedMemcpy_k8(stream, 5, 3, X, strideX, Y, strideY, width, length); return; }
	if (!(width & 3)) { img_gappedMemcpy_k4(stream, 5, 2, X, strideX, Y, strideY, width, length); return; }
	img_gappedMemcpy_k1(stream, 5, 2, X, strideX, Y, strideY, width, length);
}

#endif