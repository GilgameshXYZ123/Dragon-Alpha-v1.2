#pragma once

#ifndef IMG_EXTRACT_3CHANNELS_H
#define IMG_EXTRACT_3CHANNELS_H

//X[N, H, W, C] -> Y[N, H, W, 3(c0, c1, c2 #pad to 4)]
//extract 3 channels of X to construct Y
//C % 4 == 0, X.stride = C
//Y.stride = 4
//lengthv = N * H * W, so: lengthv % 4 == 0
#ifndef IMG_EXTRACT_3CHANNELS_CALL
#define IMG_EXTRACT_3CHANNELS_CALL

//lengthv % 16 == 0
#define img_extract_3channels_k4(stream, LB, LT, X, IC, Y, c0, c1, c2, lengthv)\
	img_extract_3channels_kernel4\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream >>>\
			(X, IC, Y, c0, c1, c2, lengthv)

//lengthv % 8 == 0
#define img_extract_3channels_k2(stream, LB, LT, X, IC, Y, c0, c1, c2, lengthv)\
	img_extract_3channels_kernel2\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream >>>\
			(X, IC, Y, c0, c1, c2, lengthv)

//common
#define img_extract_3channels_k1(stream, LB, LT, X, IC, Y, c0, c1, c2, lengthv)\
	img_extract_3channels_kernel1\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream >>>\
			(X, IC, Y, c0, c1, c2, lengthv)

//common
#define img_extract_3channels_k1_small(stream, X, IC, Y, c0, c1, c2, lengthv)\
	img_extract_3channels_kernel1\
		<<< 1, lengthv, 0, stream >>>\
			(X, IC, Y, c0, c1, c2, lengthv)

#endif


//lengthv % 16 == 0
#ifndef IMG_EXTRACT_3CHANNELS_KERNEL4
#define IMG_EXTRACT_3CHANNELS_KERNEL4

//(5, 2): Size = 16, Time = 0.103 mesc, Speed = 151.699 GB/s
__global__ void img_extract_3channels_kernel4(
	const char* __restrict__ X, int IC,
	      char* __restrict__ Y, int c0, int c1, int c2,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int xoffset0 = index4 * IC;//index[n, ih, iw]
		const int xoffset1 = xoffset0 + IC;
		const int xoffset2 = xoffset1 + IC;
		const int xoffset3 = xoffset2 + IC;
		char  x0 = X[xoffset0 + c0];
		char  x1 = X[xoffset0 + c1];
		char  x2 = X[xoffset0 + c2];
		char  x3 = X[xoffset1 + c0];
		char  x4 = X[xoffset1 + c1];
		char  x5 = X[xoffset1 + c2];
		char  x6 = X[xoffset2 + c0];
		char  x7 = X[xoffset2 + c1];
		char  x8 = X[xoffset2 + c2];
		char  x9 = X[xoffset3 + c0];
		char x10 = X[xoffset3 + c1];
		char x11 = X[xoffset3 + c2];
		char16 xv = { x0, x1, x2, 0, x3,  x4,  x5, 0,
			          x6, x7, x8, 0, x9, x10, x11, 0 };

		const int yoffset = index4 << 2;//Y.stride = 4, [n, h, iw, 0-3]
		*(char16*)(Y + yoffset) = xv;
	}
}

#endif


//lengthv % 8 == 0
#ifndef IMG_EXTRACT_3CHANNELS_KERNEL2
#define IMG_EXTRACT_3CHANNELS_KERNEL2

//(5, 2): Size = 16, Time = 0.103 mesc, Speed = 151.699 GB/s
__global__ void img_extract_3channels_kernel2(
	const char* __restrict__ X, int IC,
	char* __restrict__ Y, int c0, int c1, int c2,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step2 = step << 1;

	for (int index2 = index << 1; index2 < lengthv; index2 += step2) 
	{
		const int xoffset0 = index2 * IC;//index[n, ih, iw]
		const int xoffset1 = xoffset0 + IC;
		char x0 = X[xoffset0 + c0];
		char x1 = X[xoffset0 + c1];
		char x2 = X[xoffset0 + c2];
		char x3 = X[xoffset1 + c0];
		char x4 = X[xoffset1 + c1];
		char x5 = X[xoffset1 + c2];
		char8 xv = { x0, x1, x2, 0, x3, x4, x5 };

		const int yoffset = index2 << 2;//Y.stride = 4, [n, h, iw, 0-3]
		*(char8*)(Y + yoffset) = xv;
	}
}

#endif


//common
#ifndef IMG_EXTRACT_3CHANNELS_KERNEL1
#define IMG_EXTRACT_3CHANNELS_KERNEL1

//(5, 2): Size = 16, Time = 0.103 mesc, Speed = 151.699 GB/s
__global__ void img_extract_3channels_kernel1(
	const char* __restrict__ X, int IC,
	      char* __restrict__ Y, int c0, int c1, int c2,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x);

	for (; index < lengthv; index += step)
	{
		const int xoffset = index * IC;//index[n, ih, iw]
		char x0 = X[xoffset + c0];
		char x1 = X[xoffset + c1];
		char x2 = X[xoffset + c2];
		char4 xv = { x0, x1, x2, 0 };

		const int yoffset = index << 2;//Y.stride = 4, [n, h, iw, 0-3]
		*(char4*)(Y + yoffset) = xv;
	}
}

#endif


void __img_extract_3channels(cudaStream_t stream,
	const char* __restrict__ X, int IC,
	      char* __restrict__ Y, int c0, int c1, int c2,
	int lengthv)
{
	if (lengthv < 256) { img_extract_3channels_k1_small(stream, X, IC, Y, c0, c1, c2, lengthv); return; }
	if (!(lengthv & 15) && lengthv >= 8192) { img_extract_3channels_k4(stream, 5, 3, X, IC, Y, c0, c1, c2, lengthv); return; }
	if (!(lengthv &  7) && lengthv >= 4096) { img_extract_3channels_k2(stream, 5, 2, X, IC, Y, c0, c1, c2, lengthv); return; }
	return img_extract_3channels_k1(stream, 5, 2, X, IC, Y, c0, c1, c2, lengthv);
}

#endif