#pragma once

#ifndef IMG_RESIZE_H
#define IMG_RESIZE_H

//C % 4 == 0
//lengthv = N*OH*OW*C, so lengthv % 4 == 0
#ifndef IMG_RESIZE_CALL
#define IMG_RESIZE_CALL

//OC % 16 == 0
#define img_resize_k16(stream, LB, LT, X, IH, IW, Y, OH, OW, C, lengthv)\
	img_resize_kernel_16\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,IH,IW, Y,OH,OW,C, lengthv)

//OC % 8 == 0
#define img_resize_k8(stream, LB, LT, X, IH, IW, Y, OH, OW, C, lengthv)\
	img_resize_kernel_8\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,IH,IW, Y,OH,OW,C, lengthv)

#define img_resize_k4(stream, LB, LT, X, IH, IW, Y, OH, OW, C, lengthv)\
	img_resize_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X,IH,IW, Y,OH,OW,C, lengthv)

#define img_resize_k4_small(stream, X, IH, IW, Y, OH, OW, C, lengthv)\
	img_resize_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X,IH,IW, Y,OH,OW,C, lengthv)

#endif


#ifndef IMG_RESIZE_KERNEL16
#define IMG_RESIZE_KERNEL16

//(5, 4): Size = 20, Time = 0.066 mesc, Speed = 295.928GB/s
__global__ void img_resize_kernel_16(
	const char* __restrict__ X, int IH, int IW,
	char* __restrict__ Y, int OH, int OW, int C,
	int lengthv)
{
	int step = gridDim.x*blockDim.x, step16 = step << 4;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	const int OW_C = OW * C, OH_OW_C = OH * OW_C;
	const float fOH = OH, fOW = OW;
	const int IH_m1 = IH - 1, IW_m1 = IW - 1;
	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		const int yindex = index16;//[n, oh, ow, c]
		int n = yindex / OH_OW_C, yr = yindex - n * OH_OW_C;
		int oh = yr / OW_C; yr -= oh * OW_C;
		int ow = yr / C, c = yr - ow * C;

		int ih = lroundf(oh * IH / fOH);//find the nearset pixel
		int iw = lroundf(ow * IW / fOW);
		ih = IF_int((ih < IH_m1), ih, IH_m1);//ih <= IH_m1
		iw = IF_int((iw < IW_m1), iw, IW_m1);//iw <= IW_m1

		const int xindex = ((n*IH + ih)*IW + iw)*C + c;//[n, ih, iw, c]
		*(uchar16*)(Y + yindex) = *(uchar16*)(X + xindex);
	}
}

#endif


#ifndef IMG_RESIZE_KERNEL8
#define IMG_RESIZE_KERNEL8

//(5, 3): Size = 20, Time = 0.116 mesc, Speed = 168.373GB/s
__global__ void img_resize_kernel_8(
	const char* __restrict__ X, int IH, int IW,
	char* __restrict__ Y, int OH, int OW, int C,
	int lengthv)
{
	int step = gridDim.x*blockDim.x, step8 = step << 3;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	const int OW_C = OW * C, OH_OW_C = OH * OW_C;
	const float fOH = OH, fOW = OW;
	const int IH_m1 = IH - 1, IW_m1 = IW - 1;
	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		const int yindex = index8;//[n, oh, ow, c]
		int n = yindex / OH_OW_C, yr = yindex - n * OH_OW_C;
		int oh = yr / OW_C; yr -= oh * OW_C;
		int ow = yr / C, c = yr - ow * C;

		int ih = lroundf(oh * IH / fOH);//find the nearset pixel
		int iw = lroundf(ow * IW / fOW);
		ih = IF_int((ih < IH_m1), ih, IH_m1);//ih <= IH_m1
		iw = IF_int((iw < IW_m1), iw, IW_m1);//iw <= IW_m1

		const int xindex = ((n*IH + ih)*IW + iw)*C + c;//[n, ih, iw, c]
		*(uchar8*)(Y + yindex) = *(uchar8*)(X + xindex);
	}
}

#endif


#ifndef IMG_RESIZE_KERNEL4
#define IMG_RESIZE_KERNEL4

//(5, 3): Size = 20, Time = 0.171 mesc, Speed = 114.218GB/s
//(5, 2): Size = 20, Time = 0.212 mesc, Speed = 92.1285GB/s
__global__ void img_resize_kernel_4(
	const char* __restrict__ X, int IH, int IW,
	      char* __restrict__ Y, int OH, int OW, int C,
	int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	const int OW_C = OW * C, OH_OW_C = OH * OW_C;
	const float fOH = OH, fOW = OW; 
	const int IH_m1 = IH - 1, IW_m1 = IW - 1;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int yindex = index4;//[n, oh, ow, c]
		int n = yindex / OH_OW_C, yr = yindex - n * OH_OW_C;
		int oh = yr / OW_C; yr -= oh * OW_C;
		int ow = yr / C, c = yr - ow * C;

		int ih = lroundf(oh * IH / fOH);//find the nearset pixel
		int iw = lroundf(ow * IW / fOW);
		ih = IF_int((ih < IH_m1), ih, IH_m1);//ih <= IH_m1
		iw = IF_int((iw < IW_m1), iw, IW_m1);//iw <= IW_m1

		const int xindex = ((n*IH + ih)*IW + iw)*C + c;//[n, ih, iw, c]
		*(uchar4*)(Y + yindex) = *(uchar4*)(X + xindex);
	}
}

#endif


void __img_resize(cudaStream_t stream,
	const char* __restrict__ X, int IH, int IW,
	      char* __restrict__ Y, int OH, int OW,
	int N, int C)
{
	int lengthv = N * OH*OW*C;
	if (lengthv < 256) { img_resize_k4_small(stream, X, IH, IW, Y, OH, OW, C, lengthv); return; }
	if (!(C & 15) && lengthv >= 16384) { img_resize_k16(stream, 5, 4, X, IH, IW, Y, OH, OW, C, lengthv); return; }
	if (!(C &  7) && lengthv >=  8192) { img_resize_k8(stream, 5, 3, X, IH, IW, Y, OH, OW, C, lengthv); return; }
	if (lengthv >= 8192) { img_resize_k4(stream, 5, 3, X, IH, IW, Y, OH, OW, C, lengthv); return; }
	img_resize_k4(stream, 5, 2, X, IH, IW, Y, OH, OW, C, lengthv);
}

#endif
