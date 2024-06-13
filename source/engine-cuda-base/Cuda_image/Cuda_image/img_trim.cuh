#pragma once

#ifndef IMG_TRIM_H
#define IMG_TRIM_H

//X[N, IH, IW, IC] -> Y[N, OH, OW, OC]
//IH = ph0 + OH + ph1 -> OH
//IW = pw0 + OW + pw1 -> OW
//IC = pc0 + OC + pc1 -> OC
//{ IC, OC } % 4 == 0
//lengthv = ON*OH*OW*OC, so: lengthv % 4 == 0

#ifndef IMG_TRIM_CALL
#define IMG_TRIM_CALL

//======[Common]==========================================================
#define img_trim_k4_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_trim_kernel4\
		<<< 1, lengthv, 0, stream >>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv) 

#define img_trim_k4(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_trim_kernel4\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv)

//OC % 8 == 0
#define img_trim_k8(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_trim_kernel8\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv)

//======[pc0 % 4 == 0]====================================================
#define img_trim_k4_pc04x_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_trim_kernel4_pc04x\
		<<< 1, lengthv, 0, stream >>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv) 

#define img_trim_k4_pc04x(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_trim_kernel4_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv)

#define img_trim_k8_pc04x(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_trim_kernel8_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv)

#define img_trim_k16_pc04x(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_trim_kernel16_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv)

#endif


//======[Common]==========================================================
#ifndef IMG_TRIM_KERNEL8
#define IMG_TRIM_KERNEL8

//OC % 8 == 0
//X ->[pn0, ph0, pw0, pc0]
//(5, 3): Size = 8, Time = 0.031 mesc, Speed = 252.016 GB/s
__global__ void img_trim_kernel8(
	const char* __restrict__ X, int IH, int IW, int IC,
	      char* __restrict__ Y, int OH_OW_OC, int OW_OC, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step8 = step << 3;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		const int yoffset = index8;//[n, oh, ow, oc0-oc3]
		int n = yoffset / OH_OW_OC, yr = yoffset - n * OH_OW_OC;
		int oh = yr / OW_OC; yr -= oh * OW_OC;
		int ow = yr / OC, oc = yr - ow * OC;

		char8 xv; int xoffset = ((n*IH + oh)*IW + ow)*IC + oc;
		xv.x0 = X[xoffset    ];//[n, ih, iw, ic0]
		xv.y0 = X[xoffset + 1];//[n, ih, iw, ic1]
		xv.z0 = X[xoffset + 2];//[n, ih, iw, ic2]
		xv.w0 = X[xoffset + 3];//[n, ih, iw, ic3]

		xv.x1 = X[xoffset + 4];//[n, ih, iw, ic4]
		xv.y1 = X[xoffset + 5];//[n, ih, iw, ic5]
		xv.z1 = X[xoffset + 6];//[n, ih, iw, ic6]
		xv.w1 = X[xoffset + 7];//[n, ih, iw, ic7]
		*(char8*)(Y + yoffset) = xv;
	}
}

#endif


#ifndef IMG_TRIM_KERNEL4
#define IMG_TRIM_KERNEL4

//X ->[pn0, ph0, pw0, pc0]
//(5, 3): Size = 8, Time = 0.034 mesc, Speed = 229.779 GB/s
//(5, 2): Size = 8, Time = 0.054 mesc, Speed = 144.676 GB/s
__global__ void img_trim_kernel4(
	const char* __restrict__ X, int IH, int IW, int IC,
	      char* __restrict__ Y, int OH_OW_OC, int OW_OC, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int yoffset = index4;//[n, oh, ow, oc0-oc3]
		int n = yoffset / OH_OW_OC, yr = yoffset - n * OH_OW_OC;
		int oh = yr / OW_OC; yr -= oh * OW_OC;
		int ow = yr / OC, oc = yr - ow * OC;

		char4 xv; int xoffset = ((n*IH + oh)*IW + ow)*IC + oc;
		xv.x = X[xoffset    ];//[n, ih, iw, ic0]
		xv.y = X[xoffset + 1];//[n, ih, iw, ic1]
		xv.z = X[xoffset + 2];//[n, ih, iw, ic2]
		xv.w = X[xoffset + 3];//[n, ih, iw, ic3]
		*(char4*)(Y + yoffset) = xv;
	}
}

#endif


//======[pc0 % 4 == 0]====================================================
#ifndef IMG_TRIM_KERNEL8_PC0_16X
#define IMG_TRIM_KERNEL8_PC0_16X

//OC % 16 == 0
//X ->[pn0, ph0, pw0, pc0]
//(5, 4): Size = 8, Time = 0.026 mesc, Speed = 300.481 GB/s
__global__ void img_trim_kernel16_pc04x(
	const char* __restrict__ X, int IH, int IW, int IC,
	char* __restrict__ Y, int OH_OW_OC, int OW_OC, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step16 = step << 4;

	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		const int yoffset = index16;//[n, oh, ow, oc0-oc3]
		int n = yoffset / OH_OW_OC, yr = yoffset - n * OH_OW_OC;
		int oh = yr / OW_OC; yr -= oh * OW_OC;
		int ow = yr / OC, oc = yr - ow * OC;

		const int xoffset = ((n*IH + oh)*IW + ow)*IC + oc;
		char4 xv0 = *(char4*)(X + xoffset);//[n, ih, iw, ic0-ic3]
		char4 xv1 = *(char4*)(X + xoffset + 4);//[n, ih, iw, ic4-ic7]
		char4 xv2 = *(char4*)(X + xoffset + 8);//[n, ih, iw, ic8-ic11]
		char4 xv3 = *(char4*)(X + xoffset + 12);//[n, ih, iw, ic12-ic15]

		*(char16*)(Y + yoffset) = {
			xv0.x, xv0.y, xv0.z, xv0.w,
			xv1.x, xv1.y, xv1.z, xv1.w,
			xv2.x, xv2.y, xv2.z, xv2.w,
			xv3.x, xv3.y, xv3.z, xv3.w,
		};
	}
}

#endif


#ifndef IMG_TRIM_KERNEL8_PC0_8X
#define IMG_TRIM_KERNEL8_PC0_8X

//OC % 8 == 0
//X ->[pn0, ph0, pw0, pc0]
//(5, 3): Size = 8, Time = 0.031 mesc, Speed = 252.016 GB/s
__global__ void img_trim_kernel8_pc04x(
	const char* __restrict__ X, int IH, int IW, int IC,
	char* __restrict__ Y, int OH_OW_OC, int OW_OC, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step8 = step << 3;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		const int yoffset = index8;//[n, oh, ow, oc0-oc3]
		int n = yoffset / OH_OW_OC, yr = yoffset - n * OH_OW_OC;
		int oh = yr / OW_OC; yr -= oh * OW_OC;
		int ow = yr / OC, oc = yr - ow * OC;

		const int xoffset = ((n*IH + oh)*IW + ow)*IC + oc;
		char4 xv0 = *(char4*)(X + xoffset);//[n, ih, iw, ic0-ic3]
		char4 xv1 = *(char4*)(X + xoffset + 4);//[n, ih, iw, ic4-ic7]
		*(char8*)(Y + yoffset) = {
			xv0.x, xv0.y, xv0.z, xv0.w,
			xv1.x, xv1.y, xv1.z, xv1.w 
		};
	}
}

#endif


#ifndef IMG_TRIM_KERNEL4_PC0_4X
#define IMG_TRIM_KERNEL4_PC0_4X

//X ->[pn0, ph0, pw0, pc0]
//(5, 3): Size = 8, Time = 0.034 mesc, Speed = 229.779 GB/s
//(5, 2): Size = 8, Time = 0.054 mesc, Speed = 144.676 GB/s
__global__ void img_trim_kernel4_pc04x(
	const char* __restrict__ X, int IH, int IW, int IC,
	char* __restrict__ Y, int OH_OW_OC, int OW_OC, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int yoffset = index4;//[n, oh, ow, oc0-oc3]
		int n = yoffset / OH_OW_OC, yr = yoffset - n * OH_OW_OC;
		int oh = yr / OW_OC; yr -= oh * OW_OC;
		int ow = yr / OC, oc = yr - ow * OC;

		const int xoffset = ((n*IH + oh)*IW + ow)*IC + oc;//[n, ih, iw, ic0-ic4]
		*(char4*)(Y + yoffset) = *(char4*)(X + xoffset);
	}
}

#endif


void __img_trim(cudaStream_t stream,
	const char* X, int IH, int IW, int IC,
	      char* Y, int OH, int OW, int OC,
	int N, int ph0, int pw0, int pc0)
{
	X += (ph0*IW + pw0)*IC + pc0;//X[pn0, ph0, pw0, pc0]
	int lengthv = N * OH * OW * OC;//Y.lengthv <= X.lengthv

	if (lengthv < 256) { 
		if (!(pc0 & 3)) img_trim_k4_pc04x_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		else img_trim_k4_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv); 
		return; 
	}
	if (!(OC & 15) && lengthv >= 16384) {
		if (!(pc0 & 3)) img_trim_k16_pc04x(stream, 5, 4, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		else img_trim_k8(stream, 5, 3, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		return;
	}
	if (!(OC &  7) && lengthv >= 8192) { 
		if (!(pc0 & 3))img_trim_k8_pc04x(stream, 5, 3, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		else img_trim_k8(stream, 5, 3, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		return; 
	}
	if (lengthv >= 8192) { 
		if (!(pc0 & 3)) img_trim_k4_pc04x(stream, 5, 3, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		else img_trim_k4(stream, 5, 3, X, IH, IW, IC, Y, OH, OW, OC, lengthv); 
		return; 
	}
	if (!(pc0 & 3)) img_trim_k4_pc04x(stream, 5, 2, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
	else img_trim_k4(stream, 5, 2, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
}

#endif