#pragma once

#ifndef IMG_PAD_H
#define IMG_PAD_H

//X[N, IH, IW, IC] -> Y[N, OH, OW, OC]
//IH -> OH = ph0 + IH + ph1
//IW -> OW = pw0 + IW + pw1
//IC -> OC = pc0 + IC + pc1
//{ IC, OC } % 4 == 0
//lengthv = N*IH*IW*IC so: lengthv%4 == 0

#ifndef IMG_PAD_CALL
#define IMG_PAD_CALL

//======[Common]==========================================================
#define img_pad_k4_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_pad_kernel4\
		<<< 1, lengthv, 0, stream >>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv)

#define img_pad_k4(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_pad_kernel4\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream >>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv)

#define img_pad_k8(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_pad_kernel8\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream >>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv)

//======[pc0 % 4 == 0]====================================================
#define img_pad_k4_pc04x_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_pad_kernel4_pc04x\
		<<< 1, lengthv, 0, stream >>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv)

#define img_pad_k4_pc04x(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_pad_kernel4_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream >>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv)

#define img_pad_k8_pc04x(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_pad_kernel8_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream >>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv)

#define img_pad_k16_pc04x(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	img_pad_kernel16_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream >>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv)

#endif


//======[Common]==========================================================
#ifndef IMG_PAD_KERNEL8
#define IMG_PAD_KERNEL8

//IC % 8 == 0
//Y ->[pn0, ph0, pw0, pc0]
//(5, 3): Size = 8, Time = 0.029 mesc, Speed = 269.397GB/s
__global__ void img_pad_kernel8(
	const char* __restrict__ X, int IH_IW_IC, int IW_IC, int IC,
	      char* __restrict__ Y, int OH, int OW, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step8 = step << 3;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		const int xoffset = index8;//[in, ih, iw, ic0-ic7]
		int n = xoffset / IH_IW_IC, xr = xoffset - n * IH_IW_IC;
		int ih = xr / IW_IC; xr -= ih * IW_IC;
		int iw = xr / IC, ic = xr - iw * IC;

		const int yoffset = ((n*OH + ih)*OW + iw)*OC + ic;
		char8 xv = *(char8*)(X + xoffset);
		Y[yoffset    ] = xv.x0;//[on, oh, ow, oc0]
		Y[yoffset + 1] = xv.y0;//[on, oh, ow, oc1]
		Y[yoffset + 2] = xv.z0;//[on, oh, ow, oc2]
		Y[yoffset + 3] = xv.w0;//[on, oh, ow, oc3]

		Y[yoffset + 4] = xv.x1;//[on, oh, ow, oc4]
		Y[yoffset + 5] = xv.y1;//[on, oh, ow, oc5]
		Y[yoffset + 6] = xv.z1;//[on, oh, ow, oc6]
		Y[yoffset + 7] = xv.w1;//[on, oh, ow, oc7]
	}
}

#endif


#ifndef IMG_PAD_KERNEL4
#define IMG_PAD_KERNEL4

//Y ->[pn0, ph0, pw0, pc0]
//(5, 3): Size = 8, Time = 0.042 mesc, Speed = 186.012 GB/s
//(5, 2): Size = 8, Time = 0.048 mesc, Speed = 162.76  GB/s
__global__ void img_pad_kernel4(
	const char* __restrict__ X, int IH_IW_IC, int IW_IC, int IC,
	      char* __restrict__ Y, int OH, int OW, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int xoffset = index4;//[in, ih, iw, ic0-ic3]
		int n = xoffset / IH_IW_IC, xr = xoffset - n * IH_IW_IC;
		int ih = xr / IW_IC; xr -= ih * IW_IC;
		int iw = xr / IC, ic = xr - iw * IC;

		const int yoffset = ((n*OH + ih)*OW + iw)*OC + ic;
		char4 xv = *(char4*)(X + xoffset);
		Y[yoffset    ] = xv.x;//[on, oh, ow, oc0]
		Y[yoffset + 1] = xv.y;//[on, oh, ow, oc1]
		Y[yoffset + 2] = xv.z;//[on, oh, ow, oc2]
		Y[yoffset + 3] = xv.w;//[on, oh, ow, oc3]
	}
}

#endif


//======[pc0 % 4 == 0]====================================================
#ifndef IMG_PAD_KERNEL16_PC0_4X
#define IMG_PAD_KERNEL16_PC0_4X

//IC % 16 == 0
//Y ->[pn0, ph0, pw0, pc0]
//(5, 4): Size = 8, Time = 0.027 mesc, Speed = 289.352GB/s
__global__ void img_pad_kernel16_pc04x(
	const char* __restrict__ X, int IH_IW_IC, int IW_IC, int IC,
	char* __restrict__ Y, int OH, int OW, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step16 = step << 4;

	for (int index16 = index << 4; index16 < lengthv; index16 += step16)
	{
		const int xoffset = index16;//[in, ih, iw, ic0-ic15]
		int n = xoffset / IH_IW_IC, xr = xoffset - n * IH_IW_IC;
		int ih = xr / IW_IC; xr -= ih * IW_IC;
		int iw = xr / IC, ic = xr - iw * IC;

		const int yoffset = ((n*OH + ih)*OW + iw)*OC + ic;
		char16 xv = *(char16*)(X + xoffset);
		*(char4*)(Y + yoffset     ) = char4{ xv.x0, xv.y0, xv.z0, xv.w0 };//[on, oh, ow, oc0-oc3]
		*(char4*)(Y + yoffset +  4) = char4{ xv.x1, xv.y1, xv.z1, xv.w1 };//[on, oh, ow, oc4-oc7]
		*(char4*)(Y + yoffset +  8) = char4{ xv.x2, xv.y2, xv.z2, xv.w2 };//[on, oh, ow, oc8-oc11]
		*(char4*)(Y + yoffset + 12) = char4{ xv.x3, xv.y3, xv.z3, xv.w3 };//[on, oh, ow, oc12-oc15]
	}
}

#endif


#ifndef IMG_PAD_KERNEL8_PC0_4X
#define IMG_PAD_KERNEL8_PC0_4X

//IC % 8 == 0
//Y ->[pn0, ph0, pw0, pc0]
//(5, 3): Size = 8, Time = 0.031 mesc, Speed = 252.016GB/s
__global__ void img_pad_kernel8_pc04x(
	const char* __restrict__ X, int IH_IW_IC, int IW_IC, int IC,
	char* __restrict__ Y, int OH, int OW, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step8 = step << 3;

	for (int index8 = index << 3; index8 < lengthv; index8 += step8)
	{
		const int xoffset = index8;//[in, ih, iw, ic0-ic7]
		int n = xoffset / IH_IW_IC, xr = xoffset - n * IH_IW_IC;
		int ih = xr / IW_IC; xr -= ih * IW_IC;
		int iw = xr / IC, ic = xr - iw * IC;

		const int yoffset = ((n*OH + ih)*OW + iw)*OC + ic;
		char8 xv = *(char8*)(X + xoffset);
		*(char4*)(Y + yoffset    ) = char4{ xv.x0, xv.y0, xv.z0, xv.w0 };//[on, oh, ow, oc0-oc3]
		*(char4*)(Y + yoffset + 4) = char4{ xv.x1, xv.y1, xv.z1, xv.w1 };//[on, oh, ow, oc4-oc7]
	}
}

#endif


#ifndef IMG_PAD_KERNEL4_PC0_4X
#define IMG_PAD_KERNEL4_PC0_4X

//Y ->[pn0, ph0, pw0, pc0]
//(5, 3): Size = 8, Time = 0.037 mesc, Speed = 211.149GB/s
//(5, 2): Size = 8, Time = 0.048 mesc, Speed = 162.76  GB/s
__global__ void img_pad_kernel4_pc04x(
	const char* __restrict__ X, int IH_IW_IC, int IW_IC, int IC,
		  char* __restrict__ Y, int OH, int OW, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int xoffset = index4;//[in, ih, iw, ic0-ic3]
		int n = xoffset / IH_IW_IC, xr = xoffset - n * IH_IW_IC;
		int ih = xr / IW_IC; xr -= ih * IW_IC;
		int iw = xr / IC, ic = xr - iw * IC;

		const int yoffset = ((n*OH + ih)*OW + iw)*OC + ic;
		*(char4*)(Y + yoffset) = *(char4*)(X + xoffset);//[on, oh, ow, oc0-oc3]
	}
}

#endif

  
void __img_pad(cudaStream_t stream,
	const char* X, int IH, int IW, int IC,
	      char* Y, int OH, int OW, int OC,
	int N, int ph0, int pw0, int pc0)
{
	Y += (ph0*OW + pw0)*OC + pc0;//Y[0, ph0, pw0, pc0]
	int lengthv = N * IH * IW * IC;//X.lengthv <= Y.lengthv

	if (lengthv < 256) { 
		if (!(pc0 & 3)) img_pad_k4_pc04x_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		else img_pad_k4_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		return; 
	}
	if (!(IC & 15) && lengthv >=  16384) {
		if (!(pc0 & 3)) img_pad_k16_pc04x(stream, 5, 4, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		else img_pad_k8(stream, 5, 3, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		return;
	}
	if (!(IC &  7) && lengthv >=  8192) { 
		if (!(pc0 & 3)) img_pad_k8_pc04x(stream, 5, 3, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		else img_pad_k8(stream, 5, 3, X, IH, IW, IC, Y, OH, OW, OC, lengthv); 
		return; 
	}
	if (lengthv >= 8192) { 
		if (!(pc0 & 3)) img_pad_k4_pc04x(stream, 5, 3, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		else img_pad_k4(stream, 5, 3, X, IH, IW, IC, Y, OH, OW, OC, lengthv); 
		return;  
	}
	if (!(pc0 & 3)) img_pad_k4_pc04x(stream, 5, 2, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
	else img_pad_k4(stream, 5, 2, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
}

#endif