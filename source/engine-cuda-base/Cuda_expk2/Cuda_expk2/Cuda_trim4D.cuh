#pragma once

#ifndef TRIM_4D_H
#define TRIM_4D_H

//X[IN, IH, IW, IC] -> Y[ON, OH, OW, OC]
//IN = pn0 + ON + pn1 -> ON
//IH = ph0 + OH + ph1 -> OH
//IW = pw0 + OW + pw1 -> OW
//IC = pc0 + OC + pc1 -> OC
//<1> IN % 4 != 0 (ignore the mem-alginment on IN)
//<2> ON % 4 != 0 (ignore the mem-alginment on ON)
//<3> IC % 4 == 0
//<4> OC % 4 == 0
//lengthv = ON*OH*OW*OC, so: lengthv % 4 == 0
#ifndef TRIM_4D_CALL
#define TRIM_4D_CALL

#define trim4d_k4_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	trim4D_kernel4\
		<<< 1, lengthv, 0, stream >>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv) 

#define trim4d_k4(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	trim4D_kernel4\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv)

//pc0 % 4 == 0
#define trim4d_k4_pc04x_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	trim4D_kernel4_pc04x\
		<<< 1, lengthv, 0, stream >>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv) 

//pc0 % 4 == 0
#define trim4d_k4_pc04x(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	trim4D_kernel4_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,IH,IW,IC, Y,(OH*OW*OC),(OW*OC),OC, lengthv)

#endif


#ifndef TRIM_4D_KERNEL4
#define TRIM_4D_KERNEL4

//X ->[pn0, ph0, pw0, pc0]
__global__ void trim4D_kernel4(
	const float* __restrict__ X, int IH, int IW, int IC,
	      float* __restrict__ Y, int OH_OW_OC, int OW_OC, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int yoffset = index4;//[on, oh, ow, oc0-oc3]
		int on = yoffset / OH_OW_OC, yr = yoffset - on * OH_OW_OC;
		int oh = yr / OW_OC; yr -= oh * OW_OC;
		int ow = yr / OC, oc = yr - ow * OC;

		float4 xv; int xoffset = ((on*IH + oh)*IW + ow)*IC + oc;
		xv.x = X[xoffset    ];//[n, ih, iw, ic0]
		xv.y = X[xoffset + 1];//[n, ih, iw, ic1]
		xv.z = X[xoffset + 2];//[n, ih, iw, ic2]
		xv.w = X[xoffset + 3];//[n, ih, iw, ic3]
		*(float4*)(Y + yoffset) = xv;
	}
}

#endif


#ifndef TRIM_4D_KERNEL4_PC0_4X
#define TRIM_4D_KERNEL4_PC0_4X

//X ->[pn0, ph0, pw0, pc0]
//pc0 % 4 == 0
__global__ void trim4D_kernel4_pc04x(
	const float* __restrict__ X, int IH, int IW, int IC,
	      float* __restrict__ Y, int OH_OW_OC, int OW_OC, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int yoffset = index4;//[on, oh, ow, oc0-oc3]
		int on = yoffset / OH_OW_OC, yr = yoffset - on * OH_OW_OC;
		int oh = yr / OW_OC; yr -= oh * OW_OC;
		int ow = yr / OC, oc = yr - ow * OC;

		const int xoffset = ((on*IH + oh)*IW + ow)*IC + oc;//[in, ih, iw, ic0-ic3]
		*(float4*)(Y + yoffset) = *(float4*)(X + xoffset);
	}
}

#endif


void __trim4d(cudaStream_t stream,
	const float* X, int IN, int IH, int IW, int IC,
	      float* Y, int ON, int OH, int OW, int OC,
	int pn0, int ph0, int pw0, int pc0)
{
	X += ((pn0*IH + ph0)*IW + pw0)*IC + pc0;//X[pn0, ph0, pw0, pc0]
	int lengthv = ON * OH * OW * OC;//Y.lengthv <= X.lengthv

	if (lengthv < 256) { 
		if(!(pc0 & 3)) trim4d_k4_pc04x_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		else trim4d_k4_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv); 
		return; 
	}
	if(!(pc0 & 3)) trim4d_k4_pc04x(stream, 5, 2, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
	else trim4d_k4(stream, 5, 2, X, IH, IW, IC, Y, OH, OW, OC,lengthv);
}

#endif
