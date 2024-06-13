#pragma once

#ifndef PAD_4D_H
#define PAD_4D_H

//X[IN, IH, IW, IC] -> Y[ON, OH, OW, OC]
//IN -> ON = pn0 + IN + pn1
//IH -> OH = ph0 + IH + ph1
//IW -> OW = pw0 + IW + pw1
//IC -> OC = pc0 + IC + pc1
//<1> IN % 4 != 0 (ignore the padding on IN)
//<2> ON % 4 != 0 (ignore the padding on ON)
//<3> IC % 4 == 0
//<4> OC % 4 == 0
//lengthv = IN*IH*IW*IC, so: lengthv % 4 == 0
#ifndef PAD_4D_CALL
#define PAD_4D_CALL

#define pad4d_k4_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	pad4D_kernel4\
		<<< 1, lengthv, 0, stream >>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv) 
		
#define pad4d_k4(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	pad4D_kernel4\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv)

//pc0 % 4 == 0
#define pad4d_k4_pc04x_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	pad4D_kernel4_pc04x\
		<<< 1, lengthv, 0, stream >>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv) 

//pc0 % 4 == 0
#define pad4d_k4_pc04x(stream, LB, LT, X, IH, IW, IC, Y, OH, OW, OC, lengthv)\
	pad4D_kernel4_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,(IH*IW*IC),(IW*IC),IC, Y,OH,OW,OC, lengthv)

#endif


#ifndef PAD_4D_KERNEL4
#define PAD_4D_KERNEL4

//Y ->[pn0, ph0, pw0, pc0]
__global__ void pad4D_kernel4(
	const float* __restrict__ X, int IH_IW_IC, int IW_IC, int IC,
	      float* __restrict__ Y, int OH, int OW, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int xoffset = index4;//[in, ih, iw, ic0-ic3]
		int in = xoffset / IH_IW_IC, xr = xoffset - in * IH_IW_IC;
		int ih = xr / IW_IC; xr -= ih * IW_IC;
		int iw = xr / IC, ic = xr - iw * IC;

		const int yoffset = ((in*OH + ih)*OW + iw)*OC + ic;
		float4 xv = *(float4*)(X + xoffset);
		Y[yoffset    ] = xv.x;//[on, oh, ow, oc0]
		Y[yoffset + 1] = xv.y;//[on, oh, ow, oc1]
		Y[yoffset + 2] = xv.z;//[on, oh, ow, oc2]
		Y[yoffset + 3] = xv.w;//[on, oh, ow, oc3]
	}
}

#endif


#ifndef PAD_4D_KERNEL4_PC0_4X
#define PAD_4D_KERNEL4_PC0_4X

//Y ->[pn0, ph0, pw0, pc0]
//pc0 % 4 == 0
__global__ void pad4D_kernel4_pc04x(
	const float* __restrict__ X, int IH_IW_IC, int IW_IC, int IC,
	      float* __restrict__ Y, int OH, int OW, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int xoffset = index4;//[in, ih, iw, ic0-ic3]
		int in = xoffset / IH_IW_IC, xr = xoffset - in * IH_IW_IC;
		int ih = xr / IW_IC; xr -= ih * IW_IC;
		int iw = xr / IC, ic = xr - iw * IC;

		const int yoffset = ((in*OH + ih)*OW + iw)*OC + ic;//[n, oh, ow, oc0-oc3]
		*(float4*)(Y + yoffset) = *(float4*)(X + xoffset);
	}
}

#endif


void __pad4d(cudaStream_t stream,
	const float* X, int IN, int IH, int IW, int IC,
		  float* Y, int ON, int OH, int OW, int OC,
	int pn0, int ph0, int pw0, int pc0)
{
	Y += ((pn0*OH + ph0)*OW + pw0)*OC + pc0;//Y[pn0, ph0, pw0, pc0]
	int lengthv = IN * IH * IW * IC;//X.lengthv <= Y.lengthv

	if (lengthv < 256) {
		if(!(pc0 & 3)) pad4d_k4_pc04x_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
		else pad4d_k4_small(stream, X, IH, IW, IC, Y, OH, OW, OC, lengthv); 
		return; 
	}
	if(!(pc0 & 3)) pad4d_k4_pc04x(stream, 5, 2, X, IH, IW, IC, Y, OH, OW, OC, lengthv);
	else pad4d_k4(stream, 5, 2, X, IH, IW, IC, Y, OH, OW, OC,lengthv);
		
}

#endif