#pragma once

#ifndef CUDA_PAD_3D_H
#define CUDA_PAD_3D_H

//X[IN, IW, IC] -> Y[ON, OW, IC]
//IN -> ON = pn0 + IN + pn1
//IW -> OW = pw0 + IW + pw1
//IC -> OC = pc0 + IC + pc1
//<1> IN % 4 != 0 (ignore the mem-alginment on IN)
//<2> ON % 4 != 0 (ignore the mem-alginment on ON)
//<3> IC % 4 == 0
//<4> OC % 4 == 0
//lengthv = IN*IW*IC, so: lengthv % 4 == 0
#ifndef PAD_3D_CALL
#define PAD_3D_CALL

#define pad3d_k4_small(stream, X, IW, IC, Y, OW, OC, lengthv)\
	pad3D_kernel4\
		<<< 1, lengthv, 0, stream >>>\
			(X,(IW*IC),IC, Y,OW,OC, lengthv) 

#define pad3d_k4(stream, LB, LT, X, IW, IC, Y, OW, OC, lengthv)\
	pad3D_kernel4\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,(IW*IC),IC, Y,OW,OC, lengthv)

//pc0 % 4 == 0
#define pad3d_k4_pc04x_small(stream, X, IW, IC, Y, OW, OC, lengthv)\
	pad3D_kernel4_pc04x\
		<<< 1, lengthv, 0, stream >>>\
			(X,(IW*IC),IC, Y,OW,OC, lengthv) 

//pc0 % 4 == 0
#define pad3d_k4_pc04x(stream, LB, LT, X, IW, IC, Y, OW, OC, lengthv)\
	pad3D_kernel4_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,(IW*IC),IC, Y,OW,OC, lengthv)

#endif


#ifndef PAD_3D_KERNEL4
#define PAD_3D_KERNEL4

//Y -> [pn0, pw0, pc0]
__global__ void pad3D_kernel4(
	const float* __restrict__ X, int IW_IC, int IC,
	      float* __restrict__ Y, int OW, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int xoffset = index4;//[in, iw, ic0-ic3]
		int in = xoffset / IW_IC, xr = xoffset - in * IW_IC;
		int iw = xr / IC, ic = xr - iw * IC;

		const int yoffset = (in*OW + iw)*OC + ic;
		float4 xv = *(float4*)(X + xoffset);
		Y[yoffset    ] = xv.x;//[on, ow, oc0]
		Y[yoffset + 1] = xv.y;//[on, ow, oc1]
		Y[yoffset + 2] = xv.z;//[on, ow, oc2]
		Y[yoffset + 3] = xv.w;//[on, ow, oc3]
	}
}

#endif


#ifndef PAD_3D_KERNEL4_PC0_4X
#define PAD_3D_KERNEL4_PC0_4X

//Y -> [pn0, pw0, pc0]
//pc0 % 4 == 0
__global__ void pad3D_kernel4_pc04x(
	const float* __restrict__ X, int IW_IC, int IC,
	      float* __restrict__ Y, int OW, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int xoffset = index4;//[in, iw, ic0-ic3]
		int in = xoffset / IW_IC, xr = xoffset - in * IW_IC;
		int iw = xr / IC, ic = xr - iw * IC;

		const int yoffset = (in*OW + iw)*OC + ic;//[on, ow, oc0-oc3]
		*(float4*)(Y + yoffset) = *(float4*)(X + xoffset);
	}
}

#endif


void __pad3d(cudaStream_t stream,
	const float* X, int IN, int IW, int IC,
	float* Y, int ON, int OW, int OC,
	int pn0, int pw0, int pc0)
{
	Y += (pn0*OW + pw0)*OC + pc0;//Y[pn0, pw0, pc0]
	int lengthv = IN * IW * OC;//X.lengthv <= Y.lengthv

	if (lengthv < 256) { 
		if (!(pc0 & 3)) pad3d_k4_pc04x_small(stream, X, IW, IC, Y, OW, OC, lengthv);
		else pad3d_k4_small(stream, X, IW, IC, Y, OW, OC, lengthv); 
		return;
	}
	if(!(pc0 & 3)) pad3d_k4_pc04x(stream, 5, 2, X, IW, IC, Y, OW, OC, lengthv);
	else pad3d_k4(stream, 5, 2, X, IW, IC, Y, OW, OC, lengthv);
}

#endif