#pragma once

#ifndef TRIM_3D_H
#define TRIM_3D_H

//X[IN, IW, IC] -> Y[ON, OW, OC]
//IN = pn0 + ON + pn1 -> ON
//IW = pw0 + OW + pw1 -> OW
//IC = pc0 + OC + pc1 -> OC
//<1> IN % 4 != 0 (ignore the mem-alginment on IN)
//<2> ON % 4 != 0 (ignore the mem-alginment on ON)
//<3> IC % 4 == 0
//<4> OC % 4 == 0
//lengthv = ON*OW*OC, so: lengthv % 4 == 0
#ifndef TRIM_3D_CALL
#define TRIM_3D_CALL

#define trim3d_k4_small(stream, X, IW, IC, Y, OW, OC, lengthv)\
	trim3D_kernel4\
		<<< 1, lengthv, 0, stream >>>\
			(X,IW,IC, Y,(OW*OC),OC, lengthv) 

#define trim3d_k4(stream, LB, LT, X, IW, IC, Y, OW, OC, lengthv)\
	trim3D_kernel4\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,IW,IC, Y,(OW*OC),OC, lengthv) 

//pc0 % 4 == 0
#define trim3d_k4_pc04x_small(stream, X, IW, IC, Y, OW, OC, lengthv)\
	trim3D_kernel4_pc04x\
		<<< 1, lengthv, 0, stream >>>\
			(X,IW,IC, Y,(OW*OC),OC, lengthv) 

//pc0 % 4 == 0
#define trim3d_k4_pc04x(stream, LB, LT, X, IW, IC, Y, OW, OC, lengthv)\
	trim3D_kernel4_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X,IW,IC, Y,(OW*OC),OC, lengthv) 

#endif


#ifndef TRIM_3D_KERNEL4
#define TRIM_3D_KERNEL4

//X -> [pn0, pw0, pc0]
__global__ void trim3D_kernel4(
	const float* __restrict__ X, int IW, int IC,
	      float* __restrict__ Y, int OW_OC, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int yoffset = index4;//[on, ow, oc0-oc3]
		int on = yoffset / OW_OC, yr = yoffset - on * OW_OC;
		int ow = yr / OC, oc = yr - ow * OC;

		float4 xv; int xoffset = (on*IW + ow)*IC + oc;
		xv.x = X[xoffset    ];//[in, iw, ic0]
		xv.y = X[xoffset + 1];//[in, iw, ic1]
		xv.z = X[xoffset + 2];//[in, iw, ic2]
		xv.w = X[xoffset + 3];//[in, iw, ic3]
		*(float4*)(Y + yoffset) = xv;
	}
}

#endif


#ifndef TRIM_3D_KERNEL4_PC0_4X
#define TRIM_3D_KERNEL4_PC0_4X

//X -> [pn0, pw0, pc0]
//pc0 % 4 == 0
__global__ void trim3D_kernel4_pc04x(
	const float* __restrict__ X, int IW, int IC,
	      float* __restrict__ Y, int OW_OC, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int yoffset = index4;//[on, ow, oc0-oc3]
		int on = yoffset / OW_OC, yr = yoffset - on * OW_OC;
		int ow = yr / OC, oc = yr - ow * OC;

		const int xoffset = (on*IW + ow)*IC + oc;//[in, iw, ic0-ic3]
		*(float4*)(Y + yoffset) = *(float4*)(X + xoffset);
	}
}

#endif


void __trim3d(cudaStream_t stream,
	const float* X, int IN, int IW, int IC,
	      float* Y, int ON, int OW, int OC,
	int pn0, int pw0, int pc0)
{
	X += (pn0*IW + pw0)*IC + pc0;//X[pn0, pw0, pc0]
	int lengthv = ON * OW * OC;//Y.lengthv <= X.lengthv

	if (lengthv < 256) { 
		if(!(pc0 & 3)) trim3d_k4_pc04x_small(stream, X, IW, IC, Y, OW, OC, lengthv);
		else trim3d_k4_small(stream, X, IW, IC, Y, OW, OC, lengthv); 
		return; 
	}
	if(!(pc0 & 3)) trim3d_k4_pc04x(stream, 5, 2, X, IW, IC, Y, OW, OC, lengthv);
	else trim3d_k4(stream, 5, 2, X, IW, IC, Y, OW, OC, lengthv);
}

#endif
