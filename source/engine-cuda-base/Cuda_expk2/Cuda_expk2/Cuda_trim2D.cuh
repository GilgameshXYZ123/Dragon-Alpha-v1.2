#pragma once

#ifndef TRIM_2D_H
#define TRIM_2D_H

//X[IN, IC] -> Y[ON, OC]
//IN = pn0 + ON + pn1 -> ON
//IC = pc0 + OC + pc1 -> OC
//<1> IN % 4 != 0 (ignore the mem-alginment on IN)
//<2> ON % 4 != 0 (ignore the mem-alginment on ON)
//<3> IC % 4 == 0
//<4> OC % 4 == 0
//lengthv = ON*OC, so: lengthv%4 == 0
#ifndef TRIM_2D_CALL
#define TRIM_2D_CALL

#define trim2d_k4_small(stream, X, IC, Y, OC, lengthv)\
	trim2D_kernel4\
		<<< 1, lengthv, 0, stream >>>\
			(X, IC, Y, OC, lengthv) 

#define trim2d_k4(stream, LB, LT, X, IC, Y, OC, lengthv)\
	trim2D_kernel4\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X, IC, Y, OC, lengthv)

//pc0 % 4 == 0
#define trim2d_k4_pc04x_small(stream, X, IC, Y, OC, lengthv)\
	trim2D_kernel4_pc04x\
		<<< 1, lengthv, 0, stream >>>\
			(X, IC, Y, OC, lengthv) 

//pc0 % 4 == 0
#define trim2d_k4_pc04x(stream, LB, LT, X, IC, Y, OC, lengthv)\
	trim2D_kernel4_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X, IC, Y, OC, lengthv)

#endif


#ifndef TRIM_2D_KERNEL4
#define TRIM_2D_KERNEL4

//X -> X[pn0, pc0]
__global__ void trim2D_kernel4(
	const float* __restrict__ X, int IC,
	      float* __restrict__ Y, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int yoffset = index4;//[on, oc0-oc3]
		int on = yoffset / OC, oc = yoffset - on * OC;

		float4 xv; int xoffset = on * IC + oc;
		xv.x = X[xoffset    ];//[on, ic0]
		xv.y = X[xoffset + 1];//[on, ic1]
		xv.z = X[xoffset + 2];//[on, ic2]
		xv.w = X[xoffset + 3];//[on, ic3]
		*(float4*)(Y + yoffset) = xv;
	}
}

#endif


#ifndef TRIM_2D_KERNEL4_PC0_4X
#define TRIM_2D_KERNEL4_PC0_4X

//X -> X[pn0, pc0]
//pc0 % 4 == 0
__global__ void trim2D_kernel4_pc04x(
	const float* __restrict__ X, int IC,
	      float* __restrict__ Y, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int yoffset = index4;//[on, oc0-oc3]
		int on = yoffset / OC, oc = yoffset - on * OC;

		const int xoffset = on * IC + oc;//[in, ic0-ic3]
		*(float4*)(Y + yoffset) = *(float4*)(X + xoffset);
	}
}

#endif


void __trim2d(cudaStream_t stream,
	const float* X, int IN, int IC,
	      float* Y, int ON, int OC,
	int pn0, int pc0)
{
	X += pn0 * IC + pc0;//X[pn0, pc0]
	int lengthv = ON * OC;//Y.lengthv <= X.lengthv

	if (lengthv < 256) { 
		if(!(pc0 & 3)) trim2d_k4_pc04x_small(stream, X, IC, Y, OC, lengthv);
		else trim2d_k4_small(stream, X, IC, Y, OC, lengthv); 
		return; 
	}
	if(!(pc0 & 3)) trim2d_k4_pc04x(stream, 5, 2, X, IC, Y, OC, lengthv);
	else trim2d_k4(stream, 5, 2, X, IC, Y, OC, lengthv);
}

#endif
