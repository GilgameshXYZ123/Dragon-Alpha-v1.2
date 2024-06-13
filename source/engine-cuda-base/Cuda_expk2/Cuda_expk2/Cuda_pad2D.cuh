#pragma once

#ifndef PAD_2D_H
#define PAD_2D_H

//X[IN, IC] -> Y[ON, IC]
//IN -> ON = pn0 + IN + pn1
//IC -> OC = pc0 + IC + pc1
//<1> IN % 4 != 0 (ignore the padding on IN)
//<2> ON % 4 != 0 (ignore the padding on ON)
//<3> IC % 4 == 0
//<4> OC % 4 == 0
//lengthv = IN*IC, so: lengthv%4 == 0
#ifndef PAD_2D_CALL
#define PAD_2D_CALL

#define pad2d_k4_small(stream, X, IC, Y, OC, lengthv)\
	pad2D_kernel4\
		<<< 1, lengthv, 0, stream >>>\
			(X, IC, Y, OC, lengthv) 

#define pad2d_k4(stream, LB, LT, X, IC, Y, OC, lengthv)\
	pad2D_kernel4\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X, IC, Y, OC, lengthv)

//pc0 % 4 == 0
#define pad2d_k4_pc04x_small(stream, X, IC, Y, OC, lengthv)\
	pad2D_kernel4_pc04x\
		<<< 1, lengthv, 0, stream >>>\
			(X, IC, Y, OC, lengthv) 

//pc0 % 4 == 0
#define pad2d_k4_pc04x(stream, LB, LT, X, IC, Y, OC, lengthv)\
	pad2D_kernel4_pc04x\
		<<< (lengthv>>LB>>LT), (1<<LB), 0, stream>>>\
			(X, IC, Y, OC, lengthv)

#endif


#ifndef PAD_2D_KERNEL4
#define PAD_2D_KERNEL4

//Y -> Y[pn0, pc0]
__global__ void pad2D_kernel4(
	const float* __restrict__ X, int IC,
	      float* __restrict__ Y, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int xoffset = index4;//[in, ic0-ic3]
		int in = xoffset / IC, ic = xoffset - in * IC;

		const int yoffset = in * OC + ic;
		float4 xv = *(float4*)(X + xoffset);
		Y[yoffset    ] = xv.x;//[on, oc0]
		Y[yoffset + 1] = xv.y;//[on, oc1]
		Y[yoffset + 2] = xv.z;//[on, oc2]
		Y[yoffset + 3] = xv.w;//[on, oc3]
	}
}

#endif


#ifndef PAD_2D_KERNEL4_PC0_4X
#define PAD_2D_KERNEL4_PC0_4X

//Y -> Y[pn0, pc0]
//pc0 % 4 == 0
__global__ void pad2D_kernel4_pc04x(
	const float* __restrict__ X, int IC,
	      float* __restrict__ Y, int OC,
	int lengthv)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		const int xoffset = index4;//[in, ic0-ic3]
		int in = xoffset / IC, ic = xoffset - in * IC;

		const int yoffset = in * OC + ic;//[on, oc0-oc3]
		*(float4*)(Y + yoffset) = *(float4*)(X + xoffset);
	}
}

#endif


void __pad2d(cudaStream_t stream,
	const float* X, int IN, int IC,
	      float* Y, int ON, int OC,
	int pn0, int pc0)
{
	Y += pn0 * OC + pc0;//Y[pn0, pc0]
	int lengthv = IN * IC;//X.lengthv <= Y.lengthv

	if (lengthv < 256) { 
		if(!(pc0 & 3)) pad2d_k4_pc04x_small(stream, X, IC, Y, OC, lengthv);
		else pad2d_k4_small(stream, X, IC, Y, OC, lengthv);
		return; 
	}
	if(!(pc0 & 3)) pad2d_k4_pc04x(stream, 5, 2, X, IC, Y, OC, lengthv);
	else pad2d_k4(stream, 5, 2, X, IC, Y, OC, lengthv);
}

#endif