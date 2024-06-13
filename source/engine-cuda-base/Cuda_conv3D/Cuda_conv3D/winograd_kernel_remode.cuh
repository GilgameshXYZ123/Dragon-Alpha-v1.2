#pragma once

#ifndef WINOGRAD_F2X3_KERNEL_REMODE_H
#define WINOGRAD_F2X3_KERNEL_REMODE_H

//Remode the kernel for winograd:
//W[OC, FH: 3, FW: 3, IC] -> G[OC, FH: 3, IC, GW: 4]
//lengthv = OC * FH * IC, IC % 4 == 0, so: lengthv % 4 == 0
#ifndef WINOGRAD_F2X3_KERNEL_REMODE_CALL
#define WINOGRAD_F2X3_KERNEL_REMODE_CALL

//LB = log2(BLOCK_SIZE)
//lengthv = OC * FH * IC = OC * 3 * IC

//lengthv < 256
#define winograd_f2x3_kremode_k4_small(stream, W, G, OC, IC, lengthv)\
	winograd_f2x3_kremode_kernel4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, G, OC, IC, lengthv)

#define winograd_f2x3_kremode_k4(stream, LB, LT, W, G, OC, IC, lengthv)\
	winograd_f2x3_kremode_kernel4\
		<<< (lengthv>>LB>>LT), 1<<LB, 0, stream >>>\
			(W, G, OC, IC, lengthv)

#endif


#ifndef WINOGRAD_F2X3_KERNEL_REMODE_KERNEL
#define WINOGRAD_F2X3_KERNEL_REMODE_KERNEL

__global__ void winograd_f2x3_kremode_kernel4(
	const float* __restrict__ W,//[OC, FH: 3, FW: 3, IC]
	      float* __restrict__ G,//[OC, FH: 3, IC, GW: 4]
	int OC, int IC, int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	const int FH_IC = 3 * IC;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		int oc = index4 / FH_IC, ir = index4 - oc * FH_IC;
		int fh = ir / IC, ic = ir - fh * IC;

		const int woffset0 = (oc * 9 + fh * 3)*IC + ic;
		const int woffset1 = woffset0 + IC;
		const int woffset2 = woffset1 + IC;
		float4 w0 = *(float4*)(W + woffset0);//W[oc, fh, fw: 0, ic0 - ic3]
		float4 w1 = *(float4*)(W + woffset1);//W[oc, fh, fw: 1, ic0 - ic3]
		float4 w2 = *(float4*)(W + woffset2);//W[oc, fh, fw: 2, ic0 - ic3];

		float4 g0 = winograd_f2x3_g(w0.x, w1.x, w2.x);//G[oc, fh, ic0, gw: 0-3]
		float4 g1 = winograd_f2x3_g(w0.y, w1.y, w2.y);//G[oc, fh, ic1, gw: 0-3]
		float4 g2 = winograd_f2x3_g(w0.z, w1.z, w2.z);//G[oc, fh, ic2, gw: 0-3]
		float4 g3 = winograd_f2x3_g(w0.w, w1.w, w2.w);//G[oc, fh, ic3, gw: 0-3]
		const int goffset0 = ((oc*3 + fh)*IC + ic) << 2;
		*(float4*)(G + goffset0     ) = g0;
		*(float4*)(G + goffset0 +  4) = g1;
		*(float4*)(G + goffset0 +  8) = g2;
		*(float4*)(G + goffset0 + 12) = g3;
	}
}

#endif


//{OC, IC} >= 4, {OC, IC} % 4 == 0
void __winograd_f2x3_kernel_remode(cudaStream_t stream,
	const float* W, float* G,
	int OC, int IC)
{
	int lengthv = OC * IC * 3;
	if (lengthv < 256) { winograd_f2x3_kremode_k4_small(stream, W, G, OC, IC, lengthv); return; }
	winograd_f2x3_kremode_k4(stream, 5, 2, W, G, OC, IC, lengthv);
}

#endif