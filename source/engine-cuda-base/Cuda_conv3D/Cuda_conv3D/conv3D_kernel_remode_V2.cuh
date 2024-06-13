#pragma once

#ifndef CONV_3D_KERNEL_REMODE_V2_H
#define CONV_3D_KERNEL_REMODE_V2_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[IC, FH, FW, OC]
#ifndef CONV_3D_KERNEL_REMODE_V2_CALL
#define CONV_3D_KERNEL_REMODE_V2_CALL

//LB = log2(BLOCK_SIZE)
//lengthv = FH * FW * OC * IC

#define kremodeV2_k4(stream, LB, LT, W, CW, FH, FW, OC, IC, lengthv)\
	kremodeV2_kernel_4\
		<<< (lengthv>>LB>>LT), 1<<LB, 0, stream >>>\
			(W, CW, FH, FW, OC, IC, lengthv)

#define kremodeV2_k4_small(stream, W, CW, FH, FW, OC, IC, lengthv)\
	kremodeV2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, CW, FH, FW, OC, IC, lengthv)

//lengthv: OC*FH*FW*IC -> (OC>>2)*FH*FW*IC
#define kremodeV2_k4_4x(stream, LB, LT, W, CW, FH, FW, OC, IC, lengthv)\
	kremodeV2_kernel_4X\
		<<< ((lengthv>>2)>>LB>>LT), (1<<LB), 0, stream >>>\
			(W, CW, FH, FW, OC, IC, (lengthv>>2))

#endif


#ifndef CONV_3D_KERNEL_REMODE_V2_KERNEL_4
#define CONV_3D_KERNEL_REMODE_V2_KERNEL_4

//W[OC, FH, FW, IC] -> CW[IC, FH, FW, OC]
//lengthv = FH * FW * OC * IC

__global__ void kremodeV2_kernel_4(
	const float* __restrict__ W,
	float* __restrict__ CW,
	int FH, int FW, int OC, int IC, int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int FW_IC = FW * IC;
	int FH_FW_IC = FH * FW_IC;
	int CWstride = FH * FW*OC;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//[oc, fh, fw, ic] -> [fh, fw, ic, oc]
		int oc = index4 / FH_FW_IC, ir = index4 - oc * FH_FW_IC;
		int fh = ir / FW_IC; ir -= fh * FW_IC;
		int fw = ir / IC, ic = ir - fw * IC;

		//Woffset[oc, fh, fw, ic0 -> ic3]
		float4 w = *(float4*)(W + index4);

		//CW1 = (((fh*FW + fw)*IC + ic)*OC + oc);
		//= fh*FW*IC*OC + fw*IC*OC + ic*OC + oc
		//= fh*FW*IC*OC + (fw*IC + ic)*OC + oc
		//As: ir = fw*IC + ic
		//= (fh*FW*IC + (fw*IC + ic))*OC + oc
		//= (fh*FW*IC + ir)*OC + oc

		int CW1 = (((ic + 1)*FH + fh)*FW + fw)*OC + oc;
		CW[CW1 - CWstride] = w.x;//[fh, fw, ic0, oc]
		CW[CW1] = w.y;//[fh, fw, ic1, oc]
		CW[CW1 + CWstride] = w.z;//[fh, fw, ic2, oc]
		CW[CW1 + (CWstride << 1)] = w.w;//[fh, fw, ic3, oc]
	}
}

#endif


//OC >= 4, OC % 4 == 0
void __kernel_remodeV2(cudaStream_t stream,
	const float* W, float * CW,
	int FH, int FW, int OC, int IC)
{
	int lengthv = OC * FH * FW * IC;
	if (lengthv < 256) { kremodeV2_k4_small(stream, W, CW, FH, FW, OC, IC, lengthv); return; }
	else kremodeV2_k4(stream, 5, 2, W, CW, FH, FW, OC, IC, lengthv);
}

#endif