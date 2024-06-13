#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_REMODE_V2_H
#define DCONV3D_DX_KERNEL_SPLIT_REMODE_V2_H

//W[OC, FH, FW, IC] -> CW[y, x, CFH, CFW, OC, IC]
#ifndef DCONV3D_DX_KERNEL_SPLIT_REMODE_V2_CALL
#define DCONV3D_DX_KERNEL_SPLIT_REMODE_V2_CALL

//LB = log2(BLOCK_SIZE)

//lengthv = CFH * CFW * OC * IC

#define ks_remodev2_k4_small(stream, W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw, lengthv)\
	ks_remodev2_kernel_4\
		<<< dim3(1, (sh*sw)), ((lengthv + 3) >> 2), 0, stream >>>\
			(W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw)

#define ks_remodev2_k4(stream, LB, LT, W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw, lengthv)\
	ks_remodev2_kernel_4\
		<<< dim3((lengthv>>LB>>LT), (sh*sw)), 1<<LB, 0, stream >>>\
			(W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw)

#endif


#ifndef DCONV_3D_DX_KERNEL_SPLIT_REMODE_V2_KERNEL_4
#define DCONV_3D_DX_KERNEL_SPLIT_REMODE_V2_KERNEL_4

//(CFH, CFW) = max(CFH, CFW)
//lengthv = sh*sw * OC*CFH*CFW*IC
//W[oc, fh, fw, ic] -> CFH[y, x, fhr: CFH, fwr: CFW, oc:OC, ic:IC]
//fh = y + (oph - fhr)*sh;
//fw = x + (opw - fwr)*sw
//Rspeed = 286.102 GB/s

__global__ void ks_remodev2_kernel_4(
	const float* __restrict__  W, int FH, int FW,
	      float* __restrict__ CW, int mCFH, int mCFW,
	int OC, int IC, int sh, int sw)
{
	const int by = blockIdx.y;
	CW += (by * OC * mCFH * mCFW * IC);//Wks[y, x]

	const int y = by / sw, x = by - y * sw;
	const int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	const int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;

	const int OC_IC = OC * IC;
	const int CFW_OC_IC = CFW * OC_IC;
	const int lengthv = CFH * CFW_OC_IC;

	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		int fhr = index4 / CFW_OC_IC, ir = index4 - fhr * CFW_OC_IC;
		int fwr = ir / OC_IC; ir -= fwr * OC_IC;
		int oc = ir / IC, ic = ir - oc * IC;

		const int fh = y + (oph - fhr)*sh;
		const int fw = x + (opw - fwr)*sw;

		const int woffset = ((oc*FH + fh)*FW + fw)*IC + ic;
		bool lw = (fh < FH) && (fw < FW);
		float4 w = (lw ? *(float4*)(W + woffset) : FLOAT_ZERO4);

		*(float4*)(CW + index4) = w;
	}
}

#endif


#ifndef DECONV3D_DX_KERNEL_SPLIT_REMODE_FUNCTION
#define DECONV3D_DX_KERNEL_SPLIT_REMODE_FUNCTION

//(CFH, CFW) = max(CFH, CFW)
void __ks_remodev2(cudaStream_t stream,
	float* W, int FH, int FW,
	float * CW, int CFH, int CFW,
	int OC, int IC, int sh, int sw)
{
	int lengthv = OC * CFH * CFW * IC;
	if (lengthv < 256) { ks_remodev2_k4_small(stream, W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw, lengthv); return; }
	ks_remodev2_k4(stream, 5, 2, W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw, lengthv);
}

#endif

#endif