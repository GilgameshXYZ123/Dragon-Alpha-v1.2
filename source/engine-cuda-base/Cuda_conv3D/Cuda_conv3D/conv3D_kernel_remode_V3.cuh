#pragma once

#ifndef CONV_3D_KERNEL_REMODE_V3_H
#define CONV_3D_KERNEL_REMODE_V3_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, IC, FW, OC]
#ifndef CONV_3D_KERNEL_REMODE_V3_CALL
#define CONV_3D_KERNEL_REMODE_V3_CALL

//======[normal (1*4) method]========================================
//lengthv < 256
#define kremodeV3_k4_small(stream, W, CW, FH, FW, OC, IC, lengthv)\
	kremodeV3_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, CW, FH, FW, OC, IC, lengthv)

#define kremodeV3_k4(stream, LB, LT, W, CW, FH, FW, OC, IC, lengthv)\
	kremodeV3_kernel_4\
		<<< (lengthv>>LB>>LT), 1<<LB, 0, stream >>>\
			(W, CW, FH, FW, OC, IC, lengthv)

//======[mat transpose (4*4) method]=================================
//lengthv: OC*FH*FW*IC -> (OC>>2)*FH*FW*IC
#define kremodeV3_k4_4x4(stream, LB, LT, W, CW, FH, FW, OC, IC, lengthv)\
	kremodeV3_kernel_4X4\
		<<< ((lengthv>>2)>>LB>>LT), (1<<LB), 0, stream >>>\
			(W, CW, FH, FW, OC, IC, (lengthv>>2))

#endif


//======[normal (1*4) method]========================================
#ifndef CONV_3D_KERNEL_REMODE_V3_KERNEL_4
#define CONV_3D_KERNEL_REMODE_V3_KERNEL_4

//W[OC, FH, FW, IC] -> CW[FH, IC, FW, OC]
//lengthv = FH * FW * OC * IC

__global__ void kremodeV3_kernel_4(
	const float* __restrict__ W, 
	      float* __restrict__ CW,
	int FH, int FW, int OC, int IC, int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int FW_OC = FW * OC;
	int IC_FW_OC = IC * FW_OC;
	int Wstride = FH * FW * IC;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//[oc, fh, fw, ic] -> [fh, ic, fw, oc]
		int fh = index4 / IC_FW_OC, ir = index4 - fh * IC_FW_OC;
		int ic = ir / FW_OC; ir -= ic * FW_OC;
		int fw = ir / OC, oc = ir - fw * OC;

		//======{ load 4 elements from W[OC, FH, FW, IC] }=================
		const int W0 = ((oc*FH + fh)*FW + fw)*IC + ic;
		const int W1 = W0 + Wstride       ;
		const int W2 = W0 + (Wstride << 1);
		const int W3 = W0 + Wstride * 3   ;
		
		float w0 = W[W0];//[oc0, fh, fw, ic]
		float w1 = W[W1];//[oc1, fh, fw, ic]
		float w2 = W[W2];//[oc2, fh, fw, ic]
		float w3 = W[W3];//[oc3, fh, fw, ic]

		//======{ write 4 elements to CW[FH, IC, FW, OC] }=================
		const int CW0 = ((fh*IC + ic)*FW + fw)*OC + oc;
		*(float4*)(CW + CW0) = float4{ w0, w1, w2, w3 };//[fh, ic, fw, oc0-oc3]
	}
}

#endif


//======[mat transpose (4*4) method]=================================
#ifndef CONV_3D_KERNEL_REMODE_V3_KERNEL_4X4
#define CONV_3D_KERNEL_REMODE_V3_KERNEL_4X4

//lengthv = FH * FW * (OC >> 2) * IC
//W[OC, FH, FW, IC] -> CW[FH, IC, FW, OC]
//lengthv = FH * FW * OC * IC
//As: IC % 4 == 0, So: lengthv % 4 == 0

//[OC, FH, FW, IC] = [128, 3, 3, 128]:  37.563 GB/s
//[OC, FH, FW, IC] = [256, 3, 3, 256]: 110.177 GB/s
//[OC, FH, FW, IC] = [512, 3, 3, 512]: 222.962 GB/s

__global__ void kremodeV3_kernel_4X4(
	const float* __restrict__ W,
	      float* __restrict__ CW,
	int FH, int FW, int OC, int IC, int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	const int FW_OC = FW * OC;
	const int IC_FW_OC = (IC >> 2) * FW_OC;
	const int Wstride = FH * FW * IC;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//[oc, fh, fw, ic] -> [fh, ic, fw, oc]
		int fh = index4 / IC_FW_OC, ir = index4 - fh * IC_FW_OC;
		int ic = ir / FW_OC; ir -= ic * FW_OC;
		int fw = ir / OC, oc = ir - fw * OC;
		const int ic0 = ic << 2;

		//======{ load 4*4 elements from W[OC, FH, FW, IC] }==========
		const int W0 = ((oc*FH + fh)*FW + fw)*IC + ic0;
		const int W1 = W0 + Wstride;
		const int W2 = W0 + (Wstride << 1);
		const int W3 = W0 + Wstride * 3;

		float4 w0 = *(float4*)(W + W0);//[oc0, fh, fw, ic0 - ic3]
		float4 w1 = *(float4*)(W + W1);//[oc1, fh, fw, ic0 - ic3]
		float4 w2 = *(float4*)(W + W2);//[oc2, fh, fw, ic0 - ic3]
		float4 w3 = *(float4*)(W + W3);//[oc3, fh, fw, ic0 - ic3]

		//======{ write 4*4 elements to CW[FH, IC, FW, OC] }==========
		const int CW0 = ((fh*IC + ic0)*FW + fw)*OC + oc;
		const int CW1 = CW0 + FW_OC;
		const int CW2 = CW0 + (FW_OC << 1);
		const int CW3 = CW0 + FW_OC * 3;

		*(float4*)(CW + CW0) = { w0.x, w1.x, w2.x, w3.x };//CW[ic0, fh, fw, oc0 - oc3]
		*(float4*)(CW + CW1) = { w0.y, w1.y, w2.y, w3.y };//CW[ic1, fh, fw, oc0 - oc3]
		*(float4*)(CW + CW2) = { w0.z, w1.z, w2.z, w3.z };//CW[ic2, fh, fw, oc0 - oc3]
		*(float4*)(CW + CW3) = { w0.w, w1.w, w2.w, w3.w };//CW[ic3, fh, fw, oc0 - oc3]
	}
}

#endif


//======[integration]================================================
#ifndef CONV_3D_KERNEL_REMODE_V3_FUNCTION
#define CONV_3D_KERNEL_REMODE_V3_FUNCTION

//{OC, IC} >= 4, {OC, IC} % 4 == 0
void __kernel_remodeV3(cudaStream_t stream,
	const float* W, float *CW,
	int FH, int FW, int OC, int IC)
{
	int lengthv = OC * FH * FW * IC;
	if (lengthv < 256) { kremodeV3_k4_small(stream, W, CW, FH, FW, OC, IC, lengthv); return; }
	if ((lengthv >= 4096) & !(IC & 3)) {
		int GM = FH * IC * FW;//[FH, IC, FW, OC]
		//if (OC > 31 && GM > 15) { kremodeV3_k4_4x4A(stream, 3, 2, 2, 2, W, CW, FH, FW, OC, IC, GM); return; }//(OC: 5, GM: 4) => (32, 16)
		kremodeV3_k4_4x4(stream, 5, 2, W, CW, FH, FW, OC, IC, lengthv); return;
	}
	kremodeV3_k4(stream, 5, 2, W, CW, FH, FW, OC, IC, lengthv);
}

#endif


#endif