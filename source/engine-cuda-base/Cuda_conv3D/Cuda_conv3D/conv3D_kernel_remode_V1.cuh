#pragma once

#ifndef CONV_3D_KERNEL_REMODE_V1_H
#define CONV_3D_KERNEL_REMODE_V1_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV_3D_KERNEL_REMODE_CALL
#define CONV_3D_KERNEL_REMODE_CALL

//LB = log2(BLOCK_SIZE)
//lengthv = FH * FW * OC * IC

//======[normal (1*4) method]========================================
//lengthv < 256
#define kremode_k4_small(stream, W, CW, FH, FW, OC, IC, lengthv)\
	kremode_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, CW, FH, FW, OC, IC, lengthv)

#define kremode_k4(stream, LB, LT, W, CW, FH, FW, OC, IC, lengthv)\
	kremode_kernel_4\
		<<< (lengthv>>LB>>LT), 1<<LB, 0, stream >>>\
			(W, CW, FH, FW, OC, IC, lengthv)

//======[mat transpose (4*4) method]=================================
//lengthv: OC*FH*FW*IC -> (OC>>2)*FH*FW*IC
#define kremode_k4_4x4(stream, LB, LT, W, CW, FH, FW, OC, IC, lengthv)\
	kremode_kernel_4X4\
		<<< ((lengthv>>2)>>LB>>LT), (1<<LB), 0, stream >>>\
			(W, CW, FH, FW, OC, IC, (lengthv>>2))

//GM = FH*FW*IC
//LTX, LTY >= 2
#define kremode_k4_4x4A(stream, LBY, LBX, LTY, LTX, W, CW, FH, FW, OC, IC, GM)\
	kremode_kernel_4X4_A\
		<<< dim3((OC>>LBX>>LTX), (GM>>LBY>>LTY)), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(W, CW, FH, FW, OC, IC, GM)

#endif


//======[normal (1*4) method]========================================
#ifndef CONV_3D_KERNEL_REMODE_KERNEL_4
#define CONV_3D_KERNEL_REMODE_KERNEL_4

//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
//lengthv = FH * FW * OC * IC

__global__ void kremode_kernel_4(
	const float* __restrict__ W, 
	      float* __restrict__ CW,
	int FH, int FW, int OC, int IC, int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int FW_IC = FW * IC;
	int FH_FW_IC = FH * FW_IC;

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

		int CW1 = (((fh*FW + fw)*IC + ic + 1)*OC + oc);
		CW[CW1 - OC]        = w.x;//[fh, fw, ic0, oc]
		CW[CW1]             = w.y;//[fh, fw, ic1, oc]
		CW[CW1 + OC]        = w.z;//[fh, fw, ic2, oc]
		CW[CW1 + (OC << 1)] = w.w;//[fh, fw, ic3, oc]
	}
}

#endif


//======[mat transpose (4*4) method]=================================
#ifndef CONV_3D_KERNEL_REMODE_KERNEL_4X4
#define CONV_3D_KERNEL_REMODE_KERNEL_4X4

//lengthv = FH * FW * (OC >> 2) * IC
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
//lengthv = FH * FW * OC * IC
//As: IC % 4 == 0, So: lengthv % 4 == 0

//[OC, FH, FW, IC] = [128, 3, 3, 128]:  37.563 GB/s
//[OC, FH, FW, IC] = [256, 3, 3, 256]: 110.177 GB/s
//[OC, FH, FW, IC] = [512, 3, 3, 512]: 222.962 GB/s

__global__ void kremode_kernel_4X4(
	const float* __restrict__ W,
	      float* __restrict__ CW,
	int FH, int FW, int OC, int IC, int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	const int FH_FW_IC = FH * FW * IC;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//[oc, fh, fw, ic] -> [fh, fw, ic, oc]
		//<1> fhw = fh*FW + fw
		//<2> ir = (fh*FW + fw)*IC + ic = fhw*IC + ic
		int oc = index4 / FH_FW_IC, ir = index4 - oc * FH_FW_IC;

		const int oc0 = oc << 2;//ir(fh, fw, ic)
		const int woffset0 = oc0*FH_FW_IC + ir;  //<oc0, fh, fw, ic>
		const int woffset1 = woffset0 + FH_FW_IC;//<oc1, fh, fw, ic>
		const int woffset2 = woffset1 + FH_FW_IC;//<oc2, fh, fw, ic>
		const int woffset3 = woffset2 + FH_FW_IC;//<oc3, fh, fw, ic>

		//CW[FH, FW, IC, OC], ir(fh, fw, ic)
		const int cwoffset0 = ir * OC + oc0;   //<fh, fw, ic0, oc0-oc3>
		const int cwoffset1 = cwoffset0 + OC;//<fh, fw, ic1, oc0-oc3>
		const int cwoffset2 = cwoffset1 + OC;//<fh, fw, ic2, oc0-oc3>
		const int cwoffset3 = cwoffset2 + OC;//<fh, fw, ic3, oc0-oc3>

		float4 w0 = *(float4*)(W + woffset0);//W[oc0, fh, fw, ic0-ic3]
		float4 w1 = *(float4*)(W + woffset1);//W[oc1, fh, fw, ic0-ic3]
		float4 w2 = *(float4*)(W + woffset2);//W[oc2, fh, fw, ic0-ic3]
		float4 w3 = *(float4*)(W + woffset3);//W[oc3, fh, fw, ic0-ic3]

		float4 cw0 = float4{ w0.x, w1.x, w2.x, w3.x };//W[oc0-oc3, fh, fw, ic0]
		float4 cw1 = float4{ w0.y, w1.y, w2.y, w3.y };//W[oc0-oc3, fh, fw, ic1]
		float4 cw2 = float4{ w0.z, w1.z, w2.z, w3.z };//W[oc0-oc3, fh, fw, ic2]
		float4 cw3 = float4{ w0.w, w1.w, w2.w, w3.w };//W[oc0-oc3, fh, fw, ic3]

		*(float4*)(CW + cwoffset0) = cw0;//CW[ic0, fh, fw, oc0-oc3]
		*(float4*)(CW + cwoffset1) = cw1;//CW[ic1, fh, fw, oc0-oc3]
		*(float4*)(CW + cwoffset2) = cw2;//CW[ic2, fh, fw, oc0-oc3]
		*(float4*)(CW + cwoffset3) = cw3;//CW[ic3, fh, fw, oc0-oc3]
	}
}

#endif


#ifndef CONV_3D_KERNEL_REMODE_KERNEL_4X4_A
#define CONV_3D_KERNEL_REMODE_KERNEL_4X4_A
			
//[OC, FH, FW, IC] = [128, 3, 3, 128]:  38.410 GB/s
//[OC, FH, FW, IC] = [256, 3, 3, 256]: 112.896 GB/s
//[OC, FH, FW, IC] = [512, 3, 3, 512]: 235.66 GB/s

__global__ void kremode_kernel_4X4_A(
	const float* __restrict__ W,
	      float* __restrict__ CW,
	int FH, int FW, int OC, int IC, int GM)
{
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;

	const int stepY = blockDim.y * gridDim.y, stepY4 = stepY << 2;
	const int stepX = blockDim.x * gridDim.x, stepX4 = stepX << 2;

	for (int y4 = y << 2; y4 < OC; y4 += stepY4)//OC / 4
	for (int x4 = x << 2; x4 < GM; x4 += stepX4)//GM / 4
	{
		const int oc0 = y4, oc1 = oc0 + 1, oc2 = oc0 + 2, oc3 = oc0 + 3;
		const int j0 = x4, j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;//(fh, fw, ic)

		//======[load (4 * 4) elements from W[OC, <FH, FW, IC>] ]=============
		const int woffset0 = oc0 * GM + j0;//{ oc0, j0-j3 }
		const int woffset1 = oc1 * GM + j0;//{ oc1, j0-j3 }
		const int woffset2 = oc2 * GM + j0;//{ oc2, j0-j3 }
		const int woffset3 = oc3 * GM + j0;//{ oc3, j0-j3 }

		float4 w0 = *(float4*)(W + woffset0);//W[oc0, j0]
		float4 w1 = *(float4*)(W + woffset1);//W[oc1, fh, fw, ic0-ic3]
		float4 w2 = *(float4*)(W + woffset2);//W[oc2, fh, fw, ic0-ic3]
		float4 w3 = *(float4*)(W + woffset3);//W[oc3, fh, fw, ic0-ic3]

		//======[write (4 * 4) elements for CW [<FH, FW, IC>, OC] ]============
		const int cwoffset0 = j0 * OC + oc0;//{ j0, oc0-oc3 }
		const int cwoffset1 = j1 * OC + oc0;//{ j1, oc0-oc3 }
		const int cwoffset2 = j2 * OC + oc0;//{ j3, oc0-oc3 }
		const int cwoffset3 = j3 * OC + oc0;//{ j4, oc0-oc3 }

		*(float4*)(CW + cwoffset0) = float4{ w0.x, w1.x, w2.x, w3.x };//CW[ic0, fh, fw, oc0-oc3]
		*(float4*)(CW + cwoffset1) = float4{ w0.y, w1.y, w2.y, w3.y };//CW[ic1, fh, fw, oc0-oc3]
		*(float4*)(CW + cwoffset2) = float4{ w0.z, w1.z, w2.z, w3.z };//CW[ic2, fh, fw, oc0-oc3]
		*(float4*)(CW + cwoffset3) = float4{ w0.w, w1.w, w2.w, w3.w };//CW[ic3, fh, fw, oc0-oc3]
	}
}

#endif


//======[integration]================================================
#ifndef CONV_3D_KERNEL_REMODE_FUNCTION
#define CONV_3D_KERNEL_REMODE_FUNCTION

//{OC, IC} >= 4, {OC, IC} % 4 == 0
void __kernel_remode(cudaStream_t stream,
	const float* W, float *CW,
	int FH, int FW, int OC, int IC)
{
	int lengthv = OC * FH * FW * IC;
	if (lengthv < 256) { kremode_k4_small(stream, W, CW, FH, FW, OC, IC, lengthv); return; }
	if ((lengthv >= 4096) & !(OC & 3)) {
		int GM = FH * FW * IC;
		if (OC > 31 && GM > 15) { kremode_k4_4x4A(stream, 3, 2, 2, 2, W, CW, FH, FW, OC, IC, GM); return; }//(OC: 5, GM: 4) => (32, 16)
		kremode_k4_4x4(stream, 5, 2, W, CW, FH, FW, OC, IC, lengthv); return; 
	}	
	kremode_k4(stream, 5, 2, W, CW, FH, FW, OC, IC, lengthv);
}

#endif

#endif