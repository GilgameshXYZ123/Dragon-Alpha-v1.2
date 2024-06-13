#pragma once

#ifndef CONV_3D_GEMMV2_R_UERNEL_H
#define CONV_3D_GEMMV2_R_UERNEL_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
//Version2: for small input feature
#ifndef CONV_3D_GEMMV2_R_UERNEL_CALL
#define CONV_3D_GEMMV2_R_UERNEL_CALL

//======[Common]=============================================================
#define conv3dGemmV2_u88R(stream, LB, oc_index, n_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, N) \
	conv3dGemmV2_uernel_8_8R<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OW, OH), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, FH, FW, Y, IC, OC, sh, sw, ph, pw,\
			 oc_index, n_index, (GN>>LB>>3))

//======[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
#define conv3dGemmV2_u88RW3p1_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_uernel_8_8R_W3P1_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OW, OH), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, LIC, OC, sh, sw,\
			 oc_index, n_index, (GN>>LB>>3))

//=========[ph = pw = 1], [FH = FW = 4], [IH, IW > 2]===========================
#define conv3dGemmV2_u88RW4p1_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_uernel_8_8R_W4P1_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OW, OH), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, LIC, OC, sh, sw,\
			 oc_index, n_index, (GN>>LB>>3))

//=========[ph = pw = 2], [FH = FW = 5], [IH, IW > 2]===========================
#define conv3dGemmV2_u88RW5p2_ic2pow(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	conv3dGemmV2_uernel_8_8R_W5P2_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OW, OH), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, LIC, OC, sh, sw,\
			 oc_index, n_index, (GN>>LB>>3))

#endif


//======[Common]=============================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMMV2_UERNEL_8_8R
#define CONV_3D_GEMMV2_UERNEL_8_8R

//For [FH, FW] = 3:
//for (32, 32) -> (16, 16), [N, IC, OC] = [128, 128, 256] 
//LB = 4: Size = 9, Time = 1.76539 msec, Performace = 10947.9 GFlop/s
//LB = 3: Size = 9, Time = 1.98115 msec, Performace =  9755.6 GFlop/s
//for (16, 16) -> ( 8,  8), [N, IC, OC] = [512, 128, 256] 
//LB = 4: Size = 9, Time = 1.59078 msec, Performace = 12149.6 GFlop/s
//LB = 3: Size = 9, Time = 1.90065 msec, Performace = 10168.8 GFlop/s
//for ( 8,  8) -> ( 4,  4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.62252 msec, Performace = 11911.9 GFlop/s
//LB = 3: Size = 9, Time = 1.76556 msec, Performace = 10946.8 GFlop/s
//for ( 4,  4) -> ( 2,  2), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.38666 msec, Performace = 13938 GFlop/s
//LB = 3: Size = 9, Time = 1.53635 msec, Performace = 12580 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemmV2_uernel_8_8R(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int n_index, int GX)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for: OH * OW
	const int OH = gridDim.z, OW = gridDim.y;
	const int oh = blockIdx.z, ow = blockIdx.y;//ow = bz % OW
	const int toh = oh * sh - ph, tow = ow * sw - pw;

	//prepare for: GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_IC = tFW * IC, GK = tFH * tFW_IC;

	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += (fhs*FW + fws)*IC*OC + toc0;//CW[fhs, fws, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bn0 = (by << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int X0 = ((tn0*IH + toh + fhs)*IW + tow + fws)*IC;//X[tn0, toh0 + fhs, tow0 + fws, 0]
	const int Xstride = IH * IW * IC;
	const int X1 = X0 + Xstride;
	const int X2 = X1 + Xstride;
	const int X3 = X2 + Xstride;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yn0  = bn0  + (uy << 3);
	const int Y0 = ((yn0*OH + oh)*OW + ow)*OC + yoc0;

	//compute area------------------------------------------------------
	const int SX = (IW - tFW)*IC;
	const int SW = (FW - tFW)*IC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int fh = X_k / tFW_IC;//X_fh = W_fh = fh
	const int xoffset = fh * SX + X_k;
	float2 x0 = *(float2*)(X + X0 + xoffset);
	float2 x1 = *(float2*)(X + X1 + xoffset);
	float2 x2 = *(float2*)(X + X2 + xoffset);
	float2 x3 = *(float2*)(X + X3 + xoffset);
	Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
			float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int fh = X_k / tFW_IC;//X_fh = W_fh = fh
		const int xoffset = fh * SX + X_k;
		float2 x0 = *(float2*)(X + X0 + xoffset);
		float2 x1 = *(float2*)(X + X1 + xoffset);
		float2 x2 = *(float2*)(X + X2 + xoffset);
		float2 x3 = *(float2*)(X + X3 + xoffset);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fh*SW + W_k)*OC;
		const int woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
		float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//======[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMMV2_UERNEL_8_8R_W3_P1_IC2POW
#define CONV_3D_GEMMV2_UERNEL_8_8R_W3_P1_IC2POW

//for (32, 32) -> (16, 16), [N, IC, OC] = [256, 128, 256] 
//LB = 4: Size = 18, Time = 3.35776 msec, Performace = 11512   GFlop/s
//WB = 4: Size = 18, Time = 3.07496 msec, Performace = 12570.8 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: 
//cuDNN-NHWC-GEMM-implicit-prec: Size = 18, Time = 3.028 msec, Performance = 12765.8 GFlop/s

//for (16, 16) -> (8, 8), [N, IC, OC] = [256, 256, 512] 
//LB = 4: Size = 18, Time = 3.24523 msec, Performace = 11911.2 GFlop/s
//WB = 4: Size = 18, Time = 3.00737 msec, Performace = 12853.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: 
//cuDNN-NHWC-GEMM-implicit-prec: Size = 18, Time = 2.705 msec, Performance = 14290.1 GFlop/s

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 512] 
//LB = 4: Size = 18, Time = 3.23933 msec, Performace = 11932.9 GFlop/s
//WB = 4: Size = 18, Time = 2.94244 msec, Performace = 13137   GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 18, Time = 2.661 msec, Performance = 14526.4 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemmV2_uernel_8_8R_W3P1_ic2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	      float* __restrict__  Y,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index, int GX)//GX
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];
	
	//compute 8*8 results
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for: OH * OW
	const int OH = gridDim.z, OW = gridDim.y;
	const int oh = blockIdx.z, ow = blockIdx.y;//ow = bz % OW
	const int toh = oh * sh - 1, tow = ow * sw - 1;

	//prepare for GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;
	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += ((fhs * 3 + fws) << LIC)*OC + toc0;//CW[fhs, fws, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bn0 = (by << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int X0 = ((tn0*IH + toh + fhs)*IW + tow + fws) << LIC;//X[tn0, toh + fhs, tow + fws, 0]
	const int Xstride = (IH * IW) << LIC;//n0
	const int X1 = X0 + Xstride;//n1
	const int X2 = X1 + Xstride;//n2
	const int X3 = X2 + Xstride;//n3 

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = (vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1);
	const int uy = (vy >> 2) * 8 + ((vx & 15) >> 1);

	const int yoc0 = boc0 + (ux << 3);
	const int yn0  = bn0  + (uy << 3);
	const int Y0 = ((yn0*OH + oh)*OW + ow)*OC + yoc0;

	//compute area------------------------------------------------------
	const int SX = (IW - tFW) << LIC;
	const int SW = (3 - tFW) << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];
	const int xoffset = fh * SX + X_k;
	const int xoffset0 = xoffset + X0;
	const int xoffset1 = xoffset + X1;
	const int xoffset2 = xoffset + X2;
	const int xoffset3 = xoffset + X3;
	float2 x0 = *(float2*)(X + xoffset0);
	float2 x1 = *(float2*)(X + xoffset1);
	float2 x2 = *(float2*)(X + xoffset2);
	float2 x3 = *(float2*)(X + xoffset3);
	Xs[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll  
		for (int ik = 0; ik < STEP2; ++ik) {
			float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];
			float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];
		const int xoffset = fh * SX + X_k;
		const int xoffset0 = xoffset + X0;
		const int xoffset1 = xoffset + X1;
		const int xoffset2 = xoffset + X2;
		const int xoffset3 = xoffset + X3;
		float2 x0 = *(float2*)(X + xoffset0);
		float2 x1 = *(float2*)(X + xoffset1);
		float2 x2 = *(float2*)(X + xoffset2);
		float2 x3 = *(float2*)(X + xoffset3);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fh*SW + W_k)*OC;
		const int woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ++ik) {
		float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
		float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//=========[ph = pw = 1], [FH = FW = 4], [IH, IW > 2]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMMV2_UERNEL_8_8R_W4_P1_IC2POW
#define CONV_3D_GEMMV2_UERNEL_8_8R_W4_P1_IC2POW

//for: (32, 32) -> (16, 16), (N, IC, OC) = (128, 128, 256)
//LB = 4: Size = 16, Time = 2.92556 msec, Performace = 11744.7 GFlop/s
//LB = 3: Size = 16, Time = 3.49611 msec, Performace =  9827.9 GFlop/s
//for: (16, 16) -> ( 8,  8), (N, IC, OC) = (512, 128, 256)
//LB = 4: Size = 16, Time = 2.7722 msec, Performace = 12394.4 GFlop/s
//LB = 3: Size = 16, Time = 3.47453 msec, Performace = 9889.1 GFlop/s
//for: (12, 12) -> ( 6,  6), (N, IC, OC) = (512, 256, 256)
//LB = 4: Size = 18, Time = 3.12765 msec, Performace = 12359 GFlop/s
//LB = 3: Size = 18, Time = 3.93993 msec, Performace =  9811.0 GFlop/s
//for: ( 8,  8) -> ( 4,  4), (N, IC, OC) = (512, 256, 512)
//LB = 4: Size = 16, Time = 2.43753 msec, Performace = 14096.1 GFlop/s
//LB = 3: Size = 16, Time = 3.02667 msec, Performace = 11352.3 GFlop/s
//for: ( 4,  4) -> ( 2,  2), (N, IC, OC) = (512, 512, 1024)
//LB = 4: Size = 16, Time = 2.04361 msec, Performace = 16813.2 GFlop/s
//LB = 3: Size = 16, Time = 2.16015 msec, Performace = 15906.2 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemmV2_uernel_8_8R_W4P1_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = 4, 3
	      float* __restrict__ Y,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index, int GX)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for: OH * OW
	const int OH = gridDim.z, OW = gridDim.y;
	const int oh = blockIdx.z, ow = blockIdx.y;//ow = bz % OW
	const int toh = oh * sh - 1, tow = ow * sw - 1;

	//prepare for: GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 4), 4, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 4), 4, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	const int fhw_offset = (((tFH == 4) << 1) + (tFW == 4)) << 4;

	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += (((fhs << 2) + fws) << LIC)*OC + toc0;//CW[fhs, fws, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bn0 = (by << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int X0 = ((tn0*IH + toh + fhs)*IW + tow + fws) << LIC;//X[tn0, toh + fhs, tow + fws, 0]
	const int Xstride = (IH * IW) << LIC;
	const int X1 = X0 + Xstride;
	const int X2 = X1 + Xstride;
	const int X3 = X2 + Xstride;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yn0 = bn0 + (uy << 3);
	const int Y0 = ((yn0*OH + oh)*OW + ow)*OC + yoc0;
	
	//compute area------------------------------------------------------
	const int SX = (IW - tFW) << LIC;
	const int SW = (4 - tFW) << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int fh = XIDX_V2_W4P1_FH[fhw_offset + (X_k >> LIC)];
	const int xoffset = fh * SX + X_k;
	const int xoffset0 = X0 + xoffset;
	const int xoffset1 = X1 + xoffset;
	const int xoffset2 = X2 + xoffset;
	const int xoffset3 = X3 + xoffset;
	float2 x0 = *(float2*)(X + xoffset0);
	float2 x1 = *(float2*)(X + xoffset1);
	float2 x2 = *(float2*)(X + xoffset2);
	float2 x3 = *(float2*)(X + xoffset3);
	Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];
			float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];

			simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int fh = XIDX_V2_W4P1_FH[fhw_offset + (X_k >> LIC)];
		const int xoffset = fh * SX + X_k;
		const int xoffset0 = X0 + xoffset;
		const int xoffset1 = X1 + xoffset;
		const int xoffset2 = X2 + xoffset;
		const int xoffset3 = X3 + xoffset;
		float2 x0 = *(float2*)(X + xoffset0);
		float2 x1 = *(float2*)(X + xoffset1);
		float2 x2 = *(float2*)(X + xoffset2);
		float2 x3 = *(float2*)(X + xoffset3);
		Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fh*SW + W_k)*OC;
		const int woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
		float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

		simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//=========[ph = pw = 2], [FH = FW = 5], [IH, IW > 2]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMMV2_UERNEL_8_8R_W5_P2_IC2POW
#define CONV_3D_GEMMV2_UERNEL_8_8R_W5_P2_IC2POW

//for: (32, 32) -> (16, 16), (N, IC, OC) = (128, 128, 128)
//LB = 4: Size = 12.5, Time = 2.42103 msec, Performace = 11087.7 GFlop/s
//LB = 3: Size = 12.5, Time = 3.28171 msec, Performace =  8179.8 GFlop/s
//for: (16, 16) -> ( 8,  8), (N, IC, OC) = (512, 128, 128)
//LB = 4: Size = 12.5, Time = 2.27201 msec, Performace = 11814.9 GFlop/s
//LB = 3: Size = 12.5, Time = 2.73594 msec, Performace =  9811.5 GFlop/s
//for: (12, 12) -> ( 6,  6), (N, IC, OC) = (512, 128, 256) 
//LB = 4: Size = 14.0625, Time = 2.36863 msec, Performace = 12749.6 GFlop/s
//LB = 3: Size = 14.0625, Time = 3.0082  msec, Performace = 10038.9 GFlop/s
//for: ( 8,  8) -> ( 4,  4), (N, IC, OC) = (512, 256, 256)
//LB = 4: Size = 12.5, Time = 1.96587 msec, Performace = 13654.8 GFlop/s
//LB = 3: Size = 12.5, Time = 2.26321 msec, Performace = 11860.8 GFlop/s
//for: ( 4,  4) -> ( 2,  2), (N, IC, OC) = (512, 512, 512)
//LB = 4: Size = 12.5, Time = 1.59693 msec, Performace = 16809.5 GFlop/s
//LB = 3: Size = 12.5, Time = 1.72231 msec, Performace = 15585.8 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemmV2_uernel_8_8R_W5P2_ic2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = 5, 4, 3
	      float* __restrict__ Y,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 2
	int oc_index, int n_index, int GX)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for: OH * OW
	const int OH = gridDim.z, OW = gridDim.y;
	const int oh = blockIdx.z, ow = blockIdx.y;//ow = bz % OW
	const int toh = oh * sh - 2, tow = ow * sw - 2;

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 5), 5, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 5), 5, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	const int fh_idx = (tFH == 4) + ((tFH == 5) << 1);
	const int fw_idx = (tFW == 4) + ((tFW == 5) << 1);
	const int fhw_offset = (fh_idx * 3 + fw_idx) * 25;

	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += ((fhs * 5 + fws) << LIC) * OC + toc0;//CW[fhs, fws, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bn0 = (by << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int X0 = ((tn0*IH + toh + fhs)*IW + tow + fws) << LIC;//X[0, toh + fhs, tow + fws, 0]
	const int Xstride = (IH * IW) << LIC;
	const int X1 = X0 + Xstride;
	const int X2 = X1 + Xstride;
	const int X3 = X2 + Xstride;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2
	
	const int yoc0 = boc0 + (ux << 3);
	const int yn0  = bn0  + (uy << 3);
	const int Y0 = ((yn0*OH + oh)*OW + ow)*OC + yoc0;

	//compute area------------------------------------------------------
	const int SX = (IW - tFW) << LIC;
	const int SW = (5 - tFW) << LIC;
	const int Ystride = OH * OW * OC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;//with the same ty
	const int fh = XIDX_V2_W5P2_FH[fhw_offset + (X_k >> LIC)];
	const int xoffset = fh * SX + X_k;
	const int xoffset0 = xoffset + X0;
	const int xoffset1 = xoffset + X1;
	const int xoffset2 = xoffset + X2;
	const int xoffset3 = xoffset + X3;
	float2 x0 = *(float2*)(X + xoffset0);
	float2 x1 = *(float2*)(X + xoffset1);
	float2 x2 = *(float2*)(X + xoffset2);
	float2 x3 = *(float2*)(X + xoffset3);
	Xs[0][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;//with the same tx
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];
			float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int fh = XIDX_V2_W5P2_FH[fhw_offset + (X_k >> LIC)];
		const int xoffset = fh * SX + X_k;
		const int xoffset0 = xoffset + X0;
		const int xoffset1 = xoffset + X1;
		const int xoffset2 = xoffset + X2;
		const int xoffset3 = xoffset + X3;
		float2 x0 = *(float2*)(X + xoffset0);
		float2 x1 = *(float2*)(X + xoffset1);
		float2 x2 = *(float2*)(X + xoffset2);
		float2 x3 = *(float2*)(X + xoffset3);
		Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fh*SW + W_k)*OC;
		const int woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
		float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif

#endif