
//======[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef GSX_KERNEL1
#define GSX_KERNEL1

#define gsx_k1(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	GSX_Kernel1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw,\
			 oc_index, n_index)

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.4047 msec, Performace = 13759.1 GFlop/s
//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.67307 msec, Performace = 11552.1 GFlop/s
//for (16, 16) -> (8, 8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.69252 msec, Performace = 11419.3 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [128,  256, 256] 
//LB = 4: Size = 9, Time = 1.74843 msec, Performace = 11054.1 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void GSX_Kernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][(2 << LB)][(1 << LB) + 1];
	__shared__ float4 Xs[2][(2 << LB)][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	const int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	const int toh = oh * sh - 1, tow = ow * sw - 1;

	//prepare for GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	CW += ((fhs * 3 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	const int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	const int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	const int Xstride = (IH * IW) << LIC;
	const int X1 = X0 + Xstride;
	const int X2 = X1 + Xstride;
	const int X3 = X2 + Xstride;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int SX = (IW - tFW) << LIC;
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];//Idx =  (X_k >> LIC)
	const int xoffset = fh * SX + X_k;
	const int xoffset0 = xoffset + X0;
	const int xoffset1 = xoffset + X1;
	const int xoffset2 = xoffset + X2;
	const int xoffset3 = xoffset + X3;
	float2 x0 = *(float2*)(X + xoffset0);
	float2 x1 = *(float2*)(X + xoffset1);
	float2 x2 = *(float2*)(X + xoffset2);
	float2 x3 = *(float2*)(X + xoffset3);

	// , ty>>1
	Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int SW = (3 - tFW) << LIC;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll  
		for (int ik = 0; ik < STEP2; ++ik)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
	for (int ik = 0; ik < STEP2; ++ik)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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
#ifndef GSX_KERNEL2
#define GSX_KERNEL2

#define gsx_k2(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	GSX_Kernel2<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw,\
			 oc_index, n_index, (GN>>LB>>3))

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.4047 msec, Performace = 13759.1 GFlop/s
//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.67307 msec, Performace = 11552.1 GFlop/s
//for (16, 16) -> (8, 8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.69252 msec, Performace = 11419.3 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [128,  256, 256] 
//LB = 4: Size = 9, Time = 1.74843 msec, Performace = 11054.1 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void GSX_Kernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index, int GX)//GX
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][(2 << LB)][(1 << LB) + 1];
	__shared__ float4 Xs[2][(2 << LB)][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.y;
	const int oh = bz / 16, ow = bz - oh * 16;//ow = bz % OW
	const int toh = oh * sh - 1, tow = ow * sw - 1;

	//prepare for GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	CW += ((fhs * 3 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by*GX;

	//prepare for GN = OC
	const int oc0 = (((bx << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	const int n0 = (((by << LB) + ty) << 3) + n_index;
	const int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	const int Xstride = (IH * IW) << LIC;
	const int X1 = X0 + Xstride;
	const int X2 = X1 + Xstride;
	const int X3 = X2 + Xstride;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int SX = (IW - tFW) << LIC;
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];//Idx =  (X_k >> LIC)
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
	const int W_k = (ty & STEP_m1) << 1;
	const int SW = (3 - tFW) << LIC;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll  
		for (int ik = 0; ik < STEP2; ++ik)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
	for (int ik = 0; ik < STEP2; ++ik)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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
#ifndef GSX_KERNEL3
#define GSX_KERNEL3

#define gsx_k3(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	GSX_Kernel3<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OH, OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, LIC, OC, sh, sw,\
			 oc_index, n_index, (GN>>LB>>3))

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.4047 msec, Performace = 13759.1 GFlop/s
//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.67307 msec, Performace = 11552.1 GFlop/s
//for (16, 16) -> (8, 8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.69252 msec, Performace = 11419.3 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [128,  256, 256] 
//LB = 4: Size = 9, Time = 1.73406 msec, Performace = 11145.7 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void GSX_Kernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y, 
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index, int GX)//GX
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][(2 << LB)][(1 << LB) + 1];
	__shared__ float4 Xs[2][(2 << LB)][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for bz -> OH * OW
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
	CW += ((fhs * 3 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;

	//prepare for GN = OC
	const int oc0 = (((bx << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	const int n0 = (((by << LB) + ty) << 3) + n_index;
	const int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	const int Xstride = (IH * IW) << LIC;
	const int X1 = X0 + Xstride;
	const int X2 = X1 + Xstride;
	const int X3 = X2 + Xstride;

	//compute area------------------------------------------------------
	const int SX = (IW - tFW) << LIC;
	const int SW = (3 - tFW) << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];//Idx =  (X_k >> LIC)
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

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll  
		for (int ik = 0; ik < STEP2; ++ik)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
	for (int ik = 0; ik < STEP2; ++ik)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	const int Y4 = Y0 + (Ystride << 2);
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
#ifndef GSX_KERNEL4
#define GSX_KERNEL4

#define gsx_k4(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	GSX_Kernel4<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OH, OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, LIC, OC, sh, sw,\
			 oc_index, n_index, (GN>>LB>>3))

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.4047 msec, Performace = 13759.1 GFlop/s
//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.67307 msec, Performace = 11552.1 GFlop/s
//for (16, 16) -> (8, 8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.69252 msec, Performace = 11419.3 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [128,  256, 256] 
//LB = 4: Size = 9, Time = 1.73406 msec, Performace = 11145.7 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void GSX_Kernel4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index, int GX)//GX
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][(2 << LB)][(1 << LB) + 1];
	__shared__ float4 Xs[2][(2 << LB)][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for bz -> OH * OW
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
	CW += ((fhs * 3 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;

	//prepare for GN = OC
	const int oc0 = (((bx << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	const int n0 = (((by << LB) + ty) << 3) + n_index;
	const int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	const int Xstride = (IH * IW) << LIC;//n0
	const int X1 = X0 + Xstride;//n1
	const int X2 = X1 + Xstride;//n2
	const int X3 = X2 + Xstride;//n3 

	//compute area------------------------------------------------------
	const int SX = (IW - tFW) << LIC, SW = (3 - tFW) << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;//with the same ty
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];//Idx =  (X_k >> LIC)
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
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll  
		for (int ik = 0; ik < STEP2; ++ik)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
	for (int ik = 0; ik < STEP2; ++ik)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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
#ifndef GSX_KERNEL5
#define GSX_KERNEL5

#define gsx_k5(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	GSX_Kernel5<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OH, OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, LIC, OC, sh, sw,\
			 oc_index, n_index, (GN>>LB>>3))

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.4047 msec, Performace = 13759.1 GFlop/s
//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.67307 msec, Performace = 11552.1 GFlop/s
//for (16, 16) -> (8, 8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.69252 msec, Performace = 11419.3 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [128,  256, 256] 
//LB = 4: Size = 9, Time = 1.8588 msec, Performace = 10397.7 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void GSX_Kernel5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index, int GX)//GX
{
	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GK = FH * FW * IC
	const int oh = blockIdx.z, ow = blockIdx.y;//ow = bz % OW
	const int toh = oh * sh - 1, tow = ow * sw - 1;
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;
	CW += ((fhs * 3 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]
	
	//prepare for (by, bx, tx, ty)
	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;
	const int tx = threadIdx.x, ty = threadIdx.y;

	//prepare for GN = OC
	const int oc0 = (((bx << LB) + tx) << 3) + oc_index;//8x
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	const int n0 = (((by << LB) + ty) << 3) + n_index;
	const int OH = gridDim.z, OW = gridDim.y;
	const int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	const int tn = n0 + (tx & 3) + ((tx >= STEP) << 2);
	const int X0 = ((tn*IH + toh)*IW + tow) << LIC;

	//compute area------------------------------------------------------
	const int SX = (IW - tFW) << LIC;//stride for X
	const int SW = (3 - tFW) << LIC;//stride for W

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = ((tx & STEP_m1) >> 2 << 2);//with the same ty
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];//Idx = (X_k >> LIC)
	const int xoffset = (fh * SX + X_k) + X0;
	float4 xv = *(float4*)(X + xoffset);
	const int Xs_x = ((tx & STEP_m1) >> 2 << 2) + ((tx >= STEP)*STEP);
	*((float*)(&Xs[buf][Xs_x + 0][ty]) + (tx & 3)) = xv.x;
	*((float*)(&Xs[buf][Xs_x + 1][ty]) + (tx & 3)) = xv.y;
	*((float*)(&Xs[buf][Xs_x + 2][ty]) + (tx & 3)) = xv.z;
	*((float*)(&Xs[buf][Xs_x + 3][ty]) + (tx & 3)) = xv.w;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1);
	const int woffset0 = (fh*SW + W_k)*OC;
	Ws[0][ty][tx] = *(float4*)(CW + woffset0);
	__syncthreads();

	for (int ok = STEP; ok < GK; ok += STEP)
	{
#pragma unroll  
		for (int ik = 0; ik < STEP; ++ik)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

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
		const int X_k = ok + ((tx & STEP_m1) >> 2 << 2);
		const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];
		const int xoffset = fh * SX + X_k + X0;
		float4 xv = *(float4*)(X + xoffset);
		*((float*)(&Xs[buf][Xs_x + 0][ty]) + (tx & 3)) = xv.x;
		*((float*)(&Xs[buf][Xs_x + 1][ty]) + (tx & 3)) = xv.y;
		*((float*)(&Xs[buf][Xs_x + 2][ty]) + (tx & 3)) = xv.z;
		*((float*)(&Xs[buf][Xs_x + 3][ty]) + (tx & 3)) = xv.w;

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + (ty & STEP_m1);
		const int woffset0 = (fh*SW + W_k)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ++ik)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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


#ifndef GSX_KERNEL6
#define GSX_KERNEL6

#define gsx_k6(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	GSX_Kernel6<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OH, OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, LIC, OC, sh, sw,\
			 oc_index, n_index, (GN>>LB>>3))

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.4047 msec, Performace = 13759.1 GFlop/s
//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.67307 msec, Performace = 11552.1 GFlop/s
//for (16, 16) -> (8, 8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.69252 msec, Performace = 11419.3 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [128,  256, 256] 
//LB = 4: Size = 9, Time = 1.85414 msec, Performace = 10423.9 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void GSX_Kernel6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index, int GX)//GX
{
	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GK = FH * FW * IC
	const int oh = blockIdx.z, ow = blockIdx.y;//ow = bz % OW
	const int toh = oh * sh - 1, tow = ow * sw - 1;
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;
	CW += ((fhs * 3 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]

	//prepare for (by, bx, tx, ty)
	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;
	const int tx = threadIdx.x, ty = threadIdx.y;

	//prepare for GN = OC
	const int oc0 = (((bx << LB) + tx) << 3) + oc_index;//8x
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	const int n0 = (((by << LB) + ty) << 3) + n_index;
	const int OH = gridDim.z, OW = gridDim.y;
	const int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	const int tn = n0 + (tx & 3) + ((tx >= STEP) << 2);
	X += ((tn*IH + toh)*IW + tow) << LIC;//X[tn, toh, tow, 0]

	//compute area------------------------------------------------------
	const int SX = (IW - tFW) << LIC;//stride for X
	const int SW = (3 - tFW) << LIC;//stride for W

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = ((tx & STEP_m1) >> 2 << 2);//with the same ty
	const int Xs_x = X_k + ((tx >= STEP) << LB >> 1);
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];//Idx = (X_k >> LIC)
	const int xoffset = (fh * SX + X_k);
	float4 xv = *(float4*)(X + xoffset);
	*((float*)(&Xs[0][Xs_x + 0][ty]) + (tx & 3)) = xv.x;
	*((float*)(&Xs[0][Xs_x + 1][ty]) + (tx & 3)) = xv.y;
	*((float*)(&Xs[0][Xs_x + 2][ty]) + (tx & 3)) = xv.z;
	*((float*)(&Xs[0][Xs_x + 3][ty]) + (tx & 3)) = xv.w;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1);
	const int woffset0 = (fh*SW + W_k)*OC;
	Ws[0][ty][tx] = *(float4*)(CW + woffset0);
	__syncthreads();

	for (int ok = STEP; ok < GK; ok += STEP)
	{
#pragma unroll  
		for (int ik = 0; ik < STEP; ++ik)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

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
		const int X_k = ok + ((tx & STEP_m1) >> 2 << 2);
		const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];
		const int xoffset = fh * SX + X_k;
		float4 xv = *(float4*)(X + xoffset);
		*((float*)(&Xs[buf][Xs_x + 0][ty]) + (tx & 3)) = xv.x;
		*((float*)(&Xs[buf][Xs_x + 1][ty]) + (tx & 3)) = xv.y;
		*((float*)(&Xs[buf][Xs_x + 2][ty]) + (tx & 3)) = xv.z;
		*((float*)(&Xs[buf][Xs_x + 3][ty]) + (tx & 3)) = xv.w;

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + (ty & STEP_m1);
		const int woffset0 = (fh*SW + W_k)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ++ik)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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


#ifndef GSX_KERNEL7
#define GSX_KERNEL7

#define gsx_k7(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	GSX_Kernel7<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OH, OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, LIC, OC, sh, sw,\
			 oc_index, n_index, (GN>>LB>>3))

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.4047 msec, Performace = 13759.1 GFlop/s
//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.67307 msec, Performace = 11552.1 GFlop/s
//for (16, 16) -> (8, 8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.69252 msec, Performace = 11419.3 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [128,  256, 256] 
//LB = 4: Size = 9, Time = 1.83558 msec, Performace = 10529.3 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void GSX_Kernel7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index, int GX)//GX
{
	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(4 << LB) + 4];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GK = FH * FW * IC
	const int oh = blockIdx.z, ow = blockIdx.y;
	const int toh = oh * sh - 1, tow = ow * sw - 1;
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;
	CW += ((fhs * 3 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]

	//prepare for (by, bx, tx, ty)
	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;
	const int tx = threadIdx.x, ty = threadIdx.y;

	//prepare for GN = OC
	const int oc0 = (((bx << LB) + tx) << 3) + oc_index;//8x
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	const int n0 = (((by << LB) + ty) << 3) + n_index;
	const int OH = gridDim.z, OW = gridDim.y;
	const int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	const int tn = n0 + (tx & 3) + ((tx >= STEP) << 2);
	X += ((tn*IH + toh)*IW + tow) << LIC;//X[tn, toh, tow, 0]

	//compute area------------------------------------------------------
	const int SX = (IW - tFW) << LIC;//stride for X
	const int SW = (3 - tFW) << LIC;//stride for W

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = ((tx & STEP_m1) >> 2 << 2);//with the same ty
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];//Idx = (X_k >> LIC)
	const int xoffset = (fh * SX + X_k);
	float4 xv = *(float4*)(X + xoffset);
	const int Xs_x = X_k + ((tx >= STEP) << LB >> 1);
	const int Xs_y = (ty << 2) + (tx & 3);
	Xs[0][Xs_x + 0][Xs_y] = xv.x;
	Xs[0][Xs_x + 1][Xs_y] = xv.y;
	Xs[0][Xs_x + 2][Xs_y] = xv.z;
	Xs[0][Xs_x + 3][Xs_y] = xv.w;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1);
	const int woffset0 = (fh*SW + W_k)*OC;
	Ws[0][ty][tx] = *(float4*)(CW + woffset0);
	__syncthreads();

	for (int ok = STEP; ok < GK; ok += STEP)
	{
#pragma unroll  
		for (int ik = 0; ik < STEP; ++ik)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
			float4 b0 = *(float4*)(&Xs[buf][ik       ][ty << 2]);
			float4 b1 = *(float4*)(&Xs[buf][ik + STEP][ty << 2]);

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
		const int X_k = ok + ((tx & STEP_m1) >> 2 << 2);
		const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];
		const int xoffset = fh * SX + X_k;
		float4 xv = *(float4*)(X + xoffset);
		Xs[buf][Xs_x + 0][Xs_y] = xv.x;
		Xs[buf][Xs_x + 1][Xs_y] = xv.y;
		Xs[buf][Xs_x + 2][Xs_y] = xv.z;
		Xs[buf][Xs_x + 3][Xs_y] = xv.w;

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + (ty & STEP_m1);
		const int woffset0 = (fh*SW + W_k)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ++ik)
	{
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
		float4 b0 = *(float4*)(&Xs[buf][ik       ][ty << 2]);
		float4 b1 = *(float4*)(&Xs[buf][ik + STEP][ty << 2]);

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


#ifndef GSX_KERNEL8
#define GSX_KERNEL8

#define gsx_k8(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	GSX_Kernel8<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), OH, OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, LIC, OC, sh, sw,\
			 oc_index, n_index, (GN>>LB>>3))

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.4047 msec, Performace = 13759.1 GFlop/s
//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.67307 msec, Performace = 11552.1 GFlop/s
//for (16, 16) -> (8, 8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.69252 msec, Performace = 11419.3 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [128,  256, 256] 
//LB = 4: Size = 9, Time = 1.82304 msec, Performace = 10601.7 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void GSX_Kernel8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index, int GX)//GX
{
	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(4 << LB) + 4];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GK = FH * FW * IC
	const int oh = blockIdx.z, ow = blockIdx.y;
	const int toh = oh * sh - 1, tow = ow * sw - 1;
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;
	CW += ((fhs * 3 + fws)*OC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]

	//prepare for (by, bx, tx, ty)
	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;
	const int tx = threadIdx.x, ty = threadIdx.y;

	//prepare for GN = OC
	const int oc0 = (((bx << LB) + tx) << 3) + oc_index;//8x
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	const int n0 = (((by << LB) + ty) << 3) + n_index;
	const int OH = gridDim.z, OW = gridDim.y;
	const int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	const int tn = n0 + (tx & 3) + ((tx >= STEP) << 2);
	X += ((tn*IH + toh)*IW + tow) << LIC;//X[tn, toh, tow, 0]

	//compute area------------------------------------------------------
	const int SX = (IW - tFW) << LIC;//stride for X
	const int SW = (3 - tFW) << LIC;//stride for W

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1);
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (W_k >> LIC)];//Idx = (X_k >> LIC)
	const int woffset0 = (fh*SW + W_k)*OC;
	Ws[0][ty][tx] = *(float4*)(CW + woffset0);

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = ((tx & STEP_m1) >> 2 << 2);//with the same ty
	const int xoffset = (fh * SX + X_k);
	float4 xv = *(float4*)(X + xoffset);
	const int Xs_x = X_k + ((tx >= STEP) << LB >> 1);
	const int Xs_y = (ty << 2) + (tx & 3);
	Xs[0][Xs_x][Xs_y] = xv.x;
	Xs[0][Xs_x + 1][Xs_y] = xv.y;
	Xs[0][Xs_x + 2][Xs_y] = xv.z;
	Xs[0][Xs_x + 3][Xs_y] = xv.w;
	__syncthreads();

	for (int ok = STEP; ok < GK; ok += STEP)
	{
#pragma unroll  
		for (int ik = 0; ik < STEP; ++ik)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
			float4 b0 = *(float4*)(&Xs[buf][ik][ty << 2]);
			float4 b1 = *(float4*)(&Xs[buf][ik + STEP][ty << 2]);

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

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + (ty & STEP_m1);
		const int fh = XIDX_V2_W3P1_FH[fhw_offset + (W_k >> LIC)];
		const int woffset0 = (fh*SW + W_k)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset0);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) >> 2 << 2);
		const int xoffset = fh * SX + X_k;
		float4 xv = *(float4*)(X + xoffset);
		Xs[buf][Xs_x][Xs_y] = xv.x;
		Xs[buf][Xs_x + 1][Xs_y] = xv.y;
		Xs[buf][Xs_x + 2][Xs_y] = xv.z;
		Xs[buf][Xs_x + 3][Xs_y] = xv.w;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ++ik)
	{
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

		float4 b0 = *(float4*)(&Xs[buf][ik][ty << 2]);
		float4 b1 = *(float4*)(&Xs[buf][ik + STEP][ty << 2]);

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

