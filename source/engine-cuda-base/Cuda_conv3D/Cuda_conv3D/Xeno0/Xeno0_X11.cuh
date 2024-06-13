


//======[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef OSX_KERNEL1
#define OSX_KERNEL1

#define osx_k1(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, LOC, sh, sw, GN, N) \
	OSX_Kernel1<LB, (1<<LB>>1), (1 << LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, LOC, sh, sw,\
			 oc_index, n_index)

//for (IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 2.30918 msec, Performace = 2092.45 GFlop/s
//LB = 3: Size = 2.25, Time = 2.41398 msec, Performace = 2001.61 GFlop/s
//for (IH, IW) = (8, 8)
//LB = 4: Size = 9, Time = 1.69124 msec, Performace = 11427.9 GFlop/s
//LB = 3: Size = 2.25, Time = 2.99055 msec, Performace = 1615.7  GFlop/s
//for (16, 16) -> ( 8,  8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.71471 msec, Performace = 11271.5 GFlop/s
//LB = 3: Size = 2.25, Time = 3.15798 msec, Performace = 1530.04 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [512,  64, 256] 
//LB = 4: Size = 9, Time = 1.75179 msec, Performace = 11032.9 GFlop/s
//LB = 3: Size = 9, Time = 1.97126 msec, Performace =  9804.56 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OSX_Kernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y, int OH, int OW,
	int LIC, int LOC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - 1, tow = ow * sw - 1;

	//prepare for GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	CW += ((fhs * 3 + fws) << LOC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	const int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	const int Y0 = (((n0*OH + oh)*OW + ow) << LOC) + oc0;
	const int tn1 = n0 + ((tx >= STEP) << 2) + 1;
	X += ((tn1*IH + toh)*IW + tow) << LIC;//[tn0, toh, tow, 0]
	const int Xstride = (IH * IW) << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int SX = (IW - tFW) << LIC;
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];//Idx =  (X_k >> LIC)
	const int xoffset = fh * SX + X_k;
	float2 x0 = *(float2*)(X + (xoffset - Xstride));
	float2 x1 = *(float2*)(X + xoffset);
	float2 x2 = *(float2*)(X + (xoffset + Xstride));
	float2 x3 = *(float2*)(X + (xoffset + (Xstride << 1)));
	Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int SW = (3 - tFW) << LIC;
	const int woffset0 = (fh*SW + W_k) << LOC;
	Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + (woffset0 + (1 << LOC)));
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
		const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];
		const int xoffset = fh * SX + X_k;
		float2 x0 = *(float2*)(X + xoffset - Xstride);
		float2 x1 = *(float2*)(X + xoffset);
		float2 x2 = *(float2*)(X + xoffset + Xstride);
		float2 x3 = *(float2*)(X + xoffset + (Xstride << 1));
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fh*SW + W_k) << LOC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + (woffset0 + (1 << LOC)));
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0,  b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2,  b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4,  b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6,  b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8,  b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	const int Ystride = (OH * OW) << LOC;
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



//=========[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, IC % 8 == 0
#ifndef OSX_KERNEL2
#define OSX_KERNEL2

#define osx_k2(stream, LB, oc_index, n_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	OS_kernel2<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, \
			 oc_index, n_index)

//Size = 9, Time = 2.13534 msec, Performace = 9051.19 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void OS_kernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//tFH = tFW = 3, 2
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - 1, tow = ow * sw - 1;

	//prepare for GK = FH * FW * IC
	int fhs = -IF_int((toh < 0), toh, 0);
	int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	W += (fhs * 3 + fws) << LIC;//W[0, fhs, fws, 0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	const int Wstride = 9 << LIC;
	const int toc0 = (oc0 + ((ty >= STEP) << 2)) * Wstride;
	const int toc1 = toc0 + Wstride, toc2 = toc1 + Wstride, toc3 = toc2 + Wstride;

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	int Xstride = (IH * IW) << LIC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty & STEP_m1);
	int Idx = W_k >> LIC; int fh = XIDX_V2_W3P1[fhw_offset + Idx] >> 2;
	const int SW = (3 - tFW) << LIC;
	float4 w; int woffset = fh * SW + W_k;//X_fh = W_fh = fh
	w.x = W[toc0 + woffset];
	w.y = W[toc1 + woffset];
	w.z = W[toc2 + woffset];
	w.w = W[toc3 + woffset];
	Ws[buf][ty][tx] = w;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & STEP_m1);
	const int SX = (IW - tFW) << LIC;
	float4 x; int xoffset = fh * SX + X_k;
	x.x = X[X0 + xoffset];
	x.y = X[X1 + xoffset];
	x.z = X[X2 + xoffset];
	x.w = X[X3 + xoffset];
	Xs[buf][tx][ty] = x;
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = STEP; ok < GK; ok += STEP)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
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
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ok + (ty & STEP_m1);
		int Idx = W_k >> LIC; int fh = XIDX_V2_W3P1[fhw_offset + Idx] >> 2;
		float4 w; int woffset = fh * SW + W_k;//X_fh = W_fh = fh
		w.x = W[toc0 + woffset];
		w.y = W[toc1 + woffset];
		w.z = W[toc2 + woffset];
		w.w = W[toc3 + woffset];
		Ws[buf][ty][tx] = w;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ok + (tx & STEP_m1);
		float4 x; int xoffset = fh * SX + X_k;
		x.x = X[X0 + xoffset];
		x.y = X[X1 + xoffset];
		x.z = X[X2 + xoffset];
		x.w = X[X3 + xoffset];
		Xs[buf][tx][ty] = x;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
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

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

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



//=========[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, IC % 8 == 0
#ifndef OSX_KERNEL3
#define OSX_KERNEL3

#define osx_k3(stream, LB, oc_index, n_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	OS_kernel3<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, \
			 oc_index, n_index)

//Size = 9, Time = 2.13534 msec, Performace = 9051.19 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OS_kernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//tFH = tFW = 3, 2
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - 1, tow = ow * sw - 1;

	//prepare for GK = FH * FW * IC
	int fhs = -IF_int((toh < 0), toh, 0);
	int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	W += (fhs * 3 + fws) << LIC;//W[0, fhs, fws, 0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	const int Wstride = 9 << LIC;
	const int toc0 = (oc0 + ((ty >= STEP) << 2)) * Wstride;
	const int toc1 = toc0 + Wstride, toc2 = toc1 + Wstride, toc3 = toc2 + Wstride;

	//prepare for GM = N
	int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	int Y0 = ((n0*OH + oh)*OW + ow)*OC + oc0;
	int tn0 = n0 + ((tx >= STEP) << 2);
	int X0 = ((tn0*IH + toh)*IW + tow) << LIC;
	int Xstride = (IH * IW) << LIC;
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int fh = XIDX_V2_W3P1[fhw_offset + (W_k >> LIC)] >> 2;
	const int SW = (3 - tFW) << LIC;
	const int woffset = fh * SW + W_k;//X_fh = W_fh = fh
	float2 w0 = *(float2*)(W + toc0 + woffset);
	float2 w1 = *(float2*)(W + toc1 + woffset);
	float2 w2 = *(float2*)(W + toc2 + woffset);
	float2 w3 = *(float2*)(W + toc3 + woffset);
	Ws[buf][(ty << 1)    ][tx] = float4{ w0.x, w1.x, w2.x, w3.x };
	Ws[buf][(ty << 1) + 1][tx] = float4{ w0.y, w1.y, w2.y, w3.y };

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int SX = (IW - tFW) << LIC;
	const int xoffset = fh * SX + X_k;
	float2 x0 = *(float2*)(X + X0 + xoffset);
	float2 x1 = *(float2*)(X + X1 + xoffset);
	float2 x2 = *(float2*)(X + X2 + xoffset);
	float2 x3 = *(float2*)(X + X3 + xoffset);
	Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ok + ((ty & STEP_m1) << 1);
		int Idx = W_k >> LIC; int fh = XIDX_V2_W3P1[fhw_offset + Idx] >> 2;
		const int woffset = fh * SW + W_k;//X_fh = W_fh = fh
		float2 w0 = *(float2*)(W + toc0 + woffset);
		float2 w1 = *(float2*)(W + toc1 + woffset);
		float2 w2 = *(float2*)(W + toc2 + woffset);
		float2 w3 = *(float2*)(W + toc3 + woffset);
		Ws[buf][(ty << 1)    ][tx] = float4{ w0.x, w1.x, w2.x, w3.x };
		Ws[buf][(ty << 1) + 1][tx] = float4{ w0.y, w1.y, w2.y, w3.y };

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int xoffset = fh * SX + X_k;
		float2 x0 = *(float2*)(X + X0 + xoffset);
		float2 x1 = *(float2*)(X + X1 + xoffset);
		float2 x2 = *(float2*)(X + X2 + xoffset);
		float2 x3 = *(float2*)(X + X3 + xoffset);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
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

	int Ystride = OH * OW * OC;
	int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;
	int Y4 = Y3 + Ystride, Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

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
#ifndef OSX_KERNEL4
#define OSX_KERNEL4

#define osx_k4(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, LOC, sh, sw, GN, N) \
	OSX_Kernel4<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,  (1<<LB)+1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, LOC, sh, sw,\
			 oc_index, n_index)

//for (IH, IW) = (4, 4)
//LB = 4: Size = 2.25, Time = 2.30918 msec, Performace = 2092.45 GFlop/s
//LB = 3: Size = 2.25, Time = 2.41398 msec, Performace = 2001.61 GFlop/s
//for (IH, IW) = (8, 8)
//LB = 4: Size = 9, Time = 1.69124 msec, Performace = 11427.9 GFlop/s
//LB = 3: Size = 2.25, Time = 2.99055 msec, Performace = 1615.7  GFlop/s
//for (16, 16) -> ( 8,  8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.71471 msec, Performace = 11271.5 GFlop/s
//LB = 3: Size = 2.25, Time = 3.15798 msec, Performace = 1530.04 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [512,  64, 256] 
//LB = 4: Size = 9, Time = 1.75179 msec, Performace = 11032.9 GFlop/s
//LB = 3: Size = 9, Time = 1.97126 msec, Performace =  9804.56 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int MS>
__global__ void OSX_Kernel4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y, int OH, int OW,
	int LIC, int LOC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for bz -> OH * OW
	int bz = blockIdx.z;
	int oh = bz / OW, ow = bz - oh * OW;//ow = bz % OW
	int toh = oh * sh - 1, tow = ow * sw - 1;

	//prepare for GK = FH * FW * IC
	const int fhs = -IF_int((toh < 0), toh, 0);
	const int fws = -IF_int((tow < 0), tow, 0);
	int fhe = IH - toh; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = IW - tow; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int GK = (tFH * tFW) << LIC;
	CW += ((fhs * 3 + fws) << LOC) << LIC;//CW[fhs, fws, 0, oc0]
	X += (fhs*IW + fws) << LIC;//X[0, fhs, fws, 0]
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N
	const int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	const int Y0 = (((n0*OH + oh)*OW + ow) << LOC) + oc0;
	const int tn1 = n0 + ((tx >= STEP) << 2) + 1;
	X += ((tn1*IH + toh)*IW + tow) << LIC;//[tn0, toh, tow, 0]
	const int Xstride = (IH * IW) << LIC;

	const int ux = tx << 1;
	const int uy = ty << 1;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & STEP_m1) << 1;
	const int SX = (IW - tFW) << LIC;
	const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];//Idx =  (X_k >> LIC)
	const int xoffset = fh * SX + X_k;
	float2 x0 = *(float2*)(X + (xoffset - Xstride));
	float2 x1 = *(float2*)(X + xoffset);
	float2 x2 = *(float2*)(X + (xoffset + Xstride));
	float2 x3 = *(float2*)(X + (xoffset + (Xstride << 1)));
	Xs[buf][ux][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[buf][ux + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty & STEP_m1) << 1;
	const int SW = (3 - tFW) << LIC;
	const int woffset0 = (fh*SW + W_k) << LOC;
	Ws[buf][uy][tx] = *(float4*)(CW + woffset0);
	Ws[buf][uy + 1][tx] = *(float4*)(CW + (woffset0 + (1 << LOC)));
	X_k += STEP2; W_k += STEP2;
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = STEP2; ok < (GK >> LB); ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
		//const int X_k = ok + ((tx & STEP_m1) << 1);
		const int fh = XIDX_V2_W3P1_FH[fhw_offset + (X_k >> LIC)];
		const int xoffset = fh * SX + X_k;
		float2 x0 = *(float2*)(X + xoffset - Xstride);
		float2 x1 = *(float2*)(X + xoffset);
		float2 x2 = *(float2*)(X + xoffset + Xstride);
		float2 x3 = *(float2*)(X + xoffset + (Xstride << 1));
		Xs[buf][ux][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][ux + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		//const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fh*SW + W_k) << LOC;
		Ws[buf][uy][tx] = *(float4*)(CW + woffset0);
		Ws[buf][uy + 1][tx] = *(float4*)(CW + (woffset0 + (1 << LOC)));
		X_k += STEP2; W_k += STEP2;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
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

	const int Ystride = (OH * OW) << LOC;
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
#ifndef OSX_KERNEL5
#define OSX_KERNEL5

#define osx_k5(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	OSX_kernel5<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw,\
			 oc_index, n_index)

//using namespace nvcuda;

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.4047 msec, Performace = 13759.1 GFlop/s
//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.67307 msec, Performace = 11552.1 GFlop/s
//for (16, 16) -> (8, 8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.69252 msec, Performace = 11419.3 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [128,  256, 256] 
//LB = 4: Size = 9, Time = 1.74843 msec, Performace = 11054.1 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OSX_kernel5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ __align__(32 << LB << LB) float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ __align__(32 << LB << LB) float4 Xs[2][2 << LB][(1 << LB) + 1];

	//wmma::fragment<wmma:matrix_a, 8, 8, 16, float, wmma::row_major> X_frag;
	//wmma::fragment<wmma:matrix_b, 8, 8, 16, float, wmma::col_major> W_frag;

	//prepare for bz -> OH * OW      sss
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
	int X_k = (tx & STEP_m1) << 1;
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

	Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty & STEP_m1) << 1;
	const int SW = (3 - tFW) << LIC;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4 v[16]; memset(v, 0, sizeof(float4 ) * 16);
	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

			simdMM4(v[0], b0.x, a0);  simdMM4(v[1], b0.x, a1);
			simdMM4(v[2], b0.y, a0);  simdMM4(v[3], b0.y, a1);
			simdMM4(v[4], b0.z, a0);  simdMM4(v[5], b0.z, a1);
			simdMM4(v[6], b0.w, a0);  simdMM4(v[7], b0.w, a1);
			simdMM4(v[8], b1.x, a0);  simdMM4(v[9], b1.x, a1);
			simdMM4(v[10], b1.y, a0); simdMM4(v[11], b1.y, a1);
			simdMM4(v[12], b1.z, a0); simdMM4(v[13], b1.z, a1);
			simdMM4(v[14], b1.w, a0); simdMM4(v[15], b1.w, a1);
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
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v[0], b0.x, a0);  simdMM4(v[1], b0.x, a1);
		simdMM4(v[2], b0.y, a0);  simdMM4(v[3], b0.y, a1);
		simdMM4(v[4], b0.z, a0);  simdMM4(v[5], b0.z, a1);
		simdMM4(v[6], b0.w, a0);  simdMM4(v[7], b0.w, a1);
		simdMM4(v[8], b1.x, a0);  simdMM4(v[9], b1.x, a1);
		simdMM4(v[10], b1.y, a0); simdMM4(v[11], b1.y, a1);
		simdMM4(v[12], b1.z, a0); simdMM4(v[13], b1.z, a1);
		simdMM4(v[14], b1.w, a0); simdMM4(v[15], b1.w, a1);
	}

	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v[0];  *(float4*)(Y + Y0 + 4) = v[1];
	*(float4*)(Y + Y1) = v[2];  *(float4*)(Y + Y1 + 4) = v[3];
	*(float4*)(Y + Y2) = v[4];  *(float4*)(Y + Y2 + 4) = v[5];
	*(float4*)(Y + Y3) = v[6];  *(float4*)(Y + Y3 + 4) = v[7];
	*(float4*)(Y + Y4) = v[8];  *(float4*)(Y + Y4 + 4) = v[9];
	*(float4*)(Y + Y5) = v[10]; *(float4*)(Y + Y5 + 4) = v[11];
	*(float4*)(Y + Y6) = v[12]; *(float4*)(Y + Y6 + 4) = v[13];
	*(float4*)(Y + Y7) = v[14]; *(float4*)(Y + Y7 + 4) = v[15];
}

#endif


//======[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef OSX_KERNEL6
#define OSX_KERNEL6

#define osx_k6(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	OSX_kernel6<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, OH*OW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw,\
			 oc_index, n_index)

//using namespace nvcuda;

//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 512, 1024] 
//LB = 4: Size = 9, Time = 1.4047 msec, Performace = 13759.1 GFlop/s
//for (8, 8) -> (4, 4), [N, IC, OC] = [512, 256, 512] 
//LB = 4: Size = 9, Time = 1.67307 msec, Performace = 11552.1 GFlop/s
//for (16, 16) -> (8, 8), [N, IC, OC] = [512, 256, 256] 
//LB = 4: Size = 9, Time = 1.69252 msec, Performace = 11419.3 GFlop/s
//for (32, 32) -> (16, 16), [N, IC, OC] = [128,  256, 256] 
//LB = 4: Size = 9, Time = 1.74843 msec, Performace = 11054.1 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OSX_kernel6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ __align__(32 << LB << LB) float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ __align__(32 << LB << LB) float4 Xs[2][2 << LB][(1 << LB) + 1];
	float4 v[16]; memset(v, 0, sizeof(float4) * 16);

	//wmma::fragment<wmma:matrix_a, 8, 8, 16, float, wmma::row_major> X_frag;
	//wmma::fragment<wmma:matrix_b, 8, 8, 16, float, wmma::col_major> W_frag;
//	wmma::fragment<wmma:accumulator, 8, 8, 16, float, wmma::row_major> Y_flag;

	//prepare for bz -> OH * OW      sss
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
	int X_k = (tx & STEP_m1) << 1;
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

	Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty & STEP_m1) << 1;
	const int SW = (3 - tFW) << LIC;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
		//wmma::load_matrix_sync(X_frag, &Xs[buf][ik][ty], 4);
		//wmma::load_matrix_sync(W_frag, &Ws[buf][ik][tx], 4);
		//wmma::mma_sync(Y_frag, X_frag, W_frag, C_frag);

#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a[2] = { Ws[buf][ik][tx], Ws[buf][ik + STEP2][tx] };
			float4 b[2] = { Xs[buf][ik][ty], Xs[buf][ik + STEP2][ty] };

			simdMM4(v[0], b[0].x, a[0]);  simdMM4(v[1], b[0].x, a[1]);
			simdMM4(v[2], b[0].y, a[0]);  simdMM4(v[3], b[0].y, a[1]);
			simdMM4(v[4], b[0].z, a[0]);  simdMM4(v[5], b[0].z, a[1]);
			simdMM4(v[6], b[0].w, a[0]);  simdMM4(v[7], b[0].w, a[1]);
			simdMM4(v[8], b[1].x, a[0]);  simdMM4(v[9], b[1].x, a[1]);
			simdMM4(v[10], b[1].y, a[0]); simdMM4(v[11], b[1].y, a[1]);
			simdMM4(v[12], b[1].z, a[0]); simdMM4(v[13], b[1].z, a[1]);
			simdMM4(v[14], b[1].w, a[0]); simdMM4(v[15], b[1].w, a[1]);
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
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a[2] = { Ws[buf][ik][tx], Ws[buf][ik + STEP2][tx] };
		float4 b[2] = { Xs[buf][ik][ty], Xs[buf][ik + STEP2][ty] };

		simdMM4(v[0], b[0].x, a[0]);  simdMM4(v[1], b[0].x, a[1]);
		simdMM4(v[2], b[0].y, a[0]);  simdMM4(v[3], b[0].y, a[1]);
		simdMM4(v[4], b[0].z, a[0]);  simdMM4(v[5], b[0].z, a[1]);
		simdMM4(v[6], b[0].w, a[0]);  simdMM4(v[7], b[0].w, a[1]);
		simdMM4(v[8], b[1].x, a[0]);  simdMM4(v[9], b[1].x, a[1]);
		simdMM4(v[10], b[1].y, a[0]); simdMM4(v[11], b[1].y, a[1]);
		simdMM4(v[12], b[1].z, a[0]); simdMM4(v[13], b[1].z, a[1]);
		simdMM4(v[14], b[1].w, a[0]); simdMM4(v[15], b[1].w, a[1]);
	}

	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride, Y4 = Y3 + Ystride;
	const int Y5 = Y4 + Ystride, Y6 = Y5 + Ystride, Y7 = Y6 + Ystride;

	*(float4*)(Y + Y0) = v[0];  *(float4*)(Y + Y0 + 4) = v[1];
	*(float4*)(Y + Y1) = v[2];  *(float4*)(Y + Y1 + 4) = v[3];
	*(float4*)(Y + Y2) = v[4];  *(float4*)(Y + Y2 + 4) = v[5];
	*(float4*)(Y + Y3) = v[6];  *(float4*)(Y + Y3 + 4) = v[7];
	*(float4*)(Y + Y4) = v[8];  *(float4*)(Y + Y4 + 4) = v[9];
	*(float4*)(Y + Y5) = v[10]; *(float4*)(Y + Y5 + 4) = v[11];
	*(float4*)(Y + Y6) = v[12]; *(float4*)(Y + Y6 + 4) = v[13];
	*(float4*)(Y + Y7) = v[14]; *(float4*)(Y + Y7 + 4) = v[15];
}

#endif


//======[ph = pw = 1], [FH = FW = 3], [IH, IW > 1]===========================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef OSX_KERNEL7
#define OSX_KERNEL7


#define osx_k7(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	OSX_kernel7<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
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
__global__  void OSX_kernel7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][(2 << LB << LB) + 1];//[2 << LB][1 << LB]
	__shared__ float4 Xs[2][(2 << LB << LB) + 1];//[2 << LB][1 << LB]

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
	Xs[buf][(tx << 1 << LB) + ty        ] = float4{ x0.x, x1.x, x2.x, x3.x };//[(tx<<1)    , ty]
	Xs[buf][(tx << 1 << LB) + ty + STEP2] = float4{ x0.y, x1.y, x2.y, x3.y };//[(tx<<1) + 1, ty]

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int SW = (3 - tFW) << LIC;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1 << LB) + tx        ] = *(float4*)(CW + woffset0);//[(ty<<1)   , tx]
	Ws[buf][(ty << 1 << LB) + tx + STEP2] = *(float4*)(CW + woffset1);//[(ty<<1) + 1, tx]
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][(ik << LB) + tx], a1 = Ws[buf][((ik + STEP2) << LB) + tx];
			float4 b0 = Xs[buf][(ik << LB) + ty], b1 = Xs[buf][((ik + STEP2) << LB) + ty];

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
		Xs[buf][(tx << 1 << LB) + ty        ] = float4{ x0.x, x1.x, x2.x, x3.x };//[(tx<<1)    , ty]
		Xs[buf][(tx << 1 << LB) + ty + STEP2] = float4{ x0.y, x1.y, x2.y, x3.y };//[(tx<<1) + 1, ty]

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fh*SW + W_k)*OC;
		const int woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1 << LB) + tx        ] = *(float4*)(CW + woffset0);//[(ty<<1)   , tx]
		Ws[buf][(ty << 1 << LB) + tx + STEP2] = *(float4*)(CW + woffset1);//[(ty<<1) + 1, tx]
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][(ik << LB) + ty], b1 = Xs[buf][((ik + STEP2) << LB) + ty];
		float4 a0 = Ws[buf][(ik << LB) + tx], a1 = Ws[buf][((ik + STEP2) << LB) + tx];

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
#ifndef OSX_KERNEL8
#define OSX_KERNEL8

#define osx_k8(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	OSX_kernel8<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
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
__global__  void OSX_kernel8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][(2 << LB << LB) + 1];//[2 << LB][1 << LB]
	__shared__ float4 Xs[2][(2 << LB << LB) + 1];//[2 << LB][1 << LB]

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
	Xs[buf][(tx << 1 << LB) + ty + 1] = float4{ x0.x, x1.x, x2.x, x3.x };//[(tx<<1)    , ty]
	Xs[buf][(tx << 1 << LB) + ty + STEP2 + 1] = float4{ x0.y, x1.y, x2.y, x3.y };//[(tx<<1) + 1, ty]

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int SW = (3 - tFW) << LIC;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1 << LB) + tx + 1] = *(float4*)(CW + woffset0);//[(ty<<1)   , tx]
	Ws[buf][(ty << 1 << LB) + tx + STEP2 + 1] = *(float4*)(CW + woffset1);//[(ty<<1) + 1, tx]
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][(ik << LB) + tx + 1], a1 = Ws[buf][((ik + STEP2) << LB) + tx + 1];
			float4 b0 = Xs[buf][(ik << LB) + ty + 1], b1 = Xs[buf][((ik + STEP2) << LB) + ty + 1];

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
		Xs[buf][(tx << 1 << LB) + ty + 1] = float4{ x0.x, x1.x, x2.x, x3.x };//[(tx<<1)    , ty]
		Xs[buf][(tx << 1 << LB) + ty + STEP2 + 1] = float4{ x0.y, x1.y, x2.y, x3.y };//[(tx<<1) + 1, ty]

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fh*SW + W_k)*OC;
		const int woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1 << LB) + tx + 1] = *(float4*)(CW + woffset0);//[(ty<<1)   , tx]
		Ws[buf][(ty << 1 << LB) + tx + STEP2 + 1] = *(float4*)(CW + woffset1);//[(ty<<1) + 1, tx]
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Xs[buf][(ik << LB) + ty + 1], b1 = Xs[buf][((ik + STEP2) << LB) + ty + 1];
		float4 a0 = Ws[buf][(ik << LB) + tx + 1], a1 = Ws[buf][((ik + STEP2) << LB) + tx + 1];

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


#ifndef OSX_KERNEL9
#define OSX_KERNEL9

#define osx_k9(stream, LB, oc_index, n_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, GN, N) \
	OSX_kernel9<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, (1<<LB<<LB)>\
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
template<int LB, int STEP, int STEP2, int STEP_m1, int IK_END>
__global__  void OSX_kernel9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//tFH = tFW = {3, 2}
	float* __restrict__ Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw,//ph = pw = 1
	int oc_index, int n_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][(2 << LB << LB) + 1];//[2 << LB][1 << LB]
	__shared__ float4 Xs[2][(2 << LB << LB) + 1];//[2 << LB][1 << LB]

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

	const int Xs_idx = (tx << 1 << LB) + ty + 1;
	const int Ws_idx = (ty << 1 << LB) + tx + 1;

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
	Xs[buf][Xs_idx] = float4{ x0.x, x1.x, x2.x, x3.x };//[(tx<<1)    , ty]
	Xs[buf][Xs_idx + STEP2] = float4{ x0.y, x1.y, x2.y, x3.y };//[(tx<<1) + 1, ty]

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int SW = (3 - tFW) << LIC;
	const int woffset0 = (fh*SW + W_k)*OC;
	const int woffset1 = woffset0 + OC;
	Ws[buf][Ws_idx] = *(float4*)(CW + woffset0);//[(ty<<1)   , tx]
	Ws[buf][Ws_idx + STEP2] = *(float4*)(CW + woffset1);//[(ty<<1) + 1, tx]
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 1; ik <= IK_END; ik += STEP2)
		{
			float4 a0 = Ws[buf][ik + tx], a1 = Ws[buf][ik + IK_END + tx];
			float4 b0 = Xs[buf][ik + ty], b1 = Xs[buf][ik + IK_END + ty];

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
		Xs[buf][Xs_idx] = float4{ x0.x, x1.x, x2.x, x3.x };//[(tx<<1)    , ty]
		Xs[buf][Xs_idx + STEP2] = float4{ x0.y, x1.y, x2.y, x3.y };//[(tx<<1) + 1, ty]

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fh*SW + W_k)*OC;
		const int woffset1 = woffset0 + OC;
		Ws[buf][Ws_idx] = *(float4*)(CW + woffset0);//[(ty<<1)   , tx]
		Ws[buf][Ws_idx + STEP2] = *(float4*)(CW + woffset1);//[(ty<<1) + 1, tx]
		__syncthreads();
	}
#pragma unroll
	for (int ik = 1; ik <= IK_END; ik += STEP2)
	{
		float4 a0 = Ws[buf][ik + tx], a1 = Ws[buf][ik + IK_END + tx];
		float4 b0 = Xs[buf][ik + ty], b1 = Xs[buf][ik + IK_END + ty];

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