#pragma once

#ifndef CONV_3D_GEMMR_UERNEL_EX_H
#define CONV_3D_GEMMR_UERNEL_EX_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV_3D_GEMMR_UERNEL_EX_CALL
#define CONV_3D_GEMMR_UERNEL_EX_CALL

//LB = log2(BLOCK_SIZE)

//======[FH = FW = 3]=========================================
#define conv3dGemm_u88RW3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RW3<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#define conv3dGemm_u88RW3_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RW3_IC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//------[(OH, OW) % 4 == 0]-----------------------------------
#define conv3dGemm_u88R4W3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W3<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#define conv3dGemm_u88R4W3_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W3_IC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//======[FH = FW = 5]=========================================
#define conv3dGemm_u88RW5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RW5<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#define conv3dGemm_u88RW5_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RW5_IC2pow<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//------[(OH, OW) % 4 == 0]-----------------------------------
#define conv3dGemm_u88R4W5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W5<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#define conv3dGemm_u88R4W5_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W5_IC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)


//======[FH = FW = 7]=========================================
//------[(OH, OW) % 4 == 0]-----------------------------------
#define conv3dGemm_u88R4W7_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W7_IC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

#endif


//======[FH = FW = 3]=========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RW3
#define CONV_3D_GEMM_UERNEL_8_8RW3

//LB = 4: Size = 1.125, Time = 1.63056 msec, Performace = 1481.65 GFlop/s
//LB = 3: Size = 1.125, Time = 1.76776 msec, Performace = 1366.66 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8RW3(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
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
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0, x1, x2, x3;
		x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
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

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RW3_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8RW3_IC2POW

//LB = 4: Size = 1.125, Time = 1.59093 msec, Performace = 1518.56 GFlop/s
//LB = 3: Size = 1.125, Time = 1.77885 msec, Performace = 1358.13 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8RW3_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
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
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
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

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//------[(OH, OW) % 4 == 0]-----------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0 
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W3
#define CONV_3D_GEMM_UERNEL_8_8R4W3

//LB = 4: Size = 1.125, Time = 1.5932  msec, Performace = 1516.39 GFlop/s
//LB = 3: Size = 1.125, Time = 1.75219 msec, Performace = 1378.8 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8R4W3(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) * IC;//X += X1;
	const int sw_IC = sw * IC;

	//prepare for GK = FH * FW * IC
	const int GK = 9 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = (X_fh*IW + X_fw - Idx)*IC + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int woffset = (ty - ((ty >= STEP) << LB >> 1) << 1) * OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
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
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int woffset = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1) * OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k / IC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = (X_fh*IW + X_fw - Idx)*IC + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
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

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W3_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8R4W3_IC2POW

//for: (64, 64) -> (32, 32), [FH, FW] = 3, [N, IC, OC] = [128, 64, 128]
//LB = 4: Size = 9, Time = 1.87748 msec, Performace = 10294.3 GFlop/s
//LB = 3: Size = 9, Time = 2.15023 msec, Performace =  8988.5 GFlop/s
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.81573 msec, Performace = 10644.4 GFlop/s
//LB = 3: Size = 9, Time = 2.05602 msec, Performace =  9400.4 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R4W3_IC2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	      float* __restrict__  Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
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

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw;
	const int tow2 = tow1 + sw;
	const int tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yj0  = bj0  + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area------------------------------------------------------
	const int GK = 9 << LIC;//GK = FH * FW * IC
	const int sw_IC = sw << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	const int X_fh = fhw >> 2, X_fw = fhw & 3;
	const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int woffset = ((ty & STEP_m1) << 1) * OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];
			float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];

			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		const int woffset = (ok + ((ty & STEP_m1) << 1))*OC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		const int X_fh = fhw >> 2, X_fw = fhw & 3;
		const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
		float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

		simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);
		simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);
		simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);
		simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);
		simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

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


//======[FH = FW = 5]=========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RW5
#define CONV_3D_GEMM_UERNEL_8_8RW5

//LB = 4: Size = 1.5625, Time = 2.20161 msec, Performace = 1524.08 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.36722 msec, Performace = 1417.46 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8RW5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//prepare for GK = FH * FW * IC
	const int GK = 25 * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
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
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0, x1, x2, x3;
		x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
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

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RW5_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8RW5_IC2POW

//LB = 4: Size = 1.5625, Time = 2.15232 msec, Performace = 1558.99 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.31431 msec, Performace = 1449.87 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dGemm_uernel_8_8RW5_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	      float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((tn1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((tn3*IH + toh3)*IW + tow3) << LIC;

	//prepare for GK = FH * FW * IC
	const int GK = 25 << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
	int X_fh = fhw >> 3, X_fw = fhw & 7;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty - ((ty >= STEP) << LB >> 1) << 1);
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
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
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((((ok - (ty >= STEP)) << LB >> 1) + ty) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((((ok - (tx >= STEP)) << LB >> 1) + tx) << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
		int X_fh = fhw >> 3, X_fw = fhw & 7;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Xs[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
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

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//------[(OH, OW) % 4 == 0]-----------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W5
#define CONV_3D_GEMM_UERNEL_8_8R4W5

//for: (64, 64) -> (32, 32), [FH, FW] = 5, [N, IC, OC] = [128, 64, 128]
//LB = 4: Size = 25, Time = 5.19163 msec, Performace = 10341.1  GFlop/s
//LB = 3: Size = 25, Time = 5.90553 msec, Performace =  9090.99 GFlop/s
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 25, Time = 5.13946 msec, Performace = 10446.1 GFlop/s
//LB = 3: Size = 25, Time = 5.71696 msec, Performace =  9390.8 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R4W5(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
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

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw;
	const int tow2 = tow1 + sw;
	const int tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) * IC;//X += X1;
	
	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yj0  = bj0  + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area------------------------------------------------------
	const int GK = 25 * IC;// GK = FH * FW * IC
	const int sw_IC = sw * IC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
	const int X_fh = fhw >> 3, X_fw = fhw & 7;
	const int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC       ) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset               ) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC       ) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
			float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = X_k / IC; char fhw = XIDX_W5[Idx];
		const int X_fh = fhw >> 3, X_fw = fhw & 7;
		const int xoffset = (X_fh * IW + X_fw - Idx)*IC + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC       ) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset               ) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC       ) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
		float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

		simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);
		simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);
		simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);
		simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);
		simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W5_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8R4W5_IC2POW

//for: (64, 64) -> (32, 32), [FH, FW] = 5, [N, IC, OC] = [128, 64, 128]
//LB = 4: Size = 25, Time = 5.03271 msec, Performace = 10667.6 GFlop/s
//LB = 3: Size = 25, Time = 5.8318  msec, Performace =  9205.9 GFlop/s
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 25, Time = 4.9468 msec, Performace = 10852.9 GFlop/s
//LB = 3: Size = 25, Time = 5.58037 msec, Performace = 9620.7 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R4W5_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	      float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
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

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw;
	const int tow2 = tow1 + sw;
	const int tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X[tn0, toh0, tow1]
	
	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yj0  = bj0  + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	
	//compute area------------------------------------------------------
	const int GK = 25 << LIC;// GK = FH * FW * IC
	const int Xstride = sw << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
	const int X_fh = fhw >> 3, X_fw = fhw & 7;
	const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - Xstride) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + Xstride) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (Xstride << 1)) : F32_2_0);
	Xs[0][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int woffset = ((ty & STEP_m1) << 1) * OC;
	Ws[0][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];
			float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];

			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		const int woffset = (ok + ((ty & STEP_m1) << 1)) * OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
		const int X_fh = fhw >> 3, X_fw = fhw & 7;
		const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - Xstride) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + Xstride) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (Xstride << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
		float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

		simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);
		simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);
		simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);
		simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);
		simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

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


//======[FH = FW = 7]=========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W7_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8R4W7_IC2POW

//for: (64, 64) -> (32, 32), [FH, FW] = 5, [N, IC, OC] = [128, 64, 128]
//LB = 4: Size = 49, Time = 9.77769 msec, Performace = 10761.9 GFlop/s
//LB = 3: Size = 49, Time = 10.8633 msec, Performace =  9686.5 GFlop/s
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 49, Time =  9.4938 msec, Performace = 11083.7 GFlop/s
//LB = 3: Size = 49, Time = 11.1831 msec, Performace =  9409.4 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R4W7_IC2pow(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	      float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
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

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw;
	const int tow2 = tow1 + sw;
	const int tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X[tn0, toh0, tow1]

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yj0 = bj0 + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area------------------------------------------------------
	const int GK = 49 << LIC;// GK = FH * FW * IC
	const int Xstride = sw << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int Idx = X_k >> LIC, fhw = XIDX_W7[Idx];
	const int X_fh = fhw >> 3, X_fw = fhw & 7;
	const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - Xstride) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + Xstride) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (Xstride << 1)) : F32_2_0);
	Xs[0][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int woffset = ((ty & STEP_m1) << 1) * OC;
	Ws[0][(ty << 1)][tx] = *(float4*)(CW + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
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

		//load 4 elements from W[OC, FH, FW, IC]
		const int woffset = (ok + ((ty & STEP_m1) << 1)) * OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = X_k >> LIC, fhw = XIDX_W7[Idx];
		const int X_fh = fhw >> 3, X_fw = fhw & 7;
		const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - Xstride) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + Xstride) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (Xstride << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
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

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

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