#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_S1_H
#define DECONV3D_DX_ZERO_PADDING_UERNEL_S1_H

//Unsparse Matrix Method:
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_S1_CALL
#define DECONV3D_DX_ZERO_PADDING_UERNEL_S1_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]==============================================
//N % 8 == 0
#define u88As1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, N,IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define u88s1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8_s1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//N % 8 == 0, OC is power of 2
#define u88As1_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, N,IC,LOC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//========[FH = 3, FW = 3]====================================
//N % 8 == 0
#define u88As1W3(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1_W3<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,IH,IW, N,IC,OC, (2-ph),(2-pw),\
			ic_index,j_index)

//N % 8 == 0, OC is power of 2
#define u88As1W3_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1_W3_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,IH,IW, N,IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

//OC is power of 2
#define u88s1W3x4_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8_s1_W3_x4_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,(IH*IW),IW, IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

//======[Pure: Direct Conv]============================================
#define u84s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_4_s1_pure<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define u48s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_4_8_s1_pure<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define u44s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_4_4_s1_pure<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//========[FH = 5, FW = 5]====================================
//N % 8 == 0
#define u88As1W5(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1_W5<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (24*IC)), deltaX,IH,IW, N,IC,OC, (4-ph),(4-pw),\
			ic_index,j_index)

//N % 8 == 0, OC is power of 2
#define u88As1W5_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, N, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8A_s1_W5_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (24*IC)), deltaX,IH,IW, N,IC,LOC, (4-ph),(4-pw),\
			ic_index,j_index)

//OC is power of 2
#define u88s1W5x4_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8_s1_W5_x4_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (24*IC)), deltaX,(IH*IW),IW, IC,LOC, (4-ph),(4-pw),\
			ic_index,j_index)

#endif


//======[Common]=======================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time =  3.311 msec, Performace = 11674.6 GFlop/s
//LB = 4: Size = 72, Time = 13.084 msec, Performace = 11817.4 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 3.305 msec, Performace = 11695.8 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_8A_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	      float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = IC
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3);
	const int IW_N = IW * N; get_ih_iw_n(tj0, ih0, iw0, n0);
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3);
	get_ih_iw_n(xj0, xih0, xiw0, xn0);
	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//compute area----------------------------------------------------
	const int FW_OC = FW * OC, GK = FH * FW_OC;//GK = FH * FW * OC
	const int Wstride = FH * FW * IC;
	const int SY = (OW - FW)*OC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_oc = (ty & STEP_m1) << 1;
	const int woffset = W_oc * Wstride;//[fhr = fwr = 0]
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + Wstride);

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_oc = (tx & STEP_m1) << 1;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		int W_k = ok + ((ty & STEP_m1) << 1);
		const int fh = W_k / FW_OC, fw = (W_k -= fh * FW_OC) / OC;

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_oc = W_k - fw * OC;
		const int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + Wstride);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Xstride = IH * IW * IC;
	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % BLOCK_SIZE == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1

//for (IH, IW) = 32:
//LB = 4: Size = 18, Time = 3.442 msec, Performace = 11230.3 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1> 
__global__ void zeroPadding_uernel_8_8_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
   	      float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int ic0 = bic0 + (tx << 3);
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for delteY[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3);
	const int X0 = xj0 * IC + xic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic

	//compute area----------------------------------------------------
	const int FW_OC = FW * OC, GK = FH * FW_OC;
	const int Wstride = FH * FW * IC;
	const int Ystride = (OW - FW) * OC;
	
	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ((ty & STEP_m1) << 1);
	int fh, fw, W_oc; get_fh_fw_oc(W_k, fh, fw, W_oc);
	const int woffset0 = ((W_oc*FH - fh)*FW - fw)*IC;
	const int woffset1 = woffset0 + Wstride;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset1);

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = ((tx & STEP_m1) << 1);
	const int yoffset = fh * Ystride + Y_k;
	bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
	bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
	bool ly2 = LOAD_Y(tih2, tiw2, fh, fw);
	bool ly3 = LOAD_Y(tih3, tiw3, fh, fw);
	float2 x0 = (ly0 ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
	float2 x1 = (ly1 ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
	float2 x2 = (ly2 ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
	float2 x3 = (ly3 ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();
	
	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

			//transposed compute core: (W * dY)^T
			simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
			simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
			simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
			simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
			simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ok + ((ty & STEP_m1) << 1);
		int fh, fw, W_oc; get_fh_fw_oc(W_k, fh, fw, W_oc);
		const int woffset0 = ((W_oc*FH - fh)*FW - fw)*IC;
		const int woffset1 = woffset0 + Wstride;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset1);

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int yoffset = fh * Ystride + Y_k;
		bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
		bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
		bool ly2 = LOAD_Y(tih2, tiw2, fh, fw);
		bool ly3 = LOAD_Y(tih3, tiw3, fh, fw);
		float2 x0 = (ly0 ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly1 ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly2 ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly3 ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

		simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
		simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
		simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
		simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
		simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	const int X4 = X3 + IC, X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

	*(float4*)(deltaX + X0) =  v0; *(float4*)(deltaX + X0 + 4) =  v1;
	*(float4*)(deltaX + X1) =  v2; *(float4*)(deltaX + X1 + 4) =  v3;
	*(float4*)(deltaX + X2) =  v4; *(float4*)(deltaX + X2 + 4) =  v5;
	*(float4*)(deltaX + X3) =  v6; *(float4*)(deltaX + X3 + 4) =  v7;
	*(float4*)(deltaX + X4) =  v8; *(float4*)(deltaX + X4 + 4) =  v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0, OC is power of 2
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_OC2POW

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 3.272 msec, Performace = 11813.8 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 3.271 msec, Performace = 11817.4 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_8A_s1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	      float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = IC
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3);
	const int IW_N = IW * N; get_ih_iw_n(tj0, ih0, iw0, n0);
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3);
	get_ih_iw_n(xj0, xih0, xiw0, xn0);
	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//compute area----------------------------------------------------
	const int FW_OC = FW << LOC, GK = FH * FW_OC;//GK = FH * FW * OC
	const int SW = FH * FW * IC, SY = (OW - FW) << LOC;
	const int OC_m1 = (1 << LOC) - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_oc = (tx & STEP_m1) << 1;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_oc = (ty & STEP_m1) << 1;
	const int woffset = W_oc * SW;//[fhr = fwr = 0]
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + SW);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int fh = W_k / FW_OC, fw = (W_k - fh * FW_OC) >> LOC;
		const int woffset = (((W_k & OC_m1)*FH - fh)*FW - fw)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + SW);

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Xstride = IH * IW * IC;
	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//======[Pure: Direct Conv]============================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*4), OC % BLOCK_SIZE == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_4_S1_PURE
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_4_S1_PURE

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 3.764 msec, Performace = 10269.6 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 3.718 msec, Performace = 10396.6 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_4_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	      float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Ys[2][1 << LB][(2 << LB) + 2];//follow k44

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = IC
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	const int bj0 = (blockIdx.y << LB << 2) + j_index;
	const int tj0 = bj0 + (ty << 2) + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0 = bj0 + (uy << 2);
	const int X0 = xj0 * IC + xic0;

	//compute area----------------------------------------------------
	const int Ys_x = (tx >> 1 << 1), Ys_y = (ty << 1) + (tx & 1);
	const int SY = (OW - FW)*OC, SW = FH * FW * IC;

	for (int fh = 0; fh < FH; fh++, deltaY += SY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC) {
			//load 4 elements from W[OC, FH, FW, OC]
			const int W_oc = (ty & STEP_m1) << 1;
			const int woffset0 = W_oc * SW;
			Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset0 + SW);

			//load 2 elements from deltaY[N, OH, OW, OC]
			const int Y_oc = (tx >> 1 << 1);
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
			bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
			float2 y0 = (ly0 ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
			float2 y1 = (ly1 ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
			Ys[buf][Ys_x    ][Ys_y] = float2{ y0.x, y1.x };
			Ys[buf][Ys_x + 1][Ys_y] = float2{ y0.y, y1.y };
			__syncthreads();

			for (int ooc = STEP2; ooc < OC; ooc += STEP2) {
#pragma unroll 
				for (int ik = 0; ik < STEP2; ik++) {
					float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
					float4 y0 = *(float4*)(&Ys[buf][ik][uy << 1]);

					//transposed compute core: (W * dY)^T
					simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
					simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
					simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
					simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
				}
				buf ^= 1;
				//load 4 elements from W[OC, FH, FW, IC]
				const int W_oc = ooc + ((ty & STEP_m1) << 1);
				const int woffset0 = W_oc * SW;
				Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset0);
				Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset0 + SW);

				//load 2 elements from deltaY[N, OH, OW, OC]
				const int Y_oc = ooc + (tx >> 1 << 1);
				bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
				bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
				float2 y0 = (ly0 ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
				float2 y1 = (ly1 ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
				Ys[buf][Ys_x    ][Ys_y] = float2{ y0.x, y1.x };
				Ys[buf][Ys_x + 1][Ys_y] = float2{ y0.y, y1.y };
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP2; ik++) {
				float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
				float4 y0 = *(float4*)(&Ys[buf][ik][uy << 1]);

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
				simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
				simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
				simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			}
			buf ^= 1;
		}
	}

	const int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	*(float4*)(deltaX + X0) = v0; *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2; *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4; *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6; *(float4*)(deltaX + X3 + 4) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_4_8_S1_PURE
#define DECONV3D_DX_ZERO_PADDING_UERNEL_4_8_S1_PURE

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 3.724 msec, Performace = 10379.9 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 3.683 msec, Performace = 10495.4 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_4_8_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	      float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = IC
	const int bic0 = (blockIdx.x << LB << 2) + ic_index;
	const int tic0 = bic0 + (tx << 2) + ((ty & 1) << 1);
	W += tic0;//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 2);
	const int xj0  = bj0  + (uy << 3);
	const int X0 = xj0 * IC + xic0;

	//compute area----------------------------------------------------
	const int Ws_y = (ty >> 1 << 1), Ws_x = (tx << 1) + (ty & 1);
	const int SY = (OW - FW)*OC, SW = FH * FW * IC;

	for (int fh = 0; fh < FH; fh++, deltaY += SY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC) {
			//load 4 elements from deltaY[N, OH, OW, OC]
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
			bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
			bool ly2 = LOAD_Y(tih2, tiw2, fh, fw);
			bool ly3 = LOAD_Y(tih3, tiw3, fh, fw);
			const int Y_oc = (tx & STEP_m1) << 1;
			float2 y0 = (ly0 ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
			float2 y1 = (ly1 ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
			float2 y2 = (ly2 ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
			float2 y3 = (ly3 ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
			Ys[buf][(tx << 1)    ][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
			Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

			//load 2 elements from W[OC, FH, FW, IC]
			const int W_oc = (ty >> 1 << 1);
			const int woffset0 = W_oc * SW;
			Ws[buf][Ws_y    ][Ws_x] = *(float2*)(W + woffset0);
			Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(W + woffset0 + SW);
			__syncthreads();

			for (int ooc = STEP2; ooc < OC; ooc += STEP2) {
#pragma unroll 
				for (int ik = 0; ik < STEP2; ik++) {
					float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
					float4 w0 = *(float4*)(&Ws[buf][ik][ux << 1]);

					//transposed compute core: (W * dY)^T
					simdMM4(v0, y0.x, w0); simdMM4(v1, y0.y, w0);
					simdMM4(v2, y0.z, w0); simdMM4(v3, y0.w, w0);
					simdMM4(v4, y1.x, w0); simdMM4(v5, y1.y, w0);
					simdMM4(v6, y1.z, w0); simdMM4(v7, y1.w, w0);
				}
				buf ^= 1;

				//load 4 elements from deltaY[N, OH, OW, OC]
				const int Y_oc = ooc + ((tx & STEP_m1) << 1);
				float2 y0 = (ly0 ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
				float2 y1 = (ly1 ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
				float2 y2 = (ly2 ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
				float2 y3 = (ly3 ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
				Ys[buf][(tx << 1)    ][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
				Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

				//load 2 elements from W[OC, FH, FW, IC]
				const int W_oc = ooc + (ty >> 1 << 1);
				const int woffset0 = W_oc * SW;
				Ws[buf][Ws_y    ][Ws_x] = *(float2*)(W + woffset0);
				Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(W + woffset0 + SW);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP2; ik++) {
				float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
				float4 w0 = *(float4*)(&Ws[buf][ik][ux << 1]);

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y0.x, w0); simdMM4(v1, y0.y, w0);
				simdMM4(v2, y0.z, w0); simdMM4(v3, y0.w, w0);
				simdMM4(v4, y1.x, w0); simdMM4(v5, y1.y, w0);
				simdMM4(v6, y1.z, w0); simdMM4(v7, y1.w, w0);
			}
			buf ^= 1;
		}
	}

	const int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	const int X4 = X3 + IC, X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

	*(float4*)(deltaX + X0) = v0; *(float4*)(deltaX + X1) = v1;
	*(float4*)(deltaX + X2) = v2; *(float4*)(deltaX + X3) = v3;
	*(float4*)(deltaX + X4) = v4; *(float4*)(deltaX + X5) = v5;
	*(float4*)(deltaX + X6) = v6; *(float4*)(deltaX + X7) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_4_4_S1_PURE
#define DECONV3D_DX_ZERO_PADDING_UERNEL_4_4_S1_PURE

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 4.363 msec, Performace = 8859.66 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 4.298 msec, Performace = 8993.65 GFlop/s

template<int LB, int STEP, int STEP2>
__global__ void zeroPadding_uernel_4_4_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	      float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB][(2 << LB) + 2];

	//compute 4*4 results:
	float4 v0 = F32_4_0, v1 = F32_4_0;
	float4 v2 = F32_4_0, v3 = F32_4_0;

	//prepare for GN = IC
	const int bic0 = (blockIdx.x << LB << 2) + ic_index;
	const int tic0 = bic0 + (tx << 2) + ((ty & 1) << 1);
	W += tic0;//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	const int bj0 = (blockIdx.y << LB << 2) + j_index;
	const int tj0 = bj0 + (ty << 2) + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;

	//prepare for deltaX[N, IH, IW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 2);
	const int xj0  = bj0  + (uy << 2);
	const int X0 = xj0 * IC + xic0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area----------------------------------------------------
	const int Ws_y = (ty >> 1 << 1), Ws_x = (tx << 1) + (ty & 1);
	const int Ys_x = (tx >> 1 << 1), Ys_y = (ty << 1) + (tx & 1);
	const int SY = (OW - FW)*OC, SW = FH * FW * IC;

	for (int fh = 0; fh < FH; fh++, deltaY += SY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC) {
			//load 2 elements from deltaY[N, OH, OW, OC]
			const int Y_oc = (tx >> 1 << 1);
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
			bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
			float2 y0 = (ly0 ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
			float2 y1 = (ly1 ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
			Ys[buf][Ys_x    ][Ys_y] = float2{ y0.x, y1.x };
			Ys[buf][Ys_x + 1][Ys_y] = float2{ y0.y, y1.y };

			//load 2 elements from W[OC, FH, FW, IC]
			const int W_oc = (ty >> 1 << 1);
			const int woffset0 = W_oc * SW;
			Ws[buf][Ws_y    ][Ws_x] = *(float2*)(W + woffset0);
			Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(W + woffset0 + SW);
			__syncthreads();

			for (int ooc = STEP2; ooc < OC; ooc += STEP2) {
#pragma unroll
				for (int ik = 0; ik < STEP2; ik++) {
					float4 y = *(float4*)(&Ys[buf][ik][uy << 1]);
					float4 w = *(float4*)(&Ws[buf][ik][ux << 1]);

					//transposed compute core: (W * dY)^T
					simdMM4(v0, y.x, w);
					simdMM4(v1, y.y, w);
					simdMM4(v2, y.z, w);
					simdMM4(v3, y.w, w);
				}
				buf ^= 1;

				//load 2 elements from deltaY[N, OH, OW, OC]
				const int Y_oc = ooc + (tx >> 1 << 1);
				bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
				bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
				float2 y0 = (ly0 ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
				float2 y1 = (ly1 ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
				Ys[buf][Ys_x    ][Ys_y] = float2{ y0.x, y1.x };
				Ys[buf][Ys_x + 1][Ys_y] = float2{ y0.y, y1.y };

				//load 2 elements from W[OC, FH, FW, IC]
				const int W_oc = ooc + (ty >> 1 << 1);
				const int woffset0 = W_oc * SW;
				Ws[buf][Ws_y    ][Ws_x] = *(float2*)(W + woffset0);
				Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(W + woffset0 + SW);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP2; ik++) {
				float4 y = *(float4*)(&Ys[buf][ik][uy << 1]);
				float4 w = *(float4*)(&Ws[buf][ik][ux << 1]);

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y.x, w);
				simdMM4(v1, y.y, w);
				simdMM4(v2, y.z, w);
				simdMM4(v3, y.w, w);
			}
			buf ^= 1;
		}
	}

	const int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	*(float4*)(deltaX + X0) = v0;
	*(float4*)(deltaX + X1) = v1;
	*(float4*)(deltaX + X2) = v2;
	*(float4*)(deltaX + X3) = v3;
}

#endif


//======[FH = FW = 3]==================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W3
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W3

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 3.283 msec, Performace = 11774.2 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 3.292 msec, Performace = 11742 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_8A_s1_W3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	      float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = IC
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3);
	const int IW_N = IW * N; get_ih_iw_n(tj0, ih0, iw0, n0);
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3);
	get_ih_iw_n(xj0, xih0, xiw0, xn0);
	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//compute area----------------------------------------------------
	const int GK = 9 * OC;//GK = FH * FW * OC
	const int SY = (OW - 3)*OC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_oc = (ty & STEP_m1) << 1;
	const int woffset = W_oc * 9 * IC;//[fhr = fwr = 0]
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_oc = (tx & STEP_m1) << 1;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = Y_k / OC; char fhw = YIDX_W33[Idx];
		const int fh = fhw >> 2, fw = fhw & 3;
		const int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int W_oc = W_k - (fh * 3 + fw)*OC;
		const int woffset = ((W_oc * 3 - fh) * 3 - fw)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Xstride = IH * IW * IC;
	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0
//LB = 4: GK % 16 == 0
//LB = 3: GK %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W3_OC2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W3_OC2POW

//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 18, Time = 3.299 msec, Performace = 11717.1 GFlop/s
//LB = 3: Size = 18, Time = 3.644 msec, Performace = 10607.8 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 18, Time = 3.263 msec, Performace = 11846.4 GFlop/s
//LB = 3: Size = 18, Time = 3.632 msec, Performace = 10642.8 GFlop/s
//for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 18, Time = 3.231 msec, Performace = 11963.7 GFlop/s
//LB = 3: Size = 18, Time = 3.687 msec, Performace = 10484.1 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_8A_s1_W3_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,//FH = FW = 3
	      float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, FH - 1, FW - 1, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3);
	const int IW_N = IW * N; get_ih_iw_n(tj0, yih0, yiw0, yn0);
	const int tn0 = yn0 + ((tx >= STEP) << 2);
	const int tih = yih0 - oph, tiw = yiw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3); get_ih_iw_n(xj0, xih0, xiw0, xn0);
	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//compute area----------------------------------------------------
	const int GK = 9 << LOC;//GK = FH * FW * OC
	const int SY = (OW - 3) << LOC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_oc = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	const int woffset = W_oc * 9 * IC;//[fhr = fwr = 0]
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_oc = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

			//transposed compute core: (W * dY)^T
			simdMM4(v0,  y0.x, w0); simdMM4(v1,  y0.x, w1);
			simdMM4(v2,  y0.y, w0); simdMM4(v3,  y0.y, w1);
			simdMM4(v4,  y0.z, w0); simdMM4(v5,  y0.z, w1);
			simdMM4(v6,  y0.w, w0); simdMM4(v7,  y0.w, w1);
			simdMM4(v8,  y1.x, w0); simdMM4(v9,  y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = Y_k >> LOC; char fhw = YIDX_W33[Idx];
		const int fh = fhw >> 2, fw = fhw & 3;
		const int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int W_oc = W_k - ((fh * 3 + fw) << LOC);
		const int woffset = ((W_oc * 3 - fh) * 3 - fw)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		//transposed compute core: (W * dY)^T
		simdMM4(v0,  y0.x, w0); simdMM4(v1,  y0.x, w1);
		simdMM4(v2,  y0.y, w0); simdMM4(v3,  y0.y, w1);
		simdMM4(v4,  y0.z, w0); simdMM4(v5,  y0.z, w1);
		simdMM4(v6,  y0.w, w0); simdMM4(v7,  y0.w, w1);
		simdMM4(v8,  y1.x, w0); simdMM4(v9,  y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Xstride = IH * IW * IC;
	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, (IH, IW) % 4 == 0, OC is power of 2
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_W3_X4_OC_2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_W3_X4_OC_2POW

//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 18, Time = 3.299 msec, Performace = 11717.1 GFlop/s
//LB = 3: Size = 18, Time = 3.644 msec, Performace = 10607.8 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 18, Time = 3.357 msec, Performace = 11514.7 GFlop/s
//LB = 3: Size = 18, Time = 3.632 msec, Performace = 10642.8 GFlop/s
//for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 18, Time = 3.231 msec, Performace = 11963.7 GFlop/s
//LB = 3: Size = 18, Time = 3.687 msec, Performace = 10484.1 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_8_s1_W3_x4_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,//FH = FW = 3
	      float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, FH - 1, FW - 1, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0); 
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	deltaY += ((tn0*OH + tih0)*OW + tiw1) << LOC;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0 = bj0 + (uy << 3);
	const int Y0 = xj0 * IC + xic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic

	//compute area----------------------------------------------------
	const int GK = 9 << LOC;//GK = FH * FW * OC
	const int OC = (1 << LOC), OC_m1 = OC - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = (tx & STEP_m1) << 1;
	const int Idx = Y_k >> LOC; char fhw = YIDX_W33[Idx];
	const int fh = fhw >> 2, fw = fhw & 3;
	const int yoffset = ((fh*OW + fw - Idx) << LOC) + Y_k;
	bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
	bool ly0 = ly && (tiw0 >= -fw) && (tiw0 < OW - fw);
	bool ly1 = ly && (tiw1 >= -fw) && (tiw1 < OW - fw);
	bool ly2 = ly && (tiw2 >= -fw) && (tiw2 < OW - fw);
	bool ly3 = ly && (tiw3 >= -fw) && (tiw3 < OW - fw);
	float2 x0 = (ly0 ? *(float2*)(deltaY + yoffset - OC) : F32_2_0);
	float2 x1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);
	float2 x2 = (ly2 ? *(float2*)(deltaY + yoffset + OC) : F32_2_0);
	float2 x3 = (ly3 ? *(float2*)(deltaY + yoffset + (OC << 1)) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = ((W_k & OC_m1) * 9 - fh * 3 - fw)*IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = Y_k >> LOC; char fhw = YIDX_W33[Idx];
		const int fh = fhw >> 2, fw = fhw & 3;
		const int yoffset = ((fh*OW + fw - Idx) << LOC) + Y_k;
		bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
		bool ly0 = ly && (tiw0 >= -fw) && (tiw0 < OW - fw);
		bool ly1 = ly && (tiw1 >= -fw) && (tiw1 < OW - fw);
		bool ly2 = ly && (tiw2 >= -fw) && (tiw2 < OW - fw);
		bool ly3 = ly && (tiw3 >= -fw) && (tiw3 < OW - fw);
		float2 x0 = (ly0 ? *(float2*)(deltaY + yoffset - OC) : F32_2_0);//Y0
		float2 x1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);//Y1
		float2 x2 = (ly2 ? *(float2*)(deltaY + yoffset + OC) : F32_2_0);//Y2
		float2 x3 = (ly3 ? *(float2*)(deltaY + yoffset + (OC << 1)) : F32_2_0);//Y3
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = ((W_k & OC_m1) * 9 - fh * 3 - fw)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Y1 = Y0 + IC, Y2 = Y1 + IC, Y3 = Y2 + IC;
	const int Y4 = Y3 + IC, Y5 = Y4 + IC, Y6 = Y5 + IC, Y7 = Y6 + IC;

	*(float4*)(deltaX + Y0) =  v0; *(float4*)(deltaX + Y0 + 4) =  v1;
	*(float4*)(deltaX + Y1) =  v2; *(float4*)(deltaX + Y1 + 4) =  v3;
	*(float4*)(deltaX + Y2) =  v4; *(float4*)(deltaX + Y2 + 4) =  v5;
	*(float4*)(deltaX + Y3) =  v6; *(float4*)(deltaX + Y3 + 4) =  v7;
	*(float4*)(deltaX + Y4) =  v8; *(float4*)(deltaX + Y4 + 4) =  v9;
	*(float4*)(deltaX + Y5) = v10; *(float4*)(deltaX + Y5 + 4) = v11;
	*(float4*)(deltaX + Y6) = v12; *(float4*)(deltaX + Y6 + 4) = v13;
	*(float4*)(deltaX + Y7) = v14; *(float4*)(deltaX + Y7 + 4) = v15;
}

#endif


//======[FH = FW = 5]==================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W5
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W5

//when (FH, FW) = 5, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (64, 128, 128):
//LB = 4: Size = 25, Time = 4.581 msec, Performace = 11719.5 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (64, 256, 256):
//LB = 4: Size = 25, Time = 4.558 msec, Performace = 11778.6 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_8A_s1_W5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,//FH = FW = 5
	      float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = IC
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3);
	const int IW_N = IW * N; get_ih_iw_n(tj0, ih0, iw0, n0);
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3);
	get_ih_iw_n(xj0, xih0, xiw0, xn0);
	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//compute area----------------------------------------------------
	const int GK = 25 * OC;//GK = FH * FW * OC
	const int SY = (OW - 5)*OC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_oc = (ty & STEP_m1) << 1;
	const int woffset = W_oc * 25 * IC;//[fhr = fwr = 0]
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_oc = (tx & STEP_m1) << 1;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

			//transposed compute core: (W * dY)^T
			simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
			simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
			simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
			simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
			simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = Y_k / OC; char fhw = YIDX_W55[Idx];
		const int fh = fhw >> 3, fw = fhw & 7;
		const int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int W_oc = W_k - (fh * 5 + fw)*OC;
		const int woffset = ((W_oc * 5 - fh) * 5 - fw)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
		simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
		simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
		simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
		simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Xstride = IH * IW * IC;
	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, N % 8 == 0, OC is power of 2
//LB = 4: GK % 16 == 0
//LB = 3: GK %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W5_OC2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8A_S1_W5_OC2POW

//when (FH, FW) = 5, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (64, 128, 128):
//LB = 4: Size = 25, Time = 4.514 msec, Performace = 11893.5 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (64, 256, 256):
//LB = 4: Size = 25, Time = 4.543 msec, Performace = 11817.5 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_8A_s1_W5_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,//FH = FW = 5
	      float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, FH - 1, FW - 1, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3);
	const int IW_N = IW * N; get_ih_iw_n(tj0, yih0, yiw0, yn0);
	const int tn0 = yn0 + ((tx >= STEP) << 2);
	const int tih = yih0 - oph, tiw = yiw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3); get_ih_iw_n(xj0, xih0, xiw0, xn0);
	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//compute area----------------------------------------------------
	const int GK = 25 << LOC;//GK = FH * FW * OC
	const int SY = (OW - 5) << LOC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_oc = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	const int woffset = W_oc * 25 * IC;//[fhr = fwr = 0]
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_oc = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	bool ly = LOAD_Y(tih, tiw, 0, 0);//[fhr = fwr = 0]
	float2 x0 = (ly ? *(float2*)(deltaY + Y0 + Y_oc) : F32_2_0);
	float2 x1 = (ly ? *(float2*)(deltaY + Y1 + Y_oc) : F32_2_0);
	float2 x2 = (ly ? *(float2*)(deltaY + Y2 + Y_oc) : F32_2_0);
	float2 x3 = (ly ? *(float2*)(deltaY + Y3 + Y_oc) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

			//transposed compute core: (W * dY)^T
			simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
			simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
			simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
			simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
			simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = Y_k >> LOC; char fhw = YIDX_W55[Idx];
		const int fh = fhw >> 3, fw = fhw & 7;
		const int yoffset = Y_k + fh * SY;
		bool ly = LOAD_Y(tih, tiw, fh, fw);
		float2 x0 = (ly ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
		float2 x1 = (ly ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 x2 = (ly ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
		float2 x3 = (ly ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int W_oc = W_k - ((fh * 5 + fw) << LOC);
		const int woffset = ((W_oc * 5 - fh) * 5 - fw)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		//transposed compute core: (W * dY)^T
		simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
		simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
		simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
		simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
		simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Xstride = IH * IW * IC;
	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, (IH, IW) % 4 == 0, OC is power of 2
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_W5_X4_OC2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_W5_X4_OC2POW

//when (FH, FW) = 5, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (64, 128, 128):
//LB = 4: Size = 25, Time = 4.788 msec, Performace = 11212.8 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (64, 256, 256):
//LB = 4: Size = 25, Time = 4.627 msec, Performace = 11603 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_8_s1_W5_x4_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,//FH = FW = 5 
	      float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, FH - 1, FW - 1, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0); 
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	deltaY += ((tn0*OH + tih0)*OW + tiw1) << LOC;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3);
	const int Y0 = xj0 * IC + xic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic

	//compute area----------------------------------------------------
	const int GK = 25 << LOC;//GK = FH * FW * OC
	const int OC = (1 << LOC), OC_m1 = OC - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = (tx & STEP_m1) << 1;
	const int Idx = Y_k >> LOC; char fhw = YIDX_W55[Idx];
	const int fh = fhw >> 3, fw = fhw & 7;
	const int yoffset = ((fh*OW + fw - Idx) << LOC) + Y_k;
	bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
	bool ly0 = ly && (tiw0 >= -fw) && (tiw0 < OW - fw);
	bool ly1 = ly && (tiw1 >= -fw) && (tiw1 < OW - fw);
	bool ly2 = ly && (tiw2 >= -fw) && (tiw2 < OW - fw);
	bool ly3 = ly && (tiw3 >= -fw) && (tiw3 < OW - fw);
	float2 x0 = (ly0 ? *(float2*)(deltaY + yoffset - OC) : F32_2_0);
	float2 x1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);
	float2 x2 = (ly2 ? *(float2*)(deltaY + yoffset + OC) : F32_2_0);
	float2 x3 = (ly3 ? *(float2*)(deltaY + yoffset + (OC << 1)) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = ((W_k & OC_m1) * 25 - fh *5 - fw)*IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = Y_k >> LOC; char fhw = YIDX_W55[Idx];
		const int fh = fhw >> 3, fw = fhw & 7;
		const int yoffset = ((fh*OW + fw - Idx) << LOC) + Y_k;
		bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
		bool ly0 = ly && (tiw0 >= -fw) && (tiw0 < OW - fw);
		bool ly1 = ly && (tiw1 >= -fw) && (tiw1 < OW - fw);
		bool ly2 = ly && (tiw2 >= -fw) && (tiw2 < OW - fw);
		bool ly3 = ly && (tiw3 >= -fw) && (tiw3 < OW - fw);
		float2 x0 = (ly0 ? *(float2*)(deltaY + yoffset - OC) : F32_2_0);//Y0
		float2 x1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);//Y1
		float2 x2 = (ly2 ? *(float2*)(deltaY + yoffset + OC) : F32_2_0);//Y2
		float2 x3 = (ly3 ? *(float2*)(deltaY + yoffset + (OC << 1)) : F32_2_0);//Y3
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = ((W_k & OC_m1) * 25 - fh * 5 - fw)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 25 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Y1 = Y0 + IC, Y2 = Y1 + IC, Y3 = Y2 + IC;
	const int Y4 = Y3 + IC, Y5 = Y4 + IC, Y6 = Y5 + IC, Y7 = Y6 + IC;

	*(float4*)(deltaX + Y0) =  v0; *(float4*)(deltaX + Y0 + 4) =  v1;
	*(float4*)(deltaX + Y1) =  v2; *(float4*)(deltaX + Y1 + 4) =  v3;
	*(float4*)(deltaX + Y2) =  v4; *(float4*)(deltaX + Y2 + 4) =  v5;
	*(float4*)(deltaX + Y3) =  v6; *(float4*)(deltaX + Y3 + 4) =  v7;
	*(float4*)(deltaX + Y4) =  v8; *(float4*)(deltaX + Y4 + 4) =  v9;
	*(float4*)(deltaX + Y5) = v10; *(float4*)(deltaX + Y5 + 4) = v11;
	*(float4*)(deltaX + Y6) = v12; *(float4*)(deltaX + Y6 + 4) = v13;
	*(float4*)(deltaX + Y7) = v14; *(float4*)(deltaX + Y7 + 4) = v15;
}

#endif


#endif