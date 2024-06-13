#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_H
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_H

//Unsparse Matrix Method: 
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_CALL
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]============================================================
#define uV2_88s1(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, N) \
	zeroPaddingV2_uernel_8_8_s1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, (FH - ph - 1),(FW - pw - 1),\
			 ic_index,n_index)

#define uV2_88s1_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOC, ph, pw, GN, N) \
	zeroPaddingV2_uernel_8_8_s1_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,LOC, (FH - ph - 1),(FW - pw - 1),\
			 ic_index,n_index)

//======[ph = pw = 1], [FH = FW = 3], [OH, OW > 1] -> [oph = opw = 1]=======
#define uV2_88s1W3P1(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N) \
	zeroPaddingV2_uernel_8_8_s1_W3P1<LB, (1<<LB>>1), (1<<LB),  (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,OC,\
			 ic_index,n_index)

#define uV2_88s1W3P1_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	zeroPaddingV2_uernel_8_8_s1_W3P1_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,LOC,\
			 ic_index,n_index)

//=====[ph = pw = 2], [FH = FW = 5], [OH, OW > 2] -> [oph = opw = 2]============
#define uV2_88s1W5P2(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, GN, N) \
	zeroPaddingV2_uernel_8_8_s1_W5P2<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,OC,\
			 ic_index,n_index)

#define uV2_88s1W5P2_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	zeroPaddingV2_uernel_8_8_s1_W5P2_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,LOC,\
			 ic_index,n_index)

#endif


//======[Common]================================================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1

//[IH, IW] = 32, [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 18, Time = 3.304 msec, Performace = 11699.4 GFlop/s
//[IH, IW] = 16, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 18, Time = 3.125 msec, Performace = 12369.5 GFlop/s
//[IH, IW] =  8, [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 18, Time = 2.904 msec, Performace = 13310.8 GFlop/s
//[IH, IW] =  4, [N, IC, OC] = [512, 512, 512]
//LB = 4: Size = 18, Time = 2.734 msec, Performace = 14138.5 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPaddingV2_uernel_8_8_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__     W, int FH, int FW,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int n_index)
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

	//prepare for bz -> IH * IW
	const int bz = blockIdx.z;
	const int ih = bz / IW, iw = bz - ih * IW;
	const int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws))*IC + tic0;//W[0, -fhs, -fws, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bn0 = (blockIdx.y << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + tiw + fws) * OC;
	const int Ystride = (OH * OW) * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xn0 = bn0 + (uy << 3);
	const int X0 = ((xn0*IH + ih)*IW + iw)*IC + xic0;
	const int Xstride = IH * IW * IC;

	//compute area----------------------------------------------------
	FW = FW * IC;//FW -> FW * IC
	FH = FH * FW;//FH -> FH * FW * IC
	IC = OC * FH + IC;//IC -> OC*FH*FW*IC + IC

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ((ty & STEP_m1) << 1);
	const int fh = W_k / tFW_OC;
	const int W_fw = (W_k -= fh * tFW_OC) / OC;
	int woffset = W_k * FH - fh * FW - W_fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);

	//load 4 elem from deltaY[N, OH, OW, OC]
	const int Y_k = ((tx & STEP_m1) << 1);
	const int SY = OC * (OW - tFW);
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
	__syncthreads();
	
	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ok + ((ty & STEP_m1) << 1);
		const int fh = W_k / tFW_OC;
		const int W_fw = (W_k -= fh * tFW_OC) / OC;
		const int woffset = W_k * FH - fh * FW - W_fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
		Ys[buf][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, OC is power of 2
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_OC2POW

//FH = FW = 3
//[IH, IW] = 32, [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 18, Time = 3.124 msec, Performace = 12373.5 GFlop/s
//LB = 3: Size = 18, Time = 3.508 msec, Performace = 11019 GFlop/s
//[IH, IW] = 16, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 18, Time = 2.97  msec, Performace = 13015.1 GFlop/s
//LB = 3: Size = 18, Time = 3.334 msec, Performace = 11594.1 GFlop/s
//[IH, IW] =  8, [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 18, Time = 2.769 msec, Performace = 13959.8 GFlop/s
//LB = 3: Size = 18, Time = 3.058 msec, Performace = 12640.5 GFlop/s
//[IH, IW] =  4, [N, IC, OC] = [512, 512, 512]
//LB = 4: Size = 18, Time = 2.3   msec, Performace = 16806.4 GFlop/s
//LB = 3: Size = 18, Time = 2.526 msec, Performace = 15302.7 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPaddingV2_uernel_8_8_s1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, uint32_t FH, uint32_t FW,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int n_index)
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

	//prepare for bz -> IH * IW
	const int bz = blockIdx.z;
	const int ih = bz / IW, iw = bz - ih * IW;
	const int tih = ih - oph, tiw = iw - opw;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > FH), FH, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > FW), FW, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += ((FH - 1 - fhs) * FW + (FW - 1 - fws))*IC + tic0;//W[0, -fhs, -fws, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bn0 = (blockIdx.y << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + tiw + fws) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xn0  = bn0  + (uy << 3);
	const int X0 = ((xn0*IH + ih)*IW + iw)*IC + xic0;
	const int Xstride = IH * IW * IC;

	//compute area------------------------------------------------------
	FW = FW * IC;//FW -> FW * IC
	FH = FH * FW;//FH -> FH * FW * IC
	IC = (FH << LOC) + IC;//IC -> OC*FH*FW*IC + IC

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = ((tx & STEP_m1) << 1);
	const int SY = (OW - tFW) << LOC;
	const int fh = Y_k / tFW_OC;
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ((ty & STEP_m1) << 1);
	const int W_fw = (W_k -= fh * tFW_OC) >> LOC;
	const int woffset = W_k * FH - fh * FW - W_fw * IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
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

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int fh = Y_k / tFW_OC;
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ok + ((ty & STEP_m1) << 1);
		const int W_fw = (W_k -= fh * tFW_OC) >> LOC;
		const int woffset = W_k * FH - fh * FW - W_fw * IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

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


//======[ph = pw = 1], [FH = FW = 3], [OH, OW > 1] -> [oph = opw = 1]===========
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W3_P1
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W3_P1

//[IH, IW] = 32, [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 18, Time = 3.284 msec, Performace = 11770.6 GFlop/s
//[IH, IW] = 16, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 18, Time = 3.128 msec, Performace = 12357.6 GFlop/s
//[IH, IW] =  8, [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 18, Time = 2.898 msec, Performace = 13338.4 GFlop/s
//[IH, IW] =  4, [N, IC, OC] = [512, 512, 512]
//LB = 4: Size = 18, Time = 2.745 msec, Performace = 14081.9 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPaddingV2_uernel_8_8_s1_W3P1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,//tFH = tFW = 2 or 3
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,//oph = opw = 1
	int ic_index, int n_index)
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

	//prepare for bz -> IH * IW
	const int bz = blockIdx.z;
	const int ih = bz / IW, iw = bz - ih * IW;
	const int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;

	const int fh_idx = (tFH == 3), fw_idx = (tFW == 3);
	const int fhw_offset = ((fh_idx << 1) + fw_idx) * 9;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += ((2 - fhs) * 3 + (2 - fws))*IC + tic0;//Wr[0, fhs, frws, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bn0 = (blockIdx.y << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + fws + tiw) * OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xn0  = bn0  + (uy << 3);
	const int X0 = ((xn0*IH + ih)*IW + iw)*IC + xic0, Xstride = IH * IW * IC;

	//compute area----------------------------------------------------
	uint32_t FW = 3 * IC;//FW -> FW * IC = 3*IC
	uint32_t FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = (9 * OC + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = ((ty & STEP_m1) << 1);
	const int Idx = W_k / OC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
	const int fh = fhw >> 2, fw = fhw & 3;
	const int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = ((tx & STEP_m1) << 1);
	const int SY = (OW - tFW) * OC;
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
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

		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int Idx = W_k / OC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
		const int fh = fhw >> 2, fw = fhw & 3;
		const int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		simdMM4(v0,  y0.x, w0); simdMM4(v1,  y0.x, w1);
		simdMM4(v2,  y0.y, w0); simdMM4(v3,  y0.y, w1);
		simdMM4(v4,  y0.z, w0); simdMM4(v5,  y0.z, w1);
		simdMM4(v6,  y0.w, w0); simdMM4(v7,  y0.w, w1);
		simdMM4(v8,  y1.x, w0); simdMM4(v9,  y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, OC is power of 2
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W3_P1_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W3_P1_OC2POW

//[IH, IW] = 32, [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 18, Time = 3.106 msec, Performace = 12445.2 GFlop/s
//LB = 3: Size = 18, Time = 3.571 msec, Performace = 10824.6 GFlop/s
//[IH, IW] = 16, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 18, Time = 2.951 msec, Performace = 13098.8 GFlop/s
//LB = 3: Size = 18, Time = 3.525 msec, Performace = 10965.9 GFlop/s
//[IH, IW] =  8, [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 18, Time = 2.758 msec, Performace = 14015.5 GFlop/s
//LB = 3: Size = 18, Time = 3.278 msec, Performace = 11792.2 GFlop/s
//[IH, IW] =  4, [N, IC, OC] = [512, 512, 512]
//LB = 4: Size = 18, Time = 2.331 msec, Performace = 16582.9 GFlop/s
//LB = 3: Size = 18, Time = 2.918 msec, Performace = 13247   GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPaddingV2_uernel_8_8_s1_W3P1_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,//tFH = tFW = 2 or 3
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,//oph = opw = 1
	int ic_index, int n_index)
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

	//prepare for bz -> IH * IW
	const int bz = blockIdx.z;
	const int ih = bz / IW, iw = bz - ih * IW;
	const int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;

	const int fh_idx = (tFH == 3), fw_idx = (tFW == 3);
	const int fhw_offset = ((fh_idx << 1) + fw_idx) * 9;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += ((2 - fhs) * 3 + (2 - fws))*IC + tic0;//Wr[0, fhs, frws, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bn0 = (blockIdx.y << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + fws + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xn0  = bn0  + (uy << 3);
	const int X0 = ((xn0*IH + ih)*IW + iw)*IC + xic0, Xstride = IH * IW * IC;

	//compute area----------------------------------------------------
	uint32_t FW = 3 * IC;//FW -> FW * IC = 3*IC
	uint32_t FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = ((9 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = ((tx & STEP_m1) << 1);
	const int SY = (OW - tFW) << LOC;
	const int Idx = Y_k >> LOC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
	const int fh = fhw >> 2, fw = fhw & 3;
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = ((ty & STEP_m1) << 1);
	const int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
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

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = Y_k >> LOC; char fhw = YIDX_V2_W3P1[Idx + fhw_offset];
		const int fh = fhw >> 2, fw = fhw & 3;
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		
		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

		simdMM4(v0,  y0.x, w0); simdMM4(v1,  y0.x, w1);
		simdMM4(v2,  y0.y, w0); simdMM4(v3,  y0.y, w1);
		simdMM4(v4,  y0.z, w0); simdMM4(v5,  y0.z, w1);
		simdMM4(v6,  y0.w, w0); simdMM4(v7,  y0.w, w1);
		simdMM4(v8,  y1.x, w0); simdMM4(v9,  y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

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


//=====[ph = pw = 2], [FH = FW = 5], [OH, OW > 2] -> [oph = opw = 2]============
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W5_P2
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W5_P2

//[IH, IW] = 32, [N, IC, OC] = [128, 128,  32]
//LB = 4: Size = 25, Time = 4.389 msec, Performace = 12232.2 GFlop/s
//[IH, IW] = 16, [N, IC, OC] = [128, 256,  64]
//LB = 4: Size = 25, Time = 4.028 msec, Performace = 13328.5 GFlop/s
//[IH, IW] =  8, [N, IC, OC] = [128, 512, 256]
//LB = 4: Size = 25, Time = 3.486 msec, Performace = 15400.8 GFlop/s
//[IH, IW] =  4, [N, IC, OC] = [256, 512, 256]
//LB = 4: Size = 25, Time = 2.574 msec, Performace = 20857.5 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPaddingV2_uernel_8_8_s1_W5P2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,//tFH = tFW = 5, 4, 3
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,//oph = opw = 2
	int ic_index, int n_index)
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

	//prepare for bz -> IH * IW
	const int bz = blockIdx.z;
	const int ih = bz / IW, iw = bz - ih * IW;
	const int tih = ih - 2, tiw = iw - 2;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 5), 5, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 5), 5, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW * OC, GK = tFH * tFW_OC;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += ((4 - fhs) * 5 + (4 - fws))*IC + tic0;//W[0, -fhs, -fws, tic0]

	const int fh_idx = (tFH == 4) + ((tFH == 5) << 1);
	const int fw_idx = (tFW == 4) + ((tFW == 5) << 1);
	const int fhw_offset = ((fh_idx * 3) + fw_idx) * 25;

	//prepare for deltaY[N, OH, OW, OC]
	const int bn0 = (blockIdx.y << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + tiw + fws) * OC;//deltaY[0, fhs, fws, 0]
	const int Ystride = (OH * OW) * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xn0  = bn0  + (uy << 3);
	const int X0 = ((xn0*IH + ih)*IW + iw)*IC + xic0;
	const int Xstride = IH * IW * IC;

	//compute area---------------------------------------------------------
	uint32_t FW =  5 * IC;//FW -> FW * IC = 5*IC
	uint32_t FH = 25 * IC;//FH -> FH * FW * IC = 25*IC
	IC = (25 * OC + 1)*IC;//IC -> OC * FH * FW * IC + IC = (25*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) =FH * FH * FW * IC * FW * OC = 625*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 1310, It's hard to out of bound of memory 

	//load 4 elem from W[OC, FH, FW, IC]
	const int W_k = ((ty & STEP_m1) << 1);
	const int Idx = W_k / OC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
	const int fh = fhw >> 3, fw = fhw & 7;
	const int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);

	//load 4 elem from deltaY[N, OH, OW, OC]
	const int Y_k = ((tx & STEP_m1) << 1);
	const int SY = (OW - tFW) * OC;
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
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

		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int Idx = W_k / OC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
		const int fh = fhw >> 3, fw = fhw & 7;
		const int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, OC is power of 2
//LB = 4, OC % 16 == 0
//LB = 3, OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W5_P2_OC2POW
#define DECONV3D_DX_ZERO_PADDING_V2_UERNEL_8_8_S1_W5_P2_OC2POW

//[IH, IW] = 32, [N, IC, OC] = [128, 128,  64]
//LB = 4: Size = 25, Time = 4.131 msec, Performace = 12996.1 GFlop/s
//LB = 3: Size = 25, Time = 4.889 msec, Performace = 10981.2 GFlop/s
//[IH, IW] = 16, [N, IC, OC] = [128, 256, 128]
//LB = 4: Size = 25, Time = 3.823 msec, Performace = 14043.2 GFlop/s
//LB = 3: Size = 25, Time = 4.678 msec, Performace = 11476.5 GFlop/s
//[IH, IW] =  8, [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 25, Time = 3.746 msec, Performace = 14331.8 GFlop/s
//LB = 3: Size = 25, Time = 4.338 msec, Performace = 12376 GFlop/s
//[IH, IW] =  4, [N, IC, OC] = [256, 512, 512]
//LB = 4: Size = 25, Time = 2.503 msec, Performace = 21449.1 GFlop/s
//LB = 3: Size = 25, Time = 2.851 msec, Performace = 18831   GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPaddingV2_uernel_8_8_s1_W5P2_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,//tFH = tFW = 5, 4, 3
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,//oph = opw = 2
	int ic_index, int n_index)
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

	//prepare for bz -> IH * IW
	const int bz = blockIdx.z;
	const int ih = bz / IW, iw = bz - ih * IW;
	const int tih = ih - 2, tiw = iw - 2;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 5), 5, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 5), 5, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += ((4 - fhs) * 5 + (4 - fws))*IC + tic0;//W[0, -fhs, -fws, tic0]

	const int fh_idx = (tFH == 4) + ((tFH == 5) << 1);
	const int fw_idx = (tFW == 4) + ((tFW == 5) << 1);
	const int fhw_offset = ((fh_idx * 3) + fw_idx) * 25;

	//prepare for deltaY[N, OH, OW, OC]
	const int bn0 = (blockIdx.y << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + tiw + fws) << LOC;//deltaY[0, fhs, fws, 0]
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xn0  = bn0  + (uy << 3);
	const int X0 = ((xn0*IH + ih)*IW + iw)*IC + xic0;
	const int Xstride = IH * IW * IC;

	//compute area---------------------------------------------------------
	uint32_t FW =  5 * IC;//FW -> FW * IC = 5*IC
	uint32_t FH = 25 * IC;//FH -> FH * FW * IC = 25*IC
	IC = ((25 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (25*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) =FH * FH * FW * IC * FW * OC = 625*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 1310, It's hard to out of bound of memory 

	//load 4 elem from deltaY[N, OH, OW, OC]
	const int Y_k = ((tx & STEP_m1) << 1);
	const int SY = (OW - tFW) << LOC;
	const int Idx = Y_k >> LOC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
	const int fh = fhw >> 3, fw = fhw & 7;
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elem from W[OC, FH, FW, IC]
	const int W_k = ((ty & STEP_m1) << 1);
	const int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
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

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = Y_k >> LOC; char fhw = YIDX_V2_W5P2[Idx + fhw_offset];
		const int fh = fhw >> 3, fw = fhw & 7;
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

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

#endif
