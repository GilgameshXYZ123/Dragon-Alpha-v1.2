#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_S1_COOPERATE_H
#define DECONV3D_DX_ZERO_PADDING_UERNEL_S1_COOPERATE_H

//Unsparse Matrix Method:
//(1) FH*FW >= 2
//(2) GN = IC;            GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IWr; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW ; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
//(6) Cooperate with Im2col-Winograd, to process remainder of IW
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_S1_COOPERATE_CALL
#define DECONV3D_DX_ZERO_PADDING_UERNEL_S1_COOPERATE_CALL

//======[OC is power of 2]=============================================
//IWr % 2 == 0
#define u88s1x2C_oc2pow(stream, LB, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8_s1_x2_C_oc2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, IC,LOC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index, iw_index)

//======[Common]=======================================================
//IWr % 2 == 0
#define u88s1x2C(stream, LB, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8_s1_x2_C<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index, iw_index)

#define u88s1C(stream, LB, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8_s1_C<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index, iw_index)

//======[Pure: Direct Conv]============================================
#define u84s1C_pure(stream, LB, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_4_s1_C_pure<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index,iw_index)

#define u48s1C_pure(stream, LB, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_4_8_s1_C_pure<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index,iw_index)

#define u44s1C_pure(stream, LB, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_4_4_s1_C_pure<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,IH,IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index,iw_index)

#endif


//======[OC is power of 2]=============================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % BLOCK_SIZE == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_X2_C_OC2POW
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_X2_C_OC2POW

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 3.529 msec, Performace = 10953.4 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 3.365 msec, Performace = 11487.3 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1> 
__global__ void zeroPadding_uernel_8_8_s1_x2_C_oc2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
   	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index, int iw_index)
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
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	const int IWr = IW - iw_index, IH_IWr = IH * IWr;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr, IWr); tiw0 += iw_index; 
	get_n_ih_iw_Temp(tj2, tn2, tih2, tiw2, IH_IWr, IWr); tiw2 += iw_index;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw; int tiw1 = tiw0 + 1;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw; int tiw3 = tiw2 + 1;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0) << LOC;
	const int Y1 = ((tn0*OH + tih0)*OW + tiw1) << LOC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2) << LOC;
	const int Y3 = ((tn2*OH + tih2)*OW + tiw3) << LOC;

	//prepare for delteY[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	//compute area----------------------------------------------------
	const int FW_OC = FW << LOC, GK = FH * FW_OC;
	const int Wstride = FH * FW * IC;
	const int Ystride = (OW - FW) << LOC;
	
	int W_k = ((ty & STEP_m1) << 1);
	int fh, fw, W_oc; get_fh_fw_oc_OC2pow(W_k, fh, fw, W_oc);

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = ((tx & STEP_m1) << 1);
	const int yoffset = fh * Ystride + Y_k;
	bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
	bool ly1 = LOAD_Y(tih0, tiw1, fh, fw);
	bool ly2 = LOAD_Y(tih2, tiw2, fh, fw);
	bool ly3 = LOAD_Y(tih2, tiw3, fh, fw);
	float2 x0 = (ly0 ? *(float2*)(deltaY + Y0 + yoffset) : F32_2_0);
	float2 x1 = (ly1 ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
	float2 x2 = (ly2 ? *(float2*)(deltaY + Y2 + yoffset) : F32_2_0);
	float2 x3 = (ly3 ? *(float2*)(deltaY + Y3 + yoffset) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int woffset0 = ((W_oc*FH - fh)*FW - fw)*IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(W + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(W + woffset0 + Wstride);
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
		int fh, fw, W_oc; get_fh_fw_oc_OC2pow(W_k, fh, fw, W_oc);
		const int woffset0 = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset0 + Wstride);

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int yoffset = fh * Ystride + Y_k;
		bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
		bool ly1 = LOAD_Y(tih0, tiw1, fh, fw);
		bool ly2 = LOAD_Y(tih2, tiw2, fh, fw);
		bool ly3 = LOAD_Y(tih2, tiw3, fh, fw);
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

	const int xic0 = bic0 + (ux << 3);
	const int xj0 = bj0 + (uy << 3);
	const int xj2 = xj0 + 2, xj4 = xj0 + 4, xj6 = xj0 + 6;

	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr, IWr); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj2, xn2, xih2, xiw2, IH_IWr, IWr); xiw2 += iw_index;
	get_n_ih_iw_Temp(xj4, xn4, xih4, xiw4, IH_IWr, IWr); xiw4 += iw_index;
	get_n_ih_iw_Temp(xj6, xn6, xih6, xiw6, IH_IWr, IWr); xiw6 += iw_index;

	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0, X1 = X0 + IC;
	const int X2 = ((xn2*IH + xih2)*IW + xiw2)*IC + xic0, X3 = X2 + IC;
	const int X4 = ((xn4*IH + xih4)*IW + xiw4)*IC + xic0, X5 = X4 + IC;
	const int X6 = ((xn6*IH + xih6)*IW + xiw6)*IC + xic0, X7 = X6 + IC;

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


//======[Common]=======================================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % BLOCK_SIZE == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_X2_C
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_X2_C

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 3.56  msec, Performace = 10858.1 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 3.431 msec, Performace = 11266.3 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1> 
__global__ void zeroPadding_uernel_8_8_s1_x2_C(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
   	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index, int iw_index)
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
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	const int IWr = IW - iw_index, IH_IWr = IH * IWr;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr, IWr); tiw0 += iw_index; 
	get_n_ih_iw_Temp(tj2, tn2, tih2, tiw2, IH_IWr, IWr); tiw2 += iw_index;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw; int tiw1 = tiw0 + 1;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw; int tiw3 = tiw2 + 1;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn0*OH + tih0)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn2*OH + tih2)*OW + tiw3)*OC;

	//prepare for delteY[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

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
	bool ly1 = LOAD_Y(tih0, tiw1, fh, fw);
	bool ly2 = LOAD_Y(tih2, tiw2, fh, fw);
	bool ly3 = LOAD_Y(tih2, tiw3, fh, fw);
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
		bool ly1 = LOAD_Y(tih0, tiw1, fh, fw);
		bool ly2 = LOAD_Y(tih2, tiw2, fh, fw);
		bool ly3 = LOAD_Y(tih2, tiw3, fh, fw);
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

	const int xic0 = bic0 + (ux << 3);
	const int xj0 = bj0 + (uy << 3);
	const int xj2 = xj0 + 2, xj4 = xj0 + 4, xj6 = xj0 + 6;

	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr, IWr); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj2, xn2, xih2, xiw2, IH_IWr, IWr); xiw2 += iw_index;
	get_n_ih_iw_Temp(xj4, xn4, xih4, xiw4, IH_IWr, IWr); xiw4 += iw_index;
	get_n_ih_iw_Temp(xj6, xn6, xih6, xiw6, IH_IWr, IWr); xiw6 += iw_index;

	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0, X1 = X0 + IC;
	const int X2 = ((xn2*IH + xih2)*IW + xiw2)*IC + xic0, X3 = X2 + IC;
	const int X4 = ((xn4*IH + xih4)*IW + xiw4)*IC + xic0, X5 = X4 + IC;
	const int X6 = ((xn6*IH + xih6)*IW + xiw6)*IC + xic0, X7 = X6 + IC;

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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % BLOCK_SIZE == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_C
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_8_S1_C

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 3.577 msec, Performace = 10806.5 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 3.486 msec, Performace = 11088.6 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1> 
__global__ void zeroPadding_uernel_8_8_s1_C(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
   	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index, int iw_index)
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
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IWr = IW - iw_index, IH_IWr = IH * IWr;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr, IWr); tiw0 += iw_index;
	get_n_ih_iw_Temp(tj1, tn1, tih1, tiw1, IH_IWr, IWr); tiw1 += iw_index;
	get_n_ih_iw_Temp(tj2, tn2, tih2, tiw2, IH_IWr, IWr); tiw2 += iw_index;
	get_n_ih_iw_Temp(tj3, tn3, tih3, tiw3, IH_IWr, IWr); tiw3 += iw_index;
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

	const int xic0 = bic0 + (ux << 3);
	const int xj0 = bj0 + (uy << 3);
	const int xj1 = xj0 + 1, xj2 = xj0 + 2, xj3 = xj0 + 3;
	const int xj4 = xj0 + 4, xj5 = xj0 + 5, xj6 = xj0 + 6, xj7 = xj0 + 7;

	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr, IWr); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IWr, IWr); xiw1 += iw_index;
	get_n_ih_iw_Temp(xj2, xn2, xih2, xiw2, IH_IWr, IWr); xiw2 += iw_index;
	get_n_ih_iw_Temp(xj3, xn3, xih3, xiw3, IH_IWr, IWr); xiw3 += iw_index;
	get_n_ih_iw_Temp(xj4, xn4, xih4, xiw4, IH_IWr, IWr); xiw4 += iw_index;
	get_n_ih_iw_Temp(xj5, xn5, xih5, xiw5, IH_IWr, IWr); xiw5 += iw_index;
	get_n_ih_iw_Temp(xj6, xn6, xih6, xiw6, IH_IWr, IWr); xiw6 += iw_index;
	get_n_ih_iw_Temp(xj7, xn7, xih7, xiw7, IH_IWr, IWr); xiw7 += iw_index;

	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;
	const int X1 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;
	const int X2 = ((xn2*IH + xih2)*IW + xiw2)*IC + xic0;
	const int X3 = ((xn3*IH + xih3)*IW + xiw3)*IC + xic0;
	const int X4 = ((xn4*IH + xih4)*IW + xiw4)*IC + xic0;
	const int X5 = ((xn5*IH + xih5)*IW + xiw5)*IC + xic0;
	const int X6 = ((xn6*IH + xih6)*IW + xiw6)*IC + xic0;
	const int X7 = ((xn7*IH + xih7)*IW + xiw7)*IC + xic0;

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


//======[Pure: Direct Conv]============================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*4), OC % BLOCK_SIZE == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_8_4_S1_C_PURE
#define DECONV3D_DX_ZERO_PADDING_UERNEL_8_4_S1_C_PURE

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 3.774 msec, Performace = 10242.4 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 3.76  msec, Performace = 10280.5 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_4_s1_C_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index, int iw_index)
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
	const int IWr = (IW - iw_index), IH_IWr = IH * IWr;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr, IWr); tiw0 += iw_index;
	get_n_ih_iw_Temp(tj1, tn1, tih1, tiw1, IH_IWr, IWr); tiw1 += iw_index;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

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

	const int xic0 = bic0 + (ux << 3);//8
	const int xj0  = bj0  + (uy << 2);//4
	const int xj1 = xj0 + 1, xj2 = xj0 + 2, xj3 = xj0 + 3;

	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr, IWr); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IWr, IWr); xiw1 += iw_index;
	get_n_ih_iw_Temp(xj2, xn2, xih2, xiw2, IH_IWr, IWr); xiw2 += iw_index;
	get_n_ih_iw_Temp(xj3, xn3, xih3, xiw3, IH_IWr, IWr); xiw3 += iw_index;

	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;
	const int X1 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;
	const int X2 = ((xn2*IH + xih2)*IW + xiw2)*IC + xic0;
	const int X3 = ((xn3*IH + xih3)*IW + xiw3)*IC + xic0;

	*(float4*)(deltaX + X0) = v0; *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2; *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4; *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6; *(float4*)(deltaX + X3 + 4) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_4_8_S1_C_PURE
#define DECONV3D_DX_ZERO_PADDING_UERNEL_4_8_S1_C_PURE

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 3.782 msec, Performace = 10220.7 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 3.737 msec, Performace = 10343.8 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_4_8_s1_C_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index, int iw_index)
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
	const int IWr = IW - iw_index, IH_IWr = IH * IWr;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr, IWr); tiw0 += iw_index;
	get_n_ih_iw_Temp(tj1, tn1, tih1, tiw1, IH_IWr, IWr); tiw1 += iw_index;
	get_n_ih_iw_Temp(tj2, tn2, tih2, tiw2, IH_IWr, IWr); tiw2 += iw_index;
	get_n_ih_iw_Temp(tj3, tn3, tih3, tiw3, IH_IWr, IWr); tiw3 += iw_index;
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

	const int xic0 = bic0 + (ux << 2);
	const int xj0 = bj0 + (uy << 3);
	const int xj1 = xj0 + 1, xj2 = xj0 + 2, xj3 = xj0 + 3;
	const int xj4 = xj0 + 4, xj5 = xj0 + 5, xj6 = xj0 + 6, xj7 = xj0 + 7;

	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr, IWr); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IWr, IWr); xiw1 += iw_index;
	get_n_ih_iw_Temp(xj2, xn2, xih2, xiw2, IH_IWr, IWr); xiw2 += iw_index;
	get_n_ih_iw_Temp(xj3, xn3, xih3, xiw3, IH_IWr, IWr); xiw3 += iw_index;
	get_n_ih_iw_Temp(xj4, xn4, xih4, xiw4, IH_IWr, IWr); xiw4 += iw_index;
	get_n_ih_iw_Temp(xj5, xn5, xih5, xiw5, IH_IWr, IWr); xiw5 += iw_index;
	get_n_ih_iw_Temp(xj6, xn6, xih6, xiw6, IH_IWr, IWr); xiw6 += iw_index;
	get_n_ih_iw_Temp(xj7, xn7, xih7, xiw7, IH_IWr, IWr); xiw7 += iw_index;

	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;
	const int X1 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;
	const int X2 = ((xn2*IH + xih2)*IW + xiw2)*IC + xic0;
	const int X3 = ((xn3*IH + xih3)*IW + xiw3)*IC + xic0;
	const int X4 = ((xn4*IH + xih4)*IW + xiw4)*IC + xic0;
	const int X5 = ((xn5*IH + xih5)*IW + xiw5)*IC + xic0;
	const int X6 = ((xn6*IH + xih6)*IW + xiw6)*IC + xic0;
	const int X7 = ((xn7*IH + xih7)*IW + xiw7)*IC + xic0;

	*(float4*)(deltaX + X0) = v0; *(float4*)(deltaX + X1) = v1;
	*(float4*)(deltaX + X2) = v2; *(float4*)(deltaX + X3) = v3;
	*(float4*)(deltaX + X4) = v4; *(float4*)(deltaX + X5) = v5;
	*(float4*)(deltaX + X6) = v6; *(float4*)(deltaX + X7) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_4_4_S1_C_PURE
#define DECONV3D_DX_ZERO_PADDING_UERNEL_4_4_S1_C_PURE

//when (FH, FW) = 3, (sh, sw) = 1:
//for: feature = (32, 32), [N, IC, OC] = (128, 128, 128):
//LB = 4: Size = 18, Time = 4.359 msec, Performace = 8867.79 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = (128, 256, 256):
//LB = 4: Size = 18, Time = 4.298 msec, Performace = 8993.65 GFlop/s

template<int LB, int STEP, int STEP2>
__global__ void zeroPadding_uernel_4_4_s1_C_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index, int iw_index)
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
	const int IWr = IW - iw_index, IH_IWr = IH * IWr;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr, IWr); tiw0 += iw_index;
	get_n_ih_iw_Temp(tj1, tn1, tih1, tiw1, IH_IWr, IWr); tiw1 += iw_index;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;

	//prepare for deltaX[N, IH, IW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

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

	const int xic0 = bic0 + (ux << 2);
	const int xj0  = bj0  + (uy << 2);
	const int xj1 = xj0 + 1, xj2 = xj0 + 2, xj3 = xj0 + 3;

	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr, IWr); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IWr, IWr); xiw1 += iw_index;
	get_n_ih_iw_Temp(xj2, xn2, xih2, xiw2, IH_IWr, IWr); xiw2 += iw_index;
	get_n_ih_iw_Temp(xj3, xn3, xih3, xiw3, IH_IWr, IWr); xiw3 += iw_index;

	const int X0 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;
	const int X1 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;
	const int X2 = ((xn2*IH + xih2)*IW + xiw2)*IC + xic0;
	const int X3 = ((xn3*IH + xih3)*IW + xiw3)*IC + xic0;

	*(float4*)(deltaX + X0) = v0;
	*(float4*)(deltaX + X1) = v1;
	*(float4*)(deltaX + X2) = v2;
	*(float4*)(deltaX + X3) = v3;
}

#endif


//================[Integration: 32(IC) * 32(N*IH)]=====================
#ifndef DECONV_3D_ZERO_PADDING_S1_32X32C
#define DECONV_3D_ZERO_PADDING_S1_32X32C

#define deconv3d_dX_ZeroPadding32x32C_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index  = (GM - GMr) +  j_index;\
		deconv3D_dX_ZeroPadding_s1_32x32C(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW,IWr, N,IC,OC, ph,pw,\
			GNr, GM, next_ic_index, j_index, iw_index);\
		deconv3D_dX_ZeroPadding_s1_32x32C(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW,IWr, N,IC,OC, ph,pw,\
            GN, GMr, ic_index, next_j_index, iw_index);\
		deconv3D_dX_ZeroPadding_s1_32x32C(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW,IWr, N,IC,OC, ph,pw,\
            GNr, GMr, next_ic_index, next_j_index, iw_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		deconv3D_dX_ZeroPadding_s1_32x32C(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW,IWr, N,IC,OC, ph,pw,\
			 GNr, GM, next_ic_index, j_index, iw_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3D_dX_ZeroPadding_s1_32x32C(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW,IWr, N,IC,OC, ph,pw,\
			 GN, GMr, ic_index, next_j_index, iw_index);}}

//(1) IWr = IW - iw_index; => iw_index = IW - IWr
//(2) GN = ((IC    ) / 32 * 32);
//(3) GM = ((N * IH) / 32 * 32) * IWr;
//(4) OC % 8 == 0
//(5) sh = sw = 1

void deconv3D_dX_ZeroPadding_s1_32x32C(
	jlong* streams, int &index, int length,
	const float* deltaY, int OH, int OW,
	const float*      W, int FH, int FW,
	      float* deltaX, int IH, int IW, int IWr,
	int N, int IC, int OC,
	int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index, int iw_index)
{
	next_cudaStream(stream, streams, index, length);

#ifdef ENABLE_DECONV3D_ZERO_PADDING_S1_32X32C
	if ((GN > 127) && (GM > 127) && !(OC & 15)) {//[128, 128]
		if (!(IWr & 1)) {
			if(IS_POWER2(OC)) u88s1x2C_oc2pow(stream, 4, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			else u88s1x2C(stream, 4, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		else u88s1C(stream, 4, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		deconv3d_dX_ZeroPadding32x32C_Branch(127, 127); return;
	}
	
	if (GN > 63 && GM > 63) {//[64, 64]
		if (!(IWr & 1)) {
			if (IS_POWER2(OC)) u88s1x2C_oc2pow(stream, 3, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, LOG2(OC), ph, pw, GN, GM);
			else u88s1x2C(stream, 3, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		}
		else u88s1C(stream, 3, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		deconv3d_dX_ZeroPadding32x32C_Branch(63, 63); return;
	}

	if (GN > 63 && GM > 31) {//[64, 32]
		u84s1C_pure(stream, 3, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		deconv3d_dX_ZeroPadding32x32C_Branch(63, 31); return;
	}

	if (GN > 31 && GM > 63) {//[32, 64]
		u48s1C_pure(stream, 3, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		deconv3d_dX_ZeroPadding32x32C_Branch(31, 63); return;
	}

	if (GN > 31 && GM > 31) {//[32, 32]
		u44s1C_pure(stream, 3, ic_index, j_index, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM);
		deconv3d_dX_ZeroPadding32x32C_Branch(31, 31); return;
	}
#endif
}

#endif


#endif