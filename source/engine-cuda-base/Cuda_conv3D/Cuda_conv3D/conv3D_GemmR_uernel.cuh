#pragma once

#ifndef CONV_3D_GEMMR_UERNEL_H
#define CONV_3D_GEMMR_UERNEL_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV_3D_GEMMR_UERNEL_CALL
#define CONV_3D_GEMMR_UERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[Pure: Direct Conv]===========================================
#define conv3dPure_u84R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_uernel_8_4R<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_u48R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_uernel_4_8R<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dPure_u44R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_uernel_4_4R<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//======[Common]======================================================
#define conv3dGemm_u88R(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_u88R4(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//======[IC is power of 2]============================================
#define conv3dGemm_u88R_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R_IC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#define conv3dGemm_u88R4_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4_IC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//======[FW, IC is power of 2]=========================================
#define conv3dGemm_u88R4_fw_ic2pow(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4_FW_IC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,LFW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

#endif


//======[Pure: Direct Conv]===========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_PURE_UERNEL_8_4R
#define CONV_3D_PURE_UERNEL_8_4R

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 18, Time = 4.03341 msec, Performace = 9583.64 GFlop/s
//LB = 3: Size = 18, Time = 4.43502 msec, Performace = 8715.79 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 18, Time = 4.06261 msec, Performace = 9514.76 GFlop/s
//LB = 3: Size = 18, Time = 4.33775 msec, Performace = 8911.24 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dPure_uernel_8_4R(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Xs[2][1 << LB][(2 << LB) + 2];//follow k44

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int boc0 = (blockIdx.x << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (blockIdx.y << LB << 2) + j_index;
	const int tj0 = bj0 + (ty << 2) + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yj0  = bj0  + (uy << 2);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area----------------------------------------------------
	const int Xs_x = (tx >> 1) << 1, Xs_y = (ty << 1) + (tx & 1);
	const int SX = (IW - FW) *IC, SCW = IC * OC;

	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW) {
			//load 2 elem from X[N, IH, IW, IC]
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			const int Xic = (tx >> 1 << 1);
			float2 x0 = (lx0 ? *(float2*)(X + X0 + Xic) : F32_2_0);
			float2 x1 = (lx1 ? *(float2*)(X + X1 + Xic) : F32_2_0);
			Xs[buf][Xs_x    ][Xs_y] = float2{ x0.x, x1.x };
			Xs[buf][Xs_x + 1][Xs_y] = float2{ x0.y, x1.y };

			//load 4 elem from CW[FH, FW, IC, OC]
			const int Wic = (ty & STEP_m1) << 1;
			const int woffset0 = Wic * OC, woffset1 = woffset0 + OC;
			Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
			__syncthreads();

			for (int oic = STEP2; oic < IC; oic += STEP2) {
#pragma unroll
				for (int ik = 0; ik < STEP2; ik++) {
					float4 b0 = *(float4*)(&Xs[buf][ik][uy << 1]);
					float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

					simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
					simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
					simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
					simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				}
				buf ^= 1;

				//load 2 elem from X[N, IH, IW, IC]
				const int Xic = oic + (tx >> 1 << 1);
				float2 x0 = (lx0 ? *(float2*)(X + X0 + Xic) : F32_2_0);
				float2 x1 = (lx1 ? *(float2*)(X + X1 + Xic) : F32_2_0);
				Xs[buf][Xs_x    ][Xs_y] = float2{ x0.x, x1.x };
				Xs[buf][Xs_x + 1][Xs_y] = float2{ x0.y, x1.y };

				//load 4 elem from CW[FH, FW, IC, OC]
				const int Wic = oic + ((ty & STEP_m1) << 1);
				const int woffset0 = Wic * OC, woffset1 = woffset0 + OC;
				Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
				Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP2; ik++) {
				float4 b0 = *(float4*)(&Xs[buf][ik][uy << 1]);
				float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			}
			buf ^= 1;
		}
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_PURE_UERNEL_4_8R
#define CONV_3D_PURE_UERNEL_4_8R

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 2.01913 msec, Performace = 9572.14 GFlop/s
//LB = 3: Size = 9, Time = 2.41652 msec, Performace = 7998.02 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 2.02843 msec, Performace = 9528.23 GFlop/s
//LB = 3: Size = 9, Time = 2.40068 msec, Performace = 8050.78 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dPure_uernel_4_8R(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(2 << LB) + 2];//follow k44
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];//follow k88

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int boc0 = (blockIdx.x << LB << 2) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty & 1) << 1);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int tihs2 = toh2 * sh - ph, tiws2 = tow2 * sw - pw;
	const int tihs3 = toh3 * sh - ph, tiws3 = tow3 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;
	const int X2 = (((tn2 *IH) + tihs2) * IW + tiws2) * IC;
	const int X3 = (((tn3 *IH) + tihs3) * IW + tiws3) * IC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 2);
	const int yj0  = bj0  + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area----------------------------------------------------
	const int Ws_y = (ty >> 1) << 1, Ws_x = (tx << 1) + (ty & 1);
	const int SX = (IW - FW) *IC, SCW = IC * OC;

	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW) {
			//load 4 elem from X[N, IH, IW, IC]
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			bool lx2 = LOAD_X(tihs2, tiws2, fh, fw);
			bool lx3 = LOAD_X(tihs3, tiws3, fh, fw);
			const int Xic = (tx & STEP_m1) << 1;
			float2 x0 = (lx0 ? *(float2*)(X + X0 + Xic) : F32_2_0);
			float2 x1 = (lx1 ? *(float2*)(X + X1 + Xic) : F32_2_0);
			float2 x2 = (lx2 ? *(float2*)(X + X2 + Xic) : F32_2_0);
			float2 x3 = (lx3 ? *(float2*)(X + X3 + Xic) : F32_2_0);
			Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
			Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

			//load 2 elem from CW[FH, FW, IC, OC]
			const int Wic = (ty >> 1) << 1;
			const int woffset0 = Wic * OC, woffset1 = woffset0 + OC;
			Ws[buf][Ws_y    ][Ws_x] = *(float2*)(CW + woffset0);
			Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(CW + woffset1);
			__syncthreads();

			for (int oic = STEP2; oic < IC; oic += STEP2) {
#pragma unroll
				for (int ik = 0; ik < STEP2; ik++) {
					float4 a0 = *(float4*)(&Ws[buf][ik][ux << 1]);
					float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];

					simdMM4(v0, b0.x, a0); simdMM4(v1, b0.y, a0);
					simdMM4(v2, b0.z, a0); simdMM4(v3, b0.w, a0);
					simdMM4(v4, b1.x, a0); simdMM4(v5, b1.y, a0);
					simdMM4(v6, b1.z, a0); simdMM4(v7, b1.w, a0);
				}
				buf ^= 1;

				//load 4 elem from X[N, IH, IW, IC]
				const int Xic = oic + ((tx & STEP_m1) << 1);
				float2 x0 = (lx0 ? *(float2*)(X + X0 + Xic) : F32_2_0);
				float2 x1 = (lx1 ? *(float2*)(X + X1 + Xic) : F32_2_0);
				float2 x2 = (lx2 ? *(float2*)(X + X2 + Xic) : F32_2_0);
				float2 x3 = (lx3 ? *(float2*)(X + X3 + Xic) : F32_2_0);
				Xs[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
				Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

				//load 2 elem from W[FH, FW, IC, OC]
				const int Wic = oic + ((ty >> 1) << 1);
				const int woffset0 = Wic * OC, woffset1 = woffset0 + OC;
				Ws[buf][Ws_y    ][Ws_x] = *(float2*)(CW + woffset0);
				Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(CW + woffset1);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP2; ik++) {
				float4 a0 = *(float4*)(&Ws[buf][ik][ux << 1]);
				float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.y, a0);
				simdMM4(v2, b0.z, a0); simdMM4(v3, b0.w, a0);
				simdMM4(v4, b1.x, a0); simdMM4(v5, b1.y, a0);
				simdMM4(v6, b1.z, a0); simdMM4(v7, b1.w, a0);
			}
			buf ^= 1;
		}
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2; *(float4*)(Y + Y3) = v3;
	*(float4*)(Y + Y4) = v4; *(float4*)(Y + Y5) = v5;
	*(float4*)(Y + Y6) = v6; *(float4*)(Y + Y7) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_PURE_UKERNEL_4_4R
#define CONV_3D_PURE_UKERNEL_4_4R

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 2.32286 msec, Performace = 8320.49 GFlop/s
//LB = 3: Size = 9, Time = 2.80509 msec, Performace = 6890.11 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 2.31092 msec, Performace = 8363.47 GFlop/s
//LB = 3: Size = 9, Time = 2.82778 msec, Performace = 6834.81 GFlop/s

template<int LB, int STEP, int STEP2>
__global__ void conv3dPure_uernel_4_4R(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB][(2 << LB) + 2];

	//compute 4*4 results:
	float4 v0 = F32_4_0, v1 = F32_4_0;
	float4 v2 = F32_4_0, v3 = F32_4_0;

	//prepare for GN = OC
	const int boc0 = (blockIdx.x << LB << 2) + oc_index;
	const int toc0 = boc0 + (tx << 2);
	CW += toc0 + ((ty & 1) << 1);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (blockIdx.y << LB << 2) + j_index;
	const int tj0 = bj0 + (ty << 2) + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 2);
	const int yj0  = bj0  + (uy << 2);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area----------------------------------------------------
	const int Ws_y = (ty >> 1) << 1, Ws_x = (tx << 1) + (ty & 1);
	const int Xs_x = (tx >> 1) << 1, Xs_y = (ty << 1) + (tx & 1);
	const int SX = (IW - FW)*IC, SCW = IC * OC;

	for (int fh = 0; fh < FH; fh++, X += SX) {
		for (int fw = 0; fw < FW; fw++, X += IC, CW += SCW) {
			//load 2 elem from X[N, IH, IW, IC]
			bool lx0 = LOAD_X(tihs0, tiws0, fh, fw);
			bool lx1 = LOAD_X(tihs1, tiws1, fh, fw);
			const int Xic = (tx >> 1) << 1;
			float2 x0 = (lx0 ? *(float2*)(X + X0 + Xic) : F32_2_0);
			float2 x1 = (lx1 ? *(float2*)(X + X1 + Xic) : F32_2_0);
			Xs[buf][Xs_x    ][Xs_y] = float2{ x0.x, x1.x };
			Xs[buf][Xs_x + 1][Xs_y] = float2{ x0.y, x1.y };

			//load 2 elem from CW[FH, FW, IC, OC]
			const int Wic = (ty >> 1) << 1;
			const int woffset0 = Wic * OC, woffset1 = woffset0 + OC;
			Ws[buf][Ws_y    ][Ws_x] = *(float2*)(CW + woffset0);
			Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(CW + woffset1);
			__syncthreads();

			for (int oic = STEP2; oic < IC; oic += STEP2) {
#pragma unroll
				for (int ik = 0; ik < STEP2; ik++) {
					float4 a = *(float4*)(&Ws[buf][ik][ux << 1]);
					float4 b = *(float4*)(&Xs[buf][ik][uy << 1]);

					simdMM4(v0, b.x, a);
					simdMM4(v1, b.y, a);
					simdMM4(v2, b.z, a);
					simdMM4(v3, b.w, a);
				}
				buf ^= 1;

				//load 2 elem from X[N, IH, IW, IC]
				const int Xic = oic + ((tx >> 1) << 1);
				float2 x0 = (lx0 ? *(float2*)(X + X0 + Xic) : F32_2_0);
				float2 x1 = (lx1 ? *(float2*)(X + X1 + Xic) : F32_2_0);
				Xs[buf][Xs_x    ][Xs_y] = float2{ x0.x, x1.x };
				Xs[buf][Xs_x + 1][Xs_y] = float2{ x0.y, x1.y };

				//load 2 elem from CW[FH, FW, IC, OC]
				const int Wic = oic + ((ty >> 1) << 1);
				const int woffset0 = Wic * OC, woffset1 = woffset0 + OC;
				Ws[buf][Ws_y    ][Ws_x] = *(float2*)(CW + woffset0);
				Ws[buf][Ws_y + 1][Ws_x] = *(float2*)(CW + woffset1);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP2; ik++) {
				float4 a = *(float4*)(&Ws[buf][ik][ux << 1]);
				float4 b = *(float4*)(&Xs[buf][ik][uy << 1]);

				simdMM4(v0, b.x, a);
				simdMM4(v1, b.y, a);
				simdMM4(v2, b.z, a);
				simdMM4(v3, b.w, a);
			}
			buf ^= 1;
		}
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	*(float4*)(Y + Y0) = v0;
	*(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2;
	*(float4*)(Y + Y3) = v3;
}

#endif


//======[Common]======================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R
#define CONV_3D_GEMM_UERNEL_8_8R

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.9402  msec, Performace = 9961.51 GFlop/s
//LB = 3: Size = 9, Time = 2.20417 msec, Performace = 8768.53 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 1.94551 msec, Performace = 9934.32 GFlop/s
//LB = 3: Size = 9, Time = 2.20904 msec, Performace = 8749.21 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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

	//prepare for GM = N * OH * OW
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
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

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yj0 = bj0 + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	
	//compute area------------------------------------------------------
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int IW_IC = IW * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & STEP_m1) << 1;
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	const int xoffset = X_fh * IW_IC + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
			float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

			simdMM4( v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4( v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4( v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4( v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4( v8, b1.x, a0); simdMM4(v9, b1.x, a1);
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
		int X_k = ok + ((tx & STEP_m1) << 1);
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		const int xoffset = X_fh * IW_IC + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, (OH, OW) % 4 == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4
#define CONV_3D_GEMM_UERNEL_8_8R4

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.85501 msec, Performace = 10419   GFlop/s
//LB = 3: Size = 9, Time = 2.18345 msec, Performace = 8851.76 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 1.83793 msec, Performace = 10515.8 GFlop/s
//LB = 3: Size = 9, Time = 2.17532 msec, Performace = 8884.84 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R4(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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

	//prepare for GM = N * OH * OW
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw;
	const int tow2 = tow1 + sw;
	const int tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) * IC;//X[tn0, toh0, tow1, 0]

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yj0 =  bj0  + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area------------------------------------------------------
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int IW_IC = IW * IC, sw_IC = sw * IC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & STEP_m1) << 1;
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	const int xoffset = X_fh * IW_IC + X_k;
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
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ok + ((tx & STEP_m1) << 1);
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		const int xoffset = X_fh * IW_IC + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset        ) : F32_2_0);
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


//======[IC is power of 2]============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R_IC_2POW
#define CONV_3D_GEMM_UERNEL_8_8R_IC_2POW

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.90347 msec, Performace = 10153.8 GFlop/s
//LB = 3: Size = 9, Time = 2.14919 msec, Performace =  8992.9 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 1.88864 msec, Performace = 10233.5 GFlop/s
//LB = 3: Size = 9, Time = 2.13867 msec, Performace =  9037.1 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R_IC2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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

	//prepare for GM = N * OH * OW
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
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

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yj0 = bj0 + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area------------------------------------------------------
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & STEP_m1) << 1;
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	const int xoffset = (X_fh << LIC)*IW + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
	Xs[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
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
		int X_k = ok + ((tx & STEP_m1) << 1);
		int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
		const int xoffset = (X_fh << LIC)*IW + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, (OH, OW) % 4 == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4_IC_2POW
#define CONV_3D_GEMM_UERNEL_8_8R4_IC_2POW

//when [FH, FW] = 7, [sh, sw] = 3:
//for: (64, 64) -> (32, 32), [FH, FW] = 5, [N, IC, OC] = [128, 64, 128]
//LB = 4: Size = 49, Time =  9.9050 msec, Performace = 10623.6 GFlop/s
//LB = 3: Size = 49, Time = 11.9651 msec, Performace =  8794.5 GFlop/s
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 49, Time =  9.7794 msec, Performace = 10760.1 GFlop/s
//LB = 3: Size = 49, Time = 11.1512 msec, Performace =  9436.4 GFlop/s
//when [FH, FW] = 2, [sh, sw] = 1:
//for: (32, 32) -> (32, 32), [FH, FW] = 3, [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 32, Time = 6.39618 msec, Performace = 10743.8 GFlop/s
//for: (16, 16) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 32, Time = 6.21444 msec, Performace = 11058 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R4_IC2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
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
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
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
	const int sw_IC = sw << LIC;
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & STEP_m1) << 1;
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	const int xoffset = (X_fh << LIC)*IW + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[0][(tx << 1)    ][ty] = make_float4(x0.x, x1.x, x2.x, x3.x);
	Xs[0][(tx << 1) + 1][ty] = make_float4(x0.y, x1.y, x2.y, x3.y);

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
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
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ok + ((tx & STEP_m1) << 1);
		int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
		int xoffset = (X_fh << LIC)*IW + X_k;
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


//======[FW, IC is power of 2]=========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, (OH, OW) % 4 == 0
//LB = 4, GK % 8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4_FW_IC_2POW
#define CONV_3D_GEMM_UERNEL_8_8R4_FW_IC_2POW

//for: (32, 32) -> (32, 32), [FH, FW] = 3, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 32, Time = 6.24814 msec, Performace = 10998.4 GFlop/s
//for: (32, 32) -> (32, 32), [FH, FW] = 3, [N, IC, OC] = [128, 128, 128]
//LB = 4: Size =  8, Time = 1.68305 msec, Performace = 10207.6 GFlop/s
//for: (16, 16) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 32, Time = 6.00633 msec, Performace = 11441.2 GFlop/s
//for: (16, 16) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size =  8, Time = 1.6004  msec, Performace = 10734.7 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R4_FW_IC2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int LFW,
	      float* __restrict__  Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
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
	const int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yj0  = bj0  + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area----------------------------------------------------
	const int LFW_IC = LFW + LIC, LFW_IC_m1 = (1 << LFW_IC) - 1;
	const int GK = FH << LFW_IC;
	const int sw_IC = sw << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & STEP_m1) << 1;
	int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	const int xoffset = (X_fh << LIC)*IW + X_k;
	float2 x0 = (lx0 ? *(float2*)(X + xoffset -        sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset               ) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset +        sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = W_k * OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset     );
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
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

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = W_k * OC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset     );
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ok + ((tx & STEP_m1) << 1);
		int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
		const int xoffset = (X_fh << LIC)*IW + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh); 
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset -        sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset               ) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset +        sw_IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
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