#pragma once

#ifndef CONV_3D_GEMMR_UERNEL_COORPERATE_H
#define CONV_3D_GEMMR_UERNEL_COORPERATE_H

//(1) cooperate with Im2col-Winograd
//(2) GM = N*OH*OWr
//(3) Remode the kernel: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV_3D_GEMMR_UERNEL_COOPERARE_CALL
#define CONV_3D_GEMMR_UERNEL_COOPERARE_CALL

//LB = log2(BLOCK_SIZE)

//======[sh = sw = 1, OWr % 4 == 0]=====================================
#define conv3dGemm_u88R4CS1_ruse(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4CS1_ruse<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,OH,OW, IC,OC, ph,pw,\
			oc_index,j_index, ow_index)

//================[FH = FW = 6]=============================================
//OWr % 2 == 0
#define conv3dGemm_u88R2C_W6_ic2pow(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R2C_W6_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,OH,OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

#define conv3dGemm_u88RC_W6_ic2pow(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RC_W6_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,OH,OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

//================[FH = FW = 5]=============================================
//OWr % 2 == 0
#define conv3dGemm_u88R2C_W5_ic2pow(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R2C_W5_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,OH,OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

#define conv3dGemm_u88RC_W5_ic2pow(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RC_W5_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,OH,OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)


//================[FW, IC are power of 2]===================================
//OWr % 2 == 0
#define conv3dGemm_u88R2C_fw_ic2pow(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R2C_fw_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, LFW, ((1<<LFW)-1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH, Y,OH,OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

#define conv3dGemm_u88RC_fw_ic2pow(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RC_fw_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, LFW, ((1<<LFW)-1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH, Y,OH,OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

//================[IC is power of 2]========================================
//OWr % 2 == 0
#define conv3dGemm_u88R2C_ic2pow(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R2C_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,OH,OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

#define conv3dGemm_u88RC_ic2pow(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RC_ic2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,OH,OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

//================[Commons]=================================================
//OWr % 2 == 0
#define conv3dGemm_u88R2C(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R2C<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

#define conv3dGemm_u88RC(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8RC<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

#define conv3dPure_u84RC(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_uernel_8_4RC<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

#define conv3dPure_u48RC(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_uernel_4_8RC<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

#define conv3dPure_u44RC(stream, LB, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_uernel_4_4RC<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW,FH,FW, Y,OH,OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index, ow_index)

#endif


//======[sh = sw = 1, OWr % 4 == 0]=====================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4C_S1_RUSE
#define CONV_3D_GEMM_UERNEL_8_8R4C_S1_RUSE

//when: (FH, FW) = 9, (sh, sw) = 1
//for: Feature = (16, 16), [N, IC, OC] = [128,256, 256]
//LB = 4: Size = 162, Time = 29.4093 msec, Performace = 11829.3 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1, int FH, int FW>
__global__ void conv3dGemm_uernel_8_8R4CS1_ruse(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index, int ow_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4  v0 = F32_4_0,  v1 = F32_4_0; float4  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0; float4  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += ((ty & STEP_m1) << 1) * OC + toc0;//CW[0, 0, (ty & STEP_m1), toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;//OWr % 4 == 0
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC + ((tx & STEP_m1) << 1);//X += X1;
	
	//prepare for Y[N, OH, OW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	//compute area-----------------------------------------------------------
	for (int fh = 0; fh < FH; fh++) {
		for (int oic = 0; oic < IC; oic += STEP2) {
			//load 4 elements from X[N, IH, IW, IC]
			bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
			bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
			bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
			bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
			bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
			const int xoffset0 = fh * IW * IC + oic;
			float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
			float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
			float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
			float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
			float4 xv0 = { x0.x, x1.x, x2.x, x3.x };
			float4 xv1 = { x0.y, x1.y, x2.y, x3.y };

			//load 4 elements from CW[FH, FW, IC, OC]
			const int woffset0 = (fh * FW * IC + oic)*OC;
			float4 wv0 = *(float4*)(CW + woffset0);
			float4 wv1 = *(float4*)(CW + woffset0 + OC);

			//write to shared memory
			Xs[buf][(tx << 1)][ty] = xv0; Xs[buf][(tx << 1) + 1][ty] = xv1;
			Ws[buf][(ty << 1)][tx] = wv0; Ws[buf][(ty << 1) + 1][tx] = wv1;
			__syncthreads();

			for (int fw = 1; fw < FW; fw++) {
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
				float4 ox0 = Xs[buf][(tx << 1)    ][ty];//update_shared_memory
				float4 ox1 = Xs[buf][(tx << 1) + 1][ty];
				buf ^= 1;

				//load 4 elements from X[N, IH, IW, IC]
				bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
				float2 nx3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
				float4 xv0 = { ox0.y, ox0.z, ox0.w, nx3.x };
				float4 xv1 = { ox1.y, ox1.z, ox1.w, nx3.y };
				
				//load 4 elements from CW[FH, FW, IC, OC]
				const int woffset0 = ((fh*FW + fw)*IC + oic)*OC;
				float4 wv0 = *(float4*)(CW + woffset0);
				float4 wv1 = *(float4*)(CW + woffset0 + OC);

				//write to shared memory
				Xs[buf][(tx << 1)][ty] = xv0; Xs[buf][(tx << 1) + 1][ty] = xv1;
				Ws[buf][(ty << 1)][tx] = wv0; Ws[buf][(ty << 1) + 1][tx] = wv1;
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
			buf ^= 1;
		}
	}
	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3), yj4 = yj0 + 4;//8
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0;

	const int Y1 = Y0 + OC, Y2 = Y0 + (OC << 1), Y3 = Y0 + OC * 3;
	const int Y5 = Y4 + OC, Y6 = Y4 + (OC << 1), Y7 = Y4 + OC * 3;

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


//================[FH = FW = 6]=============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R2C_W6_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8R2C_W6_IC2POW

//when [FH, FW] = 6, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 36, Time = 7.02716 msec, Performace = 11001.5 GFlop/s
//for: (16, 16) -> ( 8,  8), [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 36, Time = 7.08311 msec, Performace = 10914.6 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R2C_W6_ic2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 6
	      float* __restrict__  Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results:
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (by << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw; int tow1 = tow0 + sw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw; int tow3 = tow2 + sw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	//compute area------------------------------------------------------
	const int GK = 36 << LIC;//GK = FH * FW * IC
	const int sw_IC = sw << LIC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int Idx = X_k >> LIC; char fhw = XIDX_W6[Idx];
	const int X_fh = fhw >> 3, X_fw = fhw & 7;
	const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh0, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh2, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset        ) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X0 + xoffset + sw_IC) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset        ) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X2 + xoffset + sw_IC) : F32_2_0);
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
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = X_k >> LIC; char fhw = XIDX_W6[Idx];
		const int X_fh = fhw >> 3, X_fw = fhw & 7;
		const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh0, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh2, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X0 + xoffset + sw_IC) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X2 + xoffset + sw_IC) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3);//8
	const int yj2 = yj0 + 2, yj4 = yj0 + 4, yj6 = yj0 + 6;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0, Y1 = Y0 + OC;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0, Y3 = Y2 + OC;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0, Y5 = Y4 + OC;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0, Y7 = Y6 + OC;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RC_W6_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8RC_W6_IC2POW

//when [FH, FW] = 6, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 36, Time = 7.18548 msec, Performace = 10759.1 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 25, Time = 4.95497 msec, Performace = 10835 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8RC_W6_ic2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 6
	      float* __restrict__  Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results:
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (by << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj1, tn1, toh1, tow1, OH_OWr, OWr); tow1 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	get_n_oh_ow_Temp(tj3, tn3, toh3, tow3, OH_OWr, OWr); tow3 += ow_index;
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

	//compute area------------------------------------------------------
	const int GK = 36 << LIC;//GK = FH * FW * IC

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int Idx = X_k >> LIC; char fhw = XIDX_W6[Idx];
	const int X_fh = fhw >> 3, X_fw = fhw & 7;
	const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
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
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = X_k >> LIC; char fhw = XIDX_W6[Idx];
		const int X_fh = fhw >> 3, X_fw = fhw & 7;
		const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3);//8
	const int yj1 = yj0 + 1, yj2 = yj0 + 2, yj3 = yj0 + 3;
	const int yj4 = yj0 + 4, yj5 = yj0 + 5, yj6 = yj0 + 6, yj7 = yj0 + 7;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr, OWr); yow1 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj3, yn3, yoh3, yow3, OH_OWr, OWr); yow3 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj5, yn5, yoh5, yow5, OH_OWr, OWr); yow5 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;
	get_n_oh_ow_Temp(yj7, yn7, yoh7, yow7, OH_OWr, OWr); yow7 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;
	const int Y1 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0;
	const int Y3 = ((yn3*OH + yoh3)*OW + yow3)*OC + yoc0;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0;
	const int Y5 = ((yn5*OH + yoh5)*OW + yow5)*OC + yoc0;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0;
	const int Y7 = ((yn7*OH + yoh7)*OW + yow7)*OC + yoc0;

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


//================[FH = FW = 5]=============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, OWr2 % 2 == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R2C_W5_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8R2C_W5_IC2POW

//when [FH, FW] = 5, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 25, Time = 4.91508 msec, Performace = 10922.9 GFlop/s
//for: (16, 16) -> ( 8,  8), [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 25, Time = 4.89289 msec, Performace = 10972.5 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R2C_W5_ic2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	      float* __restrict__  Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results:
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (by << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw; int tow1 = tow0 + sw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw; int tow3 = tow2 + sw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	//compute area------------------------------------------------------
	const int GK = 25 << LIC;//GK = FH * FW * IC
	const int sw_IC = sw << LIC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
	const int X_fh = fhw >> 3, X_fw = fhw & 7;
	const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh0, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh2, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset        ) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X0 + xoffset + sw_IC) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset        ) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X2 + xoffset + sw_IC) : F32_2_0);
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
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
		const int X_fh = fhw >> 3, X_fw = fhw & 7;
		const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh0, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh2, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X0 + xoffset + sw_IC) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X2 + xoffset + sw_IC) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3);//8
	const int yj2 = yj0 + 2, yj4 = yj0 + 4, yj6 = yj0 + 6;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0, Y1 = Y0 + OC;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0, Y3 = Y2 + OC;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0, Y5 = Y4 + OC;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0, Y7 = Y6 + OC;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RC_W5_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8RC_W5_IC2POW

//when [FH, FW] = 5, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 25, Time = 5.03784 msec, Performace = 10656.8 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 25, Time = 4.95497 msec, Performace = 10835 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8RC_W5_ic2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 5
	      float* __restrict__  Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results:
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (by << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj1, tn1, toh1, tow1, OH_OWr, OWr); tow1 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	get_n_oh_ow_Temp(tj3, tn3, toh3, tow3, OH_OWr, OWr); tow3 += ow_index;
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

	//compute area------------------------------------------------------
	const int GK = 25 << LIC;//GK = FH * FW * IC

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
	const int X_fh = fhw >> 3, X_fw = fhw & 7;
	const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
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
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = X_k >> LIC; char fhw = XIDX_W5[Idx];
		const int X_fh = fhw >> 3, X_fw = fhw & 7;
		const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3);//8
	const int yj1 = yj0 + 1, yj2 = yj0 + 2, yj3 = yj0 + 3;
	const int yj4 = yj0 + 4, yj5 = yj0 + 5, yj6 = yj0 + 6, yj7 = yj0 + 7;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr, OWr); yow1 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj3, yn3, yoh3, yow3, OH_OWr, OWr); yow3 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj5, yn5, yoh5, yow5, OH_OWr, OWr); yow5 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;
	get_n_oh_ow_Temp(yj7, yn7, yoh7, yow7, OH_OWr, OWr); yow7 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;
	const int Y1 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0;
	const int Y3 = ((yn3*OH + yoh3)*OW + yow3)*OC + yoc0;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0;
	const int Y5 = ((yn5*OH + yoh5)*OW + yow5)*OC + yoc0;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0;
	const int Y7 = ((yn7*OH + yoh7)*OW + yow7)*OC + yoc0;

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


//================[FW, IC are power of 2]===================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, OWr2 % 2 == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R2C_FW_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8R2C_FW_IC2POW

//when [FH, FW] = 4, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.87222 msec, Performace = 10323.2 GFlop/s
//for: (16, 16) -> ( 8,  8), [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 16, Time = 3.16684 msec, Performace = 10849.8 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1, int LFW, int FW_m1>
__global__ void conv3dGemm_uernel_8_8R2C_fw_ic2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH,
	      float* __restrict__  Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results:
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (by << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw; int tow1 = tow0 + sw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw; int tow3 = tow2 + sw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	//compute area------------------------------------------------------
	const int GK = FH << LFW << LIC;//GK = FH * FW * IC
	const int IC_m1 = (1 << LIC) - 1, sw_IC = sw << LIC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int X_fhw = X_k >> LIC;
	const int X_fh = X_fhw >> LFW, X_fw = X_fhw & FW_m1;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh0, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh2, tow3, X_fh, X_fw);
	const int xoffset = ((X_fh*IW + X_fw) << LIC) + (X_k & IC_m1);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset        ) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X0 + xoffset + sw_IC) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset        ) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X2 + xoffset + sw_IC) : F32_2_0);
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
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int X_fhw = X_k >> LIC;
		const int X_fh = X_fhw >> LFW, X_fw = X_fhw & FW_m1;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh0, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh2, tow3, X_fh, X_fw);
		const int xoffset = ((X_fh*IW + X_fw) << LIC) + (X_k & IC_m1);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X0 + xoffset + sw_IC) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X2 + xoffset + sw_IC) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3);//8
	const int yj2 = yj0 + 2, yj4 = yj0 + 4, yj6 = yj0 + 6;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0, Y1 = Y0 + OC;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0, Y3 = Y2 + OC;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0, Y5 = Y4 + OC;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0, Y7 = Y6 + OC;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RC_FW_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8RC_FW_IC2POW

//when [FH, FW] = 4, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.87222 msec, Performace = 10323.2 GFlop/s
//for: (16, 16) -> ( 8,  8), [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 16, Time = 3.23926 msec, Performace = 10607.3 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1, int LFW, int FW_m1>
__global__ void conv3dGemm_uernel_8_8RC_fw_ic2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH,
	      float* __restrict__  Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results:
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (by << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj1, tn1, toh1, tow1, OH_OWr, OWr); tow1 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	get_n_oh_ow_Temp(tj3, tn3, toh3, tow3, OH_OWr, OWr); tow3 += ow_index;
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

	//compute area------------------------------------------------------
	const int GK = FH << LFW << LIC;//GK = FH * FW * IC
	const int IC_m1 = (1 << LIC) - 1;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int X_fhw = X_k >> LIC;
	const int X_fh = X_fhw >> LFW, X_fw = X_fhw & FW_m1;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
	const int xoffset = ((X_fh*IW + X_fw) << LIC) + (X_k & IC_m1);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X3 + xoffset) : F32_2_0);
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
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int X_fhw = X_k >> LIC;
		const int X_fh = X_fhw >> LFW, X_fw = X_fhw & FW_m1;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh3, tow3, X_fh, X_fw);
		const int xoffset = ((X_fh*IW + X_fw) << LIC) + (X_k & IC_m1);
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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3);//8
	const int yj1 = yj0 + 1, yj2 = yj0 + 2, yj3 = yj0 + 3;
	const int yj4 = yj0 + 4, yj5 = yj0 + 5, yj6 = yj0 + 6, yj7 = yj0 + 7;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr, OWr); yow1 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj3, yn3, yoh3, yow3, OH_OWr, OWr); yow3 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj5, yn5, yoh5, yow5, OH_OWr, OWr); yow5 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;
	get_n_oh_ow_Temp(yj7, yn7, yoh7, yow7, OH_OWr, OWr); yow7 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;
	const int Y1 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0;
	const int Y3 = ((yn3*OH + yoh3)*OW + yow3)*OC + yoc0;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0;
	const int Y5 = ((yn5*OH + yoh5)*OW + yow5)*OC + yoc0;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0;
	const int Y7 = ((yn7*OH + yoh7)*OW + yow7)*OC + yoc0;

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


//================[IC is power of 2]========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, OWr % 2 == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R2C_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8R2C_IC2POW

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.86472 msec, Performace = 10364.8 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 1.83095 msec, Performace = 10555.9 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R2C_ic2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results:
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (by << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw; int tow1 = tow0 + sw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw; int tow3 = tow2 + sw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2) << LIC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	//compute area------------------------------------------------------
	const int FW_IC = FW << LIC, GK = FH * FW_IC;
	const int sw_IC = sw << LIC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & STEP_m1) << 1;
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	const int xoffset = (X_fh << LIC)*IW + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh0, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh2, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset        ) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X0 + xoffset + sw_IC) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset        ) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X2 + xoffset + sw_IC) : F32_2_0);
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
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ok + ((tx & STEP_m1) << 1);
		int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
		const int xoffset = (X_fh << LIC)*IW + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh0, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh2, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X0 + xoffset + sw_IC) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X2 + xoffset + sw_IC) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3);//8
	const int yj2 = yj0 + 2, yj4 = yj0 + 4, yj6 = yj0 + 6;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0, Y1 = Y0 + OC;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0, Y3 = Y2 + OC;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0, Y5 = Y4 + OC;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0, Y7 = Y6 + OC;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RC_IC2POW
#define CONV_3D_GEMM_UERNEL_8_8RC_IC2POW

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.87222 msec, Performace = 10323.2 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 1.84274 msec, Performace = 10488.4 GFlop/s
//when [FH, FW] = 5, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 25, Time = 5.00298 msec, Performace = 10731 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 25, Time = 5.06191 msec, Performace = 10606.1 GFlop/s
//when [FH, FW] = 6, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 36, Time = 7.27246 msec, Performace = 10630.4 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4:  Size = 36, Time = 7.22232 msec, Performace = 10704.2 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8RC_ic2pow(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y, int OH, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results:
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (by << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj1, tn1, toh1, tow1, OH_OWr, OWr); tow1 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	get_n_oh_ow_Temp(tj3, tn3, toh3, tow3, OH_OWr, OWr); tow3 += ow_index;
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

	//compute area------------------------------------------------------
	const int FW_IC = FW << LIC, GK = FH * FW_IC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & STEP_m1) << 1;
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	const int xoffset = (X_fh << LIC)*IW  + X_k;
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
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3);//8
	const int yj1 = yj0 + 1, yj2 = yj0 + 2, yj3 = yj0 + 3;
	const int yj4 = yj0 + 4, yj5 = yj0 + 5, yj6 = yj0 + 6, yj7 = yj0 + 7;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr, OWr); yow1 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj3, yn3, yoh3, yow3, OH_OWr, OWr); yow3 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj5, yn5, yoh5, yow5, OH_OWr, OWr); yow5 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;
	get_n_oh_ow_Temp(yj7, yn7, yoh7, yow7, OH_OWr, OWr); yow7 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;
	const int Y1 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0;
	const int Y3 = ((yn3*OH + yoh3)*OW + yow3)*OC + yoc0;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0;
	const int Y5 = ((yn5*OH + yoh5)*OW + yow5)*OC + yoc0;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0;
	const int Y7 = ((yn7*OH + yoh7)*OW + yow7)*OC + yoc0;

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


//================[Commons]=================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, OWr % 2 == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R2C
#define CONV_3D_GEMM_UERNEL_8_8R2C

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.93952 msec, Performace = 9965.01 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 1.88285 msec, Performace = 10264.9 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R2C(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results:
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (by << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw; int tow1 = tow0 + sw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw; int tow3 = tow2 + sw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int X2 = ((tn2*IH + toh2)*IW + tow2)*IC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	//compute area------------------------------------------------------
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int IW_IC = IW * IC, sw_IC = sw * IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & STEP_m1) << 1;
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	const int xoffset = X_fh * IW_IC + X_k;
	bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
	bool lx1 = LOAD_X(toh0, tow1, X_fh, X_fw);
	bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
	bool lx3 = LOAD_X(toh2, tow3, X_fh, X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset        ) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X0 + xoffset + sw_IC) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset        ) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + X2 + xoffset + sw_IC) : F32_2_0);
	Xs[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from CW[FH, FW, IC, OC]
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

		//load 4 elements from CW[FH, FW, IC, OC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ok + ((tx & STEP_m1) << 1);
		int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		const int xoffset = X_fh * IW_IC + X_k;
		bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
		bool lx1 = LOAD_X(toh0, tow1, X_fh, X_fw);
		bool lx2 = LOAD_X(toh2, tow2, X_fh, X_fw);
		bool lx3 = LOAD_X(toh2, tow3, X_fh, X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X0 + xoffset + sw_IC) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + X2 + xoffset) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + X2 + xoffset + sw_IC) : F32_2_0);
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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3);//8
	const int yj2 = yj0 + 2, yj4 = yj0 + 4, yj6 = yj0 + 6;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0, Y1 = Y0 + OC;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0, Y3 = Y2 + OC;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0, Y5 = Y4 + OC;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0, Y7 = Y6 + OC;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8RC
#define CONV_3D_GEMM_UERNEL_8_8RC

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8RC(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results:
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	const int bj0 = (by << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	const int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj1, tn1, toh1, tow1, OH_OWr, OWr); tow1 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	get_n_oh_ow_Temp(tj3, tn3, toh3, tow3, OH_OWr, OWr); tow3 += ow_index;
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
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 3);//8
	const int yj1 = yj0 + 1, yj2 = yj0 + 2, yj3 = yj0 + 3;
	const int yj4 = yj0 + 4, yj5 = yj0 + 5, yj6 = yj0 + 6, yj7 = yj0 + 7;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr, OWr); yow1 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj3, yn3, yoh3, yow3, OH_OWr, OWr); yow3 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj5, yn5, yoh5, yow5, OH_OWr, OWr); yow5 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;
	get_n_oh_ow_Temp(yj7, yn7, yoh7, yow7, OH_OWr, OWr); yow7 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;
	const int Y1 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0;
	const int Y3 = ((yn3*OH + yoh3)*OW + yow3)*OC + yoc0;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0;
	const int Y5 = ((yn5*OH + yoh5)*OW + yow5)*OC + yoc0;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0;
	const int Y7 = ((yn7*OH + yoh7)*OW + yow7)*OC + yoc0;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_PURE_UERNEL_8_4RC
#define CONV_3D_PURE_UERNEL_8_4RC

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 18, Time = 4.03341 msec, Performace = 9583.64 GFlop/s
//LB = 3: Size = 18, Time = 4.43502 msec, Performace = 8715.79 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 18, Time = 4.06261 msec, Performace = 9514.76 GFlop/s
//LB = 3: Size = 18, Time = 4.33775 msec, Performace = 8911.24 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dPure_uernel_8_4RC(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Xs[2][1 << LB][(2 << LB) + 2];//follow k44

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int boc0 = (blockIdx.x << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]
	
	//prepare for GM = N * OH * OW
	const int bj0 = (blockIdx.y << LB << 2) + j_index;
	const int tj0 = bj0 + (ty << 2) + ((tx & 1) << 1), tj1 = tj0 + 1;
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj1, tn1, toh1, tow1, OH_OWr, OWr); tow1 += ow_index;
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int X0 = (((tn0*IH) + tihs0)*IW + tiws0) * IC;
	const int X1 = (((tn1*IH) + tihs1)*IW + tiws1) * IC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

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

	const int yoc0 = boc0 + (ux << 3);//8
	const int yj0  = bj0  + (uy << 2);//4
	const int yj1 = yj0 + 1, yj2 = yj0 + 2, yj3 = yj0 + 3;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr, OWr); yow1 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj3, yn3, yoh3, yow3, OH_OWr, OWr); yow3 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;
	const int Y1 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0;
	const int Y3 = ((yn3*OH + yoh3)*OW + yow3)*OC + yoc0;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_PURE_UERNEL_4_8RC
#define CONV_3D_PURE_UERNEL_4_8RC

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 2.01913 msec, Performace = 9572.14 GFlop/s
//LB = 3: Size = 9, Time = 2.41652 msec, Performace = 7998.02 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 2.02843 msec, Performace = 9528.23 GFlop/s
//LB = 3: Size = 9, Time = 2.40068 msec, Performace = 8050.78 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dPure_uernel_4_8RC(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
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
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj1, tn1, toh1, tow1, OH_OWr, OWr); tow1 += ow_index;
	get_n_oh_ow_Temp(tj2, tn2, toh2, tow2, OH_OWr, OWr); tow2 += ow_index;
	get_n_oh_ow_Temp(tj3, tn3, toh3, tow3, OH_OWr, OWr); tow3 += ow_index;
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

	const int yoc0 = boc0 + (ux << 2);//4
	const int yj0  = bj0  + (uy << 3);//8
	const int yj1 = yj0 + 1, yj2 = yj0 + 2, yj3 = yj0 + 3;
	const int yj4 = yj0 + 4, yj5 = yj0 + 5, yj6 = yj0 + 6, yj7 = yj0 + 7;
	
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr, OWr); yow1 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj3, yn3, yoh3, yow3, OH_OWr, OWr); yow3 += ow_index;
	get_n_oh_ow_Temp(yj4, yn4, yoh4, yow4, OH_OWr, OWr); yow4 += ow_index;
	get_n_oh_ow_Temp(yj5, yn5, yoh5, yow5, OH_OWr, OWr); yow5 += ow_index;
	get_n_oh_ow_Temp(yj6, yn6, yoh6, yow6, OH_OWr, OWr); yow6 += ow_index;
	get_n_oh_ow_Temp(yj7, yn7, yoh7, yow7, OH_OWr, OWr); yow7 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;
	const int Y1 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0;
	const int Y3 = ((yn3*OH + yoh3)*OW + yow3)*OC + yoc0;
	const int Y4 = ((yn4*OH + yoh4)*OW + yow4)*OC + yoc0;
	const int Y5 = ((yn5*OH + yoh5)*OW + yow5)*OC + yoc0;
	const int Y6 = ((yn6*OH + yoh6)*OW + yow6)*OC + yoc0;
	const int Y7 = ((yn7*OH + yoh7)*OW + yow7)*OC + yoc0;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2; *(float4*)(Y + Y3) = v3;
	*(float4*)(Y + Y4) = v4; *(float4*)(Y + Y5) = v5;
	*(float4*)(Y + Y6) = v6; *(float4*)(Y + Y7) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), IC % BLOCK_SIZE == 0
//LB = 4, IC % 16 == 0
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_PURE_UKERNEL_4_4RC
#define CONV_3D_PURE_UKERNEL_4_4RC

//when [FH, FW] = 3, [sh, sw] = 2:
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 2.32286 msec, Performace = 8320.49 GFlop/s
//LB = 3: Size = 9, Time = 2.80509 msec, Performace = 6890.11 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 2.31092 msec, Performace = 8363.47 GFlop/s
//LB = 3: Size = 9, Time = 2.82778 msec, Performace = 6834.81 GFlop/s

template<int LB, int STEP, int STEP2>
__global__ void conv3dPure_uernel_4_4RC(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW, int FH, int FW,
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index, int ow_index)
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
	const int OWr = OW - ow_index, OH_OWr = OH * OWr;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr, OWr); tow0 += ow_index;
	get_n_oh_ow_Temp(tj1, tn1, toh1, tow1, OH_OWr, OWr); tow1 += ow_index;
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int X0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int X1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

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

	const int yoc0 = boc0 + (ux << 2);//4
	const int yj0  = bj0  + (uy << 2);//4
	const int yj1 = yj0 + 1, yj2 = yj0 + 2, yj3 = yj0 + 3;

	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr, OWr); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr, OWr); yow1 += ow_index;
	get_n_oh_ow_Temp(yj2, yn2, yoh2, yow2, OH_OWr, OWr); yow2 += ow_index;
	get_n_oh_ow_Temp(yj3, yn3, yoh3, yow3, OH_OWr, OWr); yow3 += ow_index;

	const int Y0 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;
	const int Y1 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;
	const int Y2 = ((yn2*OH + yoh2)*OW + yow2)*OC + yoc0;
	const int Y3 = ((yn3*OH + yoh3)*OW + yow3)*OC + yoc0;

	*(float4*)(Y + Y0) = v0;
	*(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2;
	*(float4*)(Y + Y3) = v3;
}

#endif


//================[Integration: 32(OC) * 32(N*OH)]==========================
#ifndef CONV_3D_GEMMR_32X32RC
#define CONV_3D_GEMMR_32X32RC

#ifndef CONV_3D_GEMMR_32X32RC_MICRO
#define CONV_3D_GEMMR_32X32RC_MICRO

#define conv3d_Gemm32x32RC_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index  = (GM - GMr) +  j_index;\
		conv3D_Gemm_32x32RC(streams,index,length, X,IH,IW, CW,FH,FW, Y,OH,OW,OWr, IC,OC, sh,sw,ph,pw,\
			GNr, GM, next_oc_index,j_index,ow_index);\
		conv3D_Gemm_32x32RC(streams,index,length, X,IH,IW, CW,FH,FW, Y,OH,OW,OWr, IC,OC, sh,sw,ph,pw,\
            GN, GMr, oc_index,next_j_index,ow_index);\
		conv3D_Gemm_32x32RC(streams,index,length, X,IH,IW, CW,FH,FW, Y,OH,OW,OWr, IC,OC, sh,sw,ph,pw,\
            GNr, GMr, next_oc_index, next_j_index, ow_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		conv3D_Gemm_32x32RC(streams,index,length, X,IH,IW, CW,FH,FW, Y,OH,OW,OWr, IC,OC, sh,sw,ph,pw,\
			 GNr, GM, next_oc_index,j_index,ow_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		conv3D_Gemm_32x32RC(streams,index,length, X,IH,IW, CW,FH,FW, Y,OH,OW,OWr, IC,OC, sh,sw,ph,pw,\
			 GN, GMr, oc_index,next_j_index,ow_index);}}

#endif

//(1) OWr = OW - ow_index; => ow_index = OW - OWr
//(2) GN = ((OC    ) / 32 * 32);
//(3) GM = ((N * OH) / 32 * 32) * OWr;
//(4) IC % 8 == 0
void conv3D_Gemm_32x32RC(jlong* streams, int &index, int length,
	const float*  X, int IH, int IW,
	const float* CW, int FH, int FW,
	      float*  Y, int OH, int OW, int OWr,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int oc_index, int j_index, int ow_index)
{
	next_cudaStream(stream, streams, index, length);

#ifdef ENABLE_CONV3D_GEMM_UERNEL_32X32_RC
	if ((GN > 127) && (GM > 127) && !(IC & 15)) {//[128, 128]
		//======[OWr % 4 == 0]======================================================================
		if (!(OWr & 3) && sh == 1 && sw == 1) {
			if (FH == 9 && FW == 9) { conv3dGemm_u88R4CS1_ruse(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, 9, 9, Y, OH, OW, IC, OC, ph, pw, GN, GM); goto END_127x127; }
			if (FH == 8 && FW == 8) { conv3dGemm_u88R4CS1_ruse(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, 8, 8, Y, OH, OW, IC, OC, ph, pw, GN, GM); goto END_127x127; }
		}
		//======[OWr % 2 == 0]======================================================================
		if (!(OWr & 1)) {
			if (IS_POWER2(IC)) {
				if      (FH == 6 && FW == 6) conv3dGemm_u88R2C_W6_ic2pow(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else if (FH == 5 && FW == 5) conv3dGemm_u88R2C_W5_ic2pow(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else if (FW == 4) conv3dGemm_u88R2C_fw_ic2pow(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, FH, 2, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_u88R2C_ic2pow(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else conv3dGemm_u88R2C(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[Common]============================================================================
		else if (IS_POWER2(IC)) {
			if      (FH == 6 && FW == 6) conv3dGemm_u88RC_W6_ic2pow(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else if (FH == 5 && FW == 5) conv3dGemm_u88RC_W5_ic2pow(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else if (FW == 4) conv3dGemm_u88RC_fw_ic2pow(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, FH, 2, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_u88RC_ic2pow(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		else conv3dGemm_u88RC(stream, 4, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
END_127x127:
		conv3d_Gemm32x32RC_Branch(127, 127); return;
	}
	
	if (GN > 63 && GM > 63) {//[64, 64]
		//======[OWr % 4 == 0]======================================================================
		if (!(OWr & 3) && sh == 1 && sw == 1) {
			if (FH == 9 && FW == 9) { conv3dGemm_u88R4CS1_ruse(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, 9, 9, Y, OH, OW, IC, OC, ph, pw, GN, GM); goto END_63x63; }
		}
		//======[OWr % 2 == 0]======================================================================
		if (!(OWr & 1)) {
			if (IS_POWER2(IC)) {
				if      (FH == 6 && FW == 6) conv3dGemm_u88R2C_W6_ic2pow(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else if (FH == 5 && FW == 5) conv3dGemm_u88R2C_W5_ic2pow(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else if (FW == 4) conv3dGemm_u88R2C_fw_ic2pow(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, FH, 2, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
				else conv3dGemm_u88R2C_ic2pow(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			}
			else conv3dGemm_u88R2C(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		//======[Common]============================================================================
		else if (IS_POWER2(IC)) {
			if      (FH == 6 && FW == 6) conv3dGemm_u88RC_W6_ic2pow(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else if (FH == 5 && FW == 5) conv3dGemm_u88RC_W5_ic2pow(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else if (FW == 4) conv3dGemm_u88RC_fw_ic2pow(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, FH, 2, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
			else conv3dGemm_u88RC_ic2pow(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, LOG2(IC), OC, sh, sw, ph, pw, GN, GM);
		}
		else conv3dGemm_u88RC(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
END_63x63:
		conv3d_Gemm32x32RC_Branch(63, 63); return;
	}

	if (GN > 63 && GM > 31) {//[64, 32]
		conv3dPure_u84RC(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3d_Gemm32x32RC_Branch(63, 31); return;
	}

	if (GN > 31 && GM > 63) {//[32, 64]
		conv3dPure_u48RC(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3d_Gemm32x32RC_Branch(31, 63); return;
	}

	if (GN > 31 && GM > 31) {//[32, 32]
		conv3dPure_u44RC(stream, 3, oc_index, j_index, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM);
		conv3d_Gemm32x32RC_Branch(31, 31); return;
	}

#endif 
}

#endif


#endif