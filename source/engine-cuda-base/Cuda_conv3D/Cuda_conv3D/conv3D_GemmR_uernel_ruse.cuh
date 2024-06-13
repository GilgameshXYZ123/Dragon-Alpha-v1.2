#pragma once

#ifndef CONV_3D_GEMMR_UERNEL_RUSE_H
#define CONV_3D_GEMMR_UERNEL_RUSE_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV_3D_GEMMR_UERNEL_RUSE_CALL
#define CONV_3D_GEMMR_UERNEL_RUSE_CALL

//======[sh = sw = 1]==================================================
//OW % 4 == 0
#define conv3dGemm_u88R4S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4S1_ruse<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//======[FH = FW = 3, sh = sw = 1]======================================
//OW % 4 == 0
#define conv3dGemm_u88R4W3S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W3S1_ruse<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//======[FH = FW = 2, sh = sw = 1]=====================================
//OW % 4 == 0
#define conv3dGemm_u88R4W2S1_ruse(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dGemm_uernel_8_8R4W2S1_ruse<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

#endif


//======[sh = sw = 1]===================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, OW % 4 == 0
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4_S1_RUSE
#define CONV_3D_GEMM_UERNEL_8_8R4_S1_RUSE

//for: Feature = (64, 64), [N, IC, OC] = [32, 128, 128]
//LB = 4: Size = 18, Time = 3.31976 msec, Performace = 11643.8 GFlop/s
//LB = 3: Size = 18, Time = 3.70716 msec, Performace = 10427 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [32, 256, 256]
//LB = 4: Size = 18, Time = 3.34785 msec, Performace = 11546.1 GFlop/s
//LB = 3: Size = 18, Time = 3.67378 msec, Performace = 10521.8 GFlop/s
//for: Feature = (28, 28), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 27.5625, Time = 5.0107 msec, Performace = 11812.7 GFlop/s
//LB = 3: Size = 27.5625, Time = 5.6547 msec, Performace = 10467.4 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128,256, 256]
//LB = 4: Size = 162, Time = 28.9759 msec, Performace = 12006.3 GFlop/s
//LB = 3: Size = 18, Time = 3.68788 msec, Performace = 10481.5 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1, int FH, int FW>
__global__ void conv3dGemm_uernel_8_8R4S1_ruse(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
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
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;
	
	//prepare for Y[N, OH, OW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);//oc = f(by)
	const int yj0  = bj0 + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;

	//compute area-----------------------------------------------------------
	for (int fh = 0; fh < FH; fh++) {
		for (int oic = 0; oic < IC; oic += STEP2) {
			//load 4 elements from CW[FH, FW, IC, OC]
			const int woffset0 = (fh * FW * IC + oic)*OC;
			Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

			//load 4 elements from X[N, IH, IW, IC]
			const int xic = ((tx & STEP_m1) << 1) + oic;
			bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
			bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
			bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
			bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
			bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
			const int xoffset0 = fh * IW * IC + xic;
			float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
			float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
			float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
			float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
			Xs[buf][(tx << 1)    ][ty] = { x0.x, x1.x, x2.x, x3.x };
			Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };
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

				//load 4 elements from CW[FH, FW, IC, OC]
				const int woffset0 = ((fh*FW + fw)*IC + oic)*OC;
				Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
				Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

				//load 4 elements from X[N, IH, IW, IC]
				bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
				float2 nx3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
				Xs[buf][(tx << 1)    ][ty] = { ox0.y, ox0.z, ox0.w, nx3.x };
				Xs[buf][(tx << 1) + 1][ty] = { ox1.y, ox1.z, ox1.w, nx3.y };
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


//======[FH = FW = 3, sh = sw = 1]======================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, OW % 4 == 0
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W3_S1_RUSE
#define CONV_3D_GEMM_UERNEL_8_8R4W3_S1_RUSE

//for: Feature = (64, 64), [N, IC, OC] = [32, 128, 128]
//LB = 4: Size = 18, Time = 3.31976 msec, Performace = 11643.8 GFlop/s
//LB = 3: Size = 18, Time = 3.70716 msec, Performace = 10427 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [32, 256, 256]
//LB = 4: Size = 18, Time = 3.34785 msec, Performace = 11546.1 GFlop/s
//LB = 3: Size = 18, Time = 3.67378 msec, Performace = 10521.8 GFlop/s
//for: Feature = (28, 28), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 27.5625, Time = 5.0107 msec, Performace = 11812.7 GFlop/s
//LB = 3: Size = 27.5625, Time = 5.6547 msec, Performace = 10467.4 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128,256, 256]
//LB = 4: Size = 18, Time = 3.3314  msec, Performace = 11603.1 GFlop/s
//LB = 3: Size = 18, Time = 3.68788 msec, Performace = 10481.5 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R4W3S1_ruse(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
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
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC + ((tx & STEP_m1) << 1);//X += X1;
	
	//prepare for Y[N, OH, OW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);//oc = f(by)
	const int yj0  = bj0  + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;

	//compute area-----------------------------------------------------------
	const int IC_OC = IC * OC, IW_IC = IW * IC;
	for (int gk = 0, GK = (IC >> LB) * 3; gk < GK; gk++) {//oic_group (16channels) * FH
		const int oic_group = gk / 3, fh = gk - oic_group * 3;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		int woffset0 = ((fh * 3 * IC) + oic)*OC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		const int xoffset0 = fh * IW_IC + oic;
		float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = { x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < 3; fw++) {
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

			//load 4 elements from CW[FH, FW, IC, OC]
			woffset0 += IC_OC;
			Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

			//load 4 elements from X[N, IH, IW, IC]
			bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
			float2 nx3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
			Xs[buf][(tx << 1)    ][ty] = { ox0.y, ox0.z, ox0.w, nx3.x };
			Xs[buf][(tx << 1) + 1][ty] = { ox1.y, ox1.z, ox1.w, nx3.y };
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


//======[FH = FW = 2, sh = sw = 1]=====================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, OW % 4 == 0
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef CONV_3D_GEMM_UERNEL_8_8R4W2_S1_RUSE
#define CONV_3D_GEMM_UERNEL_8_8R4W2_S1_RUSE

//for: (32, 32) -> (32, 32), [FH, FW] = 3, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 32, Time = 6.12001 msec, Performace = 11228.6 GFlop/s
//for: (32, 32) -> (32, 32), [FH, FW] = 3, [N, IC, OC] = [128, 128, 128]
//LB = 4: Size =  8, Time = 1.61124 msec, Performace = 10662.5 GFlop/s
//for: (16, 16) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 32, Time = 5.91251 msec, Performace = 11622.7 GFlop/s
//for: (16, 16) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size =  8, Time = 1.57437 msec, Performace = 10912.2 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void conv3dGemm_uernel_8_8R4W2S1_ruse(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 2
	      float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
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
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;
	
	//prepare for Y[N, OH, OW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);//oc = f(by)
	const int yj0  = bj0  + (uy << 3);
	const int j0 = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)

	//compute area-----------------------------------------------------------
	const int IC_OC = IC * OC, IW_IC = IW * IC;
	for (int gk = 0, GK = (IC >> LB) * 2; gk < GK; gk++) {//oic_group (16channels) * FH
		const int oic_group = gk / 2, fh = gk - oic_group * 2;
		const int oic = (oic_group << LB);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >=  0) && (tow0 < IW    );
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		const int xoffset0 = fh * IW_IC + xic;
		float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = { x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from CW[FH, FW, IC, OC]
		int woffset0 = ((fh * 2 * IC) + oic)*OC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < 2; fw++) {
#pragma unroll 
			for (int ik = 0; ik < STEP2; ik++)  {
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
			float4 ox0 = Xs[buf][(tx << 1)    ][ty];//update_shared_memory
			float4 ox1 = Xs[buf][(tx << 1) + 1][ty];
			buf ^= 1;

			//load 4 elements from X[N, IH, IW, IC]
			bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
			float2 nx3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
			Xs[buf][(tx << 1)    ][ty] = { ox0.y, ox0.z, ox0.w, nx3.x };
			Xs[buf][(tx << 1) + 1][ty] = { ox1.y, ox1.z, ox1.w, nx3.y };

			//load 4 elements from CW[FH, FW, IC, OC]
			woffset0 += IC_OC;
			Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);
			__syncthreads();
		}
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
	}

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


#endif