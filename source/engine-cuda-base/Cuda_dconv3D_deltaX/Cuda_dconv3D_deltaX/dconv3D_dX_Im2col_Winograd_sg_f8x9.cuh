#pragma once 

#ifndef DECONV_DX_3D_WINOGRAD_S16_F8X9_H
#define DECONV_DX_3D_WINOGRAD_S16_F8X9_H

//(1) sh = sw = 1
//(2) FW = 9
//(3) IW % 8: group = 8 elements
#ifndef DECONV_DX_3D_WINOGRAD_F8X9_CALL
#define DECONV_DX_3D_WINOGRAD_F8X9_CALL

//<1> (opw <= 3) -> (FW - 1 - pw <= 4) -> (pw >= 4)
//<2> (IW - OW - opw + FW - 1 <= 4) -> (IW - OW + pw <= 4) -> (OW - IW - pw + 4 >= 0)

//================[Standard: 32(IC) * 256(GM)]===============================
//IW % 8 == 0, pw >= 4
#define winograd_f8x9_k32x256_p4_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f8x9_kernel_32x256_p4_tex<FH>\
		<<< dim3(IC>>5, ((N*IH)>>5)*(IW>>3)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*9*IC), deltaX,IH,IW, IC,OC, (FH-1-ph),(8-pw))

//IW % 8 == 0, pw >= 4, template<IC>
#define winograd_f8x9_k32x256_p4_ICT_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f8x9_kernel_32x256_p4_ICT_tex<FH, IC>\
		<<< dim3(IC>>5, ((N*IH)>>5)*(IW>>3)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*9*IC), deltaX,IH,IW, OC, (FH-1-ph),(8-pw))

//IW % 8 == 0, pw >= 4, template<IC, OC>
#define winograd_f8x9_k32x256_p4_CT_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f8x9_kernel_32x256_p4_CT_tex<FH, IC, OC>\
		<<< dim3(IC>>5, ((N*IH)>>5)*(IW>>3)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*9*IC), deltaX,IH,IW, (FH-1-ph),(8-pw))

//IW % 16 == 0, pw >= 4
#define winograd_f8x9_ruse_k32x256_p4_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f8x9_ruse_kernel_32x256_p4_tex<FH>\
		<<< dim3(IC>>5, ((N*IH)>>5)*(IW>>4<<1)), dim3(16, 8), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*9*IC), deltaX,IH,IW, IC,OC, (FH-1-ph),(8-pw))

//================[Standard: 64(IC) * 256(GM)]===============================
//IW % 8 == 0, pw >= 4
#define winograd_f8x9_k64x256_p4_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f8x9_kernel_64x256_p4_tex<FH>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IW>>3)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*9*IC), deltaX,IH,IW, IC,OC, (FH-1-ph),(8-pw))

//IW % 8 == 0, pw >= 4, template<OC>
#define winograd_f8x9_k64x256_p4_OCT_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f8x9_kernel_64x256_p4_OCT_tex<FH, OC>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IW>>3)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*9*IC), deltaX,IH,IW, IC,  (FH-1-ph),(8-pw))

#endif


//================[Standard: 32(IC) * 256(GM)]===============================
//LB = 4, IC % 8 == 0, IW % 8 == 0, pw >= 4
#ifndef DECONV_3D_DX_WINOGRAD_F8X9_32X256_P4_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F8X9_32X256_P4_TEXTURE

//for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 162, Time = 14.4767 msec, Performace = 24031.2 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 162, Time = 16.9733 msec, Performace = 20496.4 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 162, Time = 16      msec, Performace = 21743.3 GFlop/s

template<int FH>
__global__ void winograd_f8x9_kernel_32x256_p4_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(ic): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (bx << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 9 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, IC]
	const int bj0 = (by << 8);//32 * 8 = 256
	const int Dk = (tx & 7), Di = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Di << 3);//OW % 8
	const int IW8 = (IW >> 3 << 3), IH_IW8 = IH * IW8;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW8, IW8);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Dk;//deltaY[tn0, toh0, tow0, Dk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW8, IW8);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 9 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
		w[8] = W[W0]; w[7] = W[W1]; w[6] = W[W2];
		w[5] = W[W3]; w[4] = W[W4]; w[3] = W[W5];
		w[2] = W[W6]; w[1] = W[W7]; w[0] = W[W8];

		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=   0) && (tow0      < OW);//pw = -4
		bool ly1 = lh0 && (tow0 >=  -1) && (tow0 +  1 < OW);//pw = -3
		bool ly2 = lh0 && (tow0 >=  -2) && (tow0 +  2 < OW);//pw = -2
		bool ly3 = lh0 && (tow0 >=  -3) && (tow0 +  3 < OW);//pw = -1
		bool lyC = lh0 && (tow0 >= -12) && (tow0 + 12 < OW);//pw = +1
		bool lyD = lh0 && (tow0 >= -13) && (tow0 + 13 < OW);//pw = +2
		bool lyE = lh0 && (tow0 >= -14) && (tow0 + 14 < OW);//pw = +3
		bool lyF = lh0 && (tow0 >= -15) && (tow0 + 15 < OW);//pw = +4
		int tY0 = Y0 + fh * OW * OC;
		int tY1 = tY0 + OC     , tY2 = tY0 + (OC << 1);
		int tY3 = tY0 + OC *  3, tY4 = tY0 + (OC << 2);
		int tY5 = tY0 + OC *  5, tY6 = tY0 + OC *  6;
		int tY7 = tY0 + OC *  7, tY8 = tY0 + (OC << 3);
		int tY9 = tY0 + OC *  9, tYA = tY0 + OC * 10;
		int tYB = tY0 + OC * 11, tYC = tY0 + OC * 12;
		int tYD = tY0 + OC * 13, tYE = tY0 + OC * 14;
		int tYF = tY0 + OC * 15;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
		tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
		tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
		tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
		tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
		tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1);
		tYC = IF_int(lyC, tYC, -1); tYD = IF_int(lyD, tYD, -1);
		tYE = IF_int(lyE, tYE, -1); tYF = IF_int(lyF, tYF, -1);
		x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
		x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
		x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
		x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
		x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
		x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
		x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
		x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);

		for (int ooc = 8; ooc < OC; ooc += 8) {
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g, w);
			winograd_f8x9_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Dk][ 0][Di] = d[ 0]; Ds[Dk][ 1][Di] = d[ 1];
			Ds[Dk][ 2][Di] = d[ 2]; Ds[Dk][ 3][Di] = d[ 3];
			Ds[Dk][ 4][Di] = d[ 4]; Ds[Dk][ 5][Di] = d[ 5];
			Ds[Dk][ 6][Di] = d[ 6]; Ds[Dk][ 7][Di] = d[ 7];
			Ds[Dk][ 8][Di] = d[ 8]; Ds[Dk][ 9][Di] = d[ 9];
			Ds[Dk][10][Di] = d[10]; Ds[Dk][11][Di] = d[11];
			Ds[Dk][12][Di] = d[12]; Ds[Dk][13][Di] = d[13];
			Ds[Dk][14][Di] = d[14]; Ds[Dk][15][Di] = d[15];
			__syncthreads();
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

				//ic[0 - 3]             ic[4 - 7]
				simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
				simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
				simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
				simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
				simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}

			//load 1 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 9 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
			w[8] = W[W0]; w[7] = W[W1]; w[6] = W[W2];
			w[5] = W[W3]; w[4] = W[W4]; w[3] = W[W5];
			w[2] = W[W6]; w[1] = W[W7]; w[0] = W[W8];

			//load 2 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc;
			int tY1 = tY0 + OC    , tY2 = tY0 + (OC << 1);
			int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
			int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6;
			int tY7 = tY0 + OC * 7, tY8 = tY0 + (OC << 3);
			int tY9 = tY0 + OC * 9, tYA = tY0 + OC * 10;
			int tYB = tY0 + OC * 11, tYC = tY0 + OC * 12;
			int tYD = tY0 + OC * 13, tYE = tY0 + OC * 14;
			int tYF = tY0 + OC * 15;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
			tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
			tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
			tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
			tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
			tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1);
			tYC = IF_int(lyC, tYC, -1); tYD = IF_int(lyD, tYD, -1);
			tYE = IF_int(lyE, tYE, -1); tYF = IF_int(lyF, tYF, -1);
			x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
			x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
			x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
			x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
			x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
			x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
			x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
			x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g, w);
			winograd_f8x9_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Dk][ 0][Di] = d[ 0]; Ds[Dk][ 1][Di] = d[ 1];
			Ds[Dk][ 2][Di] = d[ 2]; Ds[Dk][ 3][Di] = d[ 3];
			Ds[Dk][ 4][Di] = d[ 4]; Ds[Dk][ 5][Di] = d[ 5];
			Ds[Dk][ 6][Di] = d[ 6]; Ds[Dk][ 7][Di] = d[ 7];
			Ds[Dk][ 8][Di] = d[ 8]; Ds[Dk][ 9][Di] = d[ 9];
			Ds[Dk][10][Di] = d[10]; Ds[Dk][11][Di] = d[11];
			Ds[Dk][12][Di] = d[12]; Ds[Dk][13][Di] = d[13];
			Ds[Dk][14][Di] = d[14]; Ds[Dk][15][Di] = d[15];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

			//ic[0 - 3]             ic[4 - 7]
			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
		}
		__syncthreads();
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Xs0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Xs1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], x0[8], x1[8], x2[8], x3[8];//ic0, ic1, ic2, ic3
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	//write-turn0: x, [ic0, ic4]
	*(float4*)(&Xs0[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Xs0[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Xs0[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Xs0[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//write-turn1: y, [ic0, ic4]
	*(float4*)(&Xs1[ux][uy][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Xs1[ux][uy][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Xs1[ux][uy][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Xs1[ux][uy][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//read-turn0: x, [ic0, ic4]
	a[ 0] = Xs0[ 0][uy][ux]; a[ 1] = Xs0[ 1][uy][ux]; a[ 2] = Xs0[ 2][uy][ux]; a[ 3] = Xs0[ 3][uy][ux];
	a[ 4] = Xs0[ 4][uy][ux]; a[ 5] = Xs0[ 5][uy][ux]; a[ 6] = Xs0[ 6][uy][ux]; a[ 7] = Xs0[ 7][uy][ux];
	a[ 8] = Xs0[ 8][uy][ux]; a[ 9] = Xs0[ 9][uy][ux]; a[10] = Xs0[10][uy][ux]; a[11] = Xs0[11][uy][ux];
	a[12] = Xs0[12][uy][ux]; a[13] = Xs0[13][uy][ux]; a[14] = Xs0[14][uy][ux]; a[15] = Xs0[15][uy][ux];
	winograd_f8x9_Y(x0, a);

	//read-turn1: y, [ic1, ic5]
	a[ 0] = Xs1[ 0][uy][ux]; a[ 1] = Xs1[ 1][uy][ux]; a[ 2] = Xs1[ 2][uy][ux]; a[ 3] = Xs1[ 3][uy][ux];
	a[ 4] = Xs1[ 4][uy][ux]; a[ 5] = Xs1[ 5][uy][ux]; a[ 6] = Xs1[ 6][uy][ux]; a[ 7] = Xs1[ 7][uy][ux];
	a[ 8] = Xs1[ 8][uy][ux]; a[ 9] = Xs1[ 9][uy][ux]; a[10] = Xs1[10][uy][ux]; a[11] = Xs1[11][uy][ux];
	a[12] = Xs1[12][uy][ux]; a[13] = Xs1[13][uy][ux]; a[14] = Xs1[14][uy][ux]; a[15] = Xs1[15][uy][ux];
	winograd_f8x9_Y(x1, a);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [ic2, ic6]
	*(float4*)(&Xs0[ux][uy][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Xs0[ux][uy][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Xs0[ux][uy][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Xs0[ux][uy][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn1: w, [ic3, ic7]
	*(float4*)(&Xs1[ux][uy][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Xs1[ux][uy][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Xs1[ux][uy][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Xs1[ux][uy][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [ic2, ic6]
	a[ 0] = Xs0[ 0][uy][ux]; a[ 1] = Xs0[ 1][uy][ux]; a[ 2] = Xs0[ 2][uy][ux]; a[ 3] = Xs0[ 3][uy][ux];
	a[ 4] = Xs0[ 4][uy][ux]; a[ 5] = Xs0[ 5][uy][ux]; a[ 6] = Xs0[ 6][uy][ux]; a[ 7] = Xs0[ 7][uy][ux];
	a[ 8] = Xs0[ 8][uy][ux]; a[ 9] = Xs0[ 9][uy][ux]; a[10] = Xs0[10][uy][ux]; a[11] = Xs0[11][uy][ux];
	a[12] = Xs0[12][uy][ux]; a[13] = Xs0[13][uy][ux]; a[14] = Xs0[14][uy][ux]; a[15] = Xs0[15][uy][ux];
	winograd_f8x9_Y(x2, a);

	//read-turn3: w, [ic3, ic7]
	a[ 0] = Xs1[ 0][uy][ux]; a[ 1] = Xs1[ 1][uy][ux]; a[ 2] = Xs1[ 2][uy][ux]; a[ 3] = Xs1[ 3][uy][ux];
	a[ 4] = Xs1[ 4][uy][ux]; a[ 5] = Xs1[ 5][uy][ux]; a[ 6] = Xs1[ 6][uy][ux]; a[ 7] = Xs1[ 7][uy][ux];
	a[ 8] = Xs1[ 8][uy][ux]; a[ 9] = Xs1[ 9][uy][ux]; a[10] = Xs1[10][uy][ux]; a[11] = Xs1[11][uy][ux];
	a[12] = Xs1[12][uy][ux]; a[13] = Xs1[13][uy][ux]; a[14] = Xs1[14][uy][ux]; a[15] = Xs1[15][uy][ux];
	winograd_f8x9_Y(x3, a);

	//write to deltaX[N, IH, IW, IC]
	*(float4*)(deltaX + X00         ) = float4{ x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC    ) = float4{ x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2) = float4{ x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3) = float4{ x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4) = float4{ x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5) = float4{ x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6) = float4{ x0[6], x1[6], x2[6], x3[6] };
	*(float4*)(deltaX + X00 + IC * 7) = float4{ x0[7], x1[7], x2[7], x3[7] };
}

#endif


//LB = 4, IC % 8 == 0, IW % 8 == 0, pw >= 4, template<IC>
#ifndef DECONV_3D_DX_WINOGRAD_F8X9_32X256_P4_ICT_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F8X9_32X256_P4_ICT_TEXTURE

//for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 162, Time = 13.1833 msec, Performace = 26388.8 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 162, Time = 15.69   msec, Performace = 22172.9 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 162, Time = 14.6667 msec, Performace = 23719.9 GFlop/s

template<int FH, int IC>
__global__ void winograd_f8x9_kernel_32x256_p4_ICT_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW,
	int OC,
	int oph, int opw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(ic): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (bx << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 9 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, IC]
	const int bj0 = (by << 8);//32 * 8 = 256
	const int Dk = (tx & 7), Di = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Di << 3);//OW % 8
	const int IW8 = (IW >> 3 << 3), IH_IW8 = IH * IW8;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW8, IW8);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Dk;//deltaY[tn0, toh0, tow0, Dk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW8, IW8);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 9 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
		w[8] = W[W0]; w[7] = W[W1]; w[6] = W[W2];
		w[5] = W[W3]; w[4] = W[W4]; w[3] = W[W5];
		w[2] = W[W6]; w[1] = W[W7]; w[0] = W[W8];

		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=   0) && (tow0      < OW);//pw = -4
		bool ly1 = lh0 && (tow0 >=  -1) && (tow0 +  1 < OW);//pw = -3
		bool ly2 = lh0 && (tow0 >=  -2) && (tow0 +  2 < OW);//pw = -2
		bool ly3 = lh0 && (tow0 >=  -3) && (tow0 +  3 < OW);//pw = -1
		bool lyC = lh0 && (tow0 >= -12) && (tow0 + 12 < OW);//pw = +1
		bool lyD = lh0 && (tow0 >= -13) && (tow0 + 13 < OW);//pw = +2
		bool lyE = lh0 && (tow0 >= -14) && (tow0 + 14 < OW);//pw = +3
		bool lyF = lh0 && (tow0 >= -15) && (tow0 + 15 < OW);//pw = +4
		int tY0 = Y0 + fh * OW * OC;
		int tY1 = tY0 + OC     , tY2 = tY0 + (OC << 1);
		int tY3 = tY0 + OC *  3, tY4 = tY0 + (OC << 2);
		int tY5 = tY0 + OC *  5, tY6 = tY0 + OC *  6;
		int tY7 = tY0 + OC *  7, tY8 = tY0 + (OC << 3);
		int tY9 = tY0 + OC *  9, tYA = tY0 + OC * 10;
		int tYB = tY0 + OC * 11, tYC = tY0 + OC * 12;
		int tYD = tY0 + OC * 13, tYE = tY0 + OC * 14;
		int tYF = tY0 + OC * 15;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
		tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
		tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
		tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
		tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
		tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1);
		tYC = IF_int(lyC, tYC, -1); tYD = IF_int(lyD, tYD, -1);
		tYE = IF_int(lyE, tYE, -1); tYF = IF_int(lyF, tYF, -1);
		x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
		x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
		x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
		x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
		x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
		x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
		x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
		x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);

		for (int ooc = 8; ooc < OC; ooc += 8) {
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g, w);
			winograd_f8x9_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Dk][ 0][Di] = d[ 0]; Ds[Dk][ 1][Di] = d[ 1];
			Ds[Dk][ 2][Di] = d[ 2]; Ds[Dk][ 3][Di] = d[ 3];
			Ds[Dk][ 4][Di] = d[ 4]; Ds[Dk][ 5][Di] = d[ 5];
			Ds[Dk][ 6][Di] = d[ 6]; Ds[Dk][ 7][Di] = d[ 7];
			Ds[Dk][ 8][Di] = d[ 8]; Ds[Dk][ 9][Di] = d[ 9];
			Ds[Dk][10][Di] = d[10]; Ds[Dk][11][Di] = d[11];
			Ds[Dk][12][Di] = d[12]; Ds[Dk][13][Di] = d[13];
			Ds[Dk][14][Di] = d[14]; Ds[Dk][15][Di] = d[15];
			__syncthreads();
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

				//ic[0 - 3]             ic[4 - 7]
				simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
				simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
				simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
				simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
				simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}

			//load 1 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 9 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
			w[8] = W[W0]; w[7] = W[W1]; w[6] = W[W2];
			w[5] = W[W3]; w[4] = W[W4]; w[3] = W[W5];
			w[2] = W[W6]; w[1] = W[W7]; w[0] = W[W8];

			//load 2 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc;
			int tY1 = tY0 + OC    , tY2 = tY0 + (OC << 1);
			int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
			int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6;
			int tY7 = tY0 + OC * 7, tY8 = tY0 + (OC << 3);
			int tY9 = tY0 + OC * 9, tYA = tY0 + OC * 10;
			int tYB = tY0 + OC * 11, tYC = tY0 + OC * 12;
			int tYD = tY0 + OC * 13, tYE = tY0 + OC * 14;
			int tYF = tY0 + OC * 15;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
			tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
			tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
			tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
			tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
			tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1);
			tYC = IF_int(lyC, tYC, -1); tYD = IF_int(lyD, tYD, -1);
			tYE = IF_int(lyE, tYE, -1); tYF = IF_int(lyF, tYF, -1);
			x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
			x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
			x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
			x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
			x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
			x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
			x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
			x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g, w);
			winograd_f8x9_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Dk][ 0][Di] = d[ 0]; Ds[Dk][ 1][Di] = d[ 1];
			Ds[Dk][ 2][Di] = d[ 2]; Ds[Dk][ 3][Di] = d[ 3];
			Ds[Dk][ 4][Di] = d[ 4]; Ds[Dk][ 5][Di] = d[ 5];
			Ds[Dk][ 6][Di] = d[ 6]; Ds[Dk][ 7][Di] = d[ 7];
			Ds[Dk][ 8][Di] = d[ 8]; Ds[Dk][ 9][Di] = d[ 9];
			Ds[Dk][10][Di] = d[10]; Ds[Dk][11][Di] = d[11];
			Ds[Dk][12][Di] = d[12]; Ds[Dk][13][Di] = d[13];
			Ds[Dk][14][Di] = d[14]; Ds[Dk][15][Di] = d[15];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

			//ic[0 - 3]             ic[4 - 7]
			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
		}
		__syncthreads();
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Xs0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Xs1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], x0[8], x1[8], x2[8], x3[8];//ic0, ic1, ic2, ic3
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	//write-turn0: x, [ic0, ic4]
	*(float4*)(&Xs0[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Xs0[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Xs0[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Xs0[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//write-turn1: y, [ic0, ic4]
	*(float4*)(&Xs1[ux][uy][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Xs1[ux][uy][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Xs1[ux][uy][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Xs1[ux][uy][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//read-turn0: x, [ic0, ic4]
	a[ 0] = Xs0[ 0][uy][ux]; a[ 1] = Xs0[ 1][uy][ux]; a[ 2] = Xs0[ 2][uy][ux]; a[ 3] = Xs0[ 3][uy][ux];
	a[ 4] = Xs0[ 4][uy][ux]; a[ 5] = Xs0[ 5][uy][ux]; a[ 6] = Xs0[ 6][uy][ux]; a[ 7] = Xs0[ 7][uy][ux];
	a[ 8] = Xs0[ 8][uy][ux]; a[ 9] = Xs0[ 9][uy][ux]; a[10] = Xs0[10][uy][ux]; a[11] = Xs0[11][uy][ux];
	a[12] = Xs0[12][uy][ux]; a[13] = Xs0[13][uy][ux]; a[14] = Xs0[14][uy][ux]; a[15] = Xs0[15][uy][ux];
	winograd_f8x9_Y(x0, a);

	//read-turn1: y, [ic1, ic5]
	a[ 0] = Xs1[ 0][uy][ux]; a[ 1] = Xs1[ 1][uy][ux]; a[ 2] = Xs1[ 2][uy][ux]; a[ 3] = Xs1[ 3][uy][ux];
	a[ 4] = Xs1[ 4][uy][ux]; a[ 5] = Xs1[ 5][uy][ux]; a[ 6] = Xs1[ 6][uy][ux]; a[ 7] = Xs1[ 7][uy][ux];
	a[ 8] = Xs1[ 8][uy][ux]; a[ 9] = Xs1[ 9][uy][ux]; a[10] = Xs1[10][uy][ux]; a[11] = Xs1[11][uy][ux];
	a[12] = Xs1[12][uy][ux]; a[13] = Xs1[13][uy][ux]; a[14] = Xs1[14][uy][ux]; a[15] = Xs1[15][uy][ux];
	winograd_f8x9_Y(x1, a);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [ic2, ic6]
	*(float4*)(&Xs0[ux][uy][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Xs0[ux][uy][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Xs0[ux][uy][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Xs0[ux][uy][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn1: w, [ic3, ic7]
	*(float4*)(&Xs1[ux][uy][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Xs1[ux][uy][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Xs1[ux][uy][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Xs1[ux][uy][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [ic2, ic6]
	a[ 0] = Xs0[ 0][uy][ux]; a[ 1] = Xs0[ 1][uy][ux]; a[ 2] = Xs0[ 2][uy][ux]; a[ 3] = Xs0[ 3][uy][ux];
	a[ 4] = Xs0[ 4][uy][ux]; a[ 5] = Xs0[ 5][uy][ux]; a[ 6] = Xs0[ 6][uy][ux]; a[ 7] = Xs0[ 7][uy][ux];
	a[ 8] = Xs0[ 8][uy][ux]; a[ 9] = Xs0[ 9][uy][ux]; a[10] = Xs0[10][uy][ux]; a[11] = Xs0[11][uy][ux];
	a[12] = Xs0[12][uy][ux]; a[13] = Xs0[13][uy][ux]; a[14] = Xs0[14][uy][ux]; a[15] = Xs0[15][uy][ux];
	winograd_f8x9_Y(x2, a);

	//read-turn3: w, [ic3, ic7]
	a[ 0] = Xs1[ 0][uy][ux]; a[ 1] = Xs1[ 1][uy][ux]; a[ 2] = Xs1[ 2][uy][ux]; a[ 3] = Xs1[ 3][uy][ux];
	a[ 4] = Xs1[ 4][uy][ux]; a[ 5] = Xs1[ 5][uy][ux]; a[ 6] = Xs1[ 6][uy][ux]; a[ 7] = Xs1[ 7][uy][ux];
	a[ 8] = Xs1[ 8][uy][ux]; a[ 9] = Xs1[ 9][uy][ux]; a[10] = Xs1[10][uy][ux]; a[11] = Xs1[11][uy][ux];
	a[12] = Xs1[12][uy][ux]; a[13] = Xs1[13][uy][ux]; a[14] = Xs1[14][uy][ux]; a[15] = Xs1[15][uy][ux];
	winograd_f8x9_Y(x3, a);

	//write to deltaX[N, IH, IW, IC]
	*(float4*)(deltaX + X00         ) = float4{ x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC    ) = float4{ x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2) = float4{ x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3) = float4{ x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4) = float4{ x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5) = float4{ x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6) = float4{ x0[6], x1[6], x2[6], x3[6] };
	*(float4*)(deltaX + X00 + IC * 7) = float4{ x0[7], x1[7], x2[7], x3[7] };
}

#endif


//LB = 4, IC % 8 == 0, IW % 8 == 0, pw >= 4, template<IC, OC>
#ifndef DECONV_3D_DX_WINOGRAD_F8X9_32X256_P4_CT_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F8X9_32X256_P4_CT_TEXTURE

//for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 162, Time = 12.9033 msec, Performace = 26961.4 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 162, Time = 13.8167 msec, Performace = 25179.2 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 162, Time = 14.5467 msec, Performace = 23915.6 GFlop/s

template<int FH, int IC, int OC>
__global__ void winograd_f8x9_kernel_32x256_p4_CT_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW,
	int oph, int opw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(ic): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (bx << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 9 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, IC]
	const int bj0 = (by << 8);//32 * 8 = 256
	const int Dk = (tx & 7), Di = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Di << 3);//OW % 8
	const int IW8 = (IW >> 3 << 3), IH_IW8 = IH * IW8;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW8, IW8);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Dk;//deltaY[tn0, toh0, tow0, Dk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW8, IW8);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 9 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
		w[8] = W[W0]; w[7] = W[W1]; w[6] = W[W2];
		w[5] = W[W3]; w[4] = W[W4]; w[3] = W[W5];
		w[2] = W[W6]; w[1] = W[W7]; w[0] = W[W8];

		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=   0) && (tow0      < OW);//pw = -4
		bool ly1 = lh0 && (tow0 >=  -1) && (tow0 +  1 < OW);//pw = -3
		bool ly2 = lh0 && (tow0 >=  -2) && (tow0 +  2 < OW);//pw = -2
		bool ly3 = lh0 && (tow0 >=  -3) && (tow0 +  3 < OW);//pw = -1
		bool lyC = lh0 && (tow0 >= -12) && (tow0 + 12 < OW);//pw = +1
		bool lyD = lh0 && (tow0 >= -13) && (tow0 + 13 < OW);//pw = +2
		bool lyE = lh0 && (tow0 >= -14) && (tow0 + 14 < OW);//pw = +3
		bool lyF = lh0 && (tow0 >= -15) && (tow0 + 15 < OW);//pw = +4
		int tY0 = Y0 + fh * OW * OC;
		int tY1 = tY0 + OC     , tY2 = tY0 + (OC << 1);
		int tY3 = tY0 + OC *  3, tY4 = tY0 + (OC << 2);
		int tY5 = tY0 + OC *  5, tY6 = tY0 + OC *  6;
		int tY7 = tY0 + OC *  7, tY8 = tY0 + (OC << 3);
		int tY9 = tY0 + OC *  9, tYA = tY0 + OC * 10;
		int tYB = tY0 + OC * 11, tYC = tY0 + OC * 12;
		int tYD = tY0 + OC * 13, tYE = tY0 + OC * 14;
		int tYF = tY0 + OC * 15;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
		tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
		tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
		tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
		tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
		tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1);
		tYC = IF_int(lyC, tYC, -1); tYD = IF_int(lyD, tYD, -1);
		tYE = IF_int(lyE, tYE, -1); tYF = IF_int(lyF, tYF, -1);
		x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
		x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
		x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
		x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
		x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
		x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
		x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
		x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);

		for (int ooc = 8; ooc < OC; ooc += 8) {
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g, w);
			winograd_f8x9_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Dk][ 0][Di] = d[ 0]; Ds[Dk][ 1][Di] = d[ 1];
			Ds[Dk][ 2][Di] = d[ 2]; Ds[Dk][ 3][Di] = d[ 3];
			Ds[Dk][ 4][Di] = d[ 4]; Ds[Dk][ 5][Di] = d[ 5];
			Ds[Dk][ 6][Di] = d[ 6]; Ds[Dk][ 7][Di] = d[ 7];
			Ds[Dk][ 8][Di] = d[ 8]; Ds[Dk][ 9][Di] = d[ 9];
			Ds[Dk][10][Di] = d[10]; Ds[Dk][11][Di] = d[11];
			Ds[Dk][12][Di] = d[12]; Ds[Dk][13][Di] = d[13];
			Ds[Dk][14][Di] = d[14]; Ds[Dk][15][Di] = d[15];
			__syncthreads();
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

				//ic[0 - 3]             ic[4 - 7]
				simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
				simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
				simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
				simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
				simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}

			//load 1 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 9 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
			w[8] = W[W0]; w[7] = W[W1]; w[6] = W[W2];
			w[5] = W[W3]; w[4] = W[W4]; w[3] = W[W5];
			w[2] = W[W6]; w[1] = W[W7]; w[0] = W[W8];

			//load 2 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc;
			int tY1 = tY0 + OC    , tY2 = tY0 + (OC << 1);
			int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
			int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6;
			int tY7 = tY0 + OC * 7, tY8 = tY0 + (OC << 3);
			int tY9 = tY0 + OC * 9, tYA = tY0 + OC * 10;
			int tYB = tY0 + OC * 11, tYC = tY0 + OC * 12;
			int tYD = tY0 + OC * 13, tYE = tY0 + OC * 14;
			int tYF = tY0 + OC * 15;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
			tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
			tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
			tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
			tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
			tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1);
			tYC = IF_int(lyC, tYC, -1); tYD = IF_int(lyD, tYD, -1);
			tYE = IF_int(lyE, tYE, -1); tYF = IF_int(lyF, tYF, -1);
			x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
			x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
			x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
			x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
			x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
			x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
			x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
			x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g, w);
			winograd_f8x9_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Dk][ 0][Di] = d[ 0]; Ds[Dk][ 1][Di] = d[ 1];
			Ds[Dk][ 2][Di] = d[ 2]; Ds[Dk][ 3][Di] = d[ 3];
			Ds[Dk][ 4][Di] = d[ 4]; Ds[Dk][ 5][Di] = d[ 5];
			Ds[Dk][ 6][Di] = d[ 6]; Ds[Dk][ 7][Di] = d[ 7];
			Ds[Dk][ 8][Di] = d[ 8]; Ds[Dk][ 9][Di] = d[ 9];
			Ds[Dk][10][Di] = d[10]; Ds[Dk][11][Di] = d[11];
			Ds[Dk][12][Di] = d[12]; Ds[Dk][13][Di] = d[13];
			Ds[Dk][14][Di] = d[14]; Ds[Dk][15][Di] = d[15];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

			//ic[0 - 3]             ic[4 - 7]
			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
		}
		__syncthreads();
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Xs0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Xs1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], x0[8], x1[8], x2[8], x3[8];//ic0, ic1, ic2, ic3
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	//write-turn0: x, [ic0, ic4]
	*(float4*)(&Xs0[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Xs0[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Xs0[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Xs0[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//write-turn1: y, [ic0, ic4]
	*(float4*)(&Xs1[ux][uy][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Xs1[ux][uy][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Xs1[ux][uy][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Xs1[ux][uy][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//read-turn0: x, [ic0, ic4]
	a[ 0] = Xs0[ 0][uy][ux]; a[ 1] = Xs0[ 1][uy][ux]; a[ 2] = Xs0[ 2][uy][ux]; a[ 3] = Xs0[ 3][uy][ux];
	a[ 4] = Xs0[ 4][uy][ux]; a[ 5] = Xs0[ 5][uy][ux]; a[ 6] = Xs0[ 6][uy][ux]; a[ 7] = Xs0[ 7][uy][ux];
	a[ 8] = Xs0[ 8][uy][ux]; a[ 9] = Xs0[ 9][uy][ux]; a[10] = Xs0[10][uy][ux]; a[11] = Xs0[11][uy][ux];
	a[12] = Xs0[12][uy][ux]; a[13] = Xs0[13][uy][ux]; a[14] = Xs0[14][uy][ux]; a[15] = Xs0[15][uy][ux];
	winograd_f8x9_Y(x0, a);

	//read-turn1: y, [ic1, ic5]
	a[ 0] = Xs1[ 0][uy][ux]; a[ 1] = Xs1[ 1][uy][ux]; a[ 2] = Xs1[ 2][uy][ux]; a[ 3] = Xs1[ 3][uy][ux];
	a[ 4] = Xs1[ 4][uy][ux]; a[ 5] = Xs1[ 5][uy][ux]; a[ 6] = Xs1[ 6][uy][ux]; a[ 7] = Xs1[ 7][uy][ux];
	a[ 8] = Xs1[ 8][uy][ux]; a[ 9] = Xs1[ 9][uy][ux]; a[10] = Xs1[10][uy][ux]; a[11] = Xs1[11][uy][ux];
	a[12] = Xs1[12][uy][ux]; a[13] = Xs1[13][uy][ux]; a[14] = Xs1[14][uy][ux]; a[15] = Xs1[15][uy][ux];
	winograd_f8x9_Y(x1, a);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [ic2, ic6]
	*(float4*)(&Xs0[ux][uy][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Xs0[ux][uy][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Xs0[ux][uy][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Xs0[ux][uy][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn1: w, [ic3, ic7]
	*(float4*)(&Xs1[ux][uy][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Xs1[ux][uy][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Xs1[ux][uy][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Xs1[ux][uy][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [ic2, ic6]
	a[ 0] = Xs0[ 0][uy][ux]; a[ 1] = Xs0[ 1][uy][ux]; a[ 2] = Xs0[ 2][uy][ux]; a[ 3] = Xs0[ 3][uy][ux];
	a[ 4] = Xs0[ 4][uy][ux]; a[ 5] = Xs0[ 5][uy][ux]; a[ 6] = Xs0[ 6][uy][ux]; a[ 7] = Xs0[ 7][uy][ux];
	a[ 8] = Xs0[ 8][uy][ux]; a[ 9] = Xs0[ 9][uy][ux]; a[10] = Xs0[10][uy][ux]; a[11] = Xs0[11][uy][ux];
	a[12] = Xs0[12][uy][ux]; a[13] = Xs0[13][uy][ux]; a[14] = Xs0[14][uy][ux]; a[15] = Xs0[15][uy][ux];
	winograd_f8x9_Y(x2, a);

	//read-turn3: w, [ic3, ic7]
	a[ 0] = Xs1[ 0][uy][ux]; a[ 1] = Xs1[ 1][uy][ux]; a[ 2] = Xs1[ 2][uy][ux]; a[ 3] = Xs1[ 3][uy][ux];
	a[ 4] = Xs1[ 4][uy][ux]; a[ 5] = Xs1[ 5][uy][ux]; a[ 6] = Xs1[ 6][uy][ux]; a[ 7] = Xs1[ 7][uy][ux];
	a[ 8] = Xs1[ 8][uy][ux]; a[ 9] = Xs1[ 9][uy][ux]; a[10] = Xs1[10][uy][ux]; a[11] = Xs1[11][uy][ux];
	a[12] = Xs1[12][uy][ux]; a[13] = Xs1[13][uy][ux]; a[14] = Xs1[14][uy][ux]; a[15] = Xs1[15][uy][ux];
	winograd_f8x9_Y(x3, a);

	//write to deltaX[N, IH, IW, IC]
	*(float4*)(deltaX + X00         ) = float4{ x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC    ) = float4{ x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2) = float4{ x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3) = float4{ x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4) = float4{ x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5) = float4{ x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6) = float4{ x0[6], x1[6], x2[6], x3[6] };
	*(float4*)(deltaX + X00 + IC * 7) = float4{ x0[7], x1[7], x2[7], x3[7] };
}

#endif


//LB = 4, IC % 8 == 0, IW % 16 == 0, pw >= 4
#ifndef DECONV_3D_DX_WINOGRAD_F8X9_RUSE_32X256_P4_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F8X9_RUSE_32X256_P4_TEXTURE

//for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 162, Time = 12.14  msec, Performace = 28656.7 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 162, Time = 14.175 msec, Performace = 24542.7 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 162, Time = 12.8   msec, Performace = 27179.1 GFlop/s

template<int FH>
__global__ void winograd_f8x9_ruse_kernel_32x256_p4_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(ic): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*16 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	float4 v16 = F32_4_0, v17 = F32_4_0, v18 = F32_4_0, v19 = F32_4_0;
	float4 v20 = F32_4_0, v21 = F32_4_0, v22 = F32_4_0, v23 = F32_4_0;
	float4 v24 = F32_4_0, v25 = F32_4_0, v26 = F32_4_0, v27 = F32_4_0;
	float4 v28 = F32_4_0, v29 = F32_4_0, v30 = F32_4_0, v31 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (bx << 5);
	const int Gk = ty, Gi = tx << 1;//[8, 16*2 = 32]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 9 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, IC]
	const int bj0 = (by << 8);//32 * 8 = 256
	const int Dk = (tx & 7), Di = ((ty << 1) + (tx > 7)) << 1;//[8, 16*2 = 32]
	const int tj0 = bj0 + (Di << 3);//OW % 8
	const int IW16 = (IW >> 4 << 4), IH_IW16 = IH * IW16;//8 * 2 = 16
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW16, IW16);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Dk;//deltaY[tn0, toh0, tow0, Dk]
	
	//prepare for threadIdx
	const int ux = (ty << 1) + (tx > 7), uy = (tx & 7);//(2 * 8)
	const int DIdx = ((uy & 1) + ((uy >> 2) << 1)) << 3;//4
	const int GIdx = ((uy & 3) >> 1) << 4;//2

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW16, IW16);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//======[compute area1: local]======================================================
	float w0[9], w1[9], g0[16], g1[16], x[24], d0[16], d1[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 9 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
		float2 wv8 = *(float2*)(W + W0), wv7 = *(float2*)(W + W1);
		float2 wv6 = *(float2*)(W + W2), wv5 = *(float2*)(W + W3);
		float2 wv4 = *(float2*)(W + W4), wv3 = *(float2*)(W + W5);
		float2 wv2 = *(float2*)(W + W6), wv1 = *(float2*)(W + W7);
		float2 wv0 = *(float2*)(W + W8);

		w0[0] = wv0.x; w1[0] = wv0.y; w0[1] = wv1.x; w1[1] = wv1.y;
		w0[2] = wv2.x; w1[2] = wv2.y; w0[3] = wv3.x; w1[3] = wv3.y;
		w0[4] = wv4.x; w1[4] = wv4.y; w0[5] = wv5.x; w1[5] = wv5.y;
		w0[6] = wv6.x; w1[6] = wv6.y; w0[7] = wv7.x; w1[7] = wv7.y;
		w0[8] = wv8.x; w1[8] = wv8.y;

		//load 2 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=   0) && (tow0      < OW);//pw = -4
		bool ly1 = lh0 && (tow0 >=  -1) && (tow0 +  1 < OW);//pw = -3
		bool ly2 = lh0 && (tow0 >=  -2) && (tow0 +  2 < OW);//pw = -2
		bool ly3 = lh0 && (tow0 >=  -3) && (tow0 +  3 < OW);//pw = -1
		bool lyK = lh0 && (tow0 >= -20) && (tow0 + 20 < OW);//pw = +1
		bool lyL = lh0 && (tow0 >= -21) && (tow0 + 21 < OW);//pw = +2
		bool lyM = lh0 && (tow0 >= -22) && (tow0 + 22 < OW);//pw = +3
		bool lyN = lh0 && (tow0 >= -23) && (tow0 + 23 < OW);//pw = +4
		int tY0 = Y0 + fh * OW * OC, tY1 = tY0 + OC;
		int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC *  3;
		int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC *  5;
		int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC *  7;
		int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC *  9;
		int tYA = tY0 + OC * 10  , tYB = tY0 + OC * 11;
		int tYC = tY0 + OC * 12  , tYD = tY0 + OC * 13;
		int tYE = tY0 + OC * 14  , tYF = tY0 + OC * 15;
		int tYG = tY0 + (OC << 4), tYH = tY0 + OC * 17;
		int tYI = tY0 + OC * 18  , tYJ = tY0 + OC * 19;
		int tYK = tY0 + OC * 20  , tYL = tY0 + OC * 21;
		int tYM = tY0 + OC * 22  , tYN = tY0 + OC * 23;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
		tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
		tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
		tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
		tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
		tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1);
		tYC = IF_int(lh0, tYC, -1); tYD = IF_int(lh0, tYD, -1);
		tYE = IF_int(lh0, tYE, -1); tYF = IF_int(lh0, tYF, -1);
		tYG = IF_int(lh0, tYG, -1); tYH = IF_int(lh0, tYH, -1);
		tYI = IF_int(lh0, tYI, -1); tYJ = IF_int(lh0, tYJ, -1);
		tYK = IF_int(lyK, tYK, -1); tYL = IF_int(lyL, tYL, -1);
		tYM = IF_int(lyM, tYM, -1); tYN = IF_int(lyN, tYN, -1);
		x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
		x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
		x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
		x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
		x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
		x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
		x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
		x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);
		x[16] = tex1Dfetch<float>(deltaY, tYG); x[17] = tex1Dfetch<float>(deltaY, tYH);
		x[18] = tex1Dfetch<float>(deltaY, tYI); x[19] = tex1Dfetch<float>(deltaY, tYJ);
		x[20] = tex1Dfetch<float>(deltaY, tYK); x[21] = tex1Dfetch<float>(deltaY, tYL);
		x[22] = tex1Dfetch<float>(deltaY, tYM); x[23] = tex1Dfetch<float>(deltaY, tYN);

		for (int ooc = 8; ooc < OC; ooc += 8) {
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g0, w0);
			winograd_f8x9_G(g1, w1);
			winograd_f8x9_D(d0, x);//x[0 - 15]
			winograd_f8x9_D_oft(d1, x, 8);//x[8 - 23]

			//write to shread memory
			*(float2*)(&Gs[Gk][0x0][Gi]) = { g0[0x0], g1[0x0] }; *(float2*)(&Gs[Gk][0x1][Gi]) = { g0[0x1], g1[0x1] };
			*(float2*)(&Gs[Gk][0x2][Gi]) = { g0[0x2], g1[0x2] }; *(float2*)(&Gs[Gk][0x3][Gi]) = { g0[0x3], g1[0x3] };
			*(float2*)(&Gs[Gk][0x4][Gi]) = { g0[0x4], g1[0x4] }; *(float2*)(&Gs[Gk][0x5][Gi]) = { g0[0x5], g1[0x5] };
			*(float2*)(&Gs[Gk][0x6][Gi]) = { g0[0x6], g1[0x6] }; *(float2*)(&Gs[Gk][0x7][Gi]) = { g0[0x7], g1[0x7] };
			*(float2*)(&Gs[Gk][0x8][Gi]) = { g0[0x8], g1[0x8] }; *(float2*)(&Gs[Gk][0x9][Gi]) = { g0[0x9], g1[0x9] };
			*(float2*)(&Gs[Gk][0xA][Gi]) = { g0[0xA], g1[0xA] }; *(float2*)(&Gs[Gk][0xB][Gi]) = { g0[0xB], g1[0xB] };
			*(float2*)(&Gs[Gk][0xC][Gi]) = { g0[0xC], g1[0xC] }; *(float2*)(&Gs[Gk][0xD][Gi]) = { g0[0xD], g1[0xD] };
			*(float2*)(&Gs[Gk][0xE][Gi]) = { g0[0xE], g1[0xE] }; *(float2*)(&Gs[Gk][0xF][Gi]) = { g0[0xF], g1[0xF] };

			*(float2*)(&Ds[Dk][0x0][Di]) = { d0[0x0], d1[0x0] }; *(float2*)(&Ds[Dk][0x1][Di]) = { d0[0x1], d1[0x1] };
			*(float2*)(&Ds[Dk][0x2][Di]) = { d0[0x2], d1[0x2] }; *(float2*)(&Ds[Dk][0x3][Di]) = { d0[0x3], d1[0x3] };
			*(float2*)(&Ds[Dk][0x4][Di]) = { d0[0x4], d1[0x4] }; *(float2*)(&Ds[Dk][0x5][Di]) = { d0[0x5], d1[0x5] };
			*(float2*)(&Ds[Dk][0x6][Di]) = { d0[0x6], d1[0x6] }; *(float2*)(&Ds[Dk][0x7][Di]) = { d0[0x7], d1[0x7] };
			*(float2*)(&Ds[Dk][0x8][Di]) = { d0[0x8], d1[0x8] }; *(float2*)(&Ds[Dk][0x9][Di]) = { d0[0x9], d1[0x9] };
			*(float2*)(&Ds[Dk][0xA][Di]) = { d0[0xA], d1[0xA] }; *(float2*)(&Ds[Dk][0xB][Di]) = { d0[0xB], d1[0xB] };
			*(float2*)(&Ds[Dk][0xC][Di]) = { d0[0xC], d1[0xC] }; *(float2*)(&Ds[Dk][0xD][Di]) = { d0[0xD], d1[0xD] };
			*(float2*)(&Ds[Dk][0xE][Di]) = { d0[0xE], d1[0xE] }; *(float2*)(&Ds[Dk][0xF][Di]) = { d0[0xF], d1[0xF] };
			__syncthreads();
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 b0 = *(float4*)(&Ds[ik][ux][DIdx    ]);
				float4 b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

				//=======================================================
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx     ]);
				float4 a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

				//ic[0 - 3]             ic[4 - 7]
				simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
				simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
				simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
				simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
				simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7

				//=======================================================
				float4 a2 = *(float4*)(&Gs[ik][ux][GIdx +  8]);
				float4 a3 = *(float4*)(&Gs[ik][ux][GIdx + 12]);

				//ic[8 - 11]            ic[12 - 15]
				simdMM4(v16, b0.x, a2); simdMM4(v17, b0.x, a3);//j0
				simdMM4(v18, b0.y, a2); simdMM4(v19, b0.y, a3);//j1
				simdMM4(v20, b0.z, a2); simdMM4(v21, b0.z, a3);//j2
				simdMM4(v22, b0.w, a2); simdMM4(v23, b0.w, a3);//j3
				simdMM4(v24, b1.x, a2); simdMM4(v25, b1.x, a3);//j4
				simdMM4(v26, b1.y, a2); simdMM4(v27, b1.y, a3);//j5
				simdMM4(v28, b1.z, a2); simdMM4(v29, b1.z, a3);//j6
				simdMM4(v30, b1.w, a2); simdMM4(v31, b1.w, a3);//j7
			}

			//load 2 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 9 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
			float2 wv8 = *(float2*)(W + W0), wv7 = *(float2*)(W + W1);
			float2 wv6 = *(float2*)(W + W2), wv5 = *(float2*)(W + W3);
			float2 wv4 = *(float2*)(W + W4), wv3 = *(float2*)(W + W5);
			float2 wv2 = *(float2*)(W + W6), wv1 = *(float2*)(W + W7);
			float2 wv0 = *(float2*)(W + W8);

			w0[0] = wv0.x; w1[0] = wv0.y; w0[1] = wv1.x; w1[1] = wv1.y;
			w0[2] = wv2.x; w1[2] = wv2.y; w0[3] = wv3.x; w1[3] = wv3.y;
			w0[4] = wv4.x; w1[4] = wv4.y; w0[5] = wv5.x; w1[5] = wv5.y;
			w0[6] = wv6.x; w1[6] = wv6.y; w0[7] = wv7.x; w1[7] = wv7.y;
			w0[8] = wv8.x; w1[8] = wv8.y;

			//load 2 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc, tY1 = tY0 + OC;
			int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC *  3;
			int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC *  5;
			int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC *  7;
			int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC *  9;
			int tYA = tY0 + OC * 10  , tYB = tY0 + OC * 11;
			int tYC = tY0 + OC * 12  , tYD = tY0 + OC * 13;
			int tYE = tY0 + OC * 14  , tYF = tY0 + OC * 15;
			int tYG = tY0 + (OC << 4), tYH = tY0 + OC * 17;
			int tYI = tY0 + OC * 18  , tYJ = tY0 + OC * 19;
			int tYK = tY0 + OC * 20  , tYL = tY0 + OC * 21;
			int tYM = tY0 + OC * 22  , tYN = tY0 + OC * 23;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
			tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
			tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
			tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
			tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
			tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1);
			tYC = IF_int(lh0, tYC, -1); tYD = IF_int(lh0, tYD, -1);
			tYE = IF_int(lh0, tYE, -1); tYF = IF_int(lh0, tYF, -1);
			tYG = IF_int(lh0, tYG, -1); tYH = IF_int(lh0, tYH, -1);
			tYI = IF_int(lh0, tYI, -1); tYJ = IF_int(lh0, tYJ, -1);
			tYK = IF_int(lyK, tYK, -1); tYL = IF_int(lyL, tYL, -1);
			tYM = IF_int(lyM, tYM, -1); tYN = IF_int(lyN, tYN, -1);
			x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
			x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
			x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
			x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
			x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
			x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
			x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
			x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);
			x[16] = tex1Dfetch<float>(deltaY, tYG); x[17] = tex1Dfetch<float>(deltaY, tYH);
			x[18] = tex1Dfetch<float>(deltaY, tYI); x[19] = tex1Dfetch<float>(deltaY, tYJ);
			x[20] = tex1Dfetch<float>(deltaY, tYK); x[21] = tex1Dfetch<float>(deltaY, tYL);
			x[22] = tex1Dfetch<float>(deltaY, tYM); x[23] = tex1Dfetch<float>(deltaY, tYN);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g0, w0);
			winograd_f8x9_G(g1, w1);
			winograd_f8x9_D(d0, x);//x[0 - 15]
			winograd_f8x9_D_oft(d1, x, 8);//x[8 - 23]

			//write to shread memory
			*(float2*)(&Gs[Gk][0x0][Gi]) = { g0[0x0], g1[0x0] }; *(float2*)(&Gs[Gk][0x1][Gi]) = { g0[0x1], g1[0x1] };
			*(float2*)(&Gs[Gk][0x2][Gi]) = { g0[0x2], g1[0x2] }; *(float2*)(&Gs[Gk][0x3][Gi]) = { g0[0x3], g1[0x3] };
			*(float2*)(&Gs[Gk][0x4][Gi]) = { g0[0x4], g1[0x4] }; *(float2*)(&Gs[Gk][0x5][Gi]) = { g0[0x5], g1[0x5] };
			*(float2*)(&Gs[Gk][0x6][Gi]) = { g0[0x6], g1[0x6] }; *(float2*)(&Gs[Gk][0x7][Gi]) = { g0[0x7], g1[0x7] };
			*(float2*)(&Gs[Gk][0x8][Gi]) = { g0[0x8], g1[0x8] }; *(float2*)(&Gs[Gk][0x9][Gi]) = { g0[0x9], g1[0x9] };
			*(float2*)(&Gs[Gk][0xA][Gi]) = { g0[0xA], g1[0xA] }; *(float2*)(&Gs[Gk][0xB][Gi]) = { g0[0xB], g1[0xB] };
			*(float2*)(&Gs[Gk][0xC][Gi]) = { g0[0xC], g1[0xC] }; *(float2*)(&Gs[Gk][0xD][Gi]) = { g0[0xD], g1[0xD] };
			*(float2*)(&Gs[Gk][0xE][Gi]) = { g0[0xE], g1[0xE] }; *(float2*)(&Gs[Gk][0xF][Gi]) = { g0[0xF], g1[0xF] };

			*(float2*)(&Ds[Dk][0x0][Di]) = { d0[0x0], d1[0x0] }; *(float2*)(&Ds[Dk][0x1][Di]) = { d0[0x1], d1[0x1] };
			*(float2*)(&Ds[Dk][0x2][Di]) = { d0[0x2], d1[0x2] }; *(float2*)(&Ds[Dk][0x3][Di]) = { d0[0x3], d1[0x3] };
			*(float2*)(&Ds[Dk][0x4][Di]) = { d0[0x4], d1[0x4] }; *(float2*)(&Ds[Dk][0x5][Di]) = { d0[0x5], d1[0x5] };
			*(float2*)(&Ds[Dk][0x6][Di]) = { d0[0x6], d1[0x6] }; *(float2*)(&Ds[Dk][0x7][Di]) = { d0[0x7], d1[0x7] };
			*(float2*)(&Ds[Dk][0x8][Di]) = { d0[0x8], d1[0x8] }; *(float2*)(&Ds[Dk][0x9][Di]) = { d0[0x9], d1[0x9] };
			*(float2*)(&Ds[Dk][0xA][Di]) = { d0[0xA], d1[0xA] }; *(float2*)(&Ds[Dk][0xB][Di]) = { d0[0xB], d1[0xB] };
			*(float2*)(&Ds[Dk][0xC][Di]) = { d0[0xC], d1[0xC] }; *(float2*)(&Ds[Dk][0xD][Di]) = { d0[0xD], d1[0xD] };
			*(float2*)(&Ds[Dk][0xE][Di]) = { d0[0xE], d1[0xE] }; *(float2*)(&Ds[Dk][0xF][Di]) = { d0[0xF], d1[0xF] };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 b0 = *(float4*)(&Ds[ik][ux][DIdx    ]);
			float4 b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

			//=======================================================
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx     ]);
			float4 a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

			//ic[0 - 3]             ic[4 - 7]
			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7

			//=======================================================
			float4 a2 = *(float4*)(&Gs[ik][ux][GIdx +  8]);
			float4 a3 = *(float4*)(&Gs[ik][ux][GIdx + 12]);

			//ic[8 - 11]            ic[12 - 15]
			simdMM4(v16, b0.x, a2); simdMM4(v17, b0.x, a3);//j0
			simdMM4(v18, b0.y, a2); simdMM4(v19, b0.y, a3);//j1
			simdMM4(v20, b0.z, a2); simdMM4(v21, b0.z, a3);//j2
			simdMM4(v22, b0.w, a2); simdMM4(v23, b0.w, a3);//j3
			simdMM4(v24, b1.x, a2); simdMM4(v25, b1.x, a3);//j4
			simdMM4(v26, b1.y, a2); simdMM4(v27, b1.y, a3);//j5
			simdMM4(v28, b1.z, a2); simdMM4(v29, b1.z, a3);//j6
			simdMM4(v30, b1.w, a2); simdMM4(v31, b1.w, a3);//j7
		}
		__syncthreads();
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Xs0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Xs1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], x0[8], x1[8], x2[8], x3[8];//ic0, ic1, ic2, ic3
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	//write-turn0: x, [ic0, ic4]
	*(float4*)(&Xs0[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Xs0[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Xs0[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Xs0[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//write-turn1: y, [ic0, ic4]
	*(float4*)(&Xs1[ux][uy][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Xs1[ux][uy][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Xs1[ux][uy][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Xs1[ux][uy][12]) = float4{ v12.y, v13.y, v14.y, v15.y };

	//write-turn2: z, [ic2, ic6]
	*(float4*)(&Xs0[ux][uy + 8][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Xs0[ux][uy + 8][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Xs0[ux][uy + 8][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Xs0[ux][uy + 8][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn1: w, [ic3, ic7]
	*(float4*)(&Xs1[ux][uy + 8][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Xs1[ux][uy + 8][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Xs1[ux][uy + 8][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Xs1[ux][uy + 8][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn0: x, [ic0, ic4]
	a[ 0] = Xs0[ 0][uy][ux]; a[ 1] = Xs0[ 1][uy][ux]; a[ 2] = Xs0[ 2][uy][ux]; a[ 3] = Xs0[ 3][uy][ux];
	a[ 4] = Xs0[ 4][uy][ux]; a[ 5] = Xs0[ 5][uy][ux]; a[ 6] = Xs0[ 6][uy][ux]; a[ 7] = Xs0[ 7][uy][ux];
	a[ 8] = Xs0[ 8][uy][ux]; a[ 9] = Xs0[ 9][uy][ux]; a[10] = Xs0[10][uy][ux]; a[11] = Xs0[11][uy][ux];
	a[12] = Xs0[12][uy][ux]; a[13] = Xs0[13][uy][ux]; a[14] = Xs0[14][uy][ux]; a[15] = Xs0[15][uy][ux];
	winograd_f8x9_Y(x0, a);

	//read-turn1: y, [ic1, ic5]
	a[ 0] = Xs1[ 0][uy][ux]; a[ 1] = Xs1[ 1][uy][ux]; a[ 2] = Xs1[ 2][uy][ux]; a[ 3] = Xs1[ 3][uy][ux];
	a[ 4] = Xs1[ 4][uy][ux]; a[ 5] = Xs1[ 5][uy][ux]; a[ 6] = Xs1[ 6][uy][ux]; a[ 7] = Xs1[ 7][uy][ux];
	a[ 8] = Xs1[ 8][uy][ux]; a[ 9] = Xs1[ 9][uy][ux]; a[10] = Xs1[10][uy][ux]; a[11] = Xs1[11][uy][ux];
	a[12] = Xs1[12][uy][ux]; a[13] = Xs1[13][uy][ux]; a[14] = Xs1[14][uy][ux]; a[15] = Xs1[15][uy][ux];
	winograd_f8x9_Y(x1, a);

	//read-turn2: z, [ic2, ic6]
	a[ 0] = Xs0[ 0][uy + 8][ux]; a[ 1] = Xs0[ 1][uy + 8][ux]; a[ 2] = Xs0[ 2][uy + 8][ux]; a[ 3] = Xs0[ 3][uy + 8][ux];
	a[ 4] = Xs0[ 4][uy + 8][ux]; a[ 5] = Xs0[ 5][uy + 8][ux]; a[ 6] = Xs0[ 6][uy + 8][ux]; a[ 7] = Xs0[ 7][uy + 8][ux];
	a[ 8] = Xs0[ 8][uy + 8][ux]; a[ 9] = Xs0[ 9][uy + 8][ux]; a[10] = Xs0[10][uy + 8][ux]; a[11] = Xs0[11][uy + 8][ux];
	a[12] = Xs0[12][uy + 8][ux]; a[13] = Xs0[13][uy + 8][ux]; a[14] = Xs0[14][uy + 8][ux]; a[15] = Xs0[15][uy + 8][ux];
	winograd_f8x9_Y(x2, a);

	//read-turn3: w, [ic3, ic7]
	a[ 0] = Xs1[ 0][uy + 8][ux]; a[ 1] = Xs1[ 1][uy + 8][ux]; a[ 2] = Xs1[ 2][uy + 8][ux]; a[ 3] = Xs1[ 3][uy + 8][ux];
	a[ 4] = Xs1[ 4][uy + 8][ux]; a[ 5] = Xs1[ 5][uy + 8][ux]; a[ 6] = Xs1[ 6][uy + 8][ux]; a[ 7] = Xs1[ 7][uy + 8][ux];
	a[ 8] = Xs1[ 8][uy + 8][ux]; a[ 9] = Xs1[ 9][uy + 8][ux]; a[10] = Xs1[10][uy + 8][ux]; a[11] = Xs1[11][uy + 8][ux];
	a[12] = Xs1[12][uy + 8][ux]; a[13] = Xs1[13][uy + 8][ux]; a[14] = Xs1[14][uy + 8][ux]; a[15] = Xs1[15][uy + 8][ux];
	winograd_f8x9_Y(x3, a);

	//write to deltaX[N, IH, IW, IC]
	*(float4*)(deltaX + X00         ) = float4{ x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC    ) = float4{ x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2) = float4{ x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3) = float4{ x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4) = float4{ x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5) = float4{ x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6) = float4{ x0[6], x1[6], x2[6], x3[6] };
	*(float4*)(deltaX + X00 + IC * 7) = float4{ x0[7], x1[7], x2[7], x3[7] };
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	//write-turn0: x, [ic0, ic4]
	*(float4*)(&Xs0[ux][uy][ 0]) = float4{ v16.x, v17.x, v18.x, v19.x };
	*(float4*)(&Xs0[ux][uy][ 4]) = float4{ v20.x, v21.x, v22.x, v23.x };
	*(float4*)(&Xs0[ux][uy][ 8]) = float4{ v24.x, v25.x, v26.x, v27.x };
	*(float4*)(&Xs0[ux][uy][12]) = float4{ v28.x, v29.x, v30.x, v31.x };

	//write-turn1: y, [ic0, ic4]
	*(float4*)(&Xs1[ux][uy][ 0]) = float4{ v16.y, v17.y, v18.y, v19.y };
	*(float4*)(&Xs1[ux][uy][ 4]) = float4{ v20.y, v21.y, v22.y, v23.y };
	*(float4*)(&Xs1[ux][uy][ 8]) = float4{ v24.y, v25.y, v26.y, v27.y };
	*(float4*)(&Xs1[ux][uy][12]) = float4{ v28.y, v29.y, v30.y, v31.y };

	//write-turn2: z, [ic2, ic6]
	*(float4*)(&Xs0[ux][uy + 8][ 0]) = float4{ v16.z, v17.z, v18.z, v19.z };
	*(float4*)(&Xs0[ux][uy + 8][ 4]) = float4{ v20.z, v21.z, v22.z, v23.z };
	*(float4*)(&Xs0[ux][uy + 8][ 8]) = float4{ v24.z, v25.z, v26.z, v27.z };
	*(float4*)(&Xs0[ux][uy + 8][12]) = float4{ v28.z, v29.z, v30.z, v31.z };

	//write-turn1: w, [ic3, ic7]
	*(float4*)(&Xs1[ux][uy + 8][ 0]) = float4{ v16.w, v17.w, v18.w, v19.w };
	*(float4*)(&Xs1[ux][uy + 8][ 4]) = float4{ v20.w, v21.w, v22.w, v23.w };
	*(float4*)(&Xs1[ux][uy + 8][ 8]) = float4{ v24.w, v25.w, v26.w, v27.w };
	*(float4*)(&Xs1[ux][uy + 8][12]) = float4{ v28.w, v29.w, v30.w, v31.w };
	__syncthreads();

	//read-turn0: x, [ic0, ic4]
	a[ 0] = Xs0[ 0][uy][ux]; a[ 1] = Xs0[ 1][uy][ux]; a[ 2] = Xs0[ 2][uy][ux]; a[ 3] = Xs0[ 3][uy][ux];
	a[ 4] = Xs0[ 4][uy][ux]; a[ 5] = Xs0[ 5][uy][ux]; a[ 6] = Xs0[ 6][uy][ux]; a[ 7] = Xs0[ 7][uy][ux];
	a[ 8] = Xs0[ 8][uy][ux]; a[ 9] = Xs0[ 9][uy][ux]; a[10] = Xs0[10][uy][ux]; a[11] = Xs0[11][uy][ux];
	a[12] = Xs0[12][uy][ux]; a[13] = Xs0[13][uy][ux]; a[14] = Xs0[14][uy][ux]; a[15] = Xs0[15][uy][ux];
	winograd_f8x9_Y(x0, a);

	//read-turn1: y, [ic1, ic5]
	a[ 0] = Xs1[ 0][uy][ux]; a[ 1] = Xs1[ 1][uy][ux]; a[ 2] = Xs1[ 2][uy][ux]; a[ 3] = Xs1[ 3][uy][ux];
	a[ 4] = Xs1[ 4][uy][ux]; a[ 5] = Xs1[ 5][uy][ux]; a[ 6] = Xs1[ 6][uy][ux]; a[ 7] = Xs1[ 7][uy][ux];
	a[ 8] = Xs1[ 8][uy][ux]; a[ 9] = Xs1[ 9][uy][ux]; a[10] = Xs1[10][uy][ux]; a[11] = Xs1[11][uy][ux];
	a[12] = Xs1[12][uy][ux]; a[13] = Xs1[13][uy][ux]; a[14] = Xs1[14][uy][ux]; a[15] = Xs1[15][uy][ux];
	winograd_f8x9_Y(x1, a);

	//read-turn2: z, [ic2, ic6]
	a[ 0] = Xs0[ 0][uy + 8][ux]; a[ 1] = Xs0[ 1][uy + 8][ux]; a[ 2] = Xs0[ 2][uy + 8][ux]; a[ 3] = Xs0[ 3][uy + 8][ux];
	a[ 4] = Xs0[ 4][uy + 8][ux]; a[ 5] = Xs0[ 5][uy + 8][ux]; a[ 6] = Xs0[ 6][uy + 8][ux]; a[ 7] = Xs0[ 7][uy + 8][ux];
	a[ 8] = Xs0[ 8][uy + 8][ux]; a[ 9] = Xs0[ 9][uy + 8][ux]; a[10] = Xs0[10][uy + 8][ux]; a[11] = Xs0[11][uy + 8][ux];
	a[12] = Xs0[12][uy + 8][ux]; a[13] = Xs0[13][uy + 8][ux]; a[14] = Xs0[14][uy + 8][ux]; a[15] = Xs0[15][uy + 8][ux];
	winograd_f8x9_Y(x2, a);

	//read-turn3: w, [ic3, ic7]
	a[ 0] = Xs1[ 0][uy + 8][ux]; a[ 1] = Xs1[ 1][uy + 8][ux]; a[ 2] = Xs1[ 2][uy + 8][ux]; a[ 3] = Xs1[ 3][uy + 8][ux];
	a[ 4] = Xs1[ 4][uy + 8][ux]; a[ 5] = Xs1[ 5][uy + 8][ux]; a[ 6] = Xs1[ 6][uy + 8][ux]; a[ 7] = Xs1[ 7][uy + 8][ux];
	a[ 8] = Xs1[ 8][uy + 8][ux]; a[ 9] = Xs1[ 9][uy + 8][ux]; a[10] = Xs1[10][uy + 8][ux]; a[11] = Xs1[11][uy + 8][ux];
	a[12] = Xs1[12][uy + 8][ux]; a[13] = Xs1[13][uy + 8][ux]; a[14] = Xs1[14][uy + 8][ux]; a[15] = Xs1[15][uy + 8][ux];
	winograd_f8x9_Y(x3, a);

	//write to deltaX[N, IH, IW, IC]
	*(float4*)(deltaX + X00          + 8) = float4{ x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC     + 8) = float4{ x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2 + 8) = float4{ x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3 + 8) = float4{ x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4 + 8) = float4{ x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5 + 8) = float4{ x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6 + 8) = float4{ x0[6], x1[6], x2[6], x3[6] };
	*(float4*)(deltaX + X00 + IC * 7 + 8) = float4{ x0[7], x1[7], x2[7], x3[7] };
}

#endif


//================[Standard: 64(IC) * 256(GM)]===============================
//LB = 4, IC % 8 == 0, IW % 10 == 0, pw >= 4
#ifndef DECONV_3D_DX_WINOGRAD_F8X9_64X256_P4_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F8X9_64X256_P4_TEXTURE

//for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 162, Time = 10.812 msec, Performace = 32176.5 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 162, Time = 12.238 msec, Performace = 28427.2 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 162, Time = 11.636 msec, Performace = 29897.9 GFlop/s

template<int FH>
__global__ void winograd_f8x9_kernel_64x256_p4_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float smem[8 * 16 * 96];
	float(*__restrict__ Gs)[16][64] = (float(*)[16][64])(smem          );//[ik: STEP][elem: 16][g(ic): 64]
	float(*__restrict__ Ds)[16][32] = (float(*)[16][32])(smem + 8*16*64);//[ik: STEP][elem: 16][d( j): 32]

	///compute 8*16 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	float4 v16 = F32_4_0, v17 = F32_4_0, v18 = F32_4_0, v19 = F32_4_0;
	float4 v20 = F32_4_0, v21 = F32_4_0, v22 = F32_4_0, v23 = F32_4_0;
	float4 v24 = F32_4_0, v25 = F32_4_0, v26 = F32_4_0, v27 = F32_4_0;
	float4 v28 = F32_4_0, v29 = F32_4_0, v30 = F32_4_0, v31 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (bx << 6);
	const int Gk = (ty & 7), Gi = ((tx << 1) + (ty > 7) << 1);//[8, 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 9 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, IC]
	const int bj0 = (by << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 3);//OW % 8
	const int IW8 = (IW >> 3) << 3, IH_IW8 = IH * IW8;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW8, IW8);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Xk;//deltaY[tn0, toh0, tow0, Dk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 4;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);//0, + 8
	const int xj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW8, IW8);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//======[compute area1: local]======================================================
	const int Di = (Xi + (Xk << 2)) & 31;//avoid bank conflict (1/8)
	float w0[9], w1[9], g0[16], g1[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = -(fh * 9) * IC;//rotate180
		const int W1 = W0 + IC    , W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
		float2 wv8 = *(float2*)(W + W0); w0[8] = wv8.x; w1[8] = wv8.y;
		float2 wv7 = *(float2*)(W + W1); w0[7] = wv7.x; w1[7] = wv7.y;
		float2 wv6 = *(float2*)(W + W2); w0[6] = wv6.x; w1[6] = wv6.y;
		float2 wv5 = *(float2*)(W + W3); w0[5] = wv5.x; w1[5] = wv5.y;
		float2 wv4 = *(float2*)(W + W4); w0[4] = wv4.x; w1[4] = wv4.y;
		float2 wv3 = *(float2*)(W + W5); w0[3] = wv3.x; w1[3] = wv3.y;
		float2 wv2 = *(float2*)(W + W6); w0[2] = wv2.x; w1[2] = wv2.y;
		float2 wv1 = *(float2*)(W + W7); w0[1] = wv1.x; w1[1] = wv1.y;
		float2 wv0 = *(float2*)(W + W8); w0[0] = wv0.x; w1[0] = wv0.y;

		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=   0) && (tow0      < OW);//pw = -4
		bool ly1 = lh0 && (tow0 >=  -1) && (tow0 +  1 < OW);//pw = -3
		bool ly2 = lh0 && (tow0 >=  -2) && (tow0 +  2 < OW);//pw = -2
		bool ly3 = lh0 && (tow0 >=  -3) && (tow0 +  3 < OW);//pw = -1
		bool lyC = lh0 && (tow0 >= -12) && (tow0 + 12 < OW);//pw = +1
		bool lyD = lh0 && (tow0 >= -13) && (tow0 + 13 < OW);//pw = +2
		bool lyE = lh0 && (tow0 >= -14) && (tow0 + 14 < OW);//pw = +3
		bool lyF = lh0 && (tow0 >= -15) && (tow0 + 15 < OW);//pw = +4
		int tY0 = Y0 + fh * OW * OC, tY1 = tY0 + OC;
		int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC *  3;
		int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC *  5;
		int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC *  7;
		int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC *  9;
		int tYA = tY0 + OC * 10  , tYB = tY0 + OC * 11;
		int tYC = tY0 + OC * 12  , tYD = tY0 + OC * 13;
		int tYE = tY0 + OC * 14  , tYF = tY0 + OC * 15;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
		tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
		tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
		tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
		tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
		tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1); 
		tYC = IF_int(lyC, tYC, -1); tYD = IF_int(lyD, tYD, -1); 
		tYE = IF_int(lyE, tYE, -1); tYF = IF_int(lyF, tYF, -1);
		x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
		x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
		x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
		x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
		x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
		x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
		x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
		x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);

		for (int ooc = 8; ooc < OC; ooc += 8) {
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g0, w0);
			winograd_f8x9_G(g1, w1);
			winograd_f8x9_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[Gk][ 0][Gi]) = { g0[ 0], g1[ 0] };
			*(float2*)(&Gs[Gk][ 1][Gi]) = { g0[ 1], g1[ 1] };
			*(float2*)(&Gs[Gk][ 2][Gi]) = { g0[ 2], g1[ 2] };
			*(float2*)(&Gs[Gk][ 3][Gi]) = { g0[ 3], g1[ 3] };
			*(float2*)(&Gs[Gk][ 4][Gi]) = { g0[ 4], g1[ 4] };
			*(float2*)(&Gs[Gk][ 5][Gi]) = { g0[ 5], g1[ 5] };
			*(float2*)(&Gs[Gk][ 6][Gi]) = { g0[ 6], g1[ 6] };
			*(float2*)(&Gs[Gk][ 7][Gi]) = { g0[ 7], g1[ 7] };
			*(float2*)(&Gs[Gk][ 8][Gi]) = { g0[ 8], g1[ 8] };
			*(float2*)(&Gs[Gk][ 9][Gi]) = { g0[ 9], g1[ 9] };
			*(float2*)(&Gs[Gk][10][Gi]) = { g0[10], g1[10] };
			*(float2*)(&Gs[Gk][11][Gi]) = { g0[11], g1[11] };
			*(float2*)(&Gs[Gk][12][Gi]) = { g0[12], g1[12] };
			*(float2*)(&Gs[Gk][13][Gi]) = { g0[13], g1[13] };
			*(float2*)(&Gs[Gk][14][Gi]) = { g0[14], g1[14] };
			*(float2*)(&Gs[Gk][15][Gi]) = { g0[15], g1[15] };

			Ds[Xk][ 0][Di] = d[ 0]; Ds[Xk][ 1][Di] = d[ 1];
			Ds[Xk][ 2][Di] = d[ 2]; Ds[Xk][ 3][Di] = d[ 3];
			Ds[Xk][ 4][Di] = d[ 4]; Ds[Xk][ 5][Di] = d[ 5];
			Ds[Xk][ 6][Di] = d[ 6]; Ds[Xk][ 7][Di] = d[ 7];
			Ds[Xk][ 8][Di] = d[ 8]; Ds[Xk][ 9][Di] = d[ 9];
			Ds[Xk][10][Di] = d[10]; Ds[Xk][11][Di] = d[11];
			Ds[Xk][12][Di] = d[12]; Ds[Xk][13][Di] = d[13];
			Ds[Xk][14][Di] = d[14]; Ds[Xk][15][Di] = d[15];
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 b0 = *(float4*)(&Ds[ik][ux][(DIdx + (ik << 2)    ) & 31]);
				float4 b1 = *(float4*)(&Ds[ik][ux][(DIdx + (ik << 2) + 4) & 31]);

				//=======================================================
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

				//ic[0 - 3]             ic[4 - 7]
				simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
				simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
				simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
				simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
				simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7

				//=======================================================
				float4 a2 = *(float4*)(&Gs[ik][ux][GIdx +  8]);
				float4 a3 = *(float4*)(&Gs[ik][ux][GIdx + 12]);

				//ic[8 - 11]            ic[12 - 15]
				simdMM4(v16, b0.x, a2); simdMM4(v17, b0.x, a3);//j0
				simdMM4(v18, b0.y, a2); simdMM4(v19, b0.y, a3);//j1
				simdMM4(v20, b0.z, a2); simdMM4(v21, b0.z, a3);//j2
				simdMM4(v22, b0.w, a2); simdMM4(v23, b0.w, a3);//j3
				simdMM4(v24, b1.x, a2); simdMM4(v25, b1.x, a3);//j4
				simdMM4(v26, b1.y, a2); simdMM4(v27, b1.y, a3);//j5
				simdMM4(v28, b1.z, a2); simdMM4(v29, b1.z, a3);//j6
				simdMM4(v30, b1.w, a2); simdMM4(v31, b1.w, a3);//j7
			}

			//load 2 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 9 * IC;//rotate180
			const int W1 = W0 + IC    , W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
			float2 wv8 = *(float2*)(W + W0); w0[8] = wv8.x; w1[8] = wv8.y;
			float2 wv7 = *(float2*)(W + W1); w0[7] = wv7.x; w1[7] = wv7.y;
			float2 wv6 = *(float2*)(W + W2); w0[6] = wv6.x; w1[6] = wv6.y;
			float2 wv5 = *(float2*)(W + W3); w0[5] = wv5.x; w1[5] = wv5.y;
			float2 wv4 = *(float2*)(W + W4); w0[4] = wv4.x; w1[4] = wv4.y;
			float2 wv3 = *(float2*)(W + W5); w0[3] = wv3.x; w1[3] = wv3.y;
			float2 wv2 = *(float2*)(W + W6); w0[2] = wv2.x; w1[2] = wv2.y;
			float2 wv1 = *(float2*)(W + W7); w0[1] = wv1.x; w1[1] = wv1.y;
			float2 wv0 = *(float2*)(W + W8); w0[0] = wv0.x; w1[0] = wv0.y;

			//load 1 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc, tY1 = tY0 + OC;
			int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC *  3;
			int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC *  5;
			int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC *  7;
			int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC *  9;
			int tYA = tY0 + OC * 10  , tYB = tY0 + OC * 11;
			int tYC = tY0 + OC * 12  , tYD = tY0 + OC * 13;
			int tYE = tY0 + OC * 14  , tYF = tY0 + OC * 15;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
			tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
			tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
			tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
			tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
			tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1);
			tYC = IF_int(lyC, tYC, -1); tYD = IF_int(lyD, tYD, -1);
			tYE = IF_int(lyE, tYE, -1); tYF = IF_int(lyF, tYF, -1);
			x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
			x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
			x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
			x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
			x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
			x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
			x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
			x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g0, w0);
			winograd_f8x9_G(g1, w1);
			winograd_f8x9_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[Gk][ 0][Gi]) = { g0[ 0], g1[ 0] };
			*(float2*)(&Gs[Gk][ 1][Gi]) = { g0[ 1], g1[ 1] };
			*(float2*)(&Gs[Gk][ 2][Gi]) = { g0[ 2], g1[ 2] };
			*(float2*)(&Gs[Gk][ 3][Gi]) = { g0[ 3], g1[ 3] };
			*(float2*)(&Gs[Gk][ 4][Gi]) = { g0[ 4], g1[ 4] };
			*(float2*)(&Gs[Gk][ 5][Gi]) = { g0[ 5], g1[ 5] };
			*(float2*)(&Gs[Gk][ 6][Gi]) = { g0[ 6], g1[ 6] };
			*(float2*)(&Gs[Gk][ 7][Gi]) = { g0[ 7], g1[ 7] };
			*(float2*)(&Gs[Gk][ 8][Gi]) = { g0[ 8], g1[ 8] };
			*(float2*)(&Gs[Gk][ 9][Gi]) = { g0[ 9], g1[ 9] };
			*(float2*)(&Gs[Gk][10][Gi]) = { g0[10], g1[10] };
			*(float2*)(&Gs[Gk][11][Gi]) = { g0[11], g1[11] };
			*(float2*)(&Gs[Gk][12][Gi]) = { g0[12], g1[12] };
			*(float2*)(&Gs[Gk][13][Gi]) = { g0[13], g1[13] };
			*(float2*)(&Gs[Gk][14][Gi]) = { g0[14], g1[14] };
			*(float2*)(&Gs[Gk][15][Gi]) = { g0[15], g1[15] };

			Ds[Xk][ 0][Di] = d[ 0]; Ds[Xk][ 1][Di] = d[ 1];
			Ds[Xk][ 2][Di] = d[ 2]; Ds[Xk][ 3][Di] = d[ 3];
			Ds[Xk][ 4][Di] = d[ 4]; Ds[Xk][ 5][Di] = d[ 5];
			Ds[Xk][ 6][Di] = d[ 6]; Ds[Xk][ 7][Di] = d[ 7];
			Ds[Xk][ 8][Di] = d[ 8]; Ds[Xk][ 9][Di] = d[ 9];
			Ds[Xk][10][Di] = d[10]; Ds[Xk][11][Di] = d[11];
			Ds[Xk][12][Di] = d[12]; Ds[Xk][13][Di] = d[13];
			Ds[Xk][14][Di] = d[14]; Ds[Xk][15][Di] = d[15];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 b0 = *(float4*)(&Ds[ik][ux][(DIdx + (ik << 2)    ) & 31]);
			float4 b1 = *(float4*)(&Ds[ik][ux][(DIdx + (ik << 2) + 4) & 31]);

			//=======================================================
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

			//ic[0 - 3]             ic[4 - 7]
			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7

			//=======================================================
			float4 a2 = *(float4*)(&Gs[ik][ux][GIdx +  8]);
			float4 a3 = *(float4*)(&Gs[ik][ux][GIdx + 12]);

			//ic[8 - 11]            ic[12 - 15]
			simdMM4(v16, b0.x, a2); simdMM4(v17, b0.x, a3);//j0
			simdMM4(v18, b0.y, a2); simdMM4(v19, b0.y, a3);//j1
			simdMM4(v20, b0.z, a2); simdMM4(v21, b0.z, a3);//j2
			simdMM4(v22, b0.w, a2); simdMM4(v23, b0.w, a3);//j3
			simdMM4(v24, b1.x, a2); simdMM4(v25, b1.x, a3);//j4
			simdMM4(v26, b1.y, a2); simdMM4(v27, b1.y, a3);//j5
			simdMM4(v28, b1.z, a2); simdMM4(v29, b1.z, a3);//j6
			simdMM4(v30, b1.w, a2); simdMM4(v31, b1.w, a3);//j7
		}
		__syncthreads();
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Xs)[33][20] = (float(*)[33][20])(smem);
	float a[16], x0[8], x1[8], x2[8], x3[8];//ic0, ic1, ic2, ic3
	__syncthreads();

	//======[ic0 - ic7]==================================================================
	//group0-----------------------------------------------------------------------------
	*(float4*)(&Xs[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Xs[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Xs[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Xs[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };
	
	*(float4*)(&Xs[ux][uy + 16][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Xs[ux][uy + 16][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Xs[ux][uy + 16][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Xs[ux][uy + 16][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//read-turn0: x, [ic0, ic4]
	a[ 0] = Xs[ 0][uy][ux]; a[ 1] = Xs[ 1][uy][ux]; a[ 2] = Xs[ 2][uy][ux]; a[ 3] = Xs[ 3][uy][ux];
	a[ 4] = Xs[ 4][uy][ux]; a[ 5] = Xs[ 5][uy][ux]; a[ 6] = Xs[ 6][uy][ux]; a[ 7] = Xs[ 7][uy][ux];
	a[ 8] = Xs[ 8][uy][ux]; a[ 9] = Xs[ 9][uy][ux]; a[10] = Xs[10][uy][ux]; a[11] = Xs[11][uy][ux];
	a[12] = Xs[12][uy][ux]; a[13] = Xs[13][uy][ux]; a[14] = Xs[14][uy][ux]; a[15] = Xs[15][uy][ux];
	winograd_f8x9_Y(x0, a);
	
	//read-turn1: y, [ic1, ic5]
	a[ 0] = Xs[ 0][uy + 16][ux]; a[ 1] = Xs[ 1][uy + 16][ux]; a[ 2] = Xs[ 2][uy + 16][ux]; a[ 3] = Xs[ 3][uy + 16][ux];
	a[ 4] = Xs[ 4][uy + 16][ux]; a[ 5] = Xs[ 5][uy + 16][ux]; a[ 6] = Xs[ 6][uy + 16][ux]; a[ 7] = Xs[ 7][uy + 16][ux];
	a[ 8] = Xs[ 8][uy + 16][ux]; a[ 9] = Xs[ 9][uy + 16][ux]; a[10] = Xs[10][uy + 16][ux]; a[11] = Xs[11][uy + 16][ux];
	a[12] = Xs[12][uy + 16][ux]; a[13] = Xs[13][uy + 16][ux]; a[14] = Xs[14][uy + 16][ux]; a[15] = Xs[15][uy + 16][ux];
	winograd_f8x9_Y(x1, a);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	*(float4*)(&Xs[ux][uy][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Xs[ux][uy][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Xs[ux][uy][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Xs[ux][uy][12]) = float4{ v12.z, v13.z, v14.z, v15.z };
	
	*(float4*)(&Xs[ux][uy + 16][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Xs[ux][uy + 16][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Xs[ux][uy + 16][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Xs[ux][uy + 16][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[ 0] = Xs[ 0][uy][ux]; a[ 1] = Xs[ 1][uy][ux]; a[ 2] = Xs[ 2][uy][ux]; a[ 3] = Xs[ 3][uy][ux];
	a[ 4] = Xs[ 4][uy][ux]; a[ 5] = Xs[ 5][uy][ux]; a[ 6] = Xs[ 6][uy][ux]; a[ 7] = Xs[ 7][uy][ux];
	a[ 8] = Xs[ 8][uy][ux]; a[ 9] = Xs[ 9][uy][ux]; a[10] = Xs[10][uy][ux]; a[11] = Xs[11][uy][ux];
	a[12] = Xs[12][uy][ux]; a[13] = Xs[13][uy][ux]; a[14] = Xs[14][uy][ux]; a[15] = Xs[15][uy][ux];
	winograd_f8x9_Y(x2, a);

	a[ 0] = Xs[ 0][uy + 16][ux]; a[ 1] = Xs[ 1][uy + 16][ux]; a[ 2] = Xs[ 2][uy + 16][ux]; a[ 3] = Xs[ 3][uy + 16][ux];
	a[ 4] = Xs[ 4][uy + 16][ux]; a[ 5] = Xs[ 5][uy + 16][ux]; a[ 6] = Xs[ 6][uy + 16][ux]; a[ 7] = Xs[ 7][uy + 16][ux];
	a[ 8] = Xs[ 8][uy + 16][ux]; a[ 9] = Xs[ 9][uy + 16][ux]; a[10] = Xs[10][uy + 16][ux]; a[11] = Xs[11][uy + 16][ux];
	a[12] = Xs[12][uy + 16][ux]; a[13] = Xs[13][uy + 16][ux]; a[14] = Xs[14][uy + 16][ux]; a[15] = Xs[15][uy + 16][ux];
	winograd_f8x9_Y(x3, a);
	__syncthreads();

	//write to deltaX[N, IH, IW, IC]
	*(float4*)(deltaX + X00         ) = float4{ x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC    ) = float4{ x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2) = float4{ x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3) = float4{ x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4) = float4{ x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5) = float4{ x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6) = float4{ x0[6], x1[6], x2[6], x3[6] };
	*(float4*)(deltaX + X00 + IC * 7) = float4{ x0[7], x1[7], x2[7], x3[7] };

	//======[ic8 - ic15]=================================================================
	//group2-----------------------------------------------------------------------------
	*(float4*)(&Xs[ux][uy][ 0]) = float4{ v16.x, v17.x, v18.x, v19.x };
	*(float4*)(&Xs[ux][uy][ 4]) = float4{ v20.x, v21.x, v22.x, v23.x };
	*(float4*)(&Xs[ux][uy][ 8]) = float4{ v24.x, v25.x, v26.x, v27.x };
	*(float4*)(&Xs[ux][uy][12]) = float4{ v28.x, v29.x, v30.x, v31.x };
	
	*(float4*)(&Xs[ux][uy + 16][ 0]) = float4{ v16.y, v17.y, v18.y, v19.y };
	*(float4*)(&Xs[ux][uy + 16][ 4]) = float4{ v20.y, v21.y, v22.y, v23.y };
	*(float4*)(&Xs[ux][uy + 16][ 8]) = float4{ v24.y, v25.y, v26.y, v27.y };
	*(float4*)(&Xs[ux][uy + 16][12]) = float4{ v28.y, v29.y, v30.y, v31.y };
	__syncthreads();

	a[ 0] = Xs[ 0][uy][ux]; a[ 1] = Xs[ 1][uy][ux]; a[ 2] = Xs[ 2][uy][ux]; a[ 3] = Xs[ 3][uy][ux];
	a[ 4] = Xs[ 4][uy][ux]; a[ 5] = Xs[ 5][uy][ux]; a[ 6] = Xs[ 6][uy][ux]; a[ 7] = Xs[ 7][uy][ux];
	a[ 8] = Xs[ 8][uy][ux]; a[ 9] = Xs[ 9][uy][ux]; a[10] = Xs[10][uy][ux]; a[11] = Xs[11][uy][ux];
	a[12] = Xs[12][uy][ux]; a[13] = Xs[13][uy][ux]; a[14] = Xs[14][uy][ux]; a[15] = Xs[15][uy][ux];
	winograd_f8x9_Y(x0, a);
	
	a[ 0] = Xs[ 0][uy + 16][ux]; a[ 1] = Xs[ 1][uy + 16][ux]; a[ 2] = Xs[ 2][uy + 16][ux]; a[ 3] = Xs[ 3][uy + 16][ux];
	a[ 4] = Xs[ 4][uy + 16][ux]; a[ 5] = Xs[ 5][uy + 16][ux]; a[ 6] = Xs[ 6][uy + 16][ux]; a[ 7] = Xs[ 7][uy + 16][ux];
	a[ 8] = Xs[ 8][uy + 16][ux]; a[ 9] = Xs[ 9][uy + 16][ux]; a[10] = Xs[10][uy + 16][ux]; a[11] = Xs[11][uy + 16][ux];
	a[12] = Xs[12][uy + 16][ux]; a[13] = Xs[13][uy + 16][ux]; a[14] = Xs[14][uy + 16][ux]; a[15] = Xs[15][uy + 16][ux];
	winograd_f8x9_Y(x1, a);
	__syncthreads();

	//group3-----------------------------------------------------------------------------
	*(float4*)(&Xs[ux][uy][ 0]) = float4{ v16.z, v17.z, v18.z, v19.z };
	*(float4*)(&Xs[ux][uy][ 4]) = float4{ v20.z, v21.z, v22.z, v23.z };
	*(float4*)(&Xs[ux][uy][ 8]) = float4{ v24.z, v25.z, v26.z, v27.z };
	*(float4*)(&Xs[ux][uy][12]) = float4{ v28.z, v29.z, v30.z, v31.z };
	
	*(float4*)(&Xs[ux][uy + 16][ 0]) = float4{ v16.w, v17.w, v18.w, v19.w };
	*(float4*)(&Xs[ux][uy + 16][ 4]) = float4{ v20.w, v21.w, v22.w, v23.w };
	*(float4*)(&Xs[ux][uy + 16][ 8]) = float4{ v24.w, v25.w, v26.w, v27.w };
	*(float4*)(&Xs[ux][uy + 16][12]) = float4{ v28.w, v29.w, v30.w, v31.w };
	__syncthreads();

	a[ 0] = Xs[ 0][uy][ux]; a[ 1] = Xs[ 1][uy][ux]; a[ 2] = Xs[ 2][uy][ux]; a[ 3] = Xs[ 3][uy][ux];
	a[ 4] = Xs[ 4][uy][ux]; a[ 5] = Xs[ 5][uy][ux]; a[ 6] = Xs[ 6][uy][ux]; a[ 7] = Xs[ 7][uy][ux];
	a[ 8] = Xs[ 8][uy][ux]; a[ 9] = Xs[ 9][uy][ux]; a[10] = Xs[10][uy][ux]; a[11] = Xs[11][uy][ux];
	a[12] = Xs[12][uy][ux]; a[13] = Xs[13][uy][ux]; a[14] = Xs[14][uy][ux]; a[15] = Xs[15][uy][ux];
	winograd_f8x9_Y(x2, a);

	a[ 0] = Xs[ 0][uy + 16][ux]; a[ 1] = Xs[ 1][uy + 16][ux]; a[ 2] = Xs[ 2][uy + 16][ux]; a[ 3] = Xs[ 3][uy + 16][ux];
	a[ 4] = Xs[ 4][uy + 16][ux]; a[ 5] = Xs[ 5][uy + 16][ux]; a[ 6] = Xs[ 6][uy + 16][ux]; a[ 7] = Xs[ 7][uy + 16][ux];
	a[ 8] = Xs[ 8][uy + 16][ux]; a[ 9] = Xs[ 9][uy + 16][ux]; a[10] = Xs[10][uy + 16][ux]; a[11] = Xs[11][uy + 16][ux];
	a[12] = Xs[12][uy + 16][ux]; a[13] = Xs[13][uy + 16][ux]; a[14] = Xs[14][uy + 16][ux]; a[15] = Xs[15][uy + 16][ux];
	winograd_f8x9_Y(x3, a);

	//write to deltaX[N, IH, IW, IC]
	*(float4*)(deltaX + X00          + 8) = float4{ x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC     + 8) = float4{ x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2 + 8) = float4{ x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3 + 8) = float4{ x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4 + 8) = float4{ x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5 + 8) = float4{ x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6 + 8) = float4{ x0[6], x1[6], x2[6], x3[6] };
	*(float4*)(deltaX + X00 + IC * 7 + 8) = float4{ x0[7], x1[7], x2[7], x3[7] };
}

#endif


//LB = 4, IC % 8 == 0, IW % 10 == 0, pw >= 4, template<OC>
#ifndef DECONV_3D_DX_WINOGRAD_F8X9_64X256_P4_OCT_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F8X9_64X256_P4_OCT_TEXTURE

//for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 162, Time =  9.992 msec, Performace = 34817.1 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 162, Time = 11.44  msec, Performace = 30410.2 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 162, Time = 11.032 msec, Performace = 31534.8 GFlop/s

template<int FH, int OC>
__global__ void winograd_f8x9_kernel_64x256_p4_OCT_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int oph, int opw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float smem[8 * 16 * 96];
	float(*__restrict__ Gs)[16][64] = (float(*)[16][64])(smem          );//[ik: STEP][elem: 16][g(ic): 64]
	float(*__restrict__ Ds)[16][32] = (float(*)[16][32])(smem + 8*16*64);//[ik: STEP][elem: 16][d( j): 32]

	///compute 8*16 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	float4 v16 = F32_4_0, v17 = F32_4_0, v18 = F32_4_0, v19 = F32_4_0;
	float4 v20 = F32_4_0, v21 = F32_4_0, v22 = F32_4_0, v23 = F32_4_0;
	float4 v24 = F32_4_0, v25 = F32_4_0, v26 = F32_4_0, v27 = F32_4_0;
	float4 v28 = F32_4_0, v29 = F32_4_0, v30 = F32_4_0, v31 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (bx << 6);
	const int Gk = (ty & 7), Gi = ((tx << 1) + (ty > 7) << 1);//[8, 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 9 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, IC]
	const int bj0 = (by << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 3);//OW % 8
	const int IW8 = (IW >> 3) << 3, IH_IW8 = IH * IW8;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW8, IW8);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Xk;//deltaY[tn0, toh0, tow0, Dk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 4;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);//0, + 8
	const int xj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW8, IW8);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;

	//======[compute area1: local]======================================================
	const int Di = (Xi + (Xk << 2)) & 31;//avoid bank conflict (1/8)
	float w0[9], w1[9], g0[16], g1[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = -(fh * 9) * IC;//rotate180
		const int W1 = W0 + IC    , W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
		float2 wv8 = *(float2*)(W + W0); w0[8] = wv8.x; w1[8] = wv8.y;
		float2 wv7 = *(float2*)(W + W1); w0[7] = wv7.x; w1[7] = wv7.y;
		float2 wv6 = *(float2*)(W + W2); w0[6] = wv6.x; w1[6] = wv6.y;
		float2 wv5 = *(float2*)(W + W3); w0[5] = wv5.x; w1[5] = wv5.y;
		float2 wv4 = *(float2*)(W + W4); w0[4] = wv4.x; w1[4] = wv4.y;
		float2 wv3 = *(float2*)(W + W5); w0[3] = wv3.x; w1[3] = wv3.y;
		float2 wv2 = *(float2*)(W + W6); w0[2] = wv2.x; w1[2] = wv2.y;
		float2 wv1 = *(float2*)(W + W7); w0[1] = wv1.x; w1[1] = wv1.y;
		float2 wv0 = *(float2*)(W + W8); w0[0] = wv0.x; w1[0] = wv0.y;

		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=   0) && (tow0      < OW);//pw = -4
		bool ly1 = lh0 && (tow0 >=  -1) && (tow0 +  1 < OW);//pw = -3
		bool ly2 = lh0 && (tow0 >=  -2) && (tow0 +  2 < OW);//pw = -2
		bool ly3 = lh0 && (tow0 >=  -3) && (tow0 +  3 < OW);//pw = -1
		bool lyC = lh0 && (tow0 >= -12) && (tow0 + 12 < OW);//pw = +1
		bool lyD = lh0 && (tow0 >= -13) && (tow0 + 13 < OW);//pw = +2
		bool lyE = lh0 && (tow0 >= -14) && (tow0 + 14 < OW);//pw = +3
		bool lyF = lh0 && (tow0 >= -15) && (tow0 + 15 < OW);//pw = +4
		int tY0 = Y0 + fh * OW * OC, tY1 = tY0 + OC;
		int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC *  3;
		int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC *  5;
		int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC *  7;
		int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC *  9;
		int tYA = tY0 + OC * 10  , tYB = tY0 + OC * 11;
		int tYC = tY0 + OC * 12  , tYD = tY0 + OC * 13;
		int tYE = tY0 + OC * 14  , tYF = tY0 + OC * 15;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
		tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
		tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
		tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
		tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
		tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1); 
		tYC = IF_int(lyC, tYC, -1); tYD = IF_int(lyD, tYD, -1); 
		tYE = IF_int(lyE, tYE, -1); tYF = IF_int(lyF, tYF, -1);
		x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
		x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
		x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
		x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
		x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
		x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
		x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
		x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);

		for (int ooc = 8; ooc < OC; ooc += 8) {
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g0, w0);
			winograd_f8x9_G(g1, w1);
			winograd_f8x9_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[Gk][ 0][Gi]) = { g0[ 0], g1[ 0] };
			*(float2*)(&Gs[Gk][ 1][Gi]) = { g0[ 1], g1[ 1] };
			*(float2*)(&Gs[Gk][ 2][Gi]) = { g0[ 2], g1[ 2] };
			*(float2*)(&Gs[Gk][ 3][Gi]) = { g0[ 3], g1[ 3] };
			*(float2*)(&Gs[Gk][ 4][Gi]) = { g0[ 4], g1[ 4] };
			*(float2*)(&Gs[Gk][ 5][Gi]) = { g0[ 5], g1[ 5] };
			*(float2*)(&Gs[Gk][ 6][Gi]) = { g0[ 6], g1[ 6] };
			*(float2*)(&Gs[Gk][ 7][Gi]) = { g0[ 7], g1[ 7] };
			*(float2*)(&Gs[Gk][ 8][Gi]) = { g0[ 8], g1[ 8] };
			*(float2*)(&Gs[Gk][ 9][Gi]) = { g0[ 9], g1[ 9] };
			*(float2*)(&Gs[Gk][10][Gi]) = { g0[10], g1[10] };
			*(float2*)(&Gs[Gk][11][Gi]) = { g0[11], g1[11] };
			*(float2*)(&Gs[Gk][12][Gi]) = { g0[12], g1[12] };
			*(float2*)(&Gs[Gk][13][Gi]) = { g0[13], g1[13] };
			*(float2*)(&Gs[Gk][14][Gi]) = { g0[14], g1[14] };
			*(float2*)(&Gs[Gk][15][Gi]) = { g0[15], g1[15] };

			Ds[Xk][ 0][Di] = d[ 0]; Ds[Xk][ 1][Di] = d[ 1];
			Ds[Xk][ 2][Di] = d[ 2]; Ds[Xk][ 3][Di] = d[ 3];
			Ds[Xk][ 4][Di] = d[ 4]; Ds[Xk][ 5][Di] = d[ 5];
			Ds[Xk][ 6][Di] = d[ 6]; Ds[Xk][ 7][Di] = d[ 7];
			Ds[Xk][ 8][Di] = d[ 8]; Ds[Xk][ 9][Di] = d[ 9];
			Ds[Xk][10][Di] = d[10]; Ds[Xk][11][Di] = d[11];
			Ds[Xk][12][Di] = d[12]; Ds[Xk][13][Di] = d[13];
			Ds[Xk][14][Di] = d[14]; Ds[Xk][15][Di] = d[15];
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 b0 = *(float4*)(&Ds[ik][ux][(DIdx + (ik << 2)    ) & 31]);
				float4 b1 = *(float4*)(&Ds[ik][ux][(DIdx + (ik << 2) + 4) & 31]);

				//=======================================================
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

				//ic[0 - 3]             ic[4 - 7]
				simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
				simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
				simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
				simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
				simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7

				//=======================================================
				float4 a2 = *(float4*)(&Gs[ik][ux][GIdx +  8]);
				float4 a3 = *(float4*)(&Gs[ik][ux][GIdx + 12]);

				//ic[8 - 11]            ic[12 - 15]
				simdMM4(v16, b0.x, a2); simdMM4(v17, b0.x, a3);//j0
				simdMM4(v18, b0.y, a2); simdMM4(v19, b0.y, a3);//j1
				simdMM4(v20, b0.z, a2); simdMM4(v21, b0.z, a3);//j2
				simdMM4(v22, b0.w, a2); simdMM4(v23, b0.w, a3);//j3
				simdMM4(v24, b1.x, a2); simdMM4(v25, b1.x, a3);//j4
				simdMM4(v26, b1.y, a2); simdMM4(v27, b1.y, a3);//j5
				simdMM4(v28, b1.z, a2); simdMM4(v29, b1.z, a3);//j6
				simdMM4(v30, b1.w, a2); simdMM4(v31, b1.w, a3);//j7
			}

			//load 2 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 9 * IC;//rotate180
			const int W1 = W0 + IC    , W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
			float2 wv8 = *(float2*)(W + W0); w0[8] = wv8.x; w1[8] = wv8.y;
			float2 wv7 = *(float2*)(W + W1); w0[7] = wv7.x; w1[7] = wv7.y;
			float2 wv6 = *(float2*)(W + W2); w0[6] = wv6.x; w1[6] = wv6.y;
			float2 wv5 = *(float2*)(W + W3); w0[5] = wv5.x; w1[5] = wv5.y;
			float2 wv4 = *(float2*)(W + W4); w0[4] = wv4.x; w1[4] = wv4.y;
			float2 wv3 = *(float2*)(W + W5); w0[3] = wv3.x; w1[3] = wv3.y;
			float2 wv2 = *(float2*)(W + W6); w0[2] = wv2.x; w1[2] = wv2.y;
			float2 wv1 = *(float2*)(W + W7); w0[1] = wv1.x; w1[1] = wv1.y;
			float2 wv0 = *(float2*)(W + W8); w0[0] = wv0.x; w1[0] = wv0.y;

			//load 1 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc, tY1 = tY0 + OC;
			int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC *  3;
			int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC *  5;
			int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC *  7;
			int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC *  9;
			int tYA = tY0 + OC * 10  , tYB = tY0 + OC * 11;
			int tYC = tY0 + OC * 12  , tYD = tY0 + OC * 13;
			int tYE = tY0 + OC * 14  , tYF = tY0 + OC * 15;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
			tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
			tY4 = IF_int(lh0, tY4, -1); tY5 = IF_int(lh0, tY5, -1);
			tY6 = IF_int(lh0, tY6, -1); tY7 = IF_int(lh0, tY7, -1);
			tY8 = IF_int(lh0, tY8, -1); tY9 = IF_int(lh0, tY9, -1);
			tYA = IF_int(lh0, tYA, -1); tYB = IF_int(lh0, tYB, -1);
			tYC = IF_int(lyC, tYC, -1); tYD = IF_int(lyD, tYD, -1);
			tYE = IF_int(lyE, tYE, -1); tYF = IF_int(lyF, tYF, -1);
			x[ 0] = tex1Dfetch<float>(deltaY, tY0); x[ 1] = tex1Dfetch<float>(deltaY, tY1);
			x[ 2] = tex1Dfetch<float>(deltaY, tY2); x[ 3] = tex1Dfetch<float>(deltaY, tY3);
			x[ 4] = tex1Dfetch<float>(deltaY, tY4); x[ 5] = tex1Dfetch<float>(deltaY, tY5);
			x[ 6] = tex1Dfetch<float>(deltaY, tY6); x[ 7] = tex1Dfetch<float>(deltaY, tY7);
			x[ 8] = tex1Dfetch<float>(deltaY, tY8); x[ 9] = tex1Dfetch<float>(deltaY, tY9);
			x[10] = tex1Dfetch<float>(deltaY, tYA); x[11] = tex1Dfetch<float>(deltaY, tYB);
			x[12] = tex1Dfetch<float>(deltaY, tYC); x[13] = tex1Dfetch<float>(deltaY, tYD);
			x[14] = tex1Dfetch<float>(deltaY, tYE); x[15] = tex1Dfetch<float>(deltaY, tYF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), Y(16) -> D(16)
			winograd_f8x9_G(g0, w0);
			winograd_f8x9_G(g1, w1);
			winograd_f8x9_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[Gk][ 0][Gi]) = { g0[ 0], g1[ 0] };
			*(float2*)(&Gs[Gk][ 1][Gi]) = { g0[ 1], g1[ 1] };
			*(float2*)(&Gs[Gk][ 2][Gi]) = { g0[ 2], g1[ 2] };
			*(float2*)(&Gs[Gk][ 3][Gi]) = { g0[ 3], g1[ 3] };
			*(float2*)(&Gs[Gk][ 4][Gi]) = { g0[ 4], g1[ 4] };
			*(float2*)(&Gs[Gk][ 5][Gi]) = { g0[ 5], g1[ 5] };
			*(float2*)(&Gs[Gk][ 6][Gi]) = { g0[ 6], g1[ 6] };
			*(float2*)(&Gs[Gk][ 7][Gi]) = { g0[ 7], g1[ 7] };
			*(float2*)(&Gs[Gk][ 8][Gi]) = { g0[ 8], g1[ 8] };
			*(float2*)(&Gs[Gk][ 9][Gi]) = { g0[ 9], g1[ 9] };
			*(float2*)(&Gs[Gk][10][Gi]) = { g0[10], g1[10] };
			*(float2*)(&Gs[Gk][11][Gi]) = { g0[11], g1[11] };
			*(float2*)(&Gs[Gk][12][Gi]) = { g0[12], g1[12] };
			*(float2*)(&Gs[Gk][13][Gi]) = { g0[13], g1[13] };
			*(float2*)(&Gs[Gk][14][Gi]) = { g0[14], g1[14] };
			*(float2*)(&Gs[Gk][15][Gi]) = { g0[15], g1[15] };

			Ds[Xk][ 0][Di] = d[ 0]; Ds[Xk][ 1][Di] = d[ 1];
			Ds[Xk][ 2][Di] = d[ 2]; Ds[Xk][ 3][Di] = d[ 3];
			Ds[Xk][ 4][Di] = d[ 4]; Ds[Xk][ 5][Di] = d[ 5];
			Ds[Xk][ 6][Di] = d[ 6]; Ds[Xk][ 7][Di] = d[ 7];
			Ds[Xk][ 8][Di] = d[ 8]; Ds[Xk][ 9][Di] = d[ 9];
			Ds[Xk][10][Di] = d[10]; Ds[Xk][11][Di] = d[11];
			Ds[Xk][12][Di] = d[12]; Ds[Xk][13][Di] = d[13];
			Ds[Xk][14][Di] = d[14]; Ds[Xk][15][Di] = d[15];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 b0 = *(float4*)(&Ds[ik][ux][(DIdx + (ik << 2)    ) & 31]);
			float4 b1 = *(float4*)(&Ds[ik][ux][(DIdx + (ik << 2) + 4) & 31]);

			//=======================================================
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

			//ic[0 - 3]             ic[4 - 7]
			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7

			//=======================================================
			float4 a2 = *(float4*)(&Gs[ik][ux][GIdx +  8]);
			float4 a3 = *(float4*)(&Gs[ik][ux][GIdx + 12]);

			//ic[8 - 11]            ic[12 - 15]
			simdMM4(v16, b0.x, a2); simdMM4(v17, b0.x, a3);//j0
			simdMM4(v18, b0.y, a2); simdMM4(v19, b0.y, a3);//j1
			simdMM4(v20, b0.z, a2); simdMM4(v21, b0.z, a3);//j2
			simdMM4(v22, b0.w, a2); simdMM4(v23, b0.w, a3);//j3
			simdMM4(v24, b1.x, a2); simdMM4(v25, b1.x, a3);//j4
			simdMM4(v26, b1.y, a2); simdMM4(v27, b1.y, a3);//j5
			simdMM4(v28, b1.z, a2); simdMM4(v29, b1.z, a3);//j6
			simdMM4(v30, b1.w, a2); simdMM4(v31, b1.w, a3);//j7
		}
		__syncthreads();
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Xs)[33][20] = (float(*)[33][20])(smem);
	float a[16], x0[8], x1[8], x2[8], x3[8];//ic0, ic1, ic2, ic3
	__syncthreads();

	//======[ic0 - ic7]==================================================================
	//group0-----------------------------------------------------------------------------
	*(float4*)(&Xs[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Xs[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Xs[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Xs[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };
	
	*(float4*)(&Xs[ux][uy + 16][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Xs[ux][uy + 16][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Xs[ux][uy + 16][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Xs[ux][uy + 16][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//read-turn0: x, [ic0, ic4]
	a[ 0] = Xs[ 0][uy][ux]; a[ 1] = Xs[ 1][uy][ux]; a[ 2] = Xs[ 2][uy][ux]; a[ 3] = Xs[ 3][uy][ux];
	a[ 4] = Xs[ 4][uy][ux]; a[ 5] = Xs[ 5][uy][ux]; a[ 6] = Xs[ 6][uy][ux]; a[ 7] = Xs[ 7][uy][ux];
	a[ 8] = Xs[ 8][uy][ux]; a[ 9] = Xs[ 9][uy][ux]; a[10] = Xs[10][uy][ux]; a[11] = Xs[11][uy][ux];
	a[12] = Xs[12][uy][ux]; a[13] = Xs[13][uy][ux]; a[14] = Xs[14][uy][ux]; a[15] = Xs[15][uy][ux];
	winograd_f8x9_Y(x0, a);
	
	//read-turn1: y, [ic1, ic5]
	a[ 0] = Xs[ 0][uy + 16][ux]; a[ 1] = Xs[ 1][uy + 16][ux]; a[ 2] = Xs[ 2][uy + 16][ux]; a[ 3] = Xs[ 3][uy + 16][ux];
	a[ 4] = Xs[ 4][uy + 16][ux]; a[ 5] = Xs[ 5][uy + 16][ux]; a[ 6] = Xs[ 6][uy + 16][ux]; a[ 7] = Xs[ 7][uy + 16][ux];
	a[ 8] = Xs[ 8][uy + 16][ux]; a[ 9] = Xs[ 9][uy + 16][ux]; a[10] = Xs[10][uy + 16][ux]; a[11] = Xs[11][uy + 16][ux];
	a[12] = Xs[12][uy + 16][ux]; a[13] = Xs[13][uy + 16][ux]; a[14] = Xs[14][uy + 16][ux]; a[15] = Xs[15][uy + 16][ux];
	winograd_f8x9_Y(x1, a);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	*(float4*)(&Xs[ux][uy][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Xs[ux][uy][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Xs[ux][uy][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Xs[ux][uy][12]) = float4{ v12.z, v13.z, v14.z, v15.z };
	
	*(float4*)(&Xs[ux][uy + 16][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Xs[ux][uy + 16][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Xs[ux][uy + 16][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Xs[ux][uy + 16][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[ 0] = Xs[ 0][uy][ux]; a[ 1] = Xs[ 1][uy][ux]; a[ 2] = Xs[ 2][uy][ux]; a[ 3] = Xs[ 3][uy][ux];
	a[ 4] = Xs[ 4][uy][ux]; a[ 5] = Xs[ 5][uy][ux]; a[ 6] = Xs[ 6][uy][ux]; a[ 7] = Xs[ 7][uy][ux];
	a[ 8] = Xs[ 8][uy][ux]; a[ 9] = Xs[ 9][uy][ux]; a[10] = Xs[10][uy][ux]; a[11] = Xs[11][uy][ux];
	a[12] = Xs[12][uy][ux]; a[13] = Xs[13][uy][ux]; a[14] = Xs[14][uy][ux]; a[15] = Xs[15][uy][ux];
	winograd_f8x9_Y(x2, a);

	a[ 0] = Xs[ 0][uy + 16][ux]; a[ 1] = Xs[ 1][uy + 16][ux]; a[ 2] = Xs[ 2][uy + 16][ux]; a[ 3] = Xs[ 3][uy + 16][ux];
	a[ 4] = Xs[ 4][uy + 16][ux]; a[ 5] = Xs[ 5][uy + 16][ux]; a[ 6] = Xs[ 6][uy + 16][ux]; a[ 7] = Xs[ 7][uy + 16][ux];
	a[ 8] = Xs[ 8][uy + 16][ux]; a[ 9] = Xs[ 9][uy + 16][ux]; a[10] = Xs[10][uy + 16][ux]; a[11] = Xs[11][uy + 16][ux];
	a[12] = Xs[12][uy + 16][ux]; a[13] = Xs[13][uy + 16][ux]; a[14] = Xs[14][uy + 16][ux]; a[15] = Xs[15][uy + 16][ux];
	winograd_f8x9_Y(x3, a);
	__syncthreads();

	//write to deltaX[N, IH, IW, IC]
	*(float4*)(deltaX + X00         ) = float4{ x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC    ) = float4{ x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2) = float4{ x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3) = float4{ x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4) = float4{ x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5) = float4{ x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6) = float4{ x0[6], x1[6], x2[6], x3[6] };
	*(float4*)(deltaX + X00 + IC * 7) = float4{ x0[7], x1[7], x2[7], x3[7] };

	//======[ic8 - ic15]=================================================================
	//group2-----------------------------------------------------------------------------
	*(float4*)(&Xs[ux][uy][ 0]) = float4{ v16.x, v17.x, v18.x, v19.x };
	*(float4*)(&Xs[ux][uy][ 4]) = float4{ v20.x, v21.x, v22.x, v23.x };
	*(float4*)(&Xs[ux][uy][ 8]) = float4{ v24.x, v25.x, v26.x, v27.x };
	*(float4*)(&Xs[ux][uy][12]) = float4{ v28.x, v29.x, v30.x, v31.x };
	
	*(float4*)(&Xs[ux][uy + 16][ 0]) = float4{ v16.y, v17.y, v18.y, v19.y };
	*(float4*)(&Xs[ux][uy + 16][ 4]) = float4{ v20.y, v21.y, v22.y, v23.y };
	*(float4*)(&Xs[ux][uy + 16][ 8]) = float4{ v24.y, v25.y, v26.y, v27.y };
	*(float4*)(&Xs[ux][uy + 16][12]) = float4{ v28.y, v29.y, v30.y, v31.y };
	__syncthreads();

	a[ 0] = Xs[ 0][uy][ux]; a[ 1] = Xs[ 1][uy][ux]; a[ 2] = Xs[ 2][uy][ux]; a[ 3] = Xs[ 3][uy][ux];
	a[ 4] = Xs[ 4][uy][ux]; a[ 5] = Xs[ 5][uy][ux]; a[ 6] = Xs[ 6][uy][ux]; a[ 7] = Xs[ 7][uy][ux];
	a[ 8] = Xs[ 8][uy][ux]; a[ 9] = Xs[ 9][uy][ux]; a[10] = Xs[10][uy][ux]; a[11] = Xs[11][uy][ux];
	a[12] = Xs[12][uy][ux]; a[13] = Xs[13][uy][ux]; a[14] = Xs[14][uy][ux]; a[15] = Xs[15][uy][ux];
	winograd_f8x9_Y(x0, a);
	
	a[ 0] = Xs[ 0][uy + 16][ux]; a[ 1] = Xs[ 1][uy + 16][ux]; a[ 2] = Xs[ 2][uy + 16][ux]; a[ 3] = Xs[ 3][uy + 16][ux];
	a[ 4] = Xs[ 4][uy + 16][ux]; a[ 5] = Xs[ 5][uy + 16][ux]; a[ 6] = Xs[ 6][uy + 16][ux]; a[ 7] = Xs[ 7][uy + 16][ux];
	a[ 8] = Xs[ 8][uy + 16][ux]; a[ 9] = Xs[ 9][uy + 16][ux]; a[10] = Xs[10][uy + 16][ux]; a[11] = Xs[11][uy + 16][ux];
	a[12] = Xs[12][uy + 16][ux]; a[13] = Xs[13][uy + 16][ux]; a[14] = Xs[14][uy + 16][ux]; a[15] = Xs[15][uy + 16][ux];
	winograd_f8x9_Y(x1, a);
	__syncthreads();

	//group3-----------------------------------------------------------------------------
	*(float4*)(&Xs[ux][uy][ 0]) = float4{ v16.z, v17.z, v18.z, v19.z };
	*(float4*)(&Xs[ux][uy][ 4]) = float4{ v20.z, v21.z, v22.z, v23.z };
	*(float4*)(&Xs[ux][uy][ 8]) = float4{ v24.z, v25.z, v26.z, v27.z };
	*(float4*)(&Xs[ux][uy][12]) = float4{ v28.z, v29.z, v30.z, v31.z };
	
	*(float4*)(&Xs[ux][uy + 16][ 0]) = float4{ v16.w, v17.w, v18.w, v19.w };
	*(float4*)(&Xs[ux][uy + 16][ 4]) = float4{ v20.w, v21.w, v22.w, v23.w };
	*(float4*)(&Xs[ux][uy + 16][ 8]) = float4{ v24.w, v25.w, v26.w, v27.w };
	*(float4*)(&Xs[ux][uy + 16][12]) = float4{ v28.w, v29.w, v30.w, v31.w };
	__syncthreads();

	a[ 0] = Xs[ 0][uy][ux]; a[ 1] = Xs[ 1][uy][ux]; a[ 2] = Xs[ 2][uy][ux]; a[ 3] = Xs[ 3][uy][ux];
	a[ 4] = Xs[ 4][uy][ux]; a[ 5] = Xs[ 5][uy][ux]; a[ 6] = Xs[ 6][uy][ux]; a[ 7] = Xs[ 7][uy][ux];
	a[ 8] = Xs[ 8][uy][ux]; a[ 9] = Xs[ 9][uy][ux]; a[10] = Xs[10][uy][ux]; a[11] = Xs[11][uy][ux];
	a[12] = Xs[12][uy][ux]; a[13] = Xs[13][uy][ux]; a[14] = Xs[14][uy][ux]; a[15] = Xs[15][uy][ux];
	winograd_f8x9_Y(x2, a);

	a[ 0] = Xs[ 0][uy + 16][ux]; a[ 1] = Xs[ 1][uy + 16][ux]; a[ 2] = Xs[ 2][uy + 16][ux]; a[ 3] = Xs[ 3][uy + 16][ux];
	a[ 4] = Xs[ 4][uy + 16][ux]; a[ 5] = Xs[ 5][uy + 16][ux]; a[ 6] = Xs[ 6][uy + 16][ux]; a[ 7] = Xs[ 7][uy + 16][ux];
	a[ 8] = Xs[ 8][uy + 16][ux]; a[ 9] = Xs[ 9][uy + 16][ux]; a[10] = Xs[10][uy + 16][ux]; a[11] = Xs[11][uy + 16][ux];
	a[12] = Xs[12][uy + 16][ux]; a[13] = Xs[13][uy + 16][ux]; a[14] = Xs[14][uy + 16][ux]; a[15] = Xs[15][uy + 16][ux];
	winograd_f8x9_Y(x3, a);

	//write to deltaX[N, IH, IW, IC]
	*(float4*)(deltaX + X00          + 8) = float4{ x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC     + 8) = float4{ x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2 + 8) = float4{ x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3 + 8) = float4{ x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4 + 8) = float4{ x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5 + 8) = float4{ x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6 + 8) = float4{ x0[6], x1[6], x2[6], x3[6] };
	*(float4*)(deltaX + X00 + IC * 7 + 8) = float4{ x0[7], x1[7], x2[7], x3[7] };
}

#endif


#endif
