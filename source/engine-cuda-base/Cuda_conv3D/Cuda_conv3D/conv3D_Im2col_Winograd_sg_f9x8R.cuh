#pragma once 

#ifndef CONV_3D_WINOGRAD_S16_F9X8R_H
#define CONV_3D_WINOGRAD_S16_F9X8R_H

//(1) sh = sw = 1
//(2) FW = 9
//(3) OW % 8 : group = 8 elements
//(4) Remode the kernel: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV_3D_WINOGRAD_F9X8R_CALL
#define CONV_3D_WINOGRAD_F9X8R_CALL

//================[Standard: 32(OC) * 288(GM)]===============================
//pw <= 4, OW % 9 == 0
#define conv3dWinograd_f9x8_k32x288R_p4_tex(stream, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw) \
	conv3dWinograd_f9x8_kernel_32x288R_p4_tex<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW/9)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw)

//pw <= 4, OW % 9 == 0, template<IC, OC>
#define conv3dWinograd_f9x8_k32x288R_p4_CT_tex(stream, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw) \
	conv3dWinograd_f9x8_kernel_32x288R_p4_CT_tex<FH, IC, OC>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW/9)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, ph, pw)

//pw <= 4, OW % 18 == 0
#define conv3dWinograd_f9x8_ruse_k32x288R_p4_tex(stream, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw) \
	conv3dWinograd_f9x8_ruse_kernel_32x288R_p4_tex<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * ((OW/18)*2)), dim3(16, 8), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw)

//================[Cooperate: 32(OC) * 288(GM)]==============================
//pw <= 4, OWr % 8 == 0
#define conv3dWinograd_f9x8_k32x288RC_p4_tex(stream, ow_index, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr) \
	conv3dWinograd_f9x8_kernel_32x288RC_p4_tex<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OWr/9)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, ow_index)

//pw <= 4, OWr % 8 == 0,  template<IC, OC>
#define conv3dWinograd_f9x8_k32x288RC_p4_CT_tex(stream, ow_index, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr) \
	conv3dWinograd_f9x8_kernel_32x288RC_p4_CT_tex<FH, IC, OC>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OWr/9)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, ph, pw, ow_index)

//================[Standard: 64(OC) * 288(GM)]===============================
#define conv3dWinograd_f9x8_k64x288R_p4_tex(stream, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw) \
	conv3dWinograd_f9x8_kernel_64x288R_p4_tex<FH>\
		<<< dim3(OC>>6, ((N*OH)>>5) * (OW/9)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw)

#endif


//================[Standard: 32(OC) * 288(GM)]===============================
//LB = 4, IC % 8 == 0, pw <= 4, OW % 9 == 0
#ifndef CONV_3D_WINOGRAD_F9X8_32X288R_P4_TEXTURE
#define CONV_3D_WINOGRAD_F9X8_32X288R_P4_TEXTURE

//for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 81, Time = 7.23126 msec, Performace = 24054.8 GFlop/s
//WB = 4: Size = 81, Time = 6.82823 msec, Performace = 25474.6 GFlop/s
//for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 81, Time = 8.73088 msec, Performace = 19923.1 GFlop/s
//WB = 4: Size = 81, Time = 8.40672 msec, Performace = 20691.3 GFlop/s
//for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 81, Time = 8.98962 msec, Performace = 19349.7 GFlop/s
//WB = 4: Size = 81, Time = 8.55911 msec, Performace = 20322.9 GFlop/s

template<int FH>
__global__ void conv3dWinograd_f9x8_kernel_32x288R_p4_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw)//sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;
	const int bx = blockIdx.x, by = blockIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 32]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = by * 288;//32 * 9 = 288
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi * 9);//OW % 9
	const int OW9 = (OW / 9) * 9, OH_OW9 = OH * OW9;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OW9, OW9);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 9;
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OW9, OW9);
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;

	//======[compute area1: local]======================================================
	float w[8], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = (fh << 3) * IC * OC;
		const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7;
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7];

		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=   0) && (tiw0      < IW);//pw = -4
		bool ly1 = lh0 && (tiw0 >=  -1) && (tiw0 +  1 < IW);//pw = -3
		bool ly2 = lh0 && (tiw0 >=  -2) && (tiw0 +  2 < IW);//pw = -2
		bool ly3 = lh0 && (tiw0 >=  -3) && (tiw0 +  3 < IW);//pw = -1
		bool lyC = lh0 && (tiw0 >= -12) && (tiw0 + 12 < IW);//pw = +1
		bool lyD = lh0 && (tiw0 >= -13) && (tiw0 + 13 < IW);//pw = +2
		bool lyE = lh0 && (tiw0 >= -14) && (tiw0 + 14 < IW);//pw = +3
		bool lyF = lh0 && (tiw0 >= -15) && (tiw0 + 15 < IW);//pw = +4
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
		int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
		int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
		int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
		int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
		int tXF = tX0 + IC * 15;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
		tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
		tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
		tXA = IF_int(lh0, tXA, -1); tXB = IF_int(lh0, tXB, -1);
		tXC = IF_int(lyC, tXC, -1); tXD = IF_int(lyD, tXD, -1);
		tXE = IF_int(lyE, tXE, -1); tXF = IF_int(lyF, tXF, -1);
		x[0x0] = tex1Dfetch<float>(X, tX0); x[0x1] = tex1Dfetch<float>(X, tX1);
		x[0x2] = tex1Dfetch<float>(X, tX2); x[0x3] = tex1Dfetch<float>(X, tX3);
		x[0x4] = tex1Dfetch<float>(X, tX4); x[0x5] = tex1Dfetch<float>(X, tX5);
		x[0x6] = tex1Dfetch<float>(X, tX6); x[0x7] = tex1Dfetch<float>(X, tX7);
		x[0x8] = tex1Dfetch<float>(X, tX8); x[0x9] = tex1Dfetch<float>(X, tX9);
		x[0xA] = tex1Dfetch<float>(X, tXA); x[0xB] = tex1Dfetch<float>(X, tXB);
		x[0xC] = tex1Dfetch<float>(X, tXC); x[0xD] = tex1Dfetch<float>(X, tXD);
		x[0xE] = tex1Dfetch<float>(X, tXE); x[0xF] = tex1Dfetch<float>(X, tXF);

		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g, w);
			winograd_f9x8_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Xk][ 0][Xi] = d[ 0]; Ds[Xk][ 1][Xi] = d[ 1];
			Ds[Xk][ 2][Xi] = d[ 2]; Ds[Xk][ 3][Xi] = d[ 3];
			Ds[Xk][ 4][Xi] = d[ 4]; Ds[Xk][ 5][Xi] = d[ 5];
			Ds[Xk][ 6][Xi] = d[ 6]; Ds[Xk][ 7][Xi] = d[ 7];
			Ds[Xk][ 8][Xi] = d[ 8]; Ds[Xk][ 9][Xi] = d[ 9];
			Ds[Xk][10][Xi] = d[10]; Ds[Xk][11][Xi] = d[11];
			Ds[Xk][12][Xi] = d[12]; Ds[Xk][13][Xi] = d[13];
			Ds[Xk][14][Xi] = d[14]; Ds[Xk][15][Xi] = d[15];
			__syncthreads();
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

				//oc[0 - 3]             oc[4 - 7]
				simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
				simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
				simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
				simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
				simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}

			//load 1 group from CW[FH, FW, IC, OC]
			const int W0 = ((fh << 3) * IC + oic) * OC;
			const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7;
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7];

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
			int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
			int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
			int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
			int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
			int tXF = tX0 + IC * 15;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
			tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
			tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
			tXA = IF_int(lh0, tXA, -1); tXB = IF_int(lh0, tXB, -1);
			tXC = IF_int(lyC, tXC, -1); tXD = IF_int(lyD, tXD, -1);
			tXE = IF_int(lyE, tXE, -1); tXF = IF_int(lyF, tXF, -1);
			x[0x0] = tex1Dfetch<float>(X, tX0); x[0x1] = tex1Dfetch<float>(X, tX1);
			x[0x2] = tex1Dfetch<float>(X, tX2); x[0x3] = tex1Dfetch<float>(X, tX3);
			x[0x4] = tex1Dfetch<float>(X, tX4); x[0x5] = tex1Dfetch<float>(X, tX5);
			x[0x6] = tex1Dfetch<float>(X, tX6); x[0x7] = tex1Dfetch<float>(X, tX7);
			x[0x8] = tex1Dfetch<float>(X, tX8); x[0x9] = tex1Dfetch<float>(X, tX9);
			x[0xA] = tex1Dfetch<float>(X, tXA); x[0xB] = tex1Dfetch<float>(X, tXB);
			x[0xC] = tex1Dfetch<float>(X, tXC); x[0xD] = tex1Dfetch<float>(X, tXD);
			x[0xE] = tex1Dfetch<float>(X, tXE); x[0xF] = tex1Dfetch<float>(X, tXF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g, w);
			winograd_f9x8_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Xk][ 0][Xi] = d[ 0]; Ds[Xk][ 1][Xi] = d[ 1];
			Ds[Xk][ 2][Xi] = d[ 2]; Ds[Xk][ 3][Xi] = d[ 3];
			Ds[Xk][ 4][Xi] = d[ 4]; Ds[Xk][ 5][Xi] = d[ 5];
			Ds[Xk][ 6][Xi] = d[ 6]; Ds[Xk][ 7][Xi] = d[ 7];
			Ds[Xk][ 8][Xi] = d[ 8]; Ds[Xk][ 9][Xi] = d[ 9];
			Ds[Xk][10][Xi] = d[10]; Ds[Xk][11][Xi] = d[11];
			Ds[Xk][12][Xi] = d[12]; Ds[Xk][13][Xi] = d[13];
			Ds[Xk][14][Xi] = d[14]; Ds[Xk][15][Xi] = d[15];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

			//oc[0 - 3]             oc[4 - 7]
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
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[9], y1[9], y2[9], y3[9];//oc0, oc1, oc2, oc3
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(&Ys0[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Ys0[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Ys0[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Ys0[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//write-turn1: y, [oc0, oc4]
	*(float4*)(&Ys1[ux][uy][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Ys1[ux][uy][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Ys1[ux][uy][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Ys1[ux][uy][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[ 0][uy][ux]; a[ 1] = Ys0[ 1][uy][ux]; a[ 2] = Ys0[ 2][uy][ux]; a[ 3] = Ys0[ 3][uy][ux];
	a[ 4] = Ys0[ 4][uy][ux]; a[ 5] = Ys0[ 5][uy][ux]; a[ 6] = Ys0[ 6][uy][ux]; a[ 7] = Ys0[ 7][uy][ux];
	a[ 8] = Ys0[ 8][uy][ux]; a[ 9] = Ys0[ 9][uy][ux]; a[10] = Ys0[10][uy][ux]; a[11] = Ys0[11][uy][ux];
	a[12] = Ys0[12][uy][ux]; a[13] = Ys0[13][uy][ux]; a[14] = Ys0[14][uy][ux]; a[15] = Ys0[15][uy][ux];
	winograd_f9x8_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f9x8_Y(y1, a);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [oc2, oc6]
	*(float4*)(&Ys0[ux][uy][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Ys0[ux][uy][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Ys0[ux][uy][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Ys0[ux][uy][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn1: w, [oc3, oc7]
	*(float4*)(&Ys1[ux][uy][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Ys1[ux][uy][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Ys1[ux][uy][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Ys1[ux][uy][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc2, oc6]
	a[ 0] = Ys0[ 0][uy][ux]; a[ 1] = Ys0[ 1][uy][ux]; a[ 2] = Ys0[ 2][uy][ux]; a[ 3] = Ys0[ 3][uy][ux];
	a[ 4] = Ys0[ 4][uy][ux]; a[ 5] = Ys0[ 5][uy][ux]; a[ 6] = Ys0[ 6][uy][ux]; a[ 7] = Ys0[ 7][uy][ux];
	a[ 8] = Ys0[ 8][uy][ux]; a[ 9] = Ys0[ 9][uy][ux]; a[10] = Ys0[10][uy][ux]; a[11] = Ys0[11][uy][ux];
	a[12] = Ys0[12][uy][ux]; a[13] = Ys0[13][uy][ux]; a[14] = Ys0[14][uy][ux]; a[15] = Ys0[15][uy][ux];
	winograd_f9x8_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f9x8_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
	*(float4*)(Y + Y00 + OC * 8) = float4{ y0[8], y1[8], y2[8], y3[8] };
}

#endif


//LB = 4, IC % 8 == 0, pw <= 4, OW % 9 == 0, template<IC, OC>
#ifndef CONV_3D_WINOGRAD_F9X8_32X288R_P4_CT_TEXTURE
#define CONV_3D_WINOGRAD_F9X8_32X288R_P4_CT_TEXTURE

//for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 81, Time = 6.50454 msec, Performace = 26742.3 GFlop/s
//WB = 4: Size = 81, Time = 6.1473  msec, Performace = 28296.4 GFlop/s
//for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 81, Time = 7.92571 msec, Performace = 21947.1 GFlop/s
//WB = 4: Size = 81, Time = 7.5688  msec, Performace = 22982   GFlop/s
//for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 81, Time = 7.86768 msec, Performace = 22108.9 GFlop/s
//WB = 4: Size = 81, Time = 7.4951  msec, Performace = 23208   GFlop/s

template<int FH, int IC, int OC>
__global__ void conv3dWinograd_f9x8_kernel_32x288R_p4_CT_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int ph, int pw)//sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;
	const int bx = blockIdx.x, by = blockIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 32]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = by * 288;//32 * 9 = 288
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi * 9);//OW % 9
	const int OW9 = (OW / 9) * 9, OH_OW9 = OH * OW9;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OW9, OW9);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 9;
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OW9, OW9);
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;

	//======[compute area1: local]======================================================
	float w[8], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = (fh << 3) * IC * OC;
		const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7;
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7];

		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=   0) && (tiw0      < IW);//pw = -4
		bool ly1 = lh0 && (tiw0 >=  -1) && (tiw0 +  1 < IW);//pw = -3
		bool ly2 = lh0 && (tiw0 >=  -2) && (tiw0 +  2 < IW);//pw = -2
		bool ly3 = lh0 && (tiw0 >=  -3) && (tiw0 +  3 < IW);//pw = -1
		bool lyC = lh0 && (tiw0 >= -12) && (tiw0 + 12 < IW);//pw = +1
		bool lyD = lh0 && (tiw0 >= -13) && (tiw0 + 13 < IW);//pw = +2
		bool lyE = lh0 && (tiw0 >= -14) && (tiw0 + 14 < IW);//pw = +3
		bool lyF = lh0 && (tiw0 >= -15) && (tiw0 + 15 < IW);//pw = +4
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
		int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
		int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
		int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
		int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
		int tXF = tX0 + IC * 15;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
		tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
		tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
		tXA = IF_int(lh0, tXA, -1); tXB = IF_int(lh0, tXB, -1);
		tXC = IF_int(lyC, tXC, -1); tXD = IF_int(lyD, tXD, -1);
		tXE = IF_int(lyE, tXE, -1); tXF = IF_int(lyF, tXF, -1);
		x[0x0] = tex1Dfetch<float>(X, tX0); x[0x1] = tex1Dfetch<float>(X, tX1);
		x[0x2] = tex1Dfetch<float>(X, tX2); x[0x3] = tex1Dfetch<float>(X, tX3);
		x[0x4] = tex1Dfetch<float>(X, tX4); x[0x5] = tex1Dfetch<float>(X, tX5);
		x[0x6] = tex1Dfetch<float>(X, tX6); x[0x7] = tex1Dfetch<float>(X, tX7);
		x[0x8] = tex1Dfetch<float>(X, tX8); x[0x9] = tex1Dfetch<float>(X, tX9);
		x[0xA] = tex1Dfetch<float>(X, tXA); x[0xB] = tex1Dfetch<float>(X, tXB);
		x[0xC] = tex1Dfetch<float>(X, tXC); x[0xD] = tex1Dfetch<float>(X, tXD);
		x[0xE] = tex1Dfetch<float>(X, tXE); x[0xF] = tex1Dfetch<float>(X, tXF);

		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g, w);
			winograd_f9x8_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Xk][ 0][Xi] = d[ 0]; Ds[Xk][ 1][Xi] = d[ 1];
			Ds[Xk][ 2][Xi] = d[ 2]; Ds[Xk][ 3][Xi] = d[ 3];
			Ds[Xk][ 4][Xi] = d[ 4]; Ds[Xk][ 5][Xi] = d[ 5];
			Ds[Xk][ 6][Xi] = d[ 6]; Ds[Xk][ 7][Xi] = d[ 7];
			Ds[Xk][ 8][Xi] = d[ 8]; Ds[Xk][ 9][Xi] = d[ 9];
			Ds[Xk][10][Xi] = d[10]; Ds[Xk][11][Xi] = d[11];
			Ds[Xk][12][Xi] = d[12]; Ds[Xk][13][Xi] = d[13];
			Ds[Xk][14][Xi] = d[14]; Ds[Xk][15][Xi] = d[15];
			__syncthreads();
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

				//oc[0 - 3]             oc[4 - 7]
				simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
				simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
				simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
				simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
				simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}

			//load 1 group from CW[FH, FW, IC, OC]
			const int W0 = ((fh << 3) * IC + oic) * OC;
			const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7;
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7];

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
			int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
			int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
			int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
			int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
			int tXF = tX0 + IC * 15;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
			tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
			tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
			tXA = IF_int(lh0, tXA, -1); tXB = IF_int(lh0, tXB, -1);
			tXC = IF_int(lyC, tXC, -1); tXD = IF_int(lyD, tXD, -1);
			tXE = IF_int(lyE, tXE, -1); tXF = IF_int(lyF, tXF, -1);
			x[0x0] = tex1Dfetch<float>(X, tX0); x[0x1] = tex1Dfetch<float>(X, tX1);
			x[0x2] = tex1Dfetch<float>(X, tX2); x[0x3] = tex1Dfetch<float>(X, tX3);
			x[0x4] = tex1Dfetch<float>(X, tX4); x[0x5] = tex1Dfetch<float>(X, tX5);
			x[0x6] = tex1Dfetch<float>(X, tX6); x[0x7] = tex1Dfetch<float>(X, tX7);
			x[0x8] = tex1Dfetch<float>(X, tX8); x[0x9] = tex1Dfetch<float>(X, tX9);
			x[0xA] = tex1Dfetch<float>(X, tXA); x[0xB] = tex1Dfetch<float>(X, tXB);
			x[0xC] = tex1Dfetch<float>(X, tXC); x[0xD] = tex1Dfetch<float>(X, tXD);
			x[0xE] = tex1Dfetch<float>(X, tXE); x[0xF] = tex1Dfetch<float>(X, tXF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g, w);
			winograd_f9x8_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Xk][ 0][Xi] = d[ 0]; Ds[Xk][ 1][Xi] = d[ 1];
			Ds[Xk][ 2][Xi] = d[ 2]; Ds[Xk][ 3][Xi] = d[ 3];
			Ds[Xk][ 4][Xi] = d[ 4]; Ds[Xk][ 5][Xi] = d[ 5];
			Ds[Xk][ 6][Xi] = d[ 6]; Ds[Xk][ 7][Xi] = d[ 7];
			Ds[Xk][ 8][Xi] = d[ 8]; Ds[Xk][ 9][Xi] = d[ 9];
			Ds[Xk][10][Xi] = d[10]; Ds[Xk][11][Xi] = d[11];
			Ds[Xk][12][Xi] = d[12]; Ds[Xk][13][Xi] = d[13];
			Ds[Xk][14][Xi] = d[14]; Ds[Xk][15][Xi] = d[15];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

			//oc[0 - 3]             oc[4 - 7]
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
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[9], y1[9], y2[9], y3[9];//oc0, oc1, oc2, oc3
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(&Ys0[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Ys0[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Ys0[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Ys0[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//write-turn1: y, [oc0, oc4]
	*(float4*)(&Ys1[ux][uy][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Ys1[ux][uy][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Ys1[ux][uy][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Ys1[ux][uy][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[ 0][uy][ux]; a[ 1] = Ys0[ 1][uy][ux]; a[ 2] = Ys0[ 2][uy][ux]; a[ 3] = Ys0[ 3][uy][ux];
	a[ 4] = Ys0[ 4][uy][ux]; a[ 5] = Ys0[ 5][uy][ux]; a[ 6] = Ys0[ 6][uy][ux]; a[ 7] = Ys0[ 7][uy][ux];
	a[ 8] = Ys0[ 8][uy][ux]; a[ 9] = Ys0[ 9][uy][ux]; a[10] = Ys0[10][uy][ux]; a[11] = Ys0[11][uy][ux];
	a[12] = Ys0[12][uy][ux]; a[13] = Ys0[13][uy][ux]; a[14] = Ys0[14][uy][ux]; a[15] = Ys0[15][uy][ux];
	winograd_f9x8_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f9x8_Y(y1, a);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [oc2, oc6]
	*(float4*)(&Ys0[ux][uy][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Ys0[ux][uy][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Ys0[ux][uy][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Ys0[ux][uy][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn1: w, [oc3, oc7]
	*(float4*)(&Ys1[ux][uy][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Ys1[ux][uy][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Ys1[ux][uy][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Ys1[ux][uy][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc2, oc6]
	a[ 0] = Ys0[ 0][uy][ux]; a[ 1] = Ys0[ 1][uy][ux]; a[ 2] = Ys0[ 2][uy][ux]; a[ 3] = Ys0[ 3][uy][ux];
	a[ 4] = Ys0[ 4][uy][ux]; a[ 5] = Ys0[ 5][uy][ux]; a[ 6] = Ys0[ 6][uy][ux]; a[ 7] = Ys0[ 7][uy][ux];
	a[ 8] = Ys0[ 8][uy][ux]; a[ 9] = Ys0[ 9][uy][ux]; a[10] = Ys0[10][uy][ux]; a[11] = Ys0[11][uy][ux];
	a[12] = Ys0[12][uy][ux]; a[13] = Ys0[13][uy][ux]; a[14] = Ys0[14][uy][ux]; a[15] = Ys0[15][uy][ux];
	winograd_f9x8_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f9x8_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
	*(float4*)(Y + Y00 + OC * 8) = float4{ y0[8], y1[8], y2[8], y3[8] };
}

#endif


//RUSE: LB = 4, IC % 8 == 0, pw <= 4, OW % 18 == 0
#ifndef CONV_3D_WINOGRAD_F9X8_RUSE_32X288R_P4_TEXTURE
#define CONV_3D_WINOGRAD_F9X8_RUSE_32X288R_P4_TEXTURE

//for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 81, Time = 6.47682 msec, Performace = 26856.7 GFlop/s
//WB = 4: Size = 81, Time = 6.03845 msec, Performace = 28806.4 GFlop/s
//for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 81, Time = 7.30048 msec, Performace = 23826.7 GFlop/s
//WB = 4: Size = 81, Time = 6.87855 msec, Performace = 25288.2 GFlop/s
//for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 81, Time = 6.96256 msec, Performace = 24983.1 GFlop/s
//WB = 4: Size = 81, Time = 6.34777 msec, Performace = 27402.7 GFlop/s

template<int FH>
__global__ void conv3dWinograd_f9x8_ruse_kernel_32x288R_p4_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
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

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 5);
	const int Gk = ty, Gi = tx << 1;//[8, 16*2 = 32]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = by * 288;//32 * 8 = 256
	const int Xk = (tx & 7), Xi = ((ty << 1) + (tx > 7)) << 1;//[8, 16*2 = 32]
	const int tj0 = bj0 + (Xi * 9);//OW % 9
	const int OW18 = (OW / 18) * 18, OH_OW18 = OH * OW18;//9 * 2 = 18
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OW18, OW18);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = (ty << 1) + (tx > 7), uy = (tx & 7);//(2 * 8)
	const int DIdx = ((uy & 1) + ((uy >> 2) << 1)) << 3;//4
	const int GIdx = ((uy & 3) >> 1) << 4;//2

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) * 9);
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OW18, OW18);
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;

	//======[compute area1: local]======================================================
	float w0[8], w1[8], g0[16], g1[16], x[25], d0[16], d1[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 8 * IC * OC;
		const int W1 = W0 + IC * OC,     W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7;
		float2 wv0 = *(float2*)(CW + W0), wv1 = *(float2*)(CW + W1);
		float2 wv2 = *(float2*)(CW + W2), wv3 = *(float2*)(CW + W3);
		float2 wv4 = *(float2*)(CW + W4), wv5 = *(float2*)(CW + W5);
		float2 wv6 = *(float2*)(CW + W6), wv7 = *(float2*)(CW + W7);

		w0[0] = wv0.x; w1[0] = wv0.y; w0[1] = wv1.x; w1[1] = wv1.y;
		w0[2] = wv2.x; w1[2] = wv2.y; w0[3] = wv3.x; w1[3] = wv3.y;
		w0[4] = wv4.x; w1[4] = wv4.y; w0[5] = wv5.x; w1[5] = wv5.y;
		w0[6] = wv6.x; w1[6] = wv6.y; w0[7] = wv7.x; w1[7] = wv7.y;

		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=   0) && (tiw0      < IW);//pw = -4
		bool ly1 = lh0 && (tiw0 >=  -1) && (tiw0 +  1 < IW);//pw = -3
		bool ly2 = lh0 && (tiw0 >=  -2) && (tiw0 +  2 < IW);//pw = -2
		bool ly3 = lh0 && (tiw0 >=  -3) && (tiw0 +  3 < IW);//pw = -1
		bool lyL = lh0 && (tiw0 >= -21) && (tiw0 + 21 < IW);//pw = +1
		bool lyM = lh0 && (tiw0 >= -22) && (tiw0 + 22 < IW);//pw = +2
		bool lyN = lh0 && (tiw0 >= -23) && (tiw0 + 23 < IW);//pw = +3
		bool lyO = lh0 && (tiw0 >= -24) && (tiw0 + 24 < IW);//pw = +4
		int tX0 = X0 + fh * IW * IC, tX1 = tX0 + IC;
		int tX2 = tX0 + (IC << 1), tX3 = tX0 + IC *  3;
		int tX4 = tX0 + (IC << 2), tX5 = tX0 + IC *  5;
		int tX6 = tX0 + IC * 6   , tX7 = tX0 + IC *  7;
		int tX8 = tX0 + (IC << 3), tX9 = tX0 + IC *  9;
		int tXA = tX0 + IC * 10  , tXB = tX0 + IC * 11;
		int tXC = tX0 + IC * 12  , tXD = tX0 + IC * 13;
		int tXE = tX0 + IC * 14  , tXF = tX0 + IC * 15;
		int tXG = tX0 + (IC << 4), tXH = tX0 + IC * 17;
		int tXI = tX0 + IC * 18  , tXJ = tX0 + IC * 19;
		int tXK = tX0 + IC * 20  , tXL = tX0 + IC * 21;
		int tXM = tX0 + IC * 22  , tXN = tX0 + IC * 23;
		int tXO = tX0 + IC * 24;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1); tX6 = IF_int(lh0, tX6, -1); 
		tX7 = IF_int(lh0, tX7, -1); tX8 = IF_int(lh0, tX8, -1); 
		tX9 = IF_int(lh0, tX9, -1); tXA = IF_int(lh0, tXA, -1);
		tXB = IF_int(lh0, tXB, -1); tXC = IF_int(lh0, tXC, -1); tXD = IF_int(lh0, tXD, -1);
		tXE = IF_int(lh0, tXE, -1); tXF = IF_int(lh0, tXF, -1);
		tXG = IF_int(lh0, tXG, -1); tXH = IF_int(lh0, tXH, -1);
		tXI = IF_int(lh0, tXI, -1); tXJ = IF_int(lh0, tXJ, -1); tXK = IF_int(lh0, tXK, -1);
		tXL = IF_int(lyL, tXL, -1); tXM = IF_int(lyM, tXM, -1);
		tXN = IF_int(lyN, tXN, -1); tXO = IF_int(lyO, tXO, -1);
		x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
		x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
		x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
		x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
		x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
		x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
		x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
		x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);
		x[16] = tex1Dfetch<float>(X, tXG); x[17] = tex1Dfetch<float>(X, tXH);
		x[18] = tex1Dfetch<float>(X, tXI); x[19] = tex1Dfetch<float>(X, tXJ);
		x[20] = tex1Dfetch<float>(X, tXK); x[21] = tex1Dfetch<float>(X, tXL);
		x[22] = tex1Dfetch<float>(X, tXM); x[23] = tex1Dfetch<float>(X, tXN);
		x[24] = tex1Dfetch<float>(X, tXO);
		
		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g0, w0);
			winograd_f9x8_G(g1, w1);
			winograd_f9x8_D(d0, x);//x[0 - 15]
			winograd_f9x8_D_oft(d1, x, 9);//x[9 - 24]

			//write to shread memory
			*(float2*)(&Gs[Gk][0x0][Gi]) = { g0[0x0], g1[0x0] }; *(float2*)(&Gs[Gk][0x1][Gi]) = { g0[0x1], g1[0x1] };
			*(float2*)(&Gs[Gk][0x2][Gi]) = { g0[0x2], g1[0x2] }; *(float2*)(&Gs[Gk][0x3][Gi]) = { g0[0x3], g1[0x3] };
			*(float2*)(&Gs[Gk][0x4][Gi]) = { g0[0x4], g1[0x4] }; *(float2*)(&Gs[Gk][0x5][Gi]) = { g0[0x5], g1[0x5] };
			*(float2*)(&Gs[Gk][0x6][Gi]) = { g0[0x6], g1[0x6] }; *(float2*)(&Gs[Gk][0x7][Gi]) = { g0[0x7], g1[0x7] };
			*(float2*)(&Gs[Gk][0x8][Gi]) = { g0[0x8], g1[0x8] }; *(float2*)(&Gs[Gk][0x9][Gi]) = { g0[0x9], g1[0x9] };
			*(float2*)(&Gs[Gk][0xA][Gi]) = { g0[0xA], g1[0xA] }; *(float2*)(&Gs[Gk][0xB][Gi]) = { g0[0xB], g1[0xB] };
			*(float2*)(&Gs[Gk][0xC][Gi]) = { g0[0xC], g1[0xC] }; *(float2*)(&Gs[Gk][0xD][Gi]) = { g0[0xD], g1[0xD] };
			*(float2*)(&Gs[Gk][0xE][Gi]) = { g0[0xE], g1[0xE] }; *(float2*)(&Gs[Gk][0xF][Gi]) = { g0[0xF], g1[0xF] };

			*(float2*)(&Ds[Xk][0x0][Xi]) = { d0[0x0], d1[0x0] }; *(float2*)(&Ds[Xk][0x1][Xi]) = { d0[0x1], d1[0x1] };
			*(float2*)(&Ds[Xk][0x2][Xi]) = { d0[0x2], d1[0x2] }; *(float2*)(&Ds[Xk][0x3][Xi]) = { d0[0x3], d1[0x3] };
			*(float2*)(&Ds[Xk][0x4][Xi]) = { d0[0x4], d1[0x4] }; *(float2*)(&Ds[Xk][0x5][Xi]) = { d0[0x5], d1[0x5] };
			*(float2*)(&Ds[Xk][0x6][Xi]) = { d0[0x6], d1[0x6] }; *(float2*)(&Ds[Xk][0x7][Xi]) = { d0[0x7], d1[0x7] };
			*(float2*)(&Ds[Xk][0x8][Xi]) = { d0[0x8], d1[0x8] }; *(float2*)(&Ds[Xk][0x9][Xi]) = { d0[0x9], d1[0x9] };
			*(float2*)(&Ds[Xk][0xA][Xi]) = { d0[0xA], d1[0xA] }; *(float2*)(&Ds[Xk][0xB][Xi]) = { d0[0xB], d1[0xB] };
			*(float2*)(&Ds[Xk][0xC][Xi]) = { d0[0xC], d1[0xC] }; *(float2*)(&Ds[Xk][0xD][Xi]) = { d0[0xD], d1[0xD] };
			*(float2*)(&Ds[Xk][0xE][Xi]) = { d0[0xE], d1[0xE] }; *(float2*)(&Ds[Xk][0xF][Xi]) = { d0[0xF], d1[0xF] };
			__syncthreads();
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 b0 = *(float4*)(&Ds[ik][ux][DIdx    ]);
				float4 b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

				//=======================================================
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
				
				//oc[0 - 3]             oc[4 - 7]
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

				//oc[8 - 11]            oc[12 - 15]
				simdMM4(v16, b0.x, a2); simdMM4(v17, b0.x, a3);//j0
				simdMM4(v18, b0.y, a2); simdMM4(v19, b0.y, a3);//j1
				simdMM4(v20, b0.z, a2); simdMM4(v21, b0.z, a3);//j2
				simdMM4(v22, b0.w, a2); simdMM4(v23, b0.w, a3);//j3
				simdMM4(v24, b1.x, a2); simdMM4(v25, b1.x, a3);//j4
				simdMM4(v26, b1.y, a2); simdMM4(v27, b1.y, a3);//j5
				simdMM4(v28, b1.z, a2); simdMM4(v29, b1.z, a3);//j6
				simdMM4(v30, b1.w, a2); simdMM4(v31, b1.w, a3);//j7
			}

			//load 1 group from CW[FH, FW, IC, OC]
			const int W0 = (fh * 8 * IC + oic) * OC;
			const int W1 = W0 + IC * OC    , W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7;
			float2 wv0 = *(float2*)(CW + W0), wv1 = *(float2*)(CW + W1);
			float2 wv2 = *(float2*)(CW + W2), wv3 = *(float2*)(CW + W3);
			float2 wv4 = *(float2*)(CW + W4), wv5 = *(float2*)(CW + W5);
			float2 wv6 = *(float2*)(CW + W6), wv7 = *(float2*)(CW + W7);

			w0[0] = wv0.x; w1[0] = wv0.y; w0[1] = wv1.x; w1[1] = wv1.y;
			w0[2] = wv2.x; w1[2] = wv2.y; w0[3] = wv3.x; w1[3] = wv3.y;
			w0[4] = wv4.x; w1[4] = wv4.y; w0[5] = wv5.x; w1[5] = wv5.y;
			w0[6] = wv6.x; w1[6] = wv6.y; w0[7] = wv7.x; w1[7] = wv7.y;

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic, tX1 = tX0 + IC;
			int tX2 = tX0 + (IC << 1), tX3 = tX0 + IC *  3;
			int tX4 = tX0 + (IC << 2), tX5 = tX0 + IC *  5;
			int tX6 = tX0 + IC * 6   , tX7 = tX0 + IC *  7;
			int tX8 = tX0 + (IC << 3), tX9 = tX0 + IC *  9;
			int tXA = tX0 + IC * 10  , tXB = tX0 + IC * 11;
			int tXC = tX0 + IC * 12  , tXD = tX0 + IC * 13;
			int tXE = tX0 + IC * 14  , tXF = tX0 + IC * 15;
			int tXG = tX0 + (IC << 4), tXH = tX0 + IC * 17;
			int tXI = tX0 + IC * 18  , tXJ = tX0 + IC * 19;
			int tXK = tX0 + IC * 20  , tXL = tX0 + IC * 21;
			int tXM = tX0 + IC * 22  , tXN = tX0 + IC * 23;
			int tXO = tX0 + IC * 24;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1); tX6 = IF_int(lh0, tX6, -1); 
			tX7 = IF_int(lh0, tX7, -1); tX8 = IF_int(lh0, tX8, -1); 
			tX9 = IF_int(lh0, tX9, -1); tXA = IF_int(lh0, tXA, -1); 
			tXB = IF_int(lh0, tXB, -1); tXC = IF_int(lh0, tXC, -1); tXD = IF_int(lh0, tXD, -1);
			tXE = IF_int(lh0, tXE, -1); tXF = IF_int(lh0, tXF, -1);
			tXG = IF_int(lh0, tXG, -1); tXH = IF_int(lh0, tXH, -1);
			tXI = IF_int(lh0, tXI, -1); tXJ = IF_int(lh0, tXJ, -1); tXK = IF_int(lh0, tXK, -1); 
			tXL = IF_int(lyL, tXL, -1); tXM = IF_int(lyM, tXM, -1);
			tXN = IF_int(lyN, tXN, -1); tXO = IF_int(lyO, tXO, -1);

			x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
			x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
			x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
			x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
			x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
			x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
			x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
			x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);
			x[16] = tex1Dfetch<float>(X, tXG); x[17] = tex1Dfetch<float>(X, tXH);
			x[18] = tex1Dfetch<float>(X, tXI); x[19] = tex1Dfetch<float>(X, tXJ);
			x[20] = tex1Dfetch<float>(X, tXK); x[21] = tex1Dfetch<float>(X, tXL);
			x[22] = tex1Dfetch<float>(X, tXM); x[23] = tex1Dfetch<float>(X, tXN);
			x[24] = tex1Dfetch<float>(X, tXO);
			__syncthreads();
		}
		{
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g0, w0); 
			winograd_f9x8_G(g1, w1);
			winograd_f9x8_D(d0, x);//x[0 - 15]
			winograd_f9x8_D_oft(d1, x, 9);//x[9 - 24]

			//write to shread memory
			*(float2*)(&Gs[Gk][0x0][Gi]) = { g0[0x0], g1[0x0] }; *(float2*)(&Gs[Gk][0x1][Gi]) = { g0[0x1], g1[0x1] };
			*(float2*)(&Gs[Gk][0x2][Gi]) = { g0[0x2], g1[0x2] }; *(float2*)(&Gs[Gk][0x3][Gi]) = { g0[0x3], g1[0x3] };
			*(float2*)(&Gs[Gk][0x4][Gi]) = { g0[0x4], g1[0x4] }; *(float2*)(&Gs[Gk][0x5][Gi]) = { g0[0x5], g1[0x5] };
			*(float2*)(&Gs[Gk][0x6][Gi]) = { g0[0x6], g1[0x6] }; *(float2*)(&Gs[Gk][0x7][Gi]) = { g0[0x7], g1[0x7] };
			*(float2*)(&Gs[Gk][0x8][Gi]) = { g0[0x8], g1[0x8] }; *(float2*)(&Gs[Gk][0x9][Gi]) = { g0[0x9], g1[0x9] };
			*(float2*)(&Gs[Gk][0xA][Gi]) = { g0[0xA], g1[0xA] }; *(float2*)(&Gs[Gk][0xB][Gi]) = { g0[0xB], g1[0xB] };
			*(float2*)(&Gs[Gk][0xC][Gi]) = { g0[0xC], g1[0xC] }; *(float2*)(&Gs[Gk][0xD][Gi]) = { g0[0xD], g1[0xD] };
			*(float2*)(&Gs[Gk][0xE][Gi]) = { g0[0xE], g1[0xE] }; *(float2*)(&Gs[Gk][0xF][Gi]) = { g0[0xF], g1[0xF] };

			*(float2*)(&Ds[Xk][0x0][Xi]) = { d0[0x0], d1[0x0] }; *(float2*)(&Ds[Xk][0x1][Xi]) = { d0[0x1], d1[0x1] };
			*(float2*)(&Ds[Xk][0x2][Xi]) = { d0[0x2], d1[0x2] }; *(float2*)(&Ds[Xk][0x3][Xi]) = { d0[0x3], d1[0x3] };
			*(float2*)(&Ds[Xk][0x4][Xi]) = { d0[0x4], d1[0x4] }; *(float2*)(&Ds[Xk][0x5][Xi]) = { d0[0x5], d1[0x5] };
			*(float2*)(&Ds[Xk][0x6][Xi]) = { d0[0x6], d1[0x6] }; *(float2*)(&Ds[Xk][0x7][Xi]) = { d0[0x7], d1[0x7] };
			*(float2*)(&Ds[Xk][0x8][Xi]) = { d0[0x8], d1[0x8] }; *(float2*)(&Ds[Xk][0x9][Xi]) = { d0[0x9], d1[0x9] };
			*(float2*)(&Ds[Xk][0xA][Xi]) = { d0[0xA], d1[0xA] }; *(float2*)(&Ds[Xk][0xB][Xi]) = { d0[0xB], d1[0xB] };
			*(float2*)(&Ds[Xk][0xC][Xi]) = { d0[0xC], d1[0xC] }; *(float2*)(&Ds[Xk][0xD][Xi]) = { d0[0xD], d1[0xD] };
			*(float2*)(&Ds[Xk][0xE][Xi]) = { d0[0xE], d1[0xE] }; *(float2*)(&Ds[Xk][0xF][Xi]) = { d0[0xF], d1[0xF] };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]);
			float4 b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

			//=======================================================
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]);
			float4 a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

			//oc[0 - 3]             oc[4 - 7]
			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7

			//=======================================================
			float4 a2 = *(float4*)(&Gs[ik][ux][GIdx + 8]);
			float4 a3 = *(float4*)(&Gs[ik][ux][GIdx + 12]);

			//oc[8 - 11]             oc[12 - 15]
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
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[9], y1[9], y2[9], y3[9];//oc0, oc1, oc2, oc3
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	*(float4*)(&Ys0[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Ys0[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Ys0[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Ys0[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	*(float4*)(&Ys1[ux][uy][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Ys1[ux][uy][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Ys1[ux][uy][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Ys1[ux][uy][12]) = float4{ v12.y, v13.y, v14.y, v15.y };

	*(float4*)(&Ys0[ux][uy + 8][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Ys0[ux][uy + 8][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Ys0[ux][uy + 8][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Ys0[ux][uy + 8][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	*(float4*)(&Ys1[ux][uy + 8][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Ys1[ux][uy + 8][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Ys1[ux][uy + 8][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Ys1[ux][uy + 8][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[ 0][uy][ux]; a[ 1] = Ys0[ 1][uy][ux]; a[ 2] = Ys0[ 2][uy][ux]; a[ 3] = Ys0[ 3][uy][ux];
	a[ 4] = Ys0[ 4][uy][ux]; a[ 5] = Ys0[ 5][uy][ux]; a[ 6] = Ys0[ 6][uy][ux]; a[ 7] = Ys0[ 7][uy][ux];
	a[ 8] = Ys0[ 8][uy][ux]; a[ 9] = Ys0[ 9][uy][ux]; a[10] = Ys0[10][uy][ux]; a[11] = Ys0[11][uy][ux];
	a[12] = Ys0[12][uy][ux]; a[13] = Ys0[13][uy][ux]; a[14] = Ys0[14][uy][ux]; a[15] = Ys0[15][uy][ux];
	winograd_f9x8_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f9x8_Y(y1, a);
	
	//read-turn2: z, [oc2, oc6]
	a[ 0] = Ys0[ 0][uy + 8][ux]; a[ 1] = Ys0[ 1][uy + 8][ux]; a[ 2] = Ys0[ 2][uy + 8][ux]; a[ 3] = Ys0[ 3][uy + 8][ux];
	a[ 4] = Ys0[ 4][uy + 8][ux]; a[ 5] = Ys0[ 5][uy + 8][ux]; a[ 6] = Ys0[ 6][uy + 8][ux]; a[ 7] = Ys0[ 7][uy + 8][ux];
	a[ 8] = Ys0[ 8][uy + 8][ux]; a[ 9] = Ys0[ 9][uy + 8][ux]; a[10] = Ys0[10][uy + 8][ux]; a[11] = Ys0[11][uy + 8][ux];
	a[12] = Ys0[12][uy + 8][ux]; a[13] = Ys0[13][uy + 8][ux]; a[14] = Ys0[14][uy + 8][ux]; a[15] = Ys0[15][uy + 8][ux];
	winograd_f9x8_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy + 8][ux]; a[ 1] = Ys1[ 1][uy + 8][ux]; a[ 2] = Ys1[ 2][uy + 8][ux]; a[ 3] = Ys1[ 3][uy + 8][ux];
	a[ 4] = Ys1[ 4][uy + 8][ux]; a[ 5] = Ys1[ 5][uy + 8][ux]; a[ 6] = Ys1[ 6][uy + 8][ux]; a[ 7] = Ys1[ 7][uy + 8][ux];
	a[ 8] = Ys1[ 8][uy + 8][ux]; a[ 9] = Ys1[ 9][uy + 8][ux]; a[10] = Ys1[10][uy + 8][ux]; a[11] = Ys1[11][uy + 8][ux];
	a[12] = Ys1[12][uy + 8][ux]; a[13] = Ys1[13][uy + 8][ux]; a[14] = Ys1[14][uy + 8][ux]; a[15] = Ys1[15][uy + 8][ux];
	winograd_f9x8_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
	*(float4*)(Y + Y00 + OC * 8) = float4{ y0[8], y1[8], y2[8], y3[8] };
	__syncthreads();
	
	//group1-----------------------------------------------------------------------------
	*(float4*)(&Ys0[ux][uy][ 0]) = float4{ v16.x, v17.x, v18.x, v19.x };
	*(float4*)(&Ys0[ux][uy][ 4]) = float4{ v20.x, v21.x, v22.x, v23.x };
	*(float4*)(&Ys0[ux][uy][ 8]) = float4{ v24.x, v25.x, v26.x, v27.x };
	*(float4*)(&Ys0[ux][uy][12]) = float4{ v28.x, v29.x, v30.x, v31.x };

	*(float4*)(&Ys1[ux][uy][ 0]) = float4{ v16.y, v17.y, v18.y, v19.y };
	*(float4*)(&Ys1[ux][uy][ 4]) = float4{ v20.y, v21.y, v22.y, v23.y };
	*(float4*)(&Ys1[ux][uy][ 8]) = float4{ v24.y, v25.y, v26.y, v27.y };
	*(float4*)(&Ys1[ux][uy][12]) = float4{ v28.y, v29.y, v30.y, v31.y };

	*(float4*)(&Ys0[ux][uy + 8][ 0]) = float4{ v16.z, v17.z, v18.z, v19.z };
	*(float4*)(&Ys0[ux][uy + 8][ 4]) = float4{ v20.z, v21.z, v22.z, v23.z };
	*(float4*)(&Ys0[ux][uy + 8][ 8]) = float4{ v24.z, v25.z, v26.z, v27.z };
	*(float4*)(&Ys0[ux][uy + 8][12]) = float4{ v28.z, v29.z, v30.z, v31.z };

	*(float4*)(&Ys1[ux][uy + 8][ 0]) = float4{ v16.w, v17.w, v18.w, v19.w };
	*(float4*)(&Ys1[ux][uy + 8][ 4]) = float4{ v20.w, v21.w, v22.w, v23.w };
	*(float4*)(&Ys1[ux][uy + 8][ 8]) = float4{ v24.w, v25.w, v26.w, v27.w };
	*(float4*)(&Ys1[ux][uy + 8][12]) = float4{ v28.w, v29.w, v30.w, v31.w };
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[ 0][uy][ux]; a[ 1] = Ys0[ 1][uy][ux]; a[ 2] = Ys0[ 2][uy][ux]; a[ 3] = Ys0[ 3][uy][ux];
	a[ 4] = Ys0[ 4][uy][ux]; a[ 5] = Ys0[ 5][uy][ux]; a[ 6] = Ys0[ 6][uy][ux]; a[ 7] = Ys0[ 7][uy][ux];
	a[ 8] = Ys0[ 8][uy][ux]; a[ 9] = Ys0[ 9][uy][ux]; a[10] = Ys0[10][uy][ux]; a[11] = Ys0[11][uy][ux];
	a[12] = Ys0[12][uy][ux]; a[13] = Ys0[13][uy][ux]; a[14] = Ys0[14][uy][ux]; a[15] = Ys0[15][uy][ux];
	winograd_f9x8_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f9x8_Y(y1, a);
	
	//read-turn2: z, [oc2, oc6]
	a[ 0] = Ys0[ 0][uy + 8][ux]; a[ 1] = Ys0[ 1][uy + 8][ux]; a[ 2] = Ys0[ 2][uy + 8][ux]; a[ 3] = Ys0[ 3][uy + 8][ux];
	a[ 4] = Ys0[ 4][uy + 8][ux]; a[ 5] = Ys0[ 5][uy + 8][ux]; a[ 6] = Ys0[ 6][uy + 8][ux]; a[ 7] = Ys0[ 7][uy + 8][ux];
	a[ 8] = Ys0[ 8][uy + 8][ux]; a[ 9] = Ys0[ 9][uy + 8][ux]; a[10] = Ys0[10][uy + 8][ux]; a[11] = Ys0[11][uy + 8][ux];
	a[12] = Ys0[12][uy + 8][ux]; a[13] = Ys0[13][uy + 8][ux]; a[14] = Ys0[14][uy + 8][ux]; a[15] = Ys0[15][uy + 8][ux];
	winograd_f9x8_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy + 8][ux]; a[ 1] = Ys1[ 1][uy + 8][ux]; a[ 2] = Ys1[ 2][uy + 8][ux]; a[ 3] = Ys1[ 3][uy + 8][ux];
	a[ 4] = Ys1[ 4][uy + 8][ux]; a[ 5] = Ys1[ 5][uy + 8][ux]; a[ 6] = Ys1[ 6][uy + 8][ux]; a[ 7] = Ys1[ 7][uy + 8][ux];
	a[ 8] = Ys1[ 8][uy + 8][ux]; a[ 9] = Ys1[ 9][uy + 8][ux]; a[10] = Ys1[10][uy + 8][ux]; a[11] = Ys1[11][uy + 8][ux];
	a[12] = Ys1[12][uy + 8][ux]; a[13] = Ys1[13][uy + 8][ux]; a[14] = Ys1[14][uy + 8][ux]; a[15] = Ys1[15][uy + 8][ux];
	winograd_f9x8_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00          + 8) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC     + 8) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2 + 8) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3 + 8) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4 + 8) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5 + 8) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6 + 8) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7 + 8) = float4{ y0[7], y1[7], y2[7], y3[7] };
	*(float4*)(Y + Y00 + OC * 8 + 8) = float4{ y0[8], y1[8], y2[8], y3[8] };
}

#endif


//================[Cooperate: 32(OC) * 288(GM)]==============================
//LB = 4, IC % 8 == 0, pw <= 4, OWr % 9 == 0
#ifndef CONV_3D_WINOGRAD_F9X8_32X288RC_P4_TEXTURE
#define CONV_3D_WINOGRAD_F9X8_32X288RC_P4_TEXTURE

//for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//WB = 4: Size = 81, Time = 6.92828 msec, Performace = 25106.7 GFlop/s
//for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//WB = 4: Size = 81, Time = 8.32736 msec, Performace = 20888.5 GFlop/s
//for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//WB = 4: Size = 81, Time = 8.35202 msec, Performace = 20826.8 GFlop/s

template<int FH>
__global__ void conv3dWinograd_f9x8_kernel_32x288RC_p4_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int ow_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;
	const int bx = blockIdx.x, by = blockIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 32]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = by * 288;//32 * 9 = 288
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi * 9);//OWr % 9
	const int OWr9 = (OW- ow_index) / 9 * 9, OH_OWr9 = OH * OWr9;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr9, OWr9); tow0 += ow_index;
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 9;
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr9, OWr9); yow0 += ow_index;
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;

	//======[compute area1: local]======================================================
	float w[8], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = (fh << 3) * IC * OC;
		const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7;
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7];

		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=   0) && (tiw0      < IW);//pw = -4
		bool ly1 = lh0 && (tiw0 >=  -1) && (tiw0 +  1 < IW);//pw = -3
		bool ly2 = lh0 && (tiw0 >=  -2) && (tiw0 +  2 < IW);//pw = -2
		bool ly3 = lh0 && (tiw0 >=  -3) && (tiw0 +  3 < IW);//pw = -1
		bool lyC = lh0 && (tiw0 >= -12) && (tiw0 + 12 < IW);//pw = +1
		bool lyD = lh0 && (tiw0 >= -13) && (tiw0 + 13 < IW);//pw = +2
		bool lyE = lh0 && (tiw0 >= -14) && (tiw0 + 14 < IW);//pw = +3
		bool lyF = lh0 && (tiw0 >= -15) && (tiw0 + 15 < IW);//pw = +4
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
		int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
		int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
		int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
		int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
		int tXF = tX0 + IC * 15;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
		tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
		tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
		tXA = IF_int(lh0, tXA, -1); tXB = IF_int(lh0, tXB, -1);
		tXC = IF_int(lyC, tXC, -1); tXD = IF_int(lyD, tXD, -1);
		tXE = IF_int(lyE, tXE, -1); tXF = IF_int(lyF, tXF, -1);
		x[0x0] = tex1Dfetch<float>(X, tX0); x[0x1] = tex1Dfetch<float>(X, tX1);
		x[0x2] = tex1Dfetch<float>(X, tX2); x[0x3] = tex1Dfetch<float>(X, tX3);
		x[0x4] = tex1Dfetch<float>(X, tX4); x[0x5] = tex1Dfetch<float>(X, tX5);
		x[0x6] = tex1Dfetch<float>(X, tX6); x[0x7] = tex1Dfetch<float>(X, tX7);
		x[0x8] = tex1Dfetch<float>(X, tX8); x[0x9] = tex1Dfetch<float>(X, tX9);
		x[0xA] = tex1Dfetch<float>(X, tXA); x[0xB] = tex1Dfetch<float>(X, tXB);
		x[0xC] = tex1Dfetch<float>(X, tXC); x[0xD] = tex1Dfetch<float>(X, tXD);
		x[0xE] = tex1Dfetch<float>(X, tXE); x[0xF] = tex1Dfetch<float>(X, tXF);

		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g, w);
			winograd_f9x8_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Xk][ 0][Xi] = d[ 0]; Ds[Xk][ 1][Xi] = d[ 1];
			Ds[Xk][ 2][Xi] = d[ 2]; Ds[Xk][ 3][Xi] = d[ 3];
			Ds[Xk][ 4][Xi] = d[ 4]; Ds[Xk][ 5][Xi] = d[ 5];
			Ds[Xk][ 6][Xi] = d[ 6]; Ds[Xk][ 7][Xi] = d[ 7];
			Ds[Xk][ 8][Xi] = d[ 8]; Ds[Xk][ 9][Xi] = d[ 9];
			Ds[Xk][10][Xi] = d[10]; Ds[Xk][11][Xi] = d[11];
			Ds[Xk][12][Xi] = d[12]; Ds[Xk][13][Xi] = d[13];
			Ds[Xk][14][Xi] = d[14]; Ds[Xk][15][Xi] = d[15];
			__syncthreads();
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

				//oc[0 - 3]             oc[4 - 7]
				simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
				simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
				simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
				simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
				simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}

			//load 1 group from CW[FH, FW, IC, OC]
			const int W0 = ((fh << 3) * IC + oic) * OC;
			const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7;
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7];

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
			int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
			int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
			int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
			int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
			int tXF = tX0 + IC * 15;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
			tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
			tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
			tXA = IF_int(lh0, tXA, -1); tXB = IF_int(lh0, tXB, -1);
			tXC = IF_int(lyC, tXC, -1); tXD = IF_int(lyD, tXD, -1);
			tXE = IF_int(lyE, tXE, -1); tXF = IF_int(lyF, tXF, -1);
			x[0x0] = tex1Dfetch<float>(X, tX0); x[0x1] = tex1Dfetch<float>(X, tX1);
			x[0x2] = tex1Dfetch<float>(X, tX2); x[0x3] = tex1Dfetch<float>(X, tX3);
			x[0x4] = tex1Dfetch<float>(X, tX4); x[0x5] = tex1Dfetch<float>(X, tX5);
			x[0x6] = tex1Dfetch<float>(X, tX6); x[0x7] = tex1Dfetch<float>(X, tX7);
			x[0x8] = tex1Dfetch<float>(X, tX8); x[0x9] = tex1Dfetch<float>(X, tX9);
			x[0xA] = tex1Dfetch<float>(X, tXA); x[0xB] = tex1Dfetch<float>(X, tXB);
			x[0xC] = tex1Dfetch<float>(X, tXC); x[0xD] = tex1Dfetch<float>(X, tXD);
			x[0xE] = tex1Dfetch<float>(X, tXE); x[0xF] = tex1Dfetch<float>(X, tXF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g, w);
			winograd_f9x8_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Xk][ 0][Xi] = d[ 0]; Ds[Xk][ 1][Xi] = d[ 1];
			Ds[Xk][ 2][Xi] = d[ 2]; Ds[Xk][ 3][Xi] = d[ 3];
			Ds[Xk][ 4][Xi] = d[ 4]; Ds[Xk][ 5][Xi] = d[ 5];
			Ds[Xk][ 6][Xi] = d[ 6]; Ds[Xk][ 7][Xi] = d[ 7];
			Ds[Xk][ 8][Xi] = d[ 8]; Ds[Xk][ 9][Xi] = d[ 9];
			Ds[Xk][10][Xi] = d[10]; Ds[Xk][11][Xi] = d[11];
			Ds[Xk][12][Xi] = d[12]; Ds[Xk][13][Xi] = d[13];
			Ds[Xk][14][Xi] = d[14]; Ds[Xk][15][Xi] = d[15];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

			//oc[0 - 3]             oc[4 - 7]
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
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[9], y1[9], y2[9], y3[9];//oc0, oc1, oc2, oc3
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(&Ys0[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Ys0[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Ys0[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Ys0[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//write-turn1: y, [oc0, oc4]
	*(float4*)(&Ys1[ux][uy][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Ys1[ux][uy][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Ys1[ux][uy][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Ys1[ux][uy][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[ 0][uy][ux]; a[ 1] = Ys0[ 1][uy][ux]; a[ 2] = Ys0[ 2][uy][ux]; a[ 3] = Ys0[ 3][uy][ux];
	a[ 4] = Ys0[ 4][uy][ux]; a[ 5] = Ys0[ 5][uy][ux]; a[ 6] = Ys0[ 6][uy][ux]; a[ 7] = Ys0[ 7][uy][ux];
	a[ 8] = Ys0[ 8][uy][ux]; a[ 9] = Ys0[ 9][uy][ux]; a[10] = Ys0[10][uy][ux]; a[11] = Ys0[11][uy][ux];
	a[12] = Ys0[12][uy][ux]; a[13] = Ys0[13][uy][ux]; a[14] = Ys0[14][uy][ux]; a[15] = Ys0[15][uy][ux];
	winograd_f9x8_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f9x8_Y(y1, a);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [oc2, oc6]
	*(float4*)(&Ys0[ux][uy][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Ys0[ux][uy][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Ys0[ux][uy][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Ys0[ux][uy][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn1: w, [oc3, oc7]
	*(float4*)(&Ys1[ux][uy][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Ys1[ux][uy][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Ys1[ux][uy][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Ys1[ux][uy][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc2, oc6]
	a[ 0] = Ys0[ 0][uy][ux]; a[ 1] = Ys0[ 1][uy][ux]; a[ 2] = Ys0[ 2][uy][ux]; a[ 3] = Ys0[ 3][uy][ux];
	a[ 4] = Ys0[ 4][uy][ux]; a[ 5] = Ys0[ 5][uy][ux]; a[ 6] = Ys0[ 6][uy][ux]; a[ 7] = Ys0[ 7][uy][ux];
	a[ 8] = Ys0[ 8][uy][ux]; a[ 9] = Ys0[ 9][uy][ux]; a[10] = Ys0[10][uy][ux]; a[11] = Ys0[11][uy][ux];
	a[12] = Ys0[12][uy][ux]; a[13] = Ys0[13][uy][ux]; a[14] = Ys0[14][uy][ux]; a[15] = Ys0[15][uy][ux];
	winograd_f9x8_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f9x8_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
	*(float4*)(Y + Y00 + OC * 8) = float4{ y0[8], y1[8], y2[8], y3[8] };
}

#endif


//LB = 4, IC % 8 == 0, pw <= 4, OWr % 9 == 0, template<IC, OC>
#ifndef CONV_3D_WINOGRAD_F9X8_32X288RC_P4_CT_TEXTURE
#define CONV_3D_WINOGRAD_F9X8_32X288RC_P4_CT_TEXTURE

//for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//WB = 4: Size = 81, Time = 6.19815 msec, Performace = 28064.2 GFlop/s
//for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//WB = 4: Size = 81, Time = 7.71775 msec, Performace = 22538.4 GFlop/s
//for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//WB = 4: Size = 81, Time = 7.50656 msec, Performace = 23172.6 GFlop/s

template<int FH, int IC, int OC>
__global__ void conv3dWinograd_f9x8_kernel_32x288RC_p4_CT_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int ph, int pw,//sh = sw = 1
	int ow_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;
	const int bx = blockIdx.x, by = blockIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 32]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = by * 288;//32 * 9 = 288
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi * 9);//OWr % 9
	const int OWr9 = (OW- ow_index) / 9 * 9, OH_OWr9 = OH * OWr9;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr9, OWr9); tow0 += ow_index;
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 9;
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr9, OWr9); yow0 += ow_index;
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;

	//======[compute area1: local]======================================================
	float w[8], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = (fh << 3) * IC * OC;
		const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7;
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7];

		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=   0) && (tiw0      < IW);//pw = -4
		bool ly1 = lh0 && (tiw0 >=  -1) && (tiw0 +  1 < IW);//pw = -3
		bool ly2 = lh0 && (tiw0 >=  -2) && (tiw0 +  2 < IW);//pw = -2
		bool ly3 = lh0 && (tiw0 >=  -3) && (tiw0 +  3 < IW);//pw = -1
		bool lyC = lh0 && (tiw0 >= -12) && (tiw0 + 12 < IW);//pw = +1
		bool lyD = lh0 && (tiw0 >= -13) && (tiw0 + 13 < IW);//pw = +2
		bool lyE = lh0 && (tiw0 >= -14) && (tiw0 + 14 < IW);//pw = +3
		bool lyF = lh0 && (tiw0 >= -15) && (tiw0 + 15 < IW);//pw = +4
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
		int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
		int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
		int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
		int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
		int tXF = tX0 + IC * 15;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
		tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
		tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
		tXA = IF_int(lh0, tXA, -1); tXB = IF_int(lh0, tXB, -1);
		tXC = IF_int(lyC, tXC, -1); tXD = IF_int(lyD, tXD, -1);
		tXE = IF_int(lyE, tXE, -1); tXF = IF_int(lyF, tXF, -1);
		x[0x0] = tex1Dfetch<float>(X, tX0); x[0x1] = tex1Dfetch<float>(X, tX1);
		x[0x2] = tex1Dfetch<float>(X, tX2); x[0x3] = tex1Dfetch<float>(X, tX3);
		x[0x4] = tex1Dfetch<float>(X, tX4); x[0x5] = tex1Dfetch<float>(X, tX5);
		x[0x6] = tex1Dfetch<float>(X, tX6); x[0x7] = tex1Dfetch<float>(X, tX7);
		x[0x8] = tex1Dfetch<float>(X, tX8); x[0x9] = tex1Dfetch<float>(X, tX9);
		x[0xA] = tex1Dfetch<float>(X, tXA); x[0xB] = tex1Dfetch<float>(X, tXB);
		x[0xC] = tex1Dfetch<float>(X, tXC); x[0xD] = tex1Dfetch<float>(X, tXD);
		x[0xE] = tex1Dfetch<float>(X, tXE); x[0xF] = tex1Dfetch<float>(X, tXF);

		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g, w);
			winograd_f9x8_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Xk][ 0][Xi] = d[ 0]; Ds[Xk][ 1][Xi] = d[ 1];
			Ds[Xk][ 2][Xi] = d[ 2]; Ds[Xk][ 3][Xi] = d[ 3];
			Ds[Xk][ 4][Xi] = d[ 4]; Ds[Xk][ 5][Xi] = d[ 5];
			Ds[Xk][ 6][Xi] = d[ 6]; Ds[Xk][ 7][Xi] = d[ 7];
			Ds[Xk][ 8][Xi] = d[ 8]; Ds[Xk][ 9][Xi] = d[ 9];
			Ds[Xk][10][Xi] = d[10]; Ds[Xk][11][Xi] = d[11];
			Ds[Xk][12][Xi] = d[12]; Ds[Xk][13][Xi] = d[13];
			Ds[Xk][14][Xi] = d[14]; Ds[Xk][15][Xi] = d[15];
			__syncthreads();
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

				//oc[0 - 3]             oc[4 - 7]
				simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);//j0
				simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);//j1
				simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);//j2
				simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);//j3
				simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}

			//load 1 group from CW[FH, FW, IC, OC]
			const int W0 = ((fh << 3) * IC + oic) * OC;
			const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7;
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7];

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
			int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
			int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
			int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
			int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
			int tXF = tX0 + IC * 15;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
			tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
			tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
			tXA = IF_int(lh0, tXA, -1); tXB = IF_int(lh0, tXB, -1);
			tXC = IF_int(lyC, tXC, -1); tXD = IF_int(lyD, tXD, -1);
			tXE = IF_int(lyE, tXE, -1); tXF = IF_int(lyF, tXF, -1);
			x[0x0] = tex1Dfetch<float>(X, tX0); x[0x1] = tex1Dfetch<float>(X, tX1);
			x[0x2] = tex1Dfetch<float>(X, tX2); x[0x3] = tex1Dfetch<float>(X, tX3);
			x[0x4] = tex1Dfetch<float>(X, tX4); x[0x5] = tex1Dfetch<float>(X, tX5);
			x[0x6] = tex1Dfetch<float>(X, tX6); x[0x7] = tex1Dfetch<float>(X, tX7);
			x[0x8] = tex1Dfetch<float>(X, tX8); x[0x9] = tex1Dfetch<float>(X, tX9);
			x[0xA] = tex1Dfetch<float>(X, tXA); x[0xB] = tex1Dfetch<float>(X, tXB);
			x[0xC] = tex1Dfetch<float>(X, tXC); x[0xD] = tex1Dfetch<float>(X, tXD);
			x[0xE] = tex1Dfetch<float>(X, tXE); x[0xF] = tex1Dfetch<float>(X, tXF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g, w);
			winograd_f9x8_D(d, x);

			//write to shread memory
			Gs[Gk][ 0][Gi] = g[ 0]; Gs[Gk][ 1][Gi] = g[ 1];
			Gs[Gk][ 2][Gi] = g[ 2]; Gs[Gk][ 3][Gi] = g[ 3];
			Gs[Gk][ 4][Gi] = g[ 4]; Gs[Gk][ 5][Gi] = g[ 5];
			Gs[Gk][ 6][Gi] = g[ 6]; Gs[Gk][ 7][Gi] = g[ 7];
			Gs[Gk][ 8][Gi] = g[ 8]; Gs[Gk][ 9][Gi] = g[ 9];
			Gs[Gk][10][Gi] = g[10]; Gs[Gk][11][Gi] = g[11];
			Gs[Gk][12][Gi] = g[12]; Gs[Gk][13][Gi] = g[13];
			Gs[Gk][14][Gi] = g[14]; Gs[Gk][15][Gi] = g[15];

			Ds[Xk][ 0][Xi] = d[ 0]; Ds[Xk][ 1][Xi] = d[ 1];
			Ds[Xk][ 2][Xi] = d[ 2]; Ds[Xk][ 3][Xi] = d[ 3];
			Ds[Xk][ 4][Xi] = d[ 4]; Ds[Xk][ 5][Xi] = d[ 5];
			Ds[Xk][ 6][Xi] = d[ 6]; Ds[Xk][ 7][Xi] = d[ 7];
			Ds[Xk][ 8][Xi] = d[ 8]; Ds[Xk][ 9][Xi] = d[ 9];
			Ds[Xk][10][Xi] = d[10]; Ds[Xk][11][Xi] = d[11];
			Ds[Xk][12][Xi] = d[12]; Ds[Xk][13][Xi] = d[13];
			Ds[Xk][14][Xi] = d[14]; Ds[Xk][15][Xi] = d[15];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][DIdx]), b1 = *(float4*)(&Ds[ik][ux][DIdx + 4]);

			//oc[0 - 3]             oc[4 - 7]
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
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[9], y1[9], y2[9], y3[9];//oc0, oc1, oc2, oc3
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(&Ys0[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Ys0[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Ys0[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Ys0[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//write-turn1: y, [oc0, oc4]
	*(float4*)(&Ys1[ux][uy][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Ys1[ux][uy][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Ys1[ux][uy][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Ys1[ux][uy][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[ 0][uy][ux]; a[ 1] = Ys0[ 1][uy][ux]; a[ 2] = Ys0[ 2][uy][ux]; a[ 3] = Ys0[ 3][uy][ux];
	a[ 4] = Ys0[ 4][uy][ux]; a[ 5] = Ys0[ 5][uy][ux]; a[ 6] = Ys0[ 6][uy][ux]; a[ 7] = Ys0[ 7][uy][ux];
	a[ 8] = Ys0[ 8][uy][ux]; a[ 9] = Ys0[ 9][uy][ux]; a[10] = Ys0[10][uy][ux]; a[11] = Ys0[11][uy][ux];
	a[12] = Ys0[12][uy][ux]; a[13] = Ys0[13][uy][ux]; a[14] = Ys0[14][uy][ux]; a[15] = Ys0[15][uy][ux];
	winograd_f9x8_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f9x8_Y(y1, a);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [oc2, oc6]
	*(float4*)(&Ys0[ux][uy][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Ys0[ux][uy][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Ys0[ux][uy][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Ys0[ux][uy][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn1: w, [oc3, oc7]
	*(float4*)(&Ys1[ux][uy][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Ys1[ux][uy][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Ys1[ux][uy][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Ys1[ux][uy][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc2, oc6]
	a[ 0] = Ys0[ 0][uy][ux]; a[ 1] = Ys0[ 1][uy][ux]; a[ 2] = Ys0[ 2][uy][ux]; a[ 3] = Ys0[ 3][uy][ux];
	a[ 4] = Ys0[ 4][uy][ux]; a[ 5] = Ys0[ 5][uy][ux]; a[ 6] = Ys0[ 6][uy][ux]; a[ 7] = Ys0[ 7][uy][ux];
	a[ 8] = Ys0[ 8][uy][ux]; a[ 9] = Ys0[ 9][uy][ux]; a[10] = Ys0[10][uy][ux]; a[11] = Ys0[11][uy][ux];
	a[12] = Ys0[12][uy][ux]; a[13] = Ys0[13][uy][ux]; a[14] = Ys0[14][uy][ux]; a[15] = Ys0[15][uy][ux];
	winograd_f9x8_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f9x8_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
	*(float4*)(Y + Y00 + OC * 8) = float4{ y0[8], y1[8], y2[8], y3[8] };
}

#endif


//================[Standard: 64(OC) * 288(GM)]===============================
//LB = 4, IC % 8 == 0, pw <= 4, OW % 8 == 0
#ifndef CONV_3D_WINOGRAD_F9X8_64X288R_P4_TEXTURE
#define CONV_3D_WINOGRAD_F9X8_64X288R_P4_TEXTURE

//for: Feature = (72, 72), [N, IC, OC] = [64, 64, 64]
//LB = 4: Size = 81, Time = 5.58542 msec, Performace = 31142.9 GFlop/s
//WB = 4: Size = 81, Time = 5.08907 msec, Performace = 34180.4 GFlop/s
//for: Feature = (36, 36), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 81, Time = 6.44722 msec, Performace = 26980   GFlop/s
//WB = 4: Size = 81, Time = 6.04288 msec, Performace = 28785.3 GFlop/s
//for: Feature = (18, 18), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 81, Time = 6.08407 msec, Performace = 28590.4 GFlop/s
//WB = 4: Size = 81, Time = 5.66385 msec, Performace = 30711.6 GFlop/s

template<int FH>
__global__ void conv3dWinograd_f9x8_kernel_64x288R_p4_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float smem[8 * 16 * 96];
	float(*__restrict__ Gs)[16][64] = (float(*)[16][64])(smem          );//[ik: STEP][elem: 16][g(oc): 64]
	float(*__restrict__ Ds)[16][32] = (float(*)[16][32])(smem + 8*16*64);//[ik: STEP][elem: 16][d( j): 32]

	//compute 8*16 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	float4 v16 = F32_4_0, v17 = F32_4_0, v18 = F32_4_0, v19 = F32_4_0;
	float4 v20 = F32_4_0, v21 = F32_4_0, v22 = F32_4_0, v23 = F32_4_0;
	float4 v24 = F32_4_0, v25 = F32_4_0, v26 = F32_4_0, v27 = F32_4_0;
	float4 v28 = F32_4_0, v29 = F32_4_0, v30 = F32_4_0, v31 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 6);
	const int Gk = (ty & 7), Gi = ((tx << 1) + (ty > 7)) << 1;//[8, 32]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (by * 288);//32 * 9 = 288
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi * 9);//OW % 9
	const int OW9 = (OW / 9) * 9, OH_OW9 = OH * OW9;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OW9, OW9);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 4;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);//0 + 8
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) * 9);
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OW9, OW9);
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;

	//======[compute area1: local]======================================================
	const int Di = (Xi + (Xk << 2)) & 31;//avoid bank conflict (1/8)
	float w0[8], w1[8], g0[16], g1[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=   0) && (tiw0      < IW);//pw = -4
		bool ly1 = lh0 && (tiw0 >=  -1) && (tiw0 +  1 < IW);//pw = -3
		bool ly2 = lh0 && (tiw0 >=  -2) && (tiw0 +  2 < IW);//pw = -2
		bool ly3 = lh0 && (tiw0 >=  -3) && (tiw0 +  3 < IW);//pw = -1
		bool lyC = lh0 && (tiw0 >= -12) && (tiw0 + 12 < IW);//pw = +1
		bool lyD = lh0 && (tiw0 >= -13) && (tiw0 + 13 < IW);//pw = +2
		bool lyE = lh0 && (tiw0 >= -14) && (tiw0 + 14 < IW);//pw = +3
		bool lyF = lh0 && (tiw0 >= -15) && (tiw0 + 15 < IW);//pw = +4
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
		int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
		int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
		int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
		int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
		int tXF = tX0 + IC * 15;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
		tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
		tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
		tXA = IF_int(lh0, tXA, -1); tXB = IF_int(lh0, tXB, -1);
		tXC = IF_int(lyC, tXC, -1); tXD = IF_int(lyD, tXD, -1);
		tXE = IF_int(lyE, tXE, -1); tXF = IF_int(lyF, tXF, -1);
		x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
		x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
		x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
		x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
		x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
		x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
		x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
		x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);

		//load 2 groups from CW[FH, FW, IC, OC]
		const int W0 = (fh << 3) * IC * OC;
		const int W1 = W0 + IC * OC    , W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7;
		float2 wv0 = *(float2*)(CW + W0); w0[0] = wv0.x; w1[0] = wv0.y;
		float2 wv1 = *(float2*)(CW + W1); w0[1] = wv1.x; w1[1] = wv1.y;
		float2 wv2 = *(float2*)(CW + W2); w0[2] = wv2.x; w1[2] = wv2.y;
		float2 wv3 = *(float2*)(CW + W3); w0[3] = wv3.x; w1[3] = wv3.y;
		float2 wv4 = *(float2*)(CW + W4); w0[4] = wv4.x; w1[4] = wv4.y;
		float2 wv5 = *(float2*)(CW + W5); w0[5] = wv5.x; w1[5] = wv5.y;
		float2 wv6 = *(float2*)(CW + W6); w0[6] = wv6.x; w1[6] = wv6.y;
		float2 wv7 = *(float2*)(CW + W7); w0[7] = wv7.x; w1[7] = wv7.y;
		
		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g0, w0);
			winograd_f9x8_G(g1, w1);
			winograd_f9x8_D(d, x);

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

				//oc[0 - 3]             oc[4 - 7]
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

				//oc[8 - 11]            oc[12 - 15]
				simdMM4(v16, b0.x, a2); simdMM4(v17, b0.x, a3);//j0
				simdMM4(v18, b0.y, a2); simdMM4(v19, b0.y, a3);//j1
				simdMM4(v20, b0.z, a2); simdMM4(v21, b0.z, a3);//j2
				simdMM4(v22, b0.w, a2); simdMM4(v23, b0.w, a3);//j3
				simdMM4(v24, b1.x, a2); simdMM4(v25, b1.x, a3);//j4
				simdMM4(v26, b1.y, a2); simdMM4(v27, b1.y, a3);//j5
				simdMM4(v28, b1.z, a2); simdMM4(v29, b1.z, a3);//j6
				simdMM4(v30, b1.w, a2); simdMM4(v31, b1.w, a3);//j7
			}

			//load 1 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
			int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
			int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
			int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
			int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
			int tXF = tX0 + IC * 15;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
			tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
			tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
			tXA = IF_int(lh0, tXA, -1); tXB = IF_int(lh0, tXB, -1);
			tXC = IF_int(lyC, tXC, -1); tXD = IF_int(lyD, tXD, -1);
			tXE = IF_int(lyE, tXE, -1); tXF = IF_int(lyF, tXF, -1);
			x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
			x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
			x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
			x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
			x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
			x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
			x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
			x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);

			//load 2 groups from CW[FH, FW, IC, OC]
			const int W0 = ((fh << 3) * IC + oic) * OC;
			const int W1 = W0 + IC * OC    , W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7;
			float2 wv0 = *(float2*)(CW + W0); w0[0] = wv0.x; w1[0] = wv0.y;
			float2 wv1 = *(float2*)(CW + W1); w0[1] = wv1.x; w1[1] = wv1.y;
			float2 wv2 = *(float2*)(CW + W2); w0[2] = wv2.x; w1[2] = wv2.y;
			float2 wv3 = *(float2*)(CW + W3); w0[3] = wv3.x; w1[3] = wv3.y;
			float2 wv4 = *(float2*)(CW + W4); w0[4] = wv4.x; w1[4] = wv4.y;
			float2 wv5 = *(float2*)(CW + W5); w0[5] = wv5.x; w1[5] = wv5.y;
			float2 wv6 = *(float2*)(CW + W6); w0[6] = wv6.x; w1[6] = wv6.y;
			float2 wv7 = *(float2*)(CW + W7); w0[7] = wv7.x; w1[7] = wv7.y;
			__syncthreads();
		}
		{
			//Winograd Transform: W(8) -> G(16), X(16) -> D(16)
			winograd_f9x8_G(g0, w0);
			winograd_f9x8_G(g1, w1);
			winograd_f9x8_D(d, x);

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

			//oc[0 - 3]             oc[4 - 7]
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

			//oc[8 - 11]            oc[12 - 15]
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
	float(* __restrict__ Ys)[33][20] = (float(*)[33][20])(smem);//[16][32 + 1][16 + 4]
	float a[16], y0[9], y1[9], y2[9], y3[9];//oc0, oc1, oc2, oc3
	__syncthreads();

	//====[oc0 -> oc7]===================================================================
	//group0-----------------------------------------------------------------------------
	*(float4*)(&Ys[ux][uy][ 0]) = float4{  v0.x,  v1.x,  v2.x,  v3.x };
	*(float4*)(&Ys[ux][uy][ 4]) = float4{  v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Ys[ux][uy][ 8]) = float4{  v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Ys[ux][uy][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	*(float4*)(&Ys[ux][uy + 16][ 0]) = float4{  v0.y,  v1.y,  v2.y,  v3.y };
	*(float4*)(&Ys[ux][uy + 16][ 4]) = float4{  v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Ys[ux][uy + 16][ 8]) = float4{  v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Ys[ux][uy + 16][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	a[ 0] = Ys[ 0][uy][ux]; a[ 1] = Ys[ 1][uy][ux]; a[ 2] = Ys[ 2][uy][ux]; a[ 3] = Ys[ 3][uy][ux];
	a[ 4] = Ys[ 4][uy][ux]; a[ 5] = Ys[ 5][uy][ux]; a[ 6] = Ys[ 6][uy][ux]; a[ 7] = Ys[ 7][uy][ux];
	a[ 8] = Ys[ 8][uy][ux]; a[ 9] = Ys[ 9][uy][ux]; a[10] = Ys[10][uy][ux]; a[11] = Ys[11][uy][ux];
	a[12] = Ys[12][uy][ux]; a[13] = Ys[13][uy][ux]; a[14] = Ys[14][uy][ux]; a[15] = Ys[15][uy][ux];
	winograd_f9x8_Y(y0, a);

	a[ 0] = Ys[ 0][uy + 16][ux]; a[ 1] = Ys[ 1][uy + 16][ux]; a[ 2] = Ys[ 2][uy + 16][ux]; a[ 3] = Ys[ 3][uy + 16][ux];
	a[ 4] = Ys[ 4][uy + 16][ux]; a[ 5] = Ys[ 5][uy + 16][ux]; a[ 6] = Ys[ 6][uy + 16][ux]; a[ 7] = Ys[ 7][uy + 16][ux];
	a[ 8] = Ys[ 8][uy + 16][ux]; a[ 9] = Ys[ 9][uy + 16][ux]; a[10] = Ys[10][uy + 16][ux]; a[11] = Ys[11][uy + 16][ux];
	a[12] = Ys[12][uy + 16][ux]; a[13] = Ys[13][uy + 16][ux]; a[14] = Ys[14][uy + 16][ux]; a[15] = Ys[15][uy + 16][ux];
	winograd_f9x8_Y(y1, a);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	*(float4*)(&Ys[ux][uy][ 0]) = float4{  v0.z,  v1.z,  v2.z,  v3.z };
	*(float4*)(&Ys[ux][uy][ 4]) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Ys[ux][uy][ 8]) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Ys[ux][uy][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	*(float4*)(&Ys[ux][uy + 16][ 0]) = float4{  v0.w,  v1.w,  v2.w,  v3.w };
	*(float4*)(&Ys[ux][uy + 16][ 4]) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Ys[ux][uy + 16][ 8]) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Ys[ux][uy + 16][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[ 0] = Ys[ 0][uy][ux]; a[ 1] = Ys[ 1][uy][ux]; a[ 2] = Ys[ 2][uy][ux]; a[ 3] = Ys[ 3][uy][ux];
	a[ 4] = Ys[ 4][uy][ux]; a[ 5] = Ys[ 5][uy][ux]; a[ 6] = Ys[ 6][uy][ux]; a[ 7] = Ys[ 7][uy][ux];
	a[ 8] = Ys[ 8][uy][ux]; a[ 9] = Ys[ 9][uy][ux]; a[10] = Ys[10][uy][ux]; a[11] = Ys[11][uy][ux];
	a[12] = Ys[12][uy][ux]; a[13] = Ys[13][uy][ux]; a[14] = Ys[14][uy][ux]; a[15] = Ys[15][uy][ux];
	winograd_f9x8_Y(y2, a);

	a[ 0] = Ys[ 0][uy + 16][ux]; a[ 1] = Ys[ 1][uy + 16][ux]; a[ 2] = Ys[ 2][uy + 16][ux]; a[ 3] = Ys[ 3][uy + 16][ux];
	a[ 4] = Ys[ 4][uy + 16][ux]; a[ 5] = Ys[ 5][uy + 16][ux]; a[ 6] = Ys[ 6][uy + 16][ux]; a[ 7] = Ys[ 7][uy + 16][ux];
	a[ 8] = Ys[ 8][uy + 16][ux]; a[ 9] = Ys[ 9][uy + 16][ux]; a[10] = Ys[10][uy + 16][ux]; a[11] = Ys[11][uy + 16][ux];
	a[12] = Ys[12][uy + 16][ux]; a[13] = Ys[13][uy + 16][ux]; a[14] = Ys[14][uy + 16][ux]; a[15] = Ys[15][uy + 16][ux];
	winograd_f9x8_Y(y3, a);
	__syncthreads();

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
	*(float4*)(Y + Y00 + OC * 8) = float4{ y0[8], y1[8], y2[8], y3[8] };

	//====[oc8 - oc15]===================================================================
	//group2-----------------------------------------------------------------------------
	*(float4*)(&Ys[ux][uy][ 0]) = float4{ v16.x, v17.x, v18.x, v19.x };
	*(float4*)(&Ys[ux][uy][ 4]) = float4{ v20.x, v21.x, v22.x, v23.x };
	*(float4*)(&Ys[ux][uy][ 8]) = float4{ v24.x, v25.x, v26.x, v27.x };
	*(float4*)(&Ys[ux][uy][12]) = float4{ v28.x, v29.x, v30.x, v31.x };

	*(float4*)(&Ys[ux][uy + 16][ 0]) = float4{ v16.y, v17.y, v18.y, v19.y };
	*(float4*)(&Ys[ux][uy + 16][ 4]) = float4{ v20.y, v21.y, v22.y, v23.y };
	*(float4*)(&Ys[ux][uy + 16][ 8]) = float4{ v24.y, v25.y, v26.y, v27.y };
	*(float4*)(&Ys[ux][uy + 16][12]) = float4{ v28.y, v29.y, v30.y, v31.y };
	__syncthreads();

	a[ 0] = Ys[ 0][uy][ux]; a[ 1] = Ys[ 1][uy][ux]; a[ 2] = Ys[ 2][uy][ux]; a[ 3] = Ys[ 3][uy][ux];
	a[ 4] = Ys[ 4][uy][ux]; a[ 5] = Ys[ 5][uy][ux]; a[ 6] = Ys[ 6][uy][ux]; a[ 7] = Ys[ 7][uy][ux];
	a[ 8] = Ys[ 8][uy][ux]; a[ 9] = Ys[ 9][uy][ux]; a[10] = Ys[10][uy][ux]; a[11] = Ys[11][uy][ux];
	a[12] = Ys[12][uy][ux]; a[13] = Ys[13][uy][ux]; a[14] = Ys[14][uy][ux]; a[15] = Ys[15][uy][ux];
	winograd_f9x8_Y(y0, a);
	
	a[ 0] = Ys[ 0][uy + 16][ux]; a[ 1] = Ys[ 1][uy + 16][ux]; a[ 2] = Ys[ 2][uy + 16][ux]; a[ 3] = Ys[ 3][uy + 16][ux];
	a[ 4] = Ys[ 4][uy + 16][ux]; a[ 5] = Ys[ 5][uy + 16][ux]; a[ 6] = Ys[ 6][uy + 16][ux]; a[ 7] = Ys[ 7][uy + 16][ux];
	a[ 8] = Ys[ 8][uy + 16][ux]; a[ 9] = Ys[ 9][uy + 16][ux]; a[10] = Ys[10][uy + 16][ux]; a[11] = Ys[11][uy + 16][ux];
	a[12] = Ys[12][uy + 16][ux]; a[13] = Ys[13][uy + 16][ux]; a[14] = Ys[14][uy + 16][ux]; a[15] = Ys[15][uy + 16][ux];
	winograd_f9x8_Y(y1, a);
	__syncthreads();

	//group3-----------------------------------------------------------------------------
	*(float4*)(&Ys[ux][uy][ 0]) = float4{ v16.z, v17.z, v18.z, v19.z };
	*(float4*)(&Ys[ux][uy][ 4]) = float4{ v20.z, v21.z, v22.z, v23.z };
	*(float4*)(&Ys[ux][uy][ 8]) = float4{ v24.z, v25.z, v26.z, v27.z };
	*(float4*)(&Ys[ux][uy][12]) = float4{ v28.z, v29.z, v30.z, v31.z };

	*(float4*)(&Ys[ux][uy + 16][ 0]) = float4{ v16.w, v17.w, v18.w, v19.w };
	*(float4*)(&Ys[ux][uy + 16][ 4]) = float4{ v20.w, v21.w, v22.w, v23.w };
	*(float4*)(&Ys[ux][uy + 16][ 8]) = float4{ v24.w, v25.w, v26.w, v27.w };
	*(float4*)(&Ys[ux][uy + 16][12]) = float4{ v28.w, v29.w, v30.w, v31.w };
	__syncthreads();

	a[ 0] = Ys[ 0][uy][ux]; a[ 1] = Ys[ 1][uy][ux]; a[ 2] = Ys[ 2][uy][ux]; a[ 3] = Ys[ 3][uy][ux];
	a[ 4] = Ys[ 4][uy][ux]; a[ 5] = Ys[ 5][uy][ux]; a[ 6] = Ys[ 6][uy][ux]; a[ 7] = Ys[ 7][uy][ux];
	a[ 8] = Ys[ 8][uy][ux]; a[ 9] = Ys[ 9][uy][ux]; a[10] = Ys[10][uy][ux]; a[11] = Ys[11][uy][ux];
	a[12] = Ys[12][uy][ux]; a[13] = Ys[13][uy][ux]; a[14] = Ys[14][uy][ux]; a[15] = Ys[15][uy][ux];
	winograd_f9x8_Y(y2, a);

	a[ 0] = Ys[ 0][uy + 16][ux]; a[ 1] = Ys[ 1][uy + 16][ux]; a[ 2] = Ys[ 2][uy + 16][ux]; a[ 3] = Ys[ 3][uy + 16][ux];
	a[ 4] = Ys[ 4][uy + 16][ux]; a[ 5] = Ys[ 5][uy + 16][ux]; a[ 6] = Ys[ 6][uy + 16][ux]; a[ 7] = Ys[ 7][uy + 16][ux];
	a[ 8] = Ys[ 8][uy + 16][ux]; a[ 9] = Ys[ 9][uy + 16][ux]; a[10] = Ys[10][uy + 16][ux]; a[11] = Ys[11][uy + 16][ux];
	a[12] = Ys[12][uy + 16][ux]; a[13] = Ys[13][uy + 16][ux]; a[14] = Ys[14][uy + 16][ux]; a[15] = Ys[15][uy + 16][ux];
	winograd_f9x8_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00          + 8) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC     + 8) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2 + 8) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3 + 8) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4 + 8) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5 + 8) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6 + 8) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7 + 8) = float4{ y0[7], y1[7], y2[7], y3[7] };
	*(float4*)(Y + Y00 + OC * 8 + 8) = float4{ y0[8], y1[8], y2[8], y3[8] };
}

#endif


#endif