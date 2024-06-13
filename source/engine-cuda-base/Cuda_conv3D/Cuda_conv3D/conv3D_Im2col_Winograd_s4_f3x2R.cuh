#pragma once

#ifndef CONV_3D_WINOGRAD_S4_F3X2R_H
#define CONV_3D_WINOGRAD_S4_F3X2R_H

//(1) sh = sw = 1
//(2) FW = 2
//(3) OW % 3: group = 3 elements
//(3) Remode the kernel: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV_3D_WINOGRAD_F3X2R_CALL
#define CONV_3D_WINOGRAD_F3X2R_CALL

//================[Cooperate: 64(OC) * 192(GM)]===============================
//Cooperate: GM = N * OH * ((OW - ow_index) / 3 * 3)

//OWr % 3 == 0
#define conv3dWinograd_f3x2_k64x192RC_tex(stream, ow_index, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr) \
	conv3dWinograd_f3x2_kernel_64x192RC_tex<FH>\
		<<< dim3(OC>>6, ((N*OH)>>6)*(OWr/3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, ow_index)

//OWr % 6 == 0
#define conv3dWinograd_f3x2_k64x192R6C_tex(stream, ow_index, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr) \
	conv3dWinograd_f3x2_kernel_64x192R6C_tex<FH>\
		<<< dim3(OC>>6, ((N*OH)>>5)*(OWr/6)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, ow_index)

//OWr % 2 == 0, FW % 3 == 0
#define conv3dWinograd_SFW_f3x2_k64x192RC_tex(stream, ow_index, X, IH, IW, CW, FH, FW, Y, OH, OW, N, IC, OC, ph, pw, OWr) \
	conv3dWinograd_SFW_f3x2_kernel_64x192RC_tex<FW>\
		<<< dim3(OC>>6, ((N*OH)>>6)*(OWr/3)), dim3(16, 16), 0, stream >>>\
			(X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw, ow_index)

#endif


//================[Cooperate: 64(OC) * 192(GM)]===============================
//LB = 4, IC % 8 == 0, OWr % 3 == 0
#ifndef CONV_3D_WINOGRAD_F3X2_KERNEL_64X192RC_TEXTURE
#define CONV_3D_WINOGRAD_F3X2_KERNEL_64X192RC_TEXTURE

template<int FH>
__global__ void conv3dWinograd_f3x2_kernel_64x192RC_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int ow_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][g(oc): 64]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][d( j): 64]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6); 
	const int Gk = (ty & 7), Gi = (tx << 2) + ((ty > 7) << 1);
	const int toc0 = boc0 + Gi;
	CW += Gk *OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y * 192);//64 * 3 = 192
	const int Xk = (tx & 7), Xi = (ty << 2) + ((tx > 7) << 1);
	const int tj0 = bj0 + Xi * 3, tj4 = tj0 + 3;
	const int OWr3 = (OW - ow_index) / 3 * 3, OH_OWr3 = OH * OWr3;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr3, OWr3); tow0 += ow_index;
	get_n_oh_ow_Temp(tj4, tn4, toh4, tow4, OH_OWr3, OWr3); tow4 += ow_index;
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int tih4 = toh4 - ph, tiw4 = tow4 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]
	const int X4 = ((tn4*IH + tih4)*IW + tiw4)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
	const int GIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int DIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//prepare for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx + ux) * 3, yj1 = yj0 + 12;//3 * 4 = 12
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr3, OWr3); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr3, OWr3); yow1 += ow_index;
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;//ux: j0 -> j3
	const int Y10 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;//ux: j4 -> j7

	//======[compute area1: local]======================================================
	float w[4], x[8]; char ly[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = (fh << 1) * IC * OC, W1 = W0 + IC * OC;
		*(float2*)(w    ) = *(float2*)(CW + W0);//fw = 0
		*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1

		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);//group0
		ly[0] = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		ly[1] = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		ly[2] = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		ly[3] = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1), tX3 = tX0 + IC * 3;
		tX0 = IF_int(ly[0], tX0, -1); tX1 = IF_int(ly[1], tX1, -1);
		tX2 = IF_int(ly[2], tX2, -1); tX3 = IF_int(ly[3], tX3, -1);
		x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
	
		bool lh4 = (tih4 >= -fh) && (tih4 + fh < IH);//group1
		ly[4] = lh4 && (tiw4 >=  0) && (tiw4     < IW);
		ly[5] = lh4 && (tiw4 >= -1) && (tiw4 + 1 < IW);
		ly[6] = lh4 && (tiw4 >= -2) && (tiw4 + 2 < IW);
		ly[7] = lh4 && (tiw4 >= -3) && (tiw4 + 3 < IW);
		int tX4 = X4 + fh * IW * IC;
		int tX5 = tX4 + IC, tX6 = tX4 + (IC << 1), tX7 = tX4 + IC * 3;
		tX4 = IF_int(ly[4], tX4, -1); tX5 = IF_int(ly[5], tX5, -1); 
		tX6 = IF_int(ly[6], tX6, -1); tX7 = IF_int(ly[7], tX7, -1);
		x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5); 
		x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

		//Winograd transform: W(2) -> G(4); X(4) -> D(4)
		float gst0 = 0.5f*(w[0] + w[2]), gst1 = gst0 - w[2];
		float gst2 = 0.5f*(w[1] + w[3]), gst3 = gst2 - w[3];
		float g0[4]{ w[0], gst0, gst1, w[2] };
		float g1[4]{ w[1], gst2, gst3, w[3] };

		float d0[4]{ x[0] - x[2], x[1] + x[2], x[2] - x[1], x[3] - x[1] };
		float d1[4]{ x[4] - x[6], x[5] + x[6], x[6] - x[5], x[7] - x[5] };

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };

		*(float2*)(&Ds[buf][Xk][0][Xi]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Xk][1][Xi]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Xk][2][Xi]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Xk][3][Xi]) = { d0[3], d1[3] };
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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
			buf ^= 1;

			//load 2 group from CW[FH, FW, IC, OC]
			const int W0 = ((fh << 1)*IC + oic)*OC, W1 = W0 + IC * OC;
			*(float2*)(w    ) = *(float2*)(CW + W0);//fw = 0
			*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;//group0
			int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1), tX3 = tX0 + IC * 3;
			tX0 = IF_int(ly[0], tX0, -1); tX1 = IF_int(ly[1], tX1, -1);
			tX2 = IF_int(ly[2], tX2, -1); tX3 = IF_int(ly[3], tX3, -1);
			x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);

			int tX4 = X4 + fh * IW * IC + oic;//group1
			int tX5 = tX4 + IC, tX6 = tX4 + (IC << 1), tX7 = tX4 + IC * 3;
			tX4 = IF_int(ly[4], tX4, -1); tX5 = IF_int(ly[5], tX5, -1);
			tX6 = IF_int(ly[6], tX6, -1); tX7 = IF_int(ly[7], tX7, -1);
			x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

			//Winograd transform: W(2) -> G(4); X(4) -> D(4)
			float gst0 = 0.5f*(w[0] + w[2]), gst1 = gst0 - w[2];
			float gst2 = 0.5f*(w[1] + w[3]), gst3 = gst2 - w[3];
			float g0[4]{ w[0], gst0, gst1, w[2] };
			float g1[4]{ w[1], gst2, gst3, w[3] };

			float d0[4]{ x[0] - x[2], x[1] + x[2], x[2] - x[1], x[3] - x[1] };
			float d1[4]{ x[4] - x[6], x[5] + x[6], x[6] - x[5], x[7] - x[5] };

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };

			*(float2*)(&Ds[buf][Xk][0][Xi]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Xk][1][Xi]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Xk][2][Xi]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Xk][3][Xi]) = { d0[3], d1[3] };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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
		buf ^= 1;
	}
	__syncthreads();

	//======[compute area12: block]======================================================
	float4 (* __restrict__ Ys0)[65][5] = (float4(*)[65][5])(Gs);//4 * 64 * 16
	float4 (* __restrict__ Ys1)[65][5] = (float4(*)[65][5])(Ds);

	float4 a0, a1, a2, a3, a4, a5, a6, a7;
	float4 k0, k1, k2, k3, k4, k5, k6, k7;
	float3 y0, y1, y2, y3, y4, y5, y6, y7;

	//write-turn[0, 1]: j0, j1, j2, j3, [oc0 - oc3], [oc4 - oc7]
	Ys0[ux][uy][0] = v0; Ys1[ux][uy][0] = v1; 
	Ys0[ux][uy][1] = v2; Ys1[ux][uy][1] = v3;
	Ys0[ux][uy][2] = v4; Ys1[ux][uy][2] = v5;
	Ys0[ux][uy][3] = v6; Ys1[ux][uy][3] = v7;
	__syncthreads();

	//read-turn[0, 1]: j0, j1, j2, j3, [oc0 - oc3], [oc4 - oc7]
	a0 = Ys0[0][uy][ux]; a4 = Ys1[0][uy][ux];
	a1 = Ys0[1][uy][ux]; a5 = Ys1[1][uy][ux];
	a2 = Ys0[2][uy][ux]; a6 = Ys1[2][uy][ux];
	a3 = Ys0[3][uy][ux]; a7 = Ys1[3][uy][ux];

	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f3x2_y_f48_64(y0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f3x2_y_f48_64(y1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f3x2_y_f48_64(y2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f3x2_y_f48_64(y3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f3x2_y_f48_64(y4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f3x2_y_f48_64(y5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f3x2_y_f48_64(y6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f3x2_y_f48_64(y7, k7);

	*(float4*)(Y + Y00            ) = { y0.x, y1.x, y2.x, y3.x }; *(float4*)(Y + Y00             + 4) = { y4.x, y5.x, y6.x, y7.x };
	*(float4*)(Y + Y00 + OC       ) = { y0.y, y1.y, y2.y, y3.y }; *(float4*)(Y + Y00 + OC        + 4) = { y4.y, y5.y, y6.y, y7.y };
	*(float4*)(Y + Y00 + (OC << 1)) = { y0.z, y1.z, y2.z, y3.z }; *(float4*)(Y + Y00 + (OC << 1) + 4) = { y4.z, y5.z, y6.z, y7.z };
	__syncthreads();

	//write-turn2: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	Ys0[ux][uy][0] =  v8; Ys1[ux][uy][0] =  v9;
	Ys0[ux][uy][1] = v10; Ys1[ux][uy][1] = v11;
	Ys0[ux][uy][2] = v12; Ys1[ux][uy][2] = v13;
	Ys0[ux][uy][3] = v14; Ys1[ux][uy][3] = v15;
	__syncthreads();

	//write-turn[2, 3]: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	a0 = Ys0[0][uy][ux]; a4 = Ys1[0][uy][ux];
	a1 = Ys0[1][uy][ux]; a5 = Ys1[1][uy][ux];
	a2 = Ys0[2][uy][ux]; a6 = Ys1[2][uy][ux];
	a3 = Ys0[3][uy][ux]; a7 = Ys1[3][uy][ux];

	//read-turn[2, 3]: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f3x2_y_f48_64(y0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f3x2_y_f48_64(y1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f3x2_y_f48_64(y2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f3x2_y_f48_64(y3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f3x2_y_f48_64(y4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f3x2_y_f48_64(y5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f3x2_y_f48_64(y6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f3x2_y_f48_64(y7, k7);

	*(float4*)(Y + Y10            ) = { y0.x, y1.x, y2.x, y3.x }; *(float4*)(Y + Y10             + 4) = { y4.x, y5.x, y6.x, y7.x };
	*(float4*)(Y + Y10 + OC       ) = { y0.y, y1.y, y2.y, y3.y }; *(float4*)(Y + Y10 + OC        + 4) = { y4.y, y5.y, y6.y, y7.y };
	*(float4*)(Y + Y10 + (OC << 1)) = { y0.z, y1.z, y2.z, y3.z }; *(float4*)(Y + Y10 + (OC << 1) + 4) = { y4.z, y5.z, y6.z, y7.z };
}

#endif


//LB = 4, IC % 8 == 0, OWr % 6 == 0
#ifndef CONV_3D_WINOGRAD_F3X2_KERNEL_64X192R6C_TEXTURE
#define CONV_3D_WINOGRAD_F3X2_KERNEL_64X192R6C_TEXTURE

template<int FH>
__global__ void conv3dWinograd_f3x2_kernel_64x192R6C_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int ow_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][g(oc): 64]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][d( j): 64]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6); 
	const int Gk = (ty & 7), Gi = (tx << 2) + ((ty > 7) << 1);
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y * 192);
	const int Xk = (tx & 7), Xi = (ty << 2) + ((tx > 7) << 1);
	const int tj0 = bj0 + Xi * 3;
	const int OWr6 = (OW - ow_index) / 6 * 6, OH_OWr6 = OH * OWr6;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr6, OWr6); tow0 += ow_index;
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
	const int GIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int DIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//prepare for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx + ux) * 3, yj1 = yj0 + 12;//3 * 4 = 12
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr6, OWr6); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr6, OWr6); yow1 += ow_index;
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;//ux: j0 -> j3
	const int Y10 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;//ux: j4 -> j7

	//======[compute area1: local]======================================================
	float w[4], x[7]; char ly[7];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		ly[0] = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		ly[1] = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		ly[2] = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		ly[3] = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		ly[4] = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		ly[5] = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		ly[6] = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC    , tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6;
		tX0 = IF_int(ly[0], tX0, -1); tX1 = IF_int(ly[1], tX1, -1);
		tX2 = IF_int(ly[2], tX2, -1); tX3 = IF_int(ly[3], tX3, -1); 
		tX4 = IF_int(ly[4], tX4, -1); tX5 = IF_int(ly[5], tX5, -1);
		tX6 = IF_int(ly[6], tX6, -1);
		x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1); 
		x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5); 
		x[6] = tex1Dfetch<float>(X, tX6);

		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = (fh << 1) * IC * OC;
		const int W1 = W0 + IC * OC;
		*(float2*)(w    ) = *(float2*)(CW + W0);//fw = 0
		*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1

		//Winograd transform: W(2) -> G(4); X(4) -> D(4)
		float d0[4]{ x[0] - x[2], x[1] + x[2], x[2] - x[1], x[3] - x[1] };
		float d1[4]{ x[3] - x[5], x[4] + x[5], x[5] - x[4], x[6] - x[4] };

		float gst0 = 0.5f*(w[0] + w[2]), gst1 = gst0 - w[2];
		float gst2 = 0.5f*(w[1] + w[3]), gst3 = gst2 - w[3];
		float g0[4]{ w[0], gst0, gst1, w[2] };
		float g1[4]{ w[1], gst2, gst3, w[3] };

		//write to shread memory
		*(float2*)(&Ds[buf][Xk][0][Xi]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Xk][1][Xi]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Xk][2][Xi]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Xk][3][Xi]) = { d0[3], d1[3] };

		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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
			buf ^= 1;

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC    , tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6;
			tX0 = IF_int(ly[0], tX0, -1);
			tX1 = IF_int(ly[1], tX1, -1); tX2 = IF_int(ly[2], tX2, -1);
			tX3 = IF_int(ly[3], tX3, -1); tX4 = IF_int(ly[4], tX4, -1);
			tX5 = IF_int(ly[5], tX5, -1); tX6 = IF_int(ly[6], tX6, -1);
			x[0] = tex1Dfetch<float>(X, tX0);
			x[1] = tex1Dfetch<float>(X, tX1); x[2] = tex1Dfetch<float>(X, tX2);
			x[3] = tex1Dfetch<float>(X, tX3); x[4] = tex1Dfetch<float>(X, tX4);
			x[5] = tex1Dfetch<float>(X, tX5); x[6] = tex1Dfetch<float>(X, tX6);

			//load 2 group from CW[FH, FW, IC, OC]
			const int W0 = ((fh << 1) * IC + oic) * OC;
			const int W1 = W0 + IC * OC;
			*(float2*)(w    ) = *(float2*)(CW + W0);//fw = 0
			*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1

			//Winograd transform: W(2) -> G(4); X(4) -> D(4)
			float d0[4]{ x[0] - x[2], x[1] + x[2], x[2] - x[1], x[3] - x[1] };
			float d1[4]{ x[3] - x[5], x[4] + x[5], x[5] - x[4], x[6] - x[4] };

			float gst0 = 0.5f*(w[0] + w[2]), gst1 = gst0 - w[2];
			float gst2 = 0.5f*(w[1] + w[3]), gst3 = gst2 - w[3];
			float g0[4]{ w[0], gst0, gst1, w[2] };
			float g1[4]{ w[1], gst2, gst3, w[3] };

			//write to shread memory
			*(float2*)(&Ds[buf][Xk][0][Xi]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Xk][1][Xi]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Xk][2][Xi]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Xk][3][Xi]) = { d0[3], d1[3] };

			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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
		buf ^= 1;
	}
	__syncthreads();

	//======[compute area12: block]======================================================
	float4 (* __restrict__ Ys0)[65][5] = (float4(*)[65][5])(Gs);//4 * 64 * 16
	float4 (* __restrict__ Ys1)[65][5] = (float4(*)[65][5])(Ds);

	float4 a0, a1, a2, a3, a4, a5, a6, a7;
	float4 k0, k1, k2, k3, k4, k5, k6, k7;
	float3 y0, y1, y2, y3, y4, y5, y6, y7;

	//write-turn[0, 1]: j0, j1, j2, j3, [oc0 - oc3], [oc4 - oc7]
	Ys0[ux][uy][0] = v0; Ys1[ux][uy][0] = v1; 
	Ys0[ux][uy][1] = v2; Ys1[ux][uy][1] = v3;
	Ys0[ux][uy][2] = v4; Ys1[ux][uy][2] = v5;
	Ys0[ux][uy][3] = v6; Ys1[ux][uy][3] = v7;
	__syncthreads();

	//read-turn[0, 1]: j0, j1, j2, j3, [oc0 - oc3], [oc4 - oc7]
	a0 = Ys0[0][uy][ux]; a4 = Ys1[0][uy][ux];
	a1 = Ys0[1][uy][ux]; a5 = Ys1[1][uy][ux];
	a2 = Ys0[2][uy][ux]; a6 = Ys1[2][uy][ux];
	a3 = Ys0[3][uy][ux]; a7 = Ys1[3][uy][ux];

	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f3x2_y_f48_64(y0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f3x2_y_f48_64(y1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f3x2_y_f48_64(y2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f3x2_y_f48_64(y3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f3x2_y_f48_64(y4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f3x2_y_f48_64(y5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f3x2_y_f48_64(y6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f3x2_y_f48_64(y7, k7);

	*(float4*)(Y + Y00            ) = { y0.x, y1.x, y2.x, y3.x }; *(float4*)(Y + Y00             + 4) = { y4.x, y5.x, y6.x, y7.x };
	*(float4*)(Y + Y00 + OC       ) = { y0.y, y1.y, y2.y, y3.y }; *(float4*)(Y + Y00 + OC        + 4) = { y4.y, y5.y, y6.y, y7.y };
	*(float4*)(Y + Y00 + (OC << 1)) = { y0.z, y1.z, y2.z, y3.z }; *(float4*)(Y + Y00 + (OC << 1) + 4) = { y4.z, y5.z, y6.z, y7.z };
	__syncthreads();

	//write-turn2: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	Ys0[ux][uy][0] =  v8; Ys1[ux][uy][0] =  v9;
	Ys0[ux][uy][1] = v10; Ys1[ux][uy][1] = v11;
	Ys0[ux][uy][2] = v12; Ys1[ux][uy][2] = v13;
	Ys0[ux][uy][3] = v14; Ys1[ux][uy][3] = v15;
	__syncthreads();

	//write-turn[2, 3]: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	a0 = Ys0[0][uy][ux]; a4 = Ys1[0][uy][ux];
	a1 = Ys0[1][uy][ux]; a5 = Ys1[1][uy][ux];
	a2 = Ys0[2][uy][ux]; a6 = Ys1[2][uy][ux];
	a3 = Ys0[3][uy][ux]; a7 = Ys1[3][uy][ux];

	//read-turn[2, 3]: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f3x2_y_f48_64(y0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f3x2_y_f48_64(y1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f3x2_y_f48_64(y2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f3x2_y_f48_64(y3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f3x2_y_f48_64(y4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f3x2_y_f48_64(y5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f3x2_y_f48_64(y6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f3x2_y_f48_64(y7, k7);

	*(float4*)(Y + Y10            ) = { y0.x, y1.x, y2.x, y3.x }; *(float4*)(Y + Y10             + 4) = { y4.x, y5.x, y6.x, y7.x };
	*(float4*)(Y + Y10 + OC       ) = { y0.y, y1.y, y2.y, y3.y }; *(float4*)(Y + Y10 + OC        + 4) = { y4.y, y5.y, y6.y, y7.y };
	*(float4*)(Y + Y10 + (OC << 1)) = { y0.z, y1.z, y2.z, y3.z }; *(float4*)(Y + Y10 + (OC << 1) + 4) = { y4.z, y5.z, y6.z, y7.z };
}

#endif


//Split FW: LB = 4, IC % 8 == 0, OWr % 3 == 0
#ifndef CONV_3D_WINOGRAD_SFW_F3X2_KERNEL_64X192RC_TEXTURE
#define CONV_3D_WINOGRAD_SFW_F3X2_KERNEL_64X192RC_TEXTURE

//for: [FH, FW] = 4, Feature = (24, 24), [N, IC, OC] = [64, 256, 256],
//LB = 4: Size = 72, Time = 10.7845 msec, Performace = 14337.1 GFlop/s

template<int FW>
__global__ void conv3dWinograd_SFW_f3x2_kernel_64x192RC_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW, int FH,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int ow_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][g(oc): 64]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][d( j): 64]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6); 
	const int Gk = (ty & 7), Gi = (tx << 2) + ((ty > 7) << 1);
	const int toc0 = boc0 + Gi;
	CW += Gk *OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y * 192);//64 * 3 = 192
	const int Xk = (tx & 7), Xi = (ty << 2) + ((tx > 7) << 1);
	const int tj0 = bj0 + Xi * 3, tj4 = tj0 + 3;
	const int OWr3 = (OW - ow_index) / 3 * 3, OH_OWr3 = OH * OWr3;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr3, OWr3); tow0 += ow_index;
	get_n_oh_ow_Temp(tj4, tn4, toh4, tow4, OH_OWr3, OWr3); tow4 += ow_index;
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int tih4 = toh4 - ph, tiw4 = tow4 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]
	const int X4 = ((tn4*IH + tih4)*IW + tiw4)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
	const int GIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int DIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//prepare for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx + ux) * 3, yj1 = yj0 + 12;//3 * 4 = 12
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr3, OWr3); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr3, OWr3); yow1 += ow_index;
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;//ux: j0 -> j3
	const int Y10 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;//ux: j4 -> j7

	//======[compute area1: local]======================================================
	float w[4], x[8]; char ly[8]; const int FHW = FH * FW;
	for (int fhw = 0; fhw < FHW; fhw += 2) {
		const int fh = fhw / FW, fw = fhw - fh * FW;

		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = (fh*FW + fw)*IC*OC, W1 = W0 + IC * OC;
		*(float2*)(w    ) = *(float2*)(CW + W0);//fw = 0
		*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1

		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);//group0
		ly[0] = lh0 && (tiw0 + fw >=  0) && (tiw0 + fw     < IW);
		ly[1] = lh0 && (tiw0 + fw >= -1) && (tiw0 + fw + 1 < IW);
		ly[2] = lh0 && (tiw0 + fw >= -2) && (tiw0 + fw + 2 < IW);
		ly[3] = lh0 && (tiw0 + fw >= -3) && (tiw0 + fw + 3 < IW);
		int tX0 = X0 + (fh*IW + fw)*IC;
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1), tX3 = tX0 + IC * 3;
		tX0 = IF_int(ly[0], tX0, -1); tX1 = IF_int(ly[1], tX1, -1);
		tX2 = IF_int(ly[2], tX2, -1); tX3 = IF_int(ly[3], tX3, -1);
		x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
	
		bool lh4 = (tih4 >= -fh) && (tih4 + fh < IH);//group1
		ly[4] = lh4 && (tiw4 + fw >=  0) && (tiw4 + fw     < IW);
		ly[5] = lh4 && (tiw4 + fw >= -1) && (tiw4 + fw + 1 < IW);
		ly[6] = lh4 && (tiw4 + fw >= -2) && (tiw4 + fw + 2 < IW);
		ly[7] = lh4 && (tiw4 + fw >= -3) && (tiw4 + fw + 3 < IW);
		int tX4 = X4 + (fh*IW + fw)*IC;
		int tX5 = tX4 + IC, tX6 = tX4 + (IC << 1), tX7 = tX4 + IC * 3;
		tX4 = IF_int(ly[4], tX4, -1); tX5 = IF_int(ly[5], tX5, -1); 
		tX6 = IF_int(ly[6], tX6, -1); tX7 = IF_int(ly[7], tX7, -1);
		x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5); 
		x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

		//Winograd transform: W(2) -> G(4); X(4) -> D(4)
		float gst0 = 0.5f*(w[0] + w[2]), gst1 = gst0 - w[2];
		float gst2 = 0.5f*(w[1] + w[3]), gst3 = gst2 - w[3];
		float g0[4]{ w[0], gst0, gst1, w[2] };
		float g1[4]{ w[1], gst2, gst3, w[3] };

		float d0[4]{ x[0] - x[2], x[1] + x[2], x[2] - x[1], x[3] - x[1] };
		float d1[4]{ x[4] - x[6], x[5] + x[6], x[6] - x[5], x[7] - x[5] };

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };

		*(float2*)(&Ds[buf][Xk][0][Xi]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Xk][1][Xi]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Xk][2][Xi]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Xk][3][Xi]) = { d0[3], d1[3] };
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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
			buf ^= 1;

			//load 2 group from CW[FH, FW, IC, OC]
			const int W0 = ((fh*FW + fw)*IC + oic)*OC, W1 = W0 + IC * OC;
			*(float2*)(w    ) = *(float2*)(CW + W0);//fw = 0
			*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + (fh*IW + fw)*IC + oic;//group0
			int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1), tX3 = tX0 + IC * 3;
			tX0 = IF_int(ly[0], tX0, -1); tX1 = IF_int(ly[1], tX1, -1);
			tX2 = IF_int(ly[2], tX2, -1); tX3 = IF_int(ly[3], tX3, -1);
			x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);

			int tX4 = X4 + (fh*IW + fw)*IC + oic;//group1
			int tX5 = tX4 + IC, tX6 = tX4 + (IC << 1), tX7 = tX4 + IC * 3;
			tX4 = IF_int(ly[4], tX4, -1); tX5 = IF_int(ly[5], tX5, -1);
			tX6 = IF_int(ly[6], tX6, -1); tX7 = IF_int(ly[7], tX7, -1);
			x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

			//Winograd transform: W(2) -> G(4); X(4) -> D(4)
			float gst0 = 0.5f*(w[0] + w[2]), gst1 = gst0 - w[2];
			float gst2 = 0.5f*(w[1] + w[3]), gst3 = gst2 - w[3];
			float g0[4]{ w[0], gst0, gst1, w[2] };
			float g1[4]{ w[1], gst2, gst3, w[3] };

			float d0[4]{ x[0] - x[2], x[1] + x[2], x[2] - x[1], x[3] - x[1] };
			float d1[4]{ x[4] - x[6], x[5] + x[6], x[6] - x[5], x[7] - x[5] };

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };

			*(float2*)(&Ds[buf][Xk][0][Xi]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Xk][1][Xi]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Xk][2][Xi]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Xk][3][Xi]) = { d0[3], d1[3] };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx    ]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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
		buf ^= 1;
	}
	__syncthreads();

	//======[compute area12: block]======================================================
	float4 (* __restrict__ Ys0)[65][5] = (float4(*)[65][5])(Gs);//4 * 64 * 16
	float4 (* __restrict__ Ys1)[65][5] = (float4(*)[65][5])(Ds);

	float4 a0, a1, a2, a3, a4, a5, a6, a7;
	float4 k0, k1, k2, k3, k4, k5, k6, k7;
	float3 y0, y1, y2, y3, y4, y5, y6, y7;

	//write-turn[0, 1]: j0, j1, j2, j3, [oc0 - oc3], [oc4 - oc7]
	Ys0[ux][uy][0] = v0; Ys1[ux][uy][0] = v1; 
	Ys0[ux][uy][1] = v2; Ys1[ux][uy][1] = v3;
	Ys0[ux][uy][2] = v4; Ys1[ux][uy][2] = v5;
	Ys0[ux][uy][3] = v6; Ys1[ux][uy][3] = v7;
	__syncthreads();

	//read-turn[0, 1]: j0, j1, j2, j3, [oc0 - oc3], [oc4 - oc7]
	a0 = Ys0[0][uy][ux]; a4 = Ys1[0][uy][ux];
	a1 = Ys0[1][uy][ux]; a5 = Ys1[1][uy][ux];
	a2 = Ys0[2][uy][ux]; a6 = Ys1[2][uy][ux];
	a3 = Ys0[3][uy][ux]; a7 = Ys1[3][uy][ux];

	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f3x2_y_f48_64(y0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f3x2_y_f48_64(y1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f3x2_y_f48_64(y2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f3x2_y_f48_64(y3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f3x2_y_f48_64(y4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f3x2_y_f48_64(y5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f3x2_y_f48_64(y6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f3x2_y_f48_64(y7, k7);

	*(float4*)(Y + Y00            ) = { y0.x, y1.x, y2.x, y3.x }; *(float4*)(Y + Y00             + 4) = { y4.x, y5.x, y6.x, y7.x };
	*(float4*)(Y + Y00 + OC       ) = { y0.y, y1.y, y2.y, y3.y }; *(float4*)(Y + Y00 + OC        + 4) = { y4.y, y5.y, y6.y, y7.y };
	*(float4*)(Y + Y00 + (OC << 1)) = { y0.z, y1.z, y2.z, y3.z }; *(float4*)(Y + Y00 + (OC << 1) + 4) = { y4.z, y5.z, y6.z, y7.z };
	__syncthreads();

	//write-turn2: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	Ys0[ux][uy][0] =  v8; Ys1[ux][uy][0] =  v9;
	Ys0[ux][uy][1] = v10; Ys1[ux][uy][1] = v11;
	Ys0[ux][uy][2] = v12; Ys1[ux][uy][2] = v13;
	Ys0[ux][uy][3] = v14; Ys1[ux][uy][3] = v15;
	__syncthreads();

	//write-turn[2, 3]: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	a0 = Ys0[0][uy][ux]; a4 = Ys1[0][uy][ux];
	a1 = Ys0[1][uy][ux]; a5 = Ys1[1][uy][ux];
	a2 = Ys0[2][uy][ux]; a6 = Ys1[2][uy][ux];
	a3 = Ys0[3][uy][ux]; a7 = Ys1[3][uy][ux];

	//read-turn[2, 3]: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f3x2_y_f48_64(y0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f3x2_y_f48_64(y1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f3x2_y_f48_64(y2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f3x2_y_f48_64(y3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f3x2_y_f48_64(y4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f3x2_y_f48_64(y5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f3x2_y_f48_64(y6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f3x2_y_f48_64(y7, k7);

	*(float4*)(Y + Y10            ) = { y0.x, y1.x, y2.x, y3.x }; *(float4*)(Y + Y10             + 4) = { y4.x, y5.x, y6.x, y7.x };
	*(float4*)(Y + Y10 + OC       ) = { y0.y, y1.y, y2.y, y3.y }; *(float4*)(Y + Y10 + OC        + 4) = { y4.y, y5.y, y6.y, y7.y };
	*(float4*)(Y + Y10 + (OC << 1)) = { y0.z, y1.z, y2.z, y3.z }; *(float4*)(Y + Y10 + (OC << 1) + 4) = { y4.z, y5.z, y6.z, y7.z };
}

#endif


#endif