#pragma once

#ifndef CONV_3D_WINOGRAD_S8_F4X5R_H
#define CONV_3D_WINOGRAD_S8_F4X5R_H

//(1) sh = sw = 1
//(2) FW = 5
//(3) OW % 4 == 0: group = 4 elements
//(4) Remode the kernel: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV_3D_WINOGRAD_S8_F4X5R_CALL
#define CONV_3D_WINOGRAD_S8_F4X5R_CALL

//================[Standard: 64(OC) * 128(GM)]===============================
//OW % 4 == 0
#define conv3dWinograd_f4x5_k64x128R_tex(stream, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw) \
	conv3dWinograd_f4x5_kernel_64x128R_tex<FH>\
		<<< dim3(OC>>6, ((N*OH)>>5) * (OW>>2)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw)

//OW % 4 == 0, pw <= 2
#define conv3dWinograd_f4x5_k64x128R_p2_tex(stream, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw) \
	conv3dWinograd_f4x5_kernel_64x128R_p2_tex<FH>\
		<<< dim3(OC>>6, ((N*OH)>>5) * (OW>>2)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw)

//OW % 8 == 0, pw <= 2
#define conv3dWinograd_f4x5_ruse_k64x128R_p2_tex(stream, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw) \
	conv3dWinograd_f4x5_ruse_kernel_64x128R_p2_tex<FH>\
		<<< dim3(OC>>6, ((N*OH)>>5) * (OW>>3<<1)), dim3(16, 8), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw)

//================[Cooperate: 64(OC) * 128(GM)]==============================
//OWr % 4 == 0, pw <= 2
#define conv3dWinograd_f4x5_k64x128RC_p2_tex(stream, ow_index, X, IH, IW, CW, FH, Y, OH, OW, N, IC, OC, ph, pw, OWr) \
	conv3dWinograd_f4x5_kernel_64x128RC_p2_tex<FH>\
		<<< dim3(OC>>6, ((N*OH)>>5) * (OWr>>2)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, ow_index)

#endif


//================[Standard: 64(OC) * 128(GM)]===============================
//LB = 4, IC % 8 == 0, OW % 4 == 0
#ifndef CONV_3D_WINOGRAD_F4X5_KERNEL_64X128R_TEXTURE
#define CONV_3D_WINOGRAD_F4X5_KERNEL_64X128R_TEXTURE

//for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 50, Time = 5.78713 msec, Performace = 18554 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 50, Time = 6.826 msec, Performance = 15730.2 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 50, Time = 5.55205 msec, Performace = 19339.5 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 50, Time = 6.891 msec, Performance = 15581.8 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 50, Time = 5.78469 msec, Performace = 18561.8 GFlop/s
//cuDNN-NCHW-implicit-prec: Size = 50, Time = 6.936 msec, Performance = 15480.7 GFlop/s
//cuDNN-NHWC-implicit-prec:
//for: Feature = ( 8,  8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 50, Time = 5.80941 msec, Performace = 18482.8 GFlop/
//cuDNN-NCHW-implicit-prec: Size = 50, Time = 6.506 msec, Performance = 16503.9 GFlop/s

template<int FH>
__global__ void conv3dWinograd_f4x5_kernel_64x128R_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw)// sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 64]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6);
	const int Gk = (ty & 7), Gi = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk *OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7);//32 * 4 = 128
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + (Xi << 2);
	const int OW4 = OW >> 2 << 2, OH_OW4 = OH * OW4;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OW4, OW4);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Dk]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//4 * 8
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) << 2), yj1 = yj0 + 16;
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OW4, OW4);
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OW4, OW4);
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;//Y00 = yj0 * OC + yoc0;
	const int Y10 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;//Y10 = Y00 + OC * 16;

	//======[compute area1: local]======================================================
	const int Di = (Xi + ((Xk >> 1) << 3)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], x[8], d[8]; char ly[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		ly[0] = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		ly[1] = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		ly[2] = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		ly[3] = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		ly[4] = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		ly[5] = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		ly[6] = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		ly[7] = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;//[n, tih0 + fh, tiw0 + fw, ic]
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6, tX7 = tX0 + IC * 7;
		tX0 = IF_int(ly[0], tX0, -1); tX1 = IF_int(ly[1], tX1, -1);
		tX2 = IF_int(ly[2], tX2, -1); tX3 = IF_int(ly[3], tX3, -1);
		tX4 = IF_int(ly[4], tX4, -1); tX5 = IF_int(ly[5], tX5, -1);
		tX6 = IF_int(ly[6], tX6, -1); tX7 = IF_int(ly[7], tX7, -1);
		x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5);
		x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 5 * IC * OC;//with the same tx
		const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		float2 w3 = *(float2*)(CW + W3);
		float2 w4 = *(float2*)(CW + W4);

		//Winograd transform: W(5) -> G(8); X(8) -> D(8)
		winograd_f4x5_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x);
		winograd_f4x5_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y);
		winograd_f4x5_D(d, x);

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

		Ds[buf][Xk][0][Di] = d[0]; Ds[buf][Xk][1][Di] = d[1];
		Ds[buf][Xk][2][Di] = d[2]; Ds[buf][Xk][3][Di] = d[3];
		Ds[buf][Xk][4][Di] = d[4]; Ds[buf][Xk][5][Di] = d[5];
		Ds[buf][Xk][6][Di] = d[6]; Ds[buf][Xk][7][Di] = d[7];
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

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
			const int W0 = (fh * 5 * IC + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);
			float2 w3 = *(float2*)(CW + W3);
			float2 w4 = *(float2*)(CW + W4);

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6, tX7 = tX0 + IC * 7;
			tX0 = IF_int(ly[0], tX0, -1); tX1 = IF_int(ly[1], tX1, -1);
			tX2 = IF_int(ly[2], tX2, -1); tX3 = IF_int(ly[3], tX3, -1);
			tX4 = IF_int(ly[4], tX4, -1); tX5 = IF_int(ly[5], tX5, -1);
			tX6 = IF_int(ly[6], tX6, -1); tX7 = IF_int(ly[7], tX7, -1);
			x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
			x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

			//Winograd transform: W(5) -> G(8); X(8) -> D(8)
			winograd_f4x5_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x);
			winograd_f4x5_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y);
			winograd_f4x5_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

			Ds[buf][Xk][0][Di] = d[0]; Ds[buf][Xk][1][Di] = d[1];
			Ds[buf][Xk][2][Di] = d[2]; Ds[buf][Xk][3][Di] = d[3];
			Ds[buf][Xk][4][Di] = d[4]; Ds[buf][Xk][5][Di] = d[5];
			Ds[buf][Xk][6][Di] = d[6]; Ds[buf][Xk][7][Di] = d[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31)    ]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

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

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Ys)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float y0[4], y1[4], y2[4], y3[4];
	__syncthreads();

	//------write-read-turn0: {oc0 - oc3 | oc4 - oc7} x {j0 - j3}------------------------
	*(float4*)(&Ys[ux][uy][0]) = { v0.x, v0.y, v1.x, v1.y };//{oc0, oc1}, {oc4, oc5}}
	*(float4*)(&Ys[ux][uy][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Ys[ux][uy][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Ys[ux][uy][6]) = { v6.x, v6.y, v7.x, v7.y };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux];
	winograd_f4x5_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f4x5_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Ys[ux][uy][0]) = { v0.z, v0.w, v1.z, v1.w };//{oc2, oc3}, {oc6, oc7}}
	*(float4*)(&Ys[ux][uy][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Ys[ux][uy][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Ys[ux][uy][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux]; 
	winograd_f4x5_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f4x5_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	__syncthreads();
	
	//------write-read-turn0: {oc0 - oc3 | oc4 - oc7} x {j4 - j7}------------------------
	*(float4*)(&Ys[ux][uy][0]) = { v8.x,  v8.y,  v9.x,  v9.y };//{oc0, oc1}, {oc4, oc5}}
	*(float4*)(&Ys[ux][uy][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Ys[ux][uy][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Ys[ux][uy][6]) = { v14.x, v14.y, v15.x, v15.y };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux];
	winograd_f4x5_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f4x5_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Ys[ux][uy][0]) = { v8.z,  v8.w,  v9.z,  v9.w };//{oc2, oc3}, {oc6, oc7}}
	*(float4*)(&Ys[ux][uy][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Ys[ux][uy][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Ys[ux][uy][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux];
	winograd_f4x5_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f4x5_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
}

#endif


//LB = 4, IC % 8 == 0, OW % 4 == 0, pw <= 2
#ifndef CONV_3D_WINOGRAD_F4X5_KERNEL_64X128R_P2_TEXTURE
#define CONV_3D_WINOGRAD_F4X5_KERNEL_64X128R_P2_TEXTURE

//for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 50, Time = 5.72897 msec, Performace = 18742.3 GFlop/s
//WB = 4: Size = 50, Time = 5.27933 msec, Performace = 20338.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 7.603 msec, Performance = 14122.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.889 msec, Performance = 15586.3 GFlop/s

//for: Feature = (64, 64), [N, IC, OC] = [32, 128, 128]
//LB = 4: Size = 50, Time = 5.74325 msec, Performace = 18695.7 GFlop/s
//WB = 4: Size = 50, Time = 5.27971 msec, Performace = 20337.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.826 msec, Performance = 15730.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.466 msec, Performance = 16605.9 GFlop/s

//for: Feature = (48, 48), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 112.5, Time = 12.5088 msec, Performace = 19313.7 GFlop/s
//WB = 4: Size = 112.5, Time = 11.7694 msec, Performace = 20527.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 112.5, Time = 14.333 msec, Performance = 16855.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 112.5, Time = 14.416 msec, Performance = 16758.6 GFlop/s

//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 50, Time = 5.50963 msec, Performace = 19488.4 GFlop/s
//WB = 4: Size = 50, Time = 5.12471 msec, Performace = 20952.2 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.891 msec, Performance = 15581.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.788 msec, Performance = 15818.2// GFlop/s

//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 50, Time = 5.37508 msec, Performace = 19976.3 GFlop/s
//WB = 4: Size = 50, Time = 4.9361  msec, Performace = 21752.8 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.936 msec, Performance = 15480.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.442 msec, Performance = 16667.8 GFlop/s

//for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 50, Time = 5.59623 msec, Performace = 19186.9 GFlop/s
//WB = 4: Size = 50, Time = 5.07147 msec, Performace = 21172.2 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.506 msec, Performance = 16503.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.254 msec, Performance = 17168.8 GFlop/s

template<int FH>
__global__ void conv3dWinograd_f4x5_kernel_64x128R_p2_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 64]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 6);
	const int Gk = (ty & 7), Gi = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk *OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (by << 7);//32 group of Y
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + (Xi << 2);
	const int OW4 = OW >> 2 << 2, OH_OW4 = OH * OW4;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OW4, OW4);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Dk]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//8 * 32
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) << 2), yj1 = yj0 + 16;
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OW4, OW4);
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OW4, OW4);
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;//Y00 = yj0 * OC + yoc0;
	const int Y10 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;//Y10 = Y00 + OC * 16;

	//======[compute area1: local]======================================================
	const int Di = (Xi + ((Xk >> 1) << 3)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], x[8], d[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);//pw = -2
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);//pw = -1
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);//pw = +1
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);//pw = +2
		int tX0 = X0 + fh * IW * IC;//[n, tih0 + fh, tiw0 + fw, ic]
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6, tX7 = tX0 + IC * 7;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(lh0, tX2, -1); tX3 = IF_int(lh0, tX3, -1);
		tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
		tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
		x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5);
		x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 5 * IC * OC;
		const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		float2 w3 = *(float2*)(CW + W3);
		float2 w4 = *(float2*)(CW + W4);

		//Winograd transform: W(5) -> G(8); X(8) -> D(8)
		winograd_f4x5_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x);
		winograd_f4x5_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y);
		winograd_f4x5_D(d, x);

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

		Ds[buf][Xk][0][Di] = d[0]; Ds[buf][Xk][1][Di] = d[1];
		Ds[buf][Xk][2][Di] = d[2]; Ds[buf][Xk][3][Di] = d[3];
		Ds[buf][Xk][4][Di] = d[4]; Ds[buf][Xk][5][Di] = d[5];
		Ds[buf][Xk][6][Di] = d[6]; Ds[buf][Xk][7][Di] = d[7];
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

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
			int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6, tX7 = tX0 + IC * 7;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(lh0, tX2, -1); tX3 = IF_int(lh0, tX3, -1);
			tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
			tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
			x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
			x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

			//load 2 group from CW[FH, FW, IC, OC]
			const int W0 = (fh * 5 * IC + oic) * OC;
			const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);
			float2 w3 = *(float2*)(CW + W3);
			float2 w4 = *(float2*)(CW + W4);

			//Winograd transform: W(5) -> G(8); X(8) -> D(8)
			winograd_f4x5_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x);
			winograd_f4x5_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y);
			winograd_f4x5_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

			Ds[buf][Xk][0][Di] = d[0]; Ds[buf][Xk][1][Di] = d[1];
			Ds[buf][Xk][2][Di] = d[2]; Ds[buf][Xk][3][Di] = d[3];
			Ds[buf][Xk][4][Di] = d[4]; Ds[buf][Xk][5][Di] = d[5];
			Ds[buf][Xk][6][Di] = d[6]; Ds[buf][Xk][7][Di] = d[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31)    ]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

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

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Ys)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float y0[4], y1[4], y2[4], y3[4];
	__syncthreads();

	//------write-read-turn0: {oc0 - oc3 | oc4 - oc7} x {j0 - j3}------------------------
	*(float4*)(&Ys[ux][uy][0]) = { v0.x, v0.y, v1.x, v1.y };//{oc0, oc1}, {oc4, oc5}}
	*(float4*)(&Ys[ux][uy][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Ys[ux][uy][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Ys[ux][uy][6]) = { v6.x, v6.y, v7.x, v7.y };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux];
	winograd_f4x5_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f4x5_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Ys[ux][uy][0]) = { v0.z, v0.w, v1.z, v1.w };//{oc2, oc3}, {oc6, oc7}}
	*(float4*)(&Ys[ux][uy][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Ys[ux][uy][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Ys[ux][uy][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux]; 
	winograd_f4x5_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f4x5_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	__syncthreads();
	
	//------write-read-turn0: {oc0 - oc3 | oc4 - oc7} x {j4 - j7}------------------------
	*(float4*)(&Ys[ux][uy][0]) = { v8.x,  v8.y,  v9.x,  v9.y };//{oc0, oc1}, {oc4, oc5}}
	*(float4*)(&Ys[ux][uy][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Ys[ux][uy][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Ys[ux][uy][6]) = { v14.x, v14.y, v15.x, v15.y };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux];
	winograd_f4x5_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f4x5_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Ys[ux][uy][0]) = { v8.z,  v8.w,  v9.z,  v9.w };//{oc2, oc3}, {oc6, oc7}}
	*(float4*)(&Ys[ux][uy][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Ys[ux][uy][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Ys[ux][uy][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux];
	winograd_f4x5_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f4x5_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
}

#endif


//RUSE: LB = 4, IC % 8 == 0, OW % 8 == 0, pw <= 2
#ifndef CONV_3D_WINOGRAD_F4X5_RUSE_KERNEL_64X128R_P2_TEXTURE
#define CONV_3D_WINOGRAD_F4X5_RUSE_KERNEL_64X128R_P2_TEXTURE

//for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 50, Time = 5.01505 msec, Performace = 21410.4 GFlop/s
//WB = 4: Size = 50, Time = 4.54375 msec, Performace = 23631.2 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 7.603 msec, Performance = 14122.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.889 msec, Performance = 15586.3 GFlop/s

//for: Feature = (64, 64), [N, IC, OC] = [32, 128, 128]
//LB = 4: Size = 50, Time = 5.11399 msec, Performace = 20996.2 GFlop/s
//WB = 4: Size = 50, Time = 4.643   msec, Performace = 23126   GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.826 msec, Performance = 15730.2 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.466 msec, Performance = 16605.9 GFlop/s

//for: Feature = (48, 48), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 112.5, Time = 11.6415 msec, Performace = 20752.7 GFlop/s
//WB = 4: Size = 112.5, Time = 10.5723 msec, Performace = 22851.3 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 112.5, Time = 14.333 msec, Performance = 16855.6 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 112.5, Time = 14.416 msec, Performance = 16758.6 GFlop/s

//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 50, Time = 5.05927 msec, Performace = 21223.2 GFlop/s
//WB = 4: Size = 50, Time = 4.6013  msec, Performace = 23335.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.891 msec, Performance = 15581.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.788 msec, Performance = 15818.2// GFlop/s

//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 50, Time = 4.93715 msec, Performace = 21748.2 GFlop/s
//WB = 4: Size = 50, Time = 4.38719 msec, Performace = 24474.5 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.936 msec, Performance = 15480.7 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.442 msec, Performance = 16667.8 GFlop/s

//for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 50, Time = 4.91652 msec, Performace = 21839.5 GFlop/s
//WB = 4: Size = 50, Time = 4.32837 msec, Performace = 24807.1 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.506 msec, Performance = 16503.9 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.254 msec, Performance = 17168.8 GFlop/s

template<int FH>
__global__ void conv3dWinograd_f4x5_ruse_kernel_64x128R_p2_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 64]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

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
	const int Gk = ty, Gi = tx << 2;//[8, 16*4 = 64]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (by << 7);//32 group of Y
	const int Xk = (tx & 7), Xi = ((ty << 1) + (tx > 7)) << 1;//[8, 16*2 = 32]
	const int tj0 = bj0 + (Xi << 2);
	const int OW8 = OW >> 3 << 3, OH_OW8 = OH * OW8;//4 * 2 = 8
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OW8, OW8);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Dk]

	//prepare for threadIdx
	const int GIdx = ((tx & 1) + ((tx >> 3) << 1)) << 4;//4
	const int DIdx = ((tx & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ty & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ty >> 1)) << 2), yj1 = yj0 + 16;
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OW8, OW8);
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OW8, OW8);
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;//Y00 = yj0 * OC + yoc0;
	const int Y10 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;//Y10 = Y00 + OC * 16;

	//======[compute area1: local]======================================================
	const int Di = (Xi + ((Xk >> 1) << 3)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], g2[8], g3[8], x[12], d0[8], d1[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=   0) && (tiw0      < IW);//pw = -2
		bool ly1 = lh0 && (tiw0 >=  -1) && (tiw0 +  1 < IW);//pw = -1
		bool lyA = lh0 && (tiw0 >= -10) && (tiw0 + 10 < IW);//pw = +1
		bool lyB = lh0 && (tiw0 >= -11) && (tiw0 + 11 < IW);//pw = +2
		int tX0 = X0 + fh * IW * IC, tX1 = tX0 + IC;
		int tX2 = tX0 + (IC << 1), tX3 = tX0 + IC *  3;
		int tX4 = tX0 + (IC << 2), tX5 = tX0 + IC *  5;
		int tX6 = tX0 + IC * 6   , tX7 = tX0 + IC *  7;
		int tX8 = tX0 + (IC << 3), tX9 = tX0 + IC *  9;
		int tXA = tX0 + IC * 10  , tXB = tX0 + IC * 11;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(lh0, tX2, -1); tX3 = IF_int(lh0, tX3, -1);
		tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
		tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
		tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
		tXA = IF_int(lyA, tXA, -1); tXB = IF_int(lyB, tXB, -1);
		x[0x0] = tex1Dfetch<float>(X, tX0); x[0x1] = tex1Dfetch<float>(X, tX1);
		x[0x2] = tex1Dfetch<float>(X, tX2); x[0x3] = tex1Dfetch<float>(X, tX3);
		x[0x4] = tex1Dfetch<float>(X, tX4); x[0x5] = tex1Dfetch<float>(X, tX5);
		x[0x6] = tex1Dfetch<float>(X, tX6); x[0x7] = tex1Dfetch<float>(X, tX7);
		x[0x8] = tex1Dfetch<float>(X, tX8); x[0x9] = tex1Dfetch<float>(X, tX9);
		x[0xA] = tex1Dfetch<float>(X, tXA); x[0xB] = tex1Dfetch<float>(X, tXB);

		//load 4 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 5 * IC * OC;
		const int W1 = W0 + IC * OC    , W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		float4 w0 = *(float4*)(CW + W0);
		float4 w1 = *(float4*)(CW + W1);
		float4 w2 = *(float4*)(CW + W2);
		float4 w3 = *(float4*)(CW + W3);
		float4 w4 = *(float4*)(CW + W4);

		//Winograd transform: W(5) -> G(8); X(8) -> D(8)
		winograd_f4x5_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x);
		winograd_f4x5_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y);
		winograd_f4x5_G(g2, w0.z, w1.z, w2.z, w3.z, w4.z);
		winograd_f4x5_G(g3, w0.w, w1.w, w2.w, w3.w, w4.w);
		winograd_f4x5_D(d0, x);//x[0 - 7]
		winograd_f4x5_D_oft(d1, x, 4);//x[4 - 11]

		//write to shread memory
		*(float2*)(&Ds[buf][Xk][0][Di]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Xk][1][Di]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Xk][2][Di]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Xk][3][Di]) = { d0[3], d1[3] };
		*(float2*)(&Ds[buf][Xk][4][Di]) = { d0[4], d1[4] };
		*(float2*)(&Ds[buf][Xk][5][Di]) = { d0[5], d1[5] };
		*(float2*)(&Ds[buf][Xk][6][Di]) = { d0[6], d1[6] };
		*(float2*)(&Ds[buf][Xk][7][Di]) = { d0[7], d1[7] };

		*(float4*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0], g2[0], g3[0] };
		*(float4*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1], g2[1], g3[1] };
		*(float4*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2], g2[2], g3[2] };
		*(float4*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3], g2[3], g3[3] };
		*(float4*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4], g2[4], g3[4] };
		*(float4*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5], g2[5], g3[5] };
		*(float4*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6], g2[6], g3[6] };
		*(float4*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7], g2[7], g3[7] };
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 b0 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

				//=======================================================
				float4 a0 = *(float4*)(&Gs[buf][ik][ty][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ty][GIdx + 4]);

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
				float4 a2 = *(float4*)(&Gs[buf][ik][ty][GIdx +  8]);
				float4 a3 = *(float4*)(&Gs[buf][ik][ty][GIdx + 12]);

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
			buf ^= 1;

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic, tX1 = tX0 + IC;
			int tX2 = tX0 + (IC << 1), tX3 = tX0 + IC *  3;
			int tX4 = tX0 + (IC << 2), tX5 = tX0 + IC *  5;
			int tX6 = tX0 + IC * 6   , tX7 = tX0 + IC *  7;
			int tX8 = tX0 + (IC << 3), tX9 = tX0 + IC *  9;
			int tXA = tX0 + IC * 10  , tXB = tX0 + IC * 11;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(lh0, tX2, -1); tX3 = IF_int(lh0, tX3, -1);
			tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
			tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
			tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
			tXA = IF_int(lyA, tXA, -1); tXB = IF_int(lyB, tXB, -1);
			x[0x0] = tex1Dfetch<float>(X, tX0); x[0x1] = tex1Dfetch<float>(X, tX1);
			x[0x2] = tex1Dfetch<float>(X, tX2); x[0x3] = tex1Dfetch<float>(X, tX3);
			x[0x4] = tex1Dfetch<float>(X, tX4); x[0x5] = tex1Dfetch<float>(X, tX5);
			x[0x6] = tex1Dfetch<float>(X, tX6); x[0x7] = tex1Dfetch<float>(X, tX7);
			x[0x8] = tex1Dfetch<float>(X, tX8); x[0x9] = tex1Dfetch<float>(X, tX9);
			x[0xA] = tex1Dfetch<float>(X, tXA); x[0xB] = tex1Dfetch<float>(X, tXB);

			//load 4 group from CW[FH, FW, IC, OC]
			const int W0 = (fh * 5 * IC + oic) * OC;
			const int W1 = W0 + IC * OC    , W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			float4 w0 = *(float4*)(CW + W0);
			float4 w1 = *(float4*)(CW + W1);
			float4 w2 = *(float4*)(CW + W2);
			float4 w3 = *(float4*)(CW + W3);
			float4 w4 = *(float4*)(CW + W4);

			//Winograd transform: W(5) -> G(8); X(8) -> D(8)
			winograd_f4x5_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x);
			winograd_f4x5_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y);
			winograd_f4x5_G(g2, w0.z, w1.z, w2.z, w3.z, w4.z);
			winograd_f4x5_G(g3, w0.w, w1.w, w2.w, w3.w, w4.w);
			winograd_f4x5_D(d0, x);//x[0 - 7]
			winograd_f4x5_D_oft(d1, x, 4);//x[4 - 11]

			//write to shread memory
			*(float2*)(&Ds[buf][Xk][0][Di]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Xk][1][Di]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Xk][2][Di]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Xk][3][Di]) = { d0[3], d1[3] };
			*(float2*)(&Ds[buf][Xk][4][Di]) = { d0[4], d1[4] };
			*(float2*)(&Ds[buf][Xk][5][Di]) = { d0[5], d1[5] };
			*(float2*)(&Ds[buf][Xk][6][Di]) = { d0[6], d1[6] };
			*(float2*)(&Ds[buf][Xk][7][Di]) = { d0[7], d1[7] };

			*(float4*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0], g2[0], g3[0] };
			*(float4*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1], g2[1], g3[1] };
			*(float4*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2], g2[2], g3[2] };
			*(float4*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3], g2[3], g3[3] };
			*(float4*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4], g2[4], g3[4] };
			*(float4*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5], g2[5], g3[5] };
			*(float4*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6], g2[6], g3[6] };
			*(float4*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7], g2[7], g3[7] };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 b0 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31)    ]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

			//=======================================================
			float4 a0 = *(float4*)(&Gs[buf][ik][ty][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[buf][ik][ty][GIdx + 4]);

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
			float4 a2 = *(float4*)(&Gs[buf][ik][ty][GIdx +  8]);
			float4 a3 = *(float4*)(&Gs[buf][ik][ty][GIdx + 12]);

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
		buf ^= 1;
	}

	//======[compute area2: block]=======================================================
	float2(*__restrict__ Ys)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float y0[4], y1[4], y2[4], y3[4];
	__syncthreads();
	
	//------write-read-turn0: {oc0 - oc7} x {j0 - j3}------------------------------------
	*(float4*)(&Ys[ty][tx][0]) = { v0.x, v0.y, v1.x, v1.y };
	*(float4*)(&Ys[ty][tx][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Ys[ty][tx][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Ys[ty][tx][6]) = { v6.x, v6.y, v7.x, v7.y };

	*(float4*)(&Ys[ty][tx + 16][0]) = { v0.z, v0.w, v1.z, v1.w };
	*(float4*)(&Ys[ty][tx + 16][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Ys[ty][tx + 16][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Ys[ty][tx + 16][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Ys[0][tx][ty]; a[1] = Ys[1][tx][ty];
	a[2] = Ys[2][tx][ty]; a[3] = Ys[3][tx][ty];
	a[4] = Ys[4][tx][ty]; a[5] = Ys[5][tx][ty];
	a[6] = Ys[6][tx][ty]; a[7] = Ys[7][tx][ty];
	winograd_f4x5_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f4x5_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }

	a[0] = Ys[0][tx + 16][ty]; a[1] = Ys[1][tx + 16][ty];
	a[2] = Ys[2][tx + 16][ty]; a[3] = Ys[3][tx + 16][ty];
	a[4] = Ys[4][tx + 16][ty]; a[5] = Ys[5][tx + 16][ty];
	a[6] = Ys[6][tx + 16][ty]; a[7] = Ys[7][tx + 16][ty];
	winograd_f4x5_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f4x5_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	__syncthreads();

	//------write-read-turn0: {oc8 - oc15} x {j0 - j3}-----------------------------------
	*(float4*)(&Ys[ty][tx][0]) = { v16.x, v16.y, v17.x, v17.y };
	*(float4*)(&Ys[ty][tx][2]) = { v18.x, v18.y, v19.x, v19.y };
	*(float4*)(&Ys[ty][tx][4]) = { v20.x, v20.y, v21.x, v21.y };
	*(float4*)(&Ys[ty][tx][6]) = { v22.x, v22.y, v23.x, v23.y };

	*(float4*)(&Ys[ty][tx + 16][0]) = { v16.z, v16.w, v17.z, v17.w };
	*(float4*)(&Ys[ty][tx + 16][2]) = { v18.z, v18.w, v19.z, v19.w };
	*(float4*)(&Ys[ty][tx + 16][4]) = { v20.z, v20.w, v21.z, v21.w };
	*(float4*)(&Ys[ty][tx + 16][6]) = { v22.z, v22.w, v23.z, v23.w };
	__syncthreads();

	a[0] = Ys[0][tx][ty]; a[1] = Ys[1][tx][ty];
	a[2] = Ys[2][tx][ty]; a[3] = Ys[3][tx][ty];
	a[4] = Ys[4][tx][ty]; a[5] = Ys[5][tx][ty];
	a[6] = Ys[6][tx][ty]; a[7] = Ys[7][tx][ty];
	winograd_f4x5_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f4x5_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }

	a[0] = Ys[0][tx + 16][ty]; a[1] = Ys[1][tx + 16][ty];
	a[2] = Ys[2][tx + 16][ty]; a[3] = Ys[3][tx + 16][ty];
	a[4] = Ys[4][tx + 16][ty]; a[5] = Ys[5][tx + 16][ty];
	a[6] = Ys[6][tx + 16][ty]; a[7] = Ys[7][tx + 16][ty];
	winograd_f4x5_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f4x5_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y00          + 8) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC * 1 + 8) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2 + 8) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3 + 8) = { y0[3], y1[3], y2[3], y3[3] };
	__syncthreads();

	//------write-read-turn0: {oc0 - oc7} x {j4 - j7}------------------------------------
	*(float4*)(&Ys[ty][tx][0]) = {  v8.x,  v8.y,  v9.x,  v9.y };
	*(float4*)(&Ys[ty][tx][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Ys[ty][tx][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Ys[ty][tx][6]) = { v14.x, v14.y, v15.x, v15.y };

	*(float4*)(&Ys[ty][tx + 16][0]) = { v8.z,  v8.w,  v9.z,  v9.w };
	*(float4*)(&Ys[ty][tx + 16][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Ys[ty][tx + 16][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Ys[ty][tx + 16][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Ys[0][tx][ty]; a[1] = Ys[1][tx][ty];
	a[2] = Ys[2][tx][ty]; a[3] = Ys[3][tx][ty];
	a[4] = Ys[4][tx][ty]; a[5] = Ys[5][tx][ty];
	a[6] = Ys[6][tx][ty]; a[7] = Ys[7][tx][ty];
	winograd_f4x5_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f4x5_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }

	a[0] = Ys[0][tx + 16][ty]; a[1] = Ys[1][tx + 16][ty];
	a[2] = Ys[2][tx + 16][ty]; a[3] = Ys[3][tx + 16][ty];
	a[4] = Ys[4][tx + 16][ty]; a[5] = Ys[5][tx + 16][ty];
	a[6] = Ys[6][tx + 16][ty]; a[7] = Ys[7][tx + 16][ty];
	winograd_f4x5_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f4x5_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	__syncthreads();

	//------write-read-turn0: {oc8 - oc15} x {j4 - j7}-----------------------------------
	*(float4*)(&Ys[ty][tx][0]) = { v24.x, v24.y, v25.x, v25.y };
	*(float4*)(&Ys[ty][tx][2]) = { v26.x, v26.y, v27.x, v27.y };
	*(float4*)(&Ys[ty][tx][4]) = { v28.x, v28.y, v29.x, v29.y };
	*(float4*)(&Ys[ty][tx][6]) = { v30.x, v30.y, v31.x, v31.y };

	*(float4*)(&Ys[ty][tx + 16][0]) = { v24.z, v24.w, v25.z, v25.w };
	*(float4*)(&Ys[ty][tx + 16][2]) = { v26.z, v26.w, v27.z, v27.w };
	*(float4*)(&Ys[ty][tx + 16][4]) = { v28.z, v28.w, v29.z, v29.w };
	*(float4*)(&Ys[ty][tx + 16][6]) = { v30.z, v30.w, v31.z, v31.w };
	__syncthreads();

	a[0] = Ys[0][tx][ty]; a[1] = Ys[1][tx][ty];
	a[2] = Ys[2][tx][ty]; a[3] = Ys[3][tx][ty];
	a[4] = Ys[4][tx][ty]; a[5] = Ys[5][tx][ty];
	a[6] = Ys[6][tx][ty]; a[7] = Ys[7][tx][ty];
	winograd_f4x5_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f4x5_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }

	a[0] = Ys[0][tx + 16][ty]; a[1] = Ys[1][tx + 16][ty];
	a[2] = Ys[2][tx + 16][ty]; a[3] = Ys[3][tx + 16][ty];
	a[4] = Ys[4][tx + 16][ty]; a[5] = Ys[5][tx + 16][ty];
	a[6] = Ys[6][tx + 16][ty]; a[7] = Ys[7][tx + 16][ty];
	winograd_f4x5_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f4x5_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y10          + 8) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y10 + OC * 1 + 8) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y10 + OC * 2 + 8) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y10 + OC * 3 + 8) = { y0[3], y1[3], y2[3], y3[3] };
}

#endif


//================[Cooperate: 64(OC) * 128(GM)]==============================
//LB = 4, IC % 8 == 0, OWr % 4 == 0, pw <= 2
#ifndef CONV_3D_WINOGRAD_F4X5_KERNEL_64X128RC_P2_TEXTURE
#define CONV_3D_WINOGRAD_F4X5_KERNEL_64X128RC_P2_TEXTURE

//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 50, Time = 5.46443 msec, Performace = 19649.7 GFlop/s
//WB = 4: Size = 50, Time = 5.13638 msec, Performace = 20904.6 GFlop/s
//cuDNN-NCHW-GEMM-implicit-prec: Size = 50, Time = 6.891 msec, Performance = 15581.8 GFlop/s
//cuDNN-NHWC-GEMM-implicit-prec: Size = 50, Time = 6.788 msec, Performance = 15818.2// GFlop/s

template<int FH>
__global__ void conv3dWinograd_f4x5_kernel_64x128RC_p2_tex(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int ow_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 64]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 6);
	const int Gk = (ty & 7), Gi = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk *OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (by << 7);//32 group of Y
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + (Xi << 2);
	const int OWr4 = (OW - ow_index) >> 2 << 2, OH_OWr4 = OH * OWr4;
	get_n_oh_ow_Temp(tj0, tn0, toh0, tow0, OH_OWr4, OWr4); tow0 += ow_index;
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Dk]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//8 * 32
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) << 2), yj1 = yj0 + 16;
	get_n_oh_ow_Temp(yj0, yn0, yoh0, yow0, OH_OWr4, OWr4); yow0 += ow_index;
	get_n_oh_ow_Temp(yj1, yn1, yoh1, yow1, OH_OWr4, OWr4); yow1 += ow_index;
	const int Y00 = ((yn0*OH + yoh0)*OW + yow0)*OC + yoc0;//Y00 = yj0 * OC + yoc0;
	const int Y10 = ((yn1*OH + yoh1)*OW + yow1)*OC + yoc0;//Y10 = Y00 + OC * 16;

	//======[compute area1: local]======================================================
	const int Di = (Xi + ((Xk >> 1) << 3)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], x[8], d[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);//pw = -2
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);//pw = -1
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);//pw = +1
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);//pw = +2
		int tX0 = X0 + fh * IW * IC;//[n, tih0 + fh, tiw0 + fw, ic]
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6, tX7 = tX0 + IC * 7;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(lh0, tX2, -1); tX3 = IF_int(lh0, tX3, -1);
		tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
		tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
		x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5);
		x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 5 * IC * OC;
		const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		float2 w3 = *(float2*)(CW + W3);
		float2 w4 = *(float2*)(CW + W4);

		//Winograd transform: W(5) -> G(8); X(8) -> D(8)
		winograd_f4x5_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x);
		winograd_f4x5_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y);
		winograd_f4x5_D(d, x);

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

		Ds[buf][Xk][0][Di] = d[0]; Ds[buf][Xk][1][Di] = d[1];
		Ds[buf][Xk][2][Di] = d[2]; Ds[buf][Xk][3][Di] = d[3];
		Ds[buf][Xk][4][Di] = d[4]; Ds[buf][Xk][5][Di] = d[5];
		Ds[buf][Xk][6][Di] = d[6]; Ds[buf][Xk][7][Di] = d[7];
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

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
			int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6, tX7 = tX0 + IC * 7;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(lh0, tX2, -1); tX3 = IF_int(lh0, tX3, -1);
			tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
			tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
			x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
			x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

			//load 2 group from CW[FH, FW, IC, OC]
			const int W0 = (fh * 5 * IC + oic) * OC;
			const int W1 = W0 + IC * OC, W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);
			float2 w3 = *(float2*)(CW + W3);
			float2 w4 = *(float2*)(CW + W4);

			//Winograd transform: W(5) -> G(8); X(8) -> D(8)
			winograd_f4x5_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x);
			winograd_f4x5_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y);
			winograd_f4x5_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

			Ds[buf][Xk][0][Di] = d[0]; Ds[buf][Xk][1][Di] = d[1];
			Ds[buf][Xk][2][Di] = d[2]; Ds[buf][Xk][3][Di] = d[3];
			Ds[buf][Xk][4][Di] = d[4]; Ds[buf][Xk][5][Di] = d[5];
			Ds[buf][Xk][6][Di] = d[6]; Ds[buf][Xk][7][Di] = d[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31)    ]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

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

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Ys)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float y0[4], y1[4], y2[4], y3[4];
	__syncthreads();

	//------write-read-turn0: {oc0 - oc3 | oc4 - oc7} x {j0 - j3}------------------------
	*(float4*)(&Ys[ux][uy][0]) = { v0.x, v0.y, v1.x, v1.y };//{oc0, oc1}, {oc4, oc5}}
	*(float4*)(&Ys[ux][uy][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Ys[ux][uy][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Ys[ux][uy][6]) = { v6.x, v6.y, v7.x, v7.y };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux];
	winograd_f4x5_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f4x5_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Ys[ux][uy][0]) = { v0.z, v0.w, v1.z, v1.w };//{oc2, oc3}, {oc6, oc7}}
	*(float4*)(&Ys[ux][uy][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Ys[ux][uy][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Ys[ux][uy][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux]; 
	winograd_f4x5_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f4x5_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	__syncthreads();
	
	//------write-read-turn0: {oc0 - oc3 | oc4 - oc7} x {j4 - j7}------------------------
	*(float4*)(&Ys[ux][uy][0]) = { v8.x,  v8.y,  v9.x,  v9.y };//{oc0, oc1}, {oc4, oc5}}
	*(float4*)(&Ys[ux][uy][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Ys[ux][uy][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Ys[ux][uy][6]) = { v14.x, v14.y, v15.x, v15.y };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux];
	winograd_f4x5_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f4x5_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Ys[ux][uy][0]) = { v8.z,  v8.w,  v9.z,  v9.w };//{oc2, oc3}, {oc6, oc7}}
	*(float4*)(&Ys[ux][uy][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Ys[ux][uy][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Ys[ux][uy][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Ys[0][uy][ux]; a[1] = Ys[1][uy][ux];
	a[2] = Ys[2][uy][ux]; a[3] = Ys[3][uy][ux];
	a[4] = Ys[4][uy][ux]; a[5] = Ys[5][uy][ux];
	a[6] = Ys[6][uy][ux]; a[7] = Ys[7][uy][ux];
	winograd_f4x5_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f4x5_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
}

#endif


#endif