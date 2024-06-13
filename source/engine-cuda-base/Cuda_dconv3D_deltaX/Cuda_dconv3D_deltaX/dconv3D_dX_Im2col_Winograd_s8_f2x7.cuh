#pragma once

#ifndef DECONV_3D_DX_WINOGRAD_S8_F2X7_H
#define DECONV_3D_DX_WINOGRAD_S8_F2X7_H

//(1) sh = sw = 1
//(2) FW = 7
//(3) OW % 2: group = 2 elements
#ifndef DECONV_3D_DX_WINOGRAD_F2X7_CALL
#define DECONV_3D_DX_WINOGRAD_F2X7_CALL

//<1> (opw <= 3) -> (FW - 1 - pw <= 3) -> (pw >= 3)
//<2> (IW - OW - opw + FW - 1 <= 3) -> (IW - OW + pw <= 3) -> (OW - IW - pw + 3 >= 0)

//================[Standard: 64(IC) * 64(GM)]==============================
//IW % 2 == 0
#define winograd_f2x7_k64x64_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f2x7_kernel_64x64_tex<FH>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IW>>1)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*7*IC), deltaX,IH,IW, IC,OC, (FH-1-ph), (6-pw))

//IW % 2 == 0, template<IC, OC>
#define winograd_f2x7_k64x64_CT_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f2x7_kernel_64x64_CT_tex<FH, IC, OC>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IW>>1)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*7*IC), deltaX,IH,IW, (FH-1-ph), (6-pw))

//IW % 4 == 0, pw >= 3
#define winograd_f2x7_ruse_k64x64_p3_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f2x7_ruse_kernel_64x64_p3_tex<FH>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IW>>2<<1)), dim3(16, 8), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*7*IC), deltaX,IH,IW, IC,OC, (FH-1-ph), (6-pw))

//IW % 4 == 0, pw >= 3, template<IC, OC>
#define winograd_f2x7_ruse_k64x64_p3_CT_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f2x7_ruse_kernel_64x64_p3_CT_tex<FH, IC, OC>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IW>>2<<1)), dim3(16, 8), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*7*IC), deltaX,IH,IW, (FH-1-ph), (6-pw))

//================[Cooperate: 64(IC) * 64(GM)]===============================
//IW % 2 == 0
#define winograd_f2x7_k64x64C_tex(stream, iw_index, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr) \
	winograd_f2x7_kernel_64x64C_tex<FH>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IWr>>1)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*7*IC), deltaX,IH,IW, IC,OC, (FH-1-ph), (6-pw), iw_index)

//IW % 2 == 0, template<IC, OC>
#define winograd_f2x7_k64x64C_CT_tex(stream, iw_index, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr) \
	winograd_f2x7_kernel_64x64C_CT_tex<FH, IC, OC>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IWr>>1)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*7*IC), deltaX,IH,IW, (FH-1-ph), (6-pw), iw_index)

//IWr % 4 == 0, pw >= 3
#define winograd_f2x7_ruse_k64x64C_p3_tex(stream, iw_index, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr) \
	winograd_f2x7_ruse_kernel_64x64C_p3_tex<FH>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IWr>>2<<1)), dim3(16, 8), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*7*IC), deltaX,IH,IW, IC,OC, (FH-1-ph), (6-pw), iw_index)

//IWr % 4 == 0, pw >= 3, template<IC, OC>
#define winograd_f2x7_ruse_k64x64C_p3_CT_tex(stream, iw_index, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw, IWr) \
	winograd_f2x7_ruse_kernel_64x64C_p3_CT_tex<FH, IC, OC>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IWr>>2<<1)), dim3(16, 8), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*7*IC), deltaX,IH,IW, (FH-1-ph), (6-pw), iw_index)

#endif


//================[Standard: 64(IC) * 64(GM)]================================
//LB = 4, OC % 8 == 0, IW % 2 == 0
#ifndef DECONV_3D_DX_WINOGRAD_F2X7_KERNEL_64X64_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F2X7_KERNEL_64X64_TEXTURE

//for: feature = (64, 64), [N, IC, OC] = [64,  64,  64]
//LB = 4: Size = 49, Time = 7.97 msec, Performace = 13202.8 GFlop/s
//for: feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 49, Time = 8.058 msec, Performace = 13058.7 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 7.866 msec, Performace = 13377.4 GFlop/s
//for: feature = ( 8,  8), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 49, Time = 8.208 msec, Performace = 12820 GFlop/s

template<int FH>
__global__ void winograd_f2x7_kernel_64x64_tex(
	cudaTextureObject_t       deltaY, int OH, int OW, 
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW, 
	int IC, int OC,
	int oph, int opw)//sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(ic): 64]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << 6);
	const int Gk = (ty & 7), Gi = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 7 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << 6);//32 * 2 = 64
	const int Yk = (tx & 7), Yi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + (Yi << 1);
	const int IW2 = IW >> 1 << 1, IH_IW2 = IH * IW2;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW2, IW2);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Yk;//deltaY[tn0, toh0, tow0, Yk]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//4 * 8
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ux >> 1)) << 1), xj1 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW2, IW2);
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IW2, IW2);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//X00 = xj0 * IC + xic0;
	const int X10 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;//X10 = X00 + IC * 8;

	//======[compute area1: local]======================================================
	const int Di = (Yi + (Yk >> 2 << 4)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], y[8], d[8]; char ly[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		ly[0] = lh0 && (tow0 >=  0) && (tow0     < OW);
		ly[1] = lh0 && (tow0 >= -1) && (tow0 + 1 < OW);
		ly[2] = lh0 && (tow0 >= -2) && (tow0 + 2 < OW);
		ly[3] = lh0 && (tow0 >= -3) && (tow0 + 3 < OW);
		ly[4] = lh0 && (tow0 >= -4) && (tow0 + 4 < OW);
		ly[5] = lh0 && (tow0 >= -5) && (tow0 + 5 < OW);
		ly[6] = lh0 && (tow0 >= -6) && (tow0 + 6 < OW);
		ly[7] = lh0 && (tow0 >= -7) && (tow0 + 7 < OW);
		int tY0 = Y0 + fh * OW * OC;
		int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1);
		int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
		int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6, tY7 = tY0 + OC * 7;
		tY0 = IF_int(ly[0], tY0, -1); tY1 = IF_int(ly[1], tY1, -1);
		tY2 = IF_int(ly[2], tY2, -1); tY3 = IF_int(ly[3], tY3, -1);
		tY4 = IF_int(ly[4], tY4, -1); tY5 = IF_int(ly[5], tY5, -1);
		tY6 = IF_int(ly[6], tY6, -1); tY7 = IF_int(ly[7], tY7, -1);
		y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
		y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
		y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
		y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 7 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1), W3 = W0 + IC * 3;
		const int W4 = W0 + (IC << 2), W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		float2 w6 = *(float2*)(W + W0);
		float2 w5 = *(float2*)(W + W1);
		float2 w4 = *(float2*)(W + W2);
		float2 w3 = *(float2*)(W + W3);
		float2 w2 = *(float2*)(W + W4);
		float2 w1 = *(float2*)(W + W5);
		float2 w0 = *(float2*)(W + W6);

		//Winograd transform: W(6) -> G(8); Y(8) -> D(8)
		winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
		winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
		winograd_f2x7_D(d, y);

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

		Ds[buf][Yk][0][Di] = d[0]; Ds[buf][Yk][1][Di] = d[1];
		Ds[buf][Yk][2][Di] = d[2]; Ds[buf][Yk][3][Di] = d[3];
		Ds[buf][Yk][4][Di] = d[4]; Ds[buf][Yk][5][Di] = d[5];
		Ds[buf][Yk][6][Di] = d[6]; Ds[buf][Yk][7][Di] = d[7];
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31) + 4]);

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
			buf ^= 1;

			//load 1 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc;
			int tY1 = tY0 + OC    , tY2 = tY0 + (OC << 1);
			int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
			int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6, tY7 = tY0 + OC * 7;
			tY0 = IF_int(ly[0], tY0, -1); tY1 = IF_int(ly[1], tY1, -1);
			tY2 = IF_int(ly[2], tY2, -1); tY3 = IF_int(ly[3], tY3, -1);
			tY4 = IF_int(ly[4], tY4, -1); tY5 = IF_int(ly[5], tY5, -1);
			tY6 = IF_int(ly[6], tY6, -1); tY7 = IF_int(ly[7], tY7, -1);
			y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
			y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
			y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
			y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

			//load 2 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 7 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1), W3 = W0 + IC * 3;
			const int W4 = W0 + (IC << 2), W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			float2 w6 = *(float2*)(W + W0);
			float2 w5 = *(float2*)(W + W1);
			float2 w4 = *(float2*)(W + W2);
			float2 w3 = *(float2*)(W + W3);
			float2 w2 = *(float2*)(W + W4);
			float2 w1 = *(float2*)(W + W5);
			float2 w0 = *(float2*)(W + W6);

			//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
			winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
			winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
			winograd_f2x7_D(d, y);

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

			Ds[buf][Yk][0][Di] = d[0]; Ds[buf][Yk][1][Di] = d[1];
			Ds[buf][Yk][2][Di] = d[2]; Ds[buf][Yk][3][Di] = d[3];
			Ds[buf][Yk][4][Di] = d[4]; Ds[buf][Yk][5][Di] = d[5];
			Ds[buf][Yk][6][Di] = d[6]; Ds[buf][Yk][7][Di] = d[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31)    ]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31) + 4]);

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
		buf ^= 1;
	}

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Xs)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float x0[2], x1[2], x2[2], x3[2];
	__syncthreads();

	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j0 - j3}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v0.x, v0.y, v1.x, v1.y };//{ic0, ic1}, {ic4, ic5}}
	*(float4*)(&Xs[ux][uy][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Xs[ux][uy][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Xs[ux][uy][6]) = { v6.x, v6.y, v7.x, v7.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v0.z, v0.w, v1.z, v1.w };//{ic2, ic3}, {ic6, ic7}}
	*(float4*)(&Xs[ux][uy][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Xs[ux][uy][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Xs[ux][uy][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j4 - j7}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v8.x,  v8.y,  v9.x,  v9.y };//{ic0, ic1}, {ic4, ic5}}
	*(float4*)(&Xs[ux][uy][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Xs[ux][uy][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Xs[ux][uy][6]) = { v14.x, v14.y, v15.x, v15.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v8.z,  v8.w,  v9.z,  v9.w };//{ic2, ic3}, {ic6, ic7}}
	*(float4*)(&Xs[ux][uy][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Xs[ux][uy][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Xs[ux][uy][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
}

#endif


//LB = 4, OC % 8 == 0, IW % 2 == 0, template<IC, OC>
#ifndef DECONV_3D_DX_WINOGRAD_F2X7_KERNEL_64X64_CT_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F2X7_KERNEL_64X64_CT_TEXTURE

//for: feature = (128, 128), [N, IC, OC] = [16,  64,  64]
//LB = 4: Size = 49, Time = 7.347 msec, Performace = 14322.4 GFlop/s
//for: feature = (64, 64), [N, IC, OC] = [64,  64,  64]
//LB = 4: Size = 49, Time = 7.314 msec, Performace = 14387 GFlop/s
//for: feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 49, Time = 7.421 msec, Performace = 14179.6 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 7.172 msec, Performace = 14671.9 GFlop/s
//for: feature = ( 8,  8), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 49, Time = 7.426 msec, Performace = 14170 GFlop/s

template<int FH, int IC, int OC>
__global__ void winograd_f2x7_kernel_64x64_CT_tex(
	cudaTextureObject_t       deltaY, int OH, int OW, 
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW, 
	int oph, int opw)//sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(ic): 64]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << 6);
	const int Gk = (ty & 7), Gi = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 7 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaX[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 6);//32 * 2 = 64
	const int Yk = (tx & 7), Yi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + (Yi << 1);
	const int IW2 = IW >> 1 << 1, IH_IW2 = IH * IW2;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW2, IW2);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Yk;//deltaY[tn0, toh0, tow0, Yk]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//4 * 8
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//prepare for deltaY[N, OH, OW, OC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ux >> 1)) << 1), xj1 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW2, IW2);
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IW2, IW2);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//X00 = xj0 * IC + xic0;
	const int X10 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;//X10 = X00 + IC * 8;

	//======[compute area1: local]======================================================
	const int Di = (Yi + (Yk >> 1 << 3)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], y[8], d[8]; char ly[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		ly[0] = lh0 && (tow0 >=  0) && (tow0     < OW);
		ly[1] = lh0 && (tow0 >= -1) && (tow0 + 1 < OW);
		ly[2] = lh0 && (tow0 >= -2) && (tow0 + 2 < OW);
		ly[3] = lh0 && (tow0 >= -3) && (tow0 + 3 < OW);
		ly[4] = lh0 && (tow0 >= -4) && (tow0 + 4 < OW);
		ly[5] = lh0 && (tow0 >= -5) && (tow0 + 5 < OW);
		ly[6] = lh0 && (tow0 >= -6) && (tow0 + 6 < OW);
		ly[7] = lh0 && (tow0 >= -7) && (tow0 + 7 < OW);
		int tY0 = Y0 + fh * OW * OC;
		int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1);
		int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
		int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6, tY7 = tY0 + OC * 7;
		tY0 = IF_int(ly[0], tY0, -1); tY1 = IF_int(ly[1], tY1, -1);
		tY2 = IF_int(ly[2], tY2, -1); tY3 = IF_int(ly[3], tY3, -1);
		tY4 = IF_int(ly[4], tY4, -1); tY5 = IF_int(ly[5], tY5, -1);
		tY6 = IF_int(ly[6], tY6, -1); tY7 = IF_int(ly[7], tY7, -1);
		y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
		y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
		y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
		y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 7 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1), W3 = W0 + IC * 3;
		const int W4 = W0 + (IC << 2), W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		float2 w6 = *(float2*)(W + W0);
		float2 w5 = *(float2*)(W + W1);
		float2 w4 = *(float2*)(W + W2);
		float2 w3 = *(float2*)(W + W3);
		float2 w2 = *(float2*)(W + W4);
		float2 w1 = *(float2*)(W + W5);
		float2 w0 = *(float2*)(W + W6);

		//Winograd transform: W(6) -> G(8); Y(8) -> D(8)
		winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
		winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
		winograd_f2x7_D(d, y);

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

		Ds[buf][Yk][0][Di] = d[0]; Ds[buf][Yk][1][Di] = d[1];
		Ds[buf][Yk][2][Di] = d[2]; Ds[buf][Yk][3][Di] = d[3];
		Ds[buf][Yk][4][Di] = d[4]; Ds[buf][Yk][5][Di] = d[5];
		Ds[buf][Yk][6][Di] = d[6]; Ds[buf][Yk][7][Di] = d[7];
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

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
			buf ^= 1;

			//load 1 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc;
			int tY1 = tY0 + OC    , tY2 = tY0 + (OC << 1);
			int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
			int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6, tY7 = tY0 + OC * 7;
			tY0 = IF_int(ly[0], tY0, -1); tY1 = IF_int(ly[1], tY1, -1);
			tY2 = IF_int(ly[2], tY2, -1); tY3 = IF_int(ly[3], tY3, -1);
			tY4 = IF_int(ly[4], tY4, -1); tY5 = IF_int(ly[5], tY5, -1);
			tY6 = IF_int(ly[6], tY6, -1); tY7 = IF_int(ly[7], tY7, -1);
			y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
			y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
			y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
			y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

			//load 2 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 7 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1), W3 = W0 + IC * 3;
			const int W4 = W0 + (IC << 2), W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			float2 w6 = *(float2*)(W + W0);
			float2 w5 = *(float2*)(W + W1);
			float2 w4 = *(float2*)(W + W2);
			float2 w3 = *(float2*)(W + W3);
			float2 w2 = *(float2*)(W + W4);
			float2 w1 = *(float2*)(W + W5);
			float2 w0 = *(float2*)(W + W6);

			//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
			winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
			winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
			winograd_f2x7_D(d, y);

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

			Ds[buf][Yk][0][Di] = d[0]; Ds[buf][Yk][1][Di] = d[1];
			Ds[buf][Yk][2][Di] = d[2]; Ds[buf][Yk][3][Di] = d[3];
			Ds[buf][Yk][4][Di] = d[4]; Ds[buf][Yk][5][Di] = d[5];
			Ds[buf][Yk][6][Di] = d[6]; Ds[buf][Yk][7][Di] = d[7];
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
	float2 (* __restrict__ Xs)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float x0[2], x1[2], x2[2], x3[2];
	__syncthreads();

	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j0 - j3}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v0.x, v0.y, v1.x, v1.y };//{ic0, ic1}, {ic4, ic5}}
	*(float4*)(&Xs[ux][uy][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Xs[ux][uy][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Xs[ux][uy][6]) = { v6.x, v6.y, v7.x, v7.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v0.z, v0.w, v1.z, v1.w };//{ic2, ic3}, {ic6, ic7}}
	*(float4*)(&Xs[ux][uy][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Xs[ux][uy][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Xs[ux][uy][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j4 - j7}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v8.x,  v8.y,  v9.x,  v9.y };//{ic0, ic1}, {ic4, ic5}}
	*(float4*)(&Xs[ux][uy][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Xs[ux][uy][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Xs[ux][uy][6]) = { v14.x, v14.y, v15.x, v15.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v8.z,  v8.w,  v9.z,  v9.w };//{ic2, ic3}, {ic6, ic7}}
	*(float4*)(&Xs[ux][uy][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Xs[ux][uy][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Xs[ux][uy][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
}

#endif


//LB = 4, OC % 8 == 0, IW % 4 == 0, pw >= 3
#ifndef DECONV_3D_DX_WINOGRAD_F2X7_RUSE_KERNEL_64X64_P3_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F2X7_RUSE_KERNEL_64X64_P3_TEXTURE

//for: feature = (64, 64), [N, IC, OC] = [64,  64,  64]
//LB = 4: Size = 49, Time = 6.528 msec, Performace = 16119.3 GFlop/s
//for: feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 49, Time = 6.436 msec, Performace = 16349.7 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 6.435 msec, Performace = 16352.2 GFlop/s
//for: feature = ( 8,  8), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 49, Time = 6.533 msec, Performace = 16106.9 GFlop/s

template<int FH>
__global__ void winograd_f2x7_ruse_kernel_64x64_p3_tex(
	cudaTextureObject_t       deltaY, int OH, int OW, 
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW, 
	int IC, int OC,
	int oph, int opw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(ic): 64]
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

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (bx << 6);
	const int Gk = ty, Gi = tx << 2;//[8, 16*4 = 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 7 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (by << 6);//32 * 2 = 64
	const int Yk = (tx & 7), Yi = ((ty << 1) + (tx > 7)) << 1;//[8, 16*2 = 32]
	const int tj0 = bj0 + (Yi << 1);
	const int IW4 = IW >> 2 << 2, IH_IW4 = IH * IW4;//2 * 2 = 4
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW4, IW4);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Yk;//deltaY[tn0, toh0, tow0, Yk]

	//prepare for threadIdx
	const int GIdx = ((tx & 1) + ((tx >> 3) << 1)) << 4;//4
	const int DIdx = ((tx & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ty & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ty >> 1)) << 1), xj1 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW4, IW4);
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IW4, IW4);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//X00 = xj0 * IC + xic0;
	const int X10 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;//X10 = X00 + IC * 8;

	//======[compute area1: local]======================================================
	const int Di = (Yi + (Yk >> 1 << 3)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], g2[8], g3[8], y[10], d0[8], d1[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 4 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 7 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		float4 w6 = *(float4*)(W + W0);
		float4 w5 = *(float4*)(W + W1);
		float4 w4 = *(float4*)(W + W2);
		float4 w3 = *(float4*)(W + W3);
		float4 w2 = *(float4*)(W + W4);
		float4 w1 = *(float4*)(W + W5);
		float4 w0 = *(float4*)(W + W6);

		//load 2 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=   0) && (tow0      < OW);//opw = -3
		bool ly1 = lh0 && (tow0 >=  -1) && (tow0 +  1 < OW);//opw = -2
		bool ly2 = lh0 && (tow0 >=  -2) && (tow0 +  2 < OW);//opw = -1
		bool ly7 = lh0 && (tow0 >=  -7) && (tow0 +  7 < OW);//opw = +1
		bool ly8 = lh0 && (tow0 >=  -8) && (tow0 +  8 < OW);//opw = +2
		bool ly9 = lh0 && (tow0 >=  -9) && (tow0 +  9 < OW);//opw = +3
		int tY0 = Y0 + fh * OW * OC, tY1 = tY0 + OC;
		int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
		int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
		int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC * 7;
		int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC * 9;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1); tY2 = IF_int(ly2, tY2, -1);
		tY3 = IF_int(lh0, tY3, -1); tY4 = IF_int(lh0, tY4, -1); 
		tY5 = IF_int(lh0, tY5, -1); tY6 = IF_int(lh0, tY6, -1); 
		tY7 = IF_int(ly7, tY7, -1); tY8 = IF_int(ly8, tY8, -1); tY9 = IF_int(ly9, tY9, -1);
		y[0x0] = tex1Dfetch<float>(deltaY, tY0); y[0x1] = tex1Dfetch<float>(deltaY, tY1);
		y[0x2] = tex1Dfetch<float>(deltaY, tY2); y[0x3] = tex1Dfetch<float>(deltaY, tY3);
		y[0x4] = tex1Dfetch<float>(deltaY, tY4); y[0x5] = tex1Dfetch<float>(deltaY, tY5);
		y[0x6] = tex1Dfetch<float>(deltaY, tY6); y[0x7] = tex1Dfetch<float>(deltaY, tY7);
		y[0x8] = tex1Dfetch<float>(deltaY, tY8); y[0x9] = tex1Dfetch<float>(deltaY, tY9);

		//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
		winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
		winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
		winograd_f2x7_G(g2, w0.z, w1.z, w2.z, w3.z, w4.z, w5.z, w6.z);
		winograd_f2x7_G(g3, w0.w, w1.w, w2.w, w3.w, w4.w, w5.w, w6.w);
		winograd_f2x7_D(d0, y);//x[0 - 7]
		winograd_f2x7_D_oft(d1, y, 2);//x[2 - 9]

		//write to shread memory
		*(float2*)(&Ds[buf][Yk][0][Di]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Yk][1][Di]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Yk][2][Di]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Yk][3][Di]) = { d0[3], d1[3] };
		*(float2*)(&Ds[buf][Yk][4][Di]) = { d0[4], d1[4] };
		*(float2*)(&Ds[buf][Yk][5][Di]) = { d0[5], d1[5] };
		*(float2*)(&Ds[buf][Yk][6][Di]) = { d0[6], d1[6] };
		*(float2*)(&Ds[buf][Yk][7][Di]) = { d0[7], d1[7] };

		*(float4*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0], g2[0], g3[0] };
		*(float4*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1], g2[1], g3[1] };
		*(float4*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2], g2[2], g3[2] };
		*(float4*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3], g2[3], g3[3] };
		*(float4*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4], g2[4], g3[4] };
		*(float4*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5], g2[5], g3[5] };
		*(float4*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6], g2[6], g3[6] };
		*(float4*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7], g2[7], g3[7] };
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 b0 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

				//=======================================================
				float4 a0 = *(float4*)(&Gs[buf][ik][ty][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ty][GIdx + 4]);

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
				float4 a2 = *(float4*)(&Gs[buf][ik][ty][GIdx +  8]);
				float4 a3 = *(float4*)(&Gs[buf][ik][ty][GIdx + 12]);
	
				//ic[8 - 11]             ic[12 - 15]
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

			//load 4 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 7 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			float4 w6 = *(float4*)(W + W0);
			float4 w5 = *(float4*)(W + W1);
			float4 w4 = *(float4*)(W + W2);
			float4 w3 = *(float4*)(W + W3);
			float4 w2 = *(float4*)(W + W4);
			float4 w1 = *(float4*)(W + W5);
			float4 w0 = *(float4*)(W + W6);

			//load 2 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc, tY1 = tY0 + OC;
			int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
			int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
			int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC * 7;
			int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC * 9;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1); tY2 = IF_int(ly2, tY2, -1);
			tY3 = IF_int(lh0, tY3, -1); tY4 = IF_int(lh0, tY4, -1); 
			tY5 = IF_int(lh0, tY5, -1); tY6 = IF_int(lh0, tY6, -1);
			tY7 = IF_int(ly7, tY7, -1); tY8 = IF_int(ly8, tY8, -1); tY9 = IF_int(ly9, tY9, -1);
			y[0x0] = tex1Dfetch<float>(deltaY, tY0); y[0x1] = tex1Dfetch<float>(deltaY, tY1);
			y[0x2] = tex1Dfetch<float>(deltaY, tY2); y[0x3] = tex1Dfetch<float>(deltaY, tY3);
			y[0x4] = tex1Dfetch<float>(deltaY, tY4); y[0x5] = tex1Dfetch<float>(deltaY, tY5);
			y[0x6] = tex1Dfetch<float>(deltaY, tY6); y[0x7] = tex1Dfetch<float>(deltaY, tY7);
			y[0x8] = tex1Dfetch<float>(deltaY, tY8); y[0x9] = tex1Dfetch<float>(deltaY, tY9);

			//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
			winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
			winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
			winograd_f2x7_G(g2, w0.z, w1.z, w2.z, w3.z, w4.z, w5.z, w6.z);
			winograd_f2x7_G(g3, w0.w, w1.w, w2.w, w3.w, w4.w, w5.w, w6.w);
			winograd_f2x7_D(d0, y);//x[0 - 7]
			winograd_f2x7_D_oft(d1, y, 2);//x[2 - 9]

			//write to shread memory
			*(float2*)(&Ds[buf][Yk][0][Di]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Yk][1][Di]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Yk][2][Di]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Yk][3][Di]) = { d0[3], d1[3] };
			*(float2*)(&Ds[buf][Yk][4][Di]) = { d0[4], d1[4] };
			*(float2*)(&Ds[buf][Yk][5][Di]) = { d0[5], d1[5] };
			*(float2*)(&Ds[buf][Yk][6][Di]) = { d0[6], d1[6] };
			*(float2*)(&Ds[buf][Yk][7][Di]) = { d0[7], d1[7] };

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
			float4 a2 = *(float4*)(&Gs[buf][ik][ty][GIdx +  8]);
			float4 a3 = *(float4*)(&Gs[buf][ik][ty][GIdx + 12]);

			//ic[8 - 11]             ic[12 - 15]
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

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Xs)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float x0[2], x1[2], x2[2], x3[2];
	__syncthreads();

	//------write-read-turn0: {ic0 - ic7} x {j0 - j3}------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v0.x, v0.y, v1.x, v1.y };
	*(float4*)(&Xs[ty][tx][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Xs[ty][tx][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Xs[ty][tx][6]) = { v6.x, v6.y, v7.x, v7.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v0.z, v0.w, v1.z, v1.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic0 - ic7} x {j4 - j7}-------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = {  v8.x,  v8.y,  v9.x,  v9.y };
	*(float4*)(&Xs[ty][tx][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Xs[ty][tx][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Xs[ty][tx][6]) = { v14.x, v14.y, v15.x, v15.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = {  v8.z,  v8.w,  v9.z,  v9.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();

	//------write-read-turn0: {ic8 - ic15} x {j0 - j3}-----------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v16.x, v16.y, v17.x, v17.y };
	*(float4*)(&Xs[ty][tx][2]) = { v18.x, v18.y, v19.x, v19.y };
	*(float4*)(&Xs[ty][tx][4]) = { v20.x, v20.y, v21.x, v21.y };
	*(float4*)(&Xs[ty][tx][6]) = { v22.x, v22.y, v23.x, v23.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v16.z, v16.w, v17.z, v17.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v18.z, v18.w, v19.z, v19.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v20.z, v20.w, v21.z, v21.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v22.z, v22.w, v23.z, v23.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00          + 8) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1 + 8) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic8 - ic15} x {j4 - j7}-------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v24.x, v24.y, v25.x, v25.y };
	*(float4*)(&Xs[ty][tx][2]) = { v26.x, v26.y, v27.x, v27.y };
	*(float4*)(&Xs[ty][tx][4]) = { v28.x, v28.y, v29.x, v29.y };
	*(float4*)(&Xs[ty][tx][6]) = { v30.x, v30.y, v31.x, v31.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v24.z, v24.w, v25.z, v25.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v26.z, v26.w, v27.z, v27.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v28.z, v28.w, v29.z, v29.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v30.z, v30.w, v31.z, v31.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10          + 8) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1 + 8) = { x0[1], x1[1], x2[1], x3[1] };
}

#endif


//LB = 4, OC % 8 == 0, IW % 4 == 0, pw >= 3, template<IC, OC>
#ifndef DECONV_3D_DX_WINOGRAD_F2X7_RUSE_KERNEL_64X64_P3_CT_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F2X7_RUSE_KERNEL_64X64_P3_CT_TEXTURE

//for: feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 49, Time = 5.999 msec, Performace = 17540.7 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 6.228 msec, Performace = 16895.7 GFlop/s
//for: feature = ( 8,  8), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 49, Time = 6.302 msec, Performace = 16697.3 GFlop/s

template<int FH, int IC, int OC>
__global__ void winograd_f2x7_ruse_kernel_64x64_p3_CT_tex(
	cudaTextureObject_t       deltaY, int OH, int OW, 
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW, 
	int oph, int opw)//sh = sw = 1
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(ic): 64]
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

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (bx << 6);
	const int Gk = ty, Gi = tx << 2;//[8, 16*4 = 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 7 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (by << 6);//32 * 2 = 64
	const int Yk = (tx & 7), Yi = ((ty << 1) + (tx > 7)) << 1;//[8, 16*2 = 32]
	const int tj0 = bj0 + (Yi << 1);
	const int IW4 = IW >> 2 << 2, IH_IW4 = IH * IW4;//2 * 2 = 4
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW4, IW4);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Yk;//deltaY[tn0, toh0, tow0, Yk]

	//prepare for threadIdx
	const int GIdx = ((tx & 1) + ((tx >> 3) << 1)) << 4;//4
	const int DIdx = ((tx & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ty & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ty >> 1)) << 1), xj1 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW4, IW4);
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IW4, IW4);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//X00 = xj0 * IC + xic0;
	const int X10 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;//X10 = X00 + IC * 8;

	//======[compute area1: local]======================================================
	const int Di = (Yi + (Yk >> 1 << 3)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], g2[8], g3[8], y[10], d0[8], d1[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 4 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 7 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		float4 w6 = *(float4*)(W + W0);
		float4 w5 = *(float4*)(W + W1);
		float4 w4 = *(float4*)(W + W2);
		float4 w3 = *(float4*)(W + W3);
		float4 w2 = *(float4*)(W + W4);
		float4 w1 = *(float4*)(W + W5);
		float4 w0 = *(float4*)(W + W6);

		//load 2 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=   0) && (tow0      < OW);//opw = -3
		bool ly1 = lh0 && (tow0 >=  -1) && (tow0 +  1 < OW);//opw = -2
		bool ly2 = lh0 && (tow0 >=  -2) && (tow0 +  2 < OW);//opw = -1
		bool ly7 = lh0 && (tow0 >=  -7) && (tow0 +  7 < OW);//opw = +1
		bool ly8 = lh0 && (tow0 >=  -8) && (tow0 +  8 < OW);//opw = +2
		bool ly9 = lh0 && (tow0 >=  -9) && (tow0 +  9 < OW);//opw = +3
		int tY0 = Y0 + fh * OW * OC, tY1 = tY0 + OC;
		int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
		int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
		int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC * 7;
		int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC * 9;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1); tY2 = IF_int(ly2, tY2, -1);
		tY3 = IF_int(lh0, tY3, -1); tY4 = IF_int(lh0, tY4, -1); 
		tY5 = IF_int(lh0, tY5, -1); tY6 = IF_int(lh0, tY6, -1); 
		tY7 = IF_int(ly7, tY7, -1); tY8 = IF_int(ly8, tY8, -1); tY9 = IF_int(ly9, tY9, -1);
		y[0x0] = tex1Dfetch<float>(deltaY, tY0); y[0x1] = tex1Dfetch<float>(deltaY, tY1);
		y[0x2] = tex1Dfetch<float>(deltaY, tY2); y[0x3] = tex1Dfetch<float>(deltaY, tY3);
		y[0x4] = tex1Dfetch<float>(deltaY, tY4); y[0x5] = tex1Dfetch<float>(deltaY, tY5);
		y[0x6] = tex1Dfetch<float>(deltaY, tY6); y[0x7] = tex1Dfetch<float>(deltaY, tY7);
		y[0x8] = tex1Dfetch<float>(deltaY, tY8); y[0x9] = tex1Dfetch<float>(deltaY, tY9);

		//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
		winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
		winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
		winograd_f2x7_G(g2, w0.z, w1.z, w2.z, w3.z, w4.z, w5.z, w6.z);
		winograd_f2x7_G(g3, w0.w, w1.w, w2.w, w3.w, w4.w, w5.w, w6.w);
		winograd_f2x7_D(d0, y);//x[0 - 7]
		winograd_f2x7_D_oft(d1, y, 2);//x[2 - 9]

		//write to shread memory
		*(float2*)(&Ds[buf][Yk][0][Di]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Yk][1][Di]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Yk][2][Di]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Yk][3][Di]) = { d0[3], d1[3] };
		*(float2*)(&Ds[buf][Yk][4][Di]) = { d0[4], d1[4] };
		*(float2*)(&Ds[buf][Yk][5][Di]) = { d0[5], d1[5] };
		*(float2*)(&Ds[buf][Yk][6][Di]) = { d0[6], d1[6] };
		*(float2*)(&Ds[buf][Yk][7][Di]) = { d0[7], d1[7] };

		*(float4*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0], g2[0], g3[0] };
		*(float4*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1], g2[1], g3[1] };
		*(float4*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2], g2[2], g3[2] };
		*(float4*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3], g2[3], g3[3] };
		*(float4*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4], g2[4], g3[4] };
		*(float4*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5], g2[5], g3[5] };
		*(float4*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6], g2[6], g3[6] };
		*(float4*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7], g2[7], g3[7] };
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 b0 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

				//=======================================================
				float4 a0 = *(float4*)(&Gs[buf][ik][ty][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ty][GIdx + 4]);

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
				float4 a2 = *(float4*)(&Gs[buf][ik][ty][GIdx +  8]);
				float4 a3 = *(float4*)(&Gs[buf][ik][ty][GIdx + 12]);
	
				//ic[8 - 11]             ic[12 - 15]
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

			//load 4 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 7 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			float4 w6 = *(float4*)(W + W0);
			float4 w5 = *(float4*)(W + W1);
			float4 w4 = *(float4*)(W + W2);
			float4 w3 = *(float4*)(W + W3);
			float4 w2 = *(float4*)(W + W4);
			float4 w1 = *(float4*)(W + W5);
			float4 w0 = *(float4*)(W + W6);

			//load 2 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc, tY1 = tY0 + OC;
			int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
			int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
			int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC * 7;
			int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC * 9;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1); tY2 = IF_int(ly2, tY2, -1);
			tY3 = IF_int(lh0, tY3, -1); tY4 = IF_int(lh0, tY4, -1); 
			tY5 = IF_int(lh0, tY5, -1); tY6 = IF_int(lh0, tY6, -1);
			tY7 = IF_int(ly7, tY7, -1); tY8 = IF_int(ly8, tY8, -1); tY9 = IF_int(ly9, tY9, -1);
			y[0x0] = tex1Dfetch<float>(deltaY, tY0); y[0x1] = tex1Dfetch<float>(deltaY, tY1);
			y[0x2] = tex1Dfetch<float>(deltaY, tY2); y[0x3] = tex1Dfetch<float>(deltaY, tY3);
			y[0x4] = tex1Dfetch<float>(deltaY, tY4); y[0x5] = tex1Dfetch<float>(deltaY, tY5);
			y[0x6] = tex1Dfetch<float>(deltaY, tY6); y[0x7] = tex1Dfetch<float>(deltaY, tY7);
			y[0x8] = tex1Dfetch<float>(deltaY, tY8); y[0x9] = tex1Dfetch<float>(deltaY, tY9);

			//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
			winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
			winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
			winograd_f2x7_G(g2, w0.z, w1.z, w2.z, w3.z, w4.z, w5.z, w6.z);
			winograd_f2x7_G(g3, w0.w, w1.w, w2.w, w3.w, w4.w, w5.w, w6.w);
			winograd_f2x7_D(d0, y);//x[0 - 7]
			winograd_f2x7_D_oft(d1, y, 2);//x[2 - 9]

			//write to shread memory
			*(float2*)(&Ds[buf][Yk][0][Di]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Yk][1][Di]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Yk][2][Di]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Yk][3][Di]) = { d0[3], d1[3] };
			*(float2*)(&Ds[buf][Yk][4][Di]) = { d0[4], d1[4] };
			*(float2*)(&Ds[buf][Yk][5][Di]) = { d0[5], d1[5] };
			*(float2*)(&Ds[buf][Yk][6][Di]) = { d0[6], d1[6] };
			*(float2*)(&Ds[buf][Yk][7][Di]) = { d0[7], d1[7] };

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
			float4 a2 = *(float4*)(&Gs[buf][ik][ty][GIdx +  8]);
			float4 a3 = *(float4*)(&Gs[buf][ik][ty][GIdx + 12]);

			//ic[8 - 11]             ic[12 - 15]
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

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Xs)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float x0[2], x1[2], x2[2], x3[2];
	__syncthreads();

	//------write-read-turn0: {ic0 - ic7} x {j0 - j3}------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v0.x, v0.y, v1.x, v1.y };
	*(float4*)(&Xs[ty][tx][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Xs[ty][tx][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Xs[ty][tx][6]) = { v6.x, v6.y, v7.x, v7.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v0.z, v0.w, v1.z, v1.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic0 - ic7} x {j4 - j7}-------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = {  v8.x,  v8.y,  v9.x,  v9.y };
	*(float4*)(&Xs[ty][tx][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Xs[ty][tx][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Xs[ty][tx][6]) = { v14.x, v14.y, v15.x, v15.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = {  v8.z,  v8.w,  v9.z,  v9.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();

	//------write-read-turn0: {ic8 - ic15} x {j0 - j3}-----------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v16.x, v16.y, v17.x, v17.y };
	*(float4*)(&Xs[ty][tx][2]) = { v18.x, v18.y, v19.x, v19.y };
	*(float4*)(&Xs[ty][tx][4]) = { v20.x, v20.y, v21.x, v21.y };
	*(float4*)(&Xs[ty][tx][6]) = { v22.x, v22.y, v23.x, v23.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v16.z, v16.w, v17.z, v17.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v18.z, v18.w, v19.z, v19.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v20.z, v20.w, v21.z, v21.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v22.z, v22.w, v23.z, v23.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00          + 8) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1 + 8) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic8 - ic15} x {j4 - j7}-------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v24.x, v24.y, v25.x, v25.y };
	*(float4*)(&Xs[ty][tx][2]) = { v26.x, v26.y, v27.x, v27.y };
	*(float4*)(&Xs[ty][tx][4]) = { v28.x, v28.y, v29.x, v29.y };
	*(float4*)(&Xs[ty][tx][6]) = { v30.x, v30.y, v31.x, v31.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v24.z, v24.w, v25.z, v25.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v26.z, v26.w, v27.z, v27.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v28.z, v28.w, v29.z, v29.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v30.z, v30.w, v31.z, v31.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10          + 8) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1 + 8) = { x0[1], x1[1], x2[1], x3[1] };
}

#endif


//================[Cooperate: 64(IC) * 64(GM)]===============================
//LB = 4, OC % 8 == 0, IWr % 2 == 0
#ifndef DECONV_3D_DX_WINOGRAD_F2X7_KERNEL_64X64C_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F2X7_KERNEL_64X64C_TEXTURE

//for: feature = (64, 64), [N, IC, OC] = [64,  64,  64]
//LB = 4: Size = 49, Time = 8.214 msec, Performace = 12810.7 GFlop/s
//for: feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 49, Time = 8.293 msec, Performace = 12688.6 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 8.202 msec, Performace = 12829.4 GFlop/s
//for: feature = ( 8,  8), [N, IC, OC] = [64, 512, 512]
//LB = 4: Size = 49, Time = 8.29  msec, Performace = 12693.2 GFlop/s

template<int FH>
__global__ void winograd_f2x7_kernel_64x64C_tex(
	cudaTextureObject_t       deltaY, int OH, int OW, 
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW, 
	int IC, int OC,
	int oph, int opw,//sh = sw = 1
	int iw_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(ic): 64]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << 6);
	const int Gk = (ty & 7), Gi = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 7 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << 6);//32 * 2 = 64
	const int Yk = (tx & 7), Yi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + (Yi << 1);
	const int IWr2 = (IW - iw_index) >> 1 << 1, IH_IWr2 = IH * IWr2;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr2, IWr2); tiw0 += iw_index;
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Yk;//deltaY[tn0, toh0, tow0, Yk]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//4 * 8
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ux >> 1)) << 1), xj1 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr2, IWr2); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IWr2, IWr2); xiw1 += iw_index;
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//X00 = xj0 * IC + xic0;
	const int X10 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;//X10 = X00 + IC * 8;

	//======[compute area1: local]======================================================
	const int Di = (Yi + (Yk >> 2 << 4)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], y[8], d[8]; char ly[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		ly[0] = lh0 && (tow0 >=  0) && (tow0     < OW);
		ly[1] = lh0 && (tow0 >= -1) && (tow0 + 1 < OW);
		ly[2] = lh0 && (tow0 >= -2) && (tow0 + 2 < OW);
		ly[3] = lh0 && (tow0 >= -3) && (tow0 + 3 < OW);
		ly[4] = lh0 && (tow0 >= -4) && (tow0 + 4 < OW);
		ly[5] = lh0 && (tow0 >= -5) && (tow0 + 5 < OW);
		ly[6] = lh0 && (tow0 >= -6) && (tow0 + 6 < OW);
		ly[7] = lh0 && (tow0 >= -7) && (tow0 + 7 < OW);
		int tY0 = Y0 + fh * OW * OC;
		int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1);
		int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
		int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6, tY7 = tY0 + OC * 7;
		tY0 = IF_int(ly[0], tY0, -1); tY1 = IF_int(ly[1], tY1, -1);
		tY2 = IF_int(ly[2], tY2, -1); tY3 = IF_int(ly[3], tY3, -1);
		tY4 = IF_int(ly[4], tY4, -1); tY5 = IF_int(ly[5], tY5, -1);
		tY6 = IF_int(ly[6], tY6, -1); tY7 = IF_int(ly[7], tY7, -1);
		y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
		y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
		y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
		y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 7 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1), W3 = W0 + IC * 3;
		const int W4 = W0 + (IC << 2), W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		float2 w6 = *(float2*)(W + W0);
		float2 w5 = *(float2*)(W + W1);
		float2 w4 = *(float2*)(W + W2);
		float2 w3 = *(float2*)(W + W3);
		float2 w2 = *(float2*)(W + W4);
		float2 w1 = *(float2*)(W + W5);
		float2 w0 = *(float2*)(W + W6);

		//Winograd transform: W(6) -> G(8); Y(8) -> D(8)
		winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
		winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
		winograd_f2x7_D(d, y);

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

		Ds[buf][Yk][0][Di] = d[0]; Ds[buf][Yk][1][Di] = d[1];
		Ds[buf][Yk][2][Di] = d[2]; Ds[buf][Yk][3][Di] = d[3];
		Ds[buf][Yk][4][Di] = d[4]; Ds[buf][Yk][5][Di] = d[5];
		Ds[buf][Yk][6][Di] = d[6]; Ds[buf][Yk][7][Di] = d[7];
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31) + 4]);

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
			buf ^= 1;

			//load 1 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc;
			int tY1 = tY0 + OC    , tY2 = tY0 + (OC << 1);
			int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
			int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6, tY7 = tY0 + OC * 7;
			tY0 = IF_int(ly[0], tY0, -1); tY1 = IF_int(ly[1], tY1, -1);
			tY2 = IF_int(ly[2], tY2, -1); tY3 = IF_int(ly[3], tY3, -1);
			tY4 = IF_int(ly[4], tY4, -1); tY5 = IF_int(ly[5], tY5, -1);
			tY6 = IF_int(ly[6], tY6, -1); tY7 = IF_int(ly[7], tY7, -1);
			y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
			y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
			y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
			y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

			//load 2 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 7 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1), W3 = W0 + IC * 3;
			const int W4 = W0 + (IC << 2), W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			float2 w6 = *(float2*)(W + W0);
			float2 w5 = *(float2*)(W + W1);
			float2 w4 = *(float2*)(W + W2);
			float2 w3 = *(float2*)(W + W3);
			float2 w2 = *(float2*)(W + W4);
			float2 w1 = *(float2*)(W + W5);
			float2 w0 = *(float2*)(W + W6);

			//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
			winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
			winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
			winograd_f2x7_D(d, y);

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

			Ds[buf][Yk][0][Di] = d[0]; Ds[buf][Yk][1][Di] = d[1];
			Ds[buf][Yk][2][Di] = d[2]; Ds[buf][Yk][3][Di] = d[3];
			Ds[buf][Yk][4][Di] = d[4]; Ds[buf][Yk][5][Di] = d[5];
			Ds[buf][Yk][6][Di] = d[6]; Ds[buf][Yk][7][Di] = d[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31)    ]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31) + 4]);

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
	float2 (* __restrict__ Xs)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float x0[2], x1[2], x2[2], x3[2];
	__syncthreads();

	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j0 - j3}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v0.x, v0.y, v1.x, v1.y };//{ic0, ic1}, {ic4, ic5}}
	*(float4*)(&Xs[ux][uy][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Xs[ux][uy][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Xs[ux][uy][6]) = { v6.x, v6.y, v7.x, v7.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v0.z, v0.w, v1.z, v1.w };//{ic2, ic3}, {ic6, ic7}}
	*(float4*)(&Xs[ux][uy][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Xs[ux][uy][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Xs[ux][uy][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j4 - j7}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v8.x,  v8.y,  v9.x,  v9.y };//{ic0, ic1}, {ic4, ic5}}
	*(float4*)(&Xs[ux][uy][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Xs[ux][uy][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Xs[ux][uy][6]) = { v14.x, v14.y, v15.x, v15.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v8.z,  v8.w,  v9.z,  v9.w };//{ic2, ic3}, {ic6, ic7}}
	*(float4*)(&Xs[ux][uy][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Xs[ux][uy][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Xs[ux][uy][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
}

#endif


//LB = 4, OC % 8 == 0, IWr % 2 == 0, template<IC, OC>
#ifndef DECONV_3D_DX_WINOGRAD_F2X7_KERNEL_64X64C_CT_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F2X7_KERNEL_64X64C_CT_TEXTURE

//for: feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 7.643 msec, Performace = 13767.7 GFlop/s

template<int FH, int IC, int OC>
__global__ void winograd_f2x7_kernel_64x64C_CT_tex(
	cudaTextureObject_t       deltaY, int OH, int OW, 
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW, 
	int oph, int opw,//sh = sw = 1
	int iw_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(ic): 64]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << 6);
	const int Gk = (ty & 7), Gi = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 7 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << 6);//32 * 2 = 64
	const int Yk = (tx & 7), Yi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + (Yi << 1);
	const int IWr2 = (IW - iw_index) >> 1 << 1, IH_IWr2 = IH * IWr2;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr2, IWr2); tiw0 += iw_index;
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Yk;//deltaY[tn0, toh0, tow0, Yk]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//4 * 8
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ux >> 1)) << 1), xj1 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr2, IWr2); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IWr2, IWr2); xiw1 += iw_index;
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//X00 = xj0 * IC + xic0;
	const int X10 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;//X10 = X00 + IC * 8;

	//======[compute area1: local]======================================================
	const int Di = (Yi + (Yk >> 2 << 4)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], y[8], d[8]; char ly[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		ly[0] = lh0 && (tow0 >=  0) && (tow0     < OW);
		ly[1] = lh0 && (tow0 >= -1) && (tow0 + 1 < OW);
		ly[2] = lh0 && (tow0 >= -2) && (tow0 + 2 < OW);
		ly[3] = lh0 && (tow0 >= -3) && (tow0 + 3 < OW);
		ly[4] = lh0 && (tow0 >= -4) && (tow0 + 4 < OW);
		ly[5] = lh0 && (tow0 >= -5) && (tow0 + 5 < OW);
		ly[6] = lh0 && (tow0 >= -6) && (tow0 + 6 < OW);
		ly[7] = lh0 && (tow0 >= -7) && (tow0 + 7 < OW);
		int tY0 = Y0 + fh * OW * OC;
		int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1);
		int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
		int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6, tY7 = tY0 + OC * 7;
		tY0 = IF_int(ly[0], tY0, -1); tY1 = IF_int(ly[1], tY1, -1);
		tY2 = IF_int(ly[2], tY2, -1); tY3 = IF_int(ly[3], tY3, -1);
		tY4 = IF_int(ly[4], tY4, -1); tY5 = IF_int(ly[5], tY5, -1);
		tY6 = IF_int(ly[6], tY6, -1); tY7 = IF_int(ly[7], tY7, -1);
		y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
		y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
		y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
		y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

		//load 2 group from W[OC, FH, 7, IC]
		const int W0 = -fh * 7 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1), W3 = W0 + IC * 3;
		const int W4 = W0 + (IC << 2), W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		float2 w6 = *(float2*)(W + W0);
		float2 w5 = *(float2*)(W + W1);
		float2 w4 = *(float2*)(W + W2);
		float2 w3 = *(float2*)(W + W3);
		float2 w2 = *(float2*)(W + W4);
		float2 w1 = *(float2*)(W + W5);
		float2 w0 = *(float2*)(W + W6);

		//Winograd transform: W(6) -> G(8); Y(8) -> D(8)
		winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
		winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
		winograd_f2x7_D(d, y);

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

		Ds[buf][Yk][0][Di] = d[0]; Ds[buf][Yk][1][Di] = d[1];
		Ds[buf][Yk][2][Di] = d[2]; Ds[buf][Yk][3][Di] = d[3];
		Ds[buf][Yk][4][Di] = d[4]; Ds[buf][Yk][5][Di] = d[5];
		Ds[buf][Yk][6][Di] = d[6]; Ds[buf][Yk][7][Di] = d[7];
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31) + 4]);

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
			buf ^= 1;

			//load 1 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc;
			int tY1 = tY0 + OC    , tY2 = tY0 + (OC << 1);
			int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2);
			int tY5 = tY0 + OC * 5, tY6 = tY0 + OC * 6, tY7 = tY0 + OC * 7;
			tY0 = IF_int(ly[0], tY0, -1); tY1 = IF_int(ly[1], tY1, -1);
			tY2 = IF_int(ly[2], tY2, -1); tY3 = IF_int(ly[3], tY3, -1);
			tY4 = IF_int(ly[4], tY4, -1); tY5 = IF_int(ly[5], tY5, -1);
			tY6 = IF_int(ly[6], tY6, -1); tY7 = IF_int(ly[7], tY7, -1);
			y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
			y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
			y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
			y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

			//load 2 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 7 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1), W3 = W0 + IC * 3;
			const int W4 = W0 + (IC << 2), W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			float2 w6 = *(float2*)(W + W0);
			float2 w5 = *(float2*)(W + W1);
			float2 w4 = *(float2*)(W + W2);
			float2 w3 = *(float2*)(W + W3);
			float2 w2 = *(float2*)(W + W4);
			float2 w1 = *(float2*)(W + W5);
			float2 w0 = *(float2*)(W + W6);

			//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
			winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
			winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
			winograd_f2x7_D(d, y);

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7] };

			Ds[buf][Yk][0][Di] = d[0]; Ds[buf][Yk][1][Di] = d[1];
			Ds[buf][Yk][2][Di] = d[2]; Ds[buf][Yk][3][Di] = d[3];
			Ds[buf][Yk][4][Di] = d[4]; Ds[buf][Yk][5][Di] = d[5];
			Ds[buf][Yk][6][Di] = d[6]; Ds[buf][Yk][7][Di] = d[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31)    ]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ux][((DIdx + (ik >> 2 << 4)) & 31) + 4]);

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
		buf ^= 1;
	}

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Xs)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float x0[2], x1[2], x2[2], x3[2];
	__syncthreads();

	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j0 - j3}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v0.x, v0.y, v1.x, v1.y };//{ic0, ic1}, {ic4, ic5}}
	*(float4*)(&Xs[ux][uy][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Xs[ux][uy][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Xs[ux][uy][6]) = { v6.x, v6.y, v7.x, v7.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v0.z, v0.w, v1.z, v1.w };//{ic2, ic3}, {ic6, ic7}}
	*(float4*)(&Xs[ux][uy][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Xs[ux][uy][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Xs[ux][uy][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j4 - j7}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v8.x,  v8.y,  v9.x,  v9.y };//{ic0, ic1}, {ic4, ic5}}
	*(float4*)(&Xs[ux][uy][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Xs[ux][uy][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Xs[ux][uy][6]) = { v14.x, v14.y, v15.x, v15.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v8.z,  v8.w,  v9.z,  v9.w };//{ic2, ic3}, {ic6, ic7}}
	*(float4*)(&Xs[ux][uy][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Xs[ux][uy][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Xs[ux][uy][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
}

#endif


//LB = 4, OC % 8 == 0, IWr % 4 == 0, pw >= 3
#ifndef DECONV_3D_DX_WINOGRAD_F2X7_RUSE_KERNEL_64X64C_P3_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F2X7_RUSE_KERNEL_64X64C_P3_TEXTURE

//for: feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 49, Time = 6.706 msec, Performace = 15691.4 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 6.665 msec, Performace = 15788   GFlop/s

template<int FH>
__global__ void winograd_f2x7_ruse_kernel_64x64C_p3_tex(
	cudaTextureObject_t       deltaY, int OH, int OW, 
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW, 
	int IC, int OC,
	int oph, int opw,//sh = sw = 1
	int iw_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(ic): 64]
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

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (bx << 6);
	const int Gk = ty, Gi = tx << 2;//[8, 16*4 = 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 7 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (by << 6);//32 * 2 = 64
	const int Yk = (tx & 7), Yi = ((ty << 1) + (tx > 7)) << 1;//[8, 16*2 = 32]
	const int tj0 = bj0 + (Yi << 1);
	const int IWr4 = (IW - iw_index) >> 2 << 2, IH_IWr4 = IH * IWr4;//2 * 2 = 4
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr4, IWr4); tiw0 += iw_index;
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Yk;//deltaY[tn0, toh0, tow0, Yk]

	//prepare for threadIdx
	const int GIdx = ((tx & 1) + ((tx >> 3) << 1)) << 4;//4
	const int DIdx = ((tx & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ty & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ty >> 1)) << 1), xj1 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr4, IWr4); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IWr4, IWr4); xiw1 += iw_index;
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//X00 = xj0 * IC + xic0;
	const int X10 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;//X10 = X00 + IC * 8;

	//======[compute area1: local]======================================================
	const int Di = (Yi + (Yk >> 1 << 3)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], g2[8], g3[8], y[10], d0[8], d1[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 4 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 7 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		float4 w6 = *(float4*)(W + W0);
		float4 w5 = *(float4*)(W + W1);
		float4 w4 = *(float4*)(W + W2);
		float4 w3 = *(float4*)(W + W3);
		float4 w2 = *(float4*)(W + W4);
		float4 w1 = *(float4*)(W + W5);
		float4 w0 = *(float4*)(W + W6);

		//load 2 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=   0) && (tow0      < OW);//opw = -3
		bool ly1 = lh0 && (tow0 >=  -1) && (tow0 +  1 < OW);//opw = -2
		bool ly2 = lh0 && (tow0 >=  -2) && (tow0 +  2 < OW);//opw = -1
		bool ly7 = lh0 && (tow0 >=  -7) && (tow0 +  7 < OW);//opw = +1
		bool ly8 = lh0 && (tow0 >=  -8) && (tow0 +  8 < OW);//opw = +2
		bool ly9 = lh0 && (tow0 >=  -9) && (tow0 +  9 < OW);//opw = +3
		int tY0 = Y0 + fh * OW * OC, tY1 = tY0 + OC;
		int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
		int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
		int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC * 7;
		int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC * 9;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1); tY2 = IF_int(ly2, tY2, -1);
		tY3 = IF_int(lh0, tY3, -1); tY4 = IF_int(lh0, tY4, -1); 
		tY5 = IF_int(lh0, tY5, -1); tY6 = IF_int(lh0, tY6, -1); 
		tY7 = IF_int(ly7, tY7, -1); tY8 = IF_int(ly8, tY8, -1); tY9 = IF_int(ly9, tY9, -1);
		y[0x0] = tex1Dfetch<float>(deltaY, tY0); y[0x1] = tex1Dfetch<float>(deltaY, tY1);
		y[0x2] = tex1Dfetch<float>(deltaY, tY2); y[0x3] = tex1Dfetch<float>(deltaY, tY3);
		y[0x4] = tex1Dfetch<float>(deltaY, tY4); y[0x5] = tex1Dfetch<float>(deltaY, tY5);
		y[0x6] = tex1Dfetch<float>(deltaY, tY6); y[0x7] = tex1Dfetch<float>(deltaY, tY7);
		y[0x8] = tex1Dfetch<float>(deltaY, tY8); y[0x9] = tex1Dfetch<float>(deltaY, tY9);

		//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
		winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
		winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
		winograd_f2x7_G(g2, w0.z, w1.z, w2.z, w3.z, w4.z, w5.z, w6.z);
		winograd_f2x7_G(g3, w0.w, w1.w, w2.w, w3.w, w4.w, w5.w, w6.w);
		winograd_f2x7_D(d0, y);//x[0 - 7]
		winograd_f2x7_D_oft(d1, y, 2);//x[2 - 9]

		//write to shread memory
		*(float2*)(&Ds[buf][Yk][0][Di]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Yk][1][Di]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Yk][2][Di]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Yk][3][Di]) = { d0[3], d1[3] };
		*(float2*)(&Ds[buf][Yk][4][Di]) = { d0[4], d1[4] };
		*(float2*)(&Ds[buf][Yk][5][Di]) = { d0[5], d1[5] };
		*(float2*)(&Ds[buf][Yk][6][Di]) = { d0[6], d1[6] };
		*(float2*)(&Ds[buf][Yk][7][Di]) = { d0[7], d1[7] };

		*(float4*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0], g2[0], g3[0] };
		*(float4*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1], g2[1], g3[1] };
		*(float4*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2], g2[2], g3[2] };
		*(float4*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3], g2[3], g3[3] };
		*(float4*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4], g2[4], g3[4] };
		*(float4*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5], g2[5], g3[5] };
		*(float4*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6], g2[6], g3[6] };
		*(float4*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7], g2[7], g3[7] };
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 b0 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

				//=======================================================
				float4 a0 = *(float4*)(&Gs[buf][ik][ty][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ty][GIdx + 4]);

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
				float4 a2 = *(float4*)(&Gs[buf][ik][ty][GIdx +  8]);
				float4 a3 = *(float4*)(&Gs[buf][ik][ty][GIdx + 12]);
	
				//ic[8 - 11]             ic[12 - 15]
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

			//load 4 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 7 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			float4 w6 = *(float4*)(W + W0);
			float4 w5 = *(float4*)(W + W1);
			float4 w4 = *(float4*)(W + W2);
			float4 w3 = *(float4*)(W + W3);
			float4 w2 = *(float4*)(W + W4);
			float4 w1 = *(float4*)(W + W5);
			float4 w0 = *(float4*)(W + W6);

			//load 2 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc, tY1 = tY0 + OC;
			int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
			int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
			int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC * 7;
			int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC * 9;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1); tY2 = IF_int(ly2, tY2, -1);
			tY3 = IF_int(lh0, tY3, -1); tY4 = IF_int(lh0, tY4, -1); 
			tY5 = IF_int(lh0, tY5, -1); tY6 = IF_int(lh0, tY6, -1);
			tY7 = IF_int(ly7, tY7, -1); tY8 = IF_int(ly8, tY8, -1); tY9 = IF_int(ly9, tY9, -1);
			y[0x0] = tex1Dfetch<float>(deltaY, tY0); y[0x1] = tex1Dfetch<float>(deltaY, tY1);
			y[0x2] = tex1Dfetch<float>(deltaY, tY2); y[0x3] = tex1Dfetch<float>(deltaY, tY3);
			y[0x4] = tex1Dfetch<float>(deltaY, tY4); y[0x5] = tex1Dfetch<float>(deltaY, tY5);
			y[0x6] = tex1Dfetch<float>(deltaY, tY6); y[0x7] = tex1Dfetch<float>(deltaY, tY7);
			y[0x8] = tex1Dfetch<float>(deltaY, tY8); y[0x9] = tex1Dfetch<float>(deltaY, tY9);

			//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
			winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
			winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
			winograd_f2x7_G(g2, w0.z, w1.z, w2.z, w3.z, w4.z, w5.z, w6.z);
			winograd_f2x7_G(g3, w0.w, w1.w, w2.w, w3.w, w4.w, w5.w, w6.w);
			winograd_f2x7_D(d0, y);//x[0 - 7]
			winograd_f2x7_D_oft(d1, y, 2);//x[2 - 9]

			//write to shread memory
			*(float2*)(&Ds[buf][Yk][0][Di]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Yk][1][Di]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Yk][2][Di]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Yk][3][Di]) = { d0[3], d1[3] };
			*(float2*)(&Ds[buf][Yk][4][Di]) = { d0[4], d1[4] };
			*(float2*)(&Ds[buf][Yk][5][Di]) = { d0[5], d1[5] };
			*(float2*)(&Ds[buf][Yk][6][Di]) = { d0[6], d1[6] };
			*(float2*)(&Ds[buf][Yk][7][Di]) = { d0[7], d1[7] };

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
			float4 a2 = *(float4*)(&Gs[buf][ik][ty][GIdx +  8]);
			float4 a3 = *(float4*)(&Gs[buf][ik][ty][GIdx + 12]);

			//ic[8 - 11]             ic[12 - 15]
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

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Xs)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float x0[2], x1[2], x2[2], x3[2];
	__syncthreads();

	//------write-read-turn0: {ic0 - ic7} x {j0 - j3}------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v0.x, v0.y, v1.x, v1.y };
	*(float4*)(&Xs[ty][tx][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Xs[ty][tx][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Xs[ty][tx][6]) = { v6.x, v6.y, v7.x, v7.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v0.z, v0.w, v1.z, v1.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic0 - ic7} x {j4 - j7}-------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = {  v8.x,  v8.y,  v9.x,  v9.y };
	*(float4*)(&Xs[ty][tx][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Xs[ty][tx][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Xs[ty][tx][6]) = { v14.x, v14.y, v15.x, v15.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = {  v8.z,  v8.w,  v9.z,  v9.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();

	//------write-read-turn0: {ic8 - ic15} x {j0 - j3}-----------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v16.x, v16.y, v17.x, v17.y };
	*(float4*)(&Xs[ty][tx][2]) = { v18.x, v18.y, v19.x, v19.y };
	*(float4*)(&Xs[ty][tx][4]) = { v20.x, v20.y, v21.x, v21.y };
	*(float4*)(&Xs[ty][tx][6]) = { v22.x, v22.y, v23.x, v23.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v16.z, v16.w, v17.z, v17.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v18.z, v18.w, v19.z, v19.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v20.z, v20.w, v21.z, v21.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v22.z, v22.w, v23.z, v23.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00          + 8) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1 + 8) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic8 - ic15} x {j4 - j7}-------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v24.x, v24.y, v25.x, v25.y };
	*(float4*)(&Xs[ty][tx][2]) = { v26.x, v26.y, v27.x, v27.y };
	*(float4*)(&Xs[ty][tx][4]) = { v28.x, v28.y, v29.x, v29.y };
	*(float4*)(&Xs[ty][tx][6]) = { v30.x, v30.y, v31.x, v31.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v24.z, v24.w, v25.z, v25.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v26.z, v26.w, v27.z, v27.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v28.z, v28.w, v29.z, v29.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v30.z, v30.w, v31.z, v31.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10          + 8) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1 + 8) = { x0[1], x1[1], x2[1], x3[1] };
}

#endif


//LB = 4, OC % 8 == 0, IWr % 4 == 0, pw >= 3, template<IC, OC>
#ifndef DECONV_3D_DX_WINOGRAD_F2X7_RUSE_KERNEL_64X64C_P3_CT_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F2X7_RUSE_KERNEL_64X64C_P3_CT_TEXTURE

//for: feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 49, Time = 6.275 msec, Performace = 16769.2 GFlop/s
//for: feature = (16, 16), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 49, Time = 6.272 msec, Performace = 16777.2 GFlop/s

template<int FH, int IC, int OC>
__global__ void winograd_f2x7_ruse_kernel_64x64C_p3_CT_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW,
	int oph, int opw,//sh = sw = 1
	int iw_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(ic): 64]
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

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (bx << 6);
	const int Gk = ty, Gi = tx << 2;//[8, 16*4 = 64]
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 7 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (by << 6);//32 * 2 = 64
	const int Yk = (tx & 7), Yi = ((ty << 1) + (tx > 7)) << 1;//[8, 16*2 = 32]
	const int tj0 = bj0 + (Yi << 1);
	const int IWr4 = (IW - iw_index) >> 2 << 2, IH_IWr4 = IH * IWr4;//2 * 2 = 4
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr4, IWr4); tiw0 += iw_index;
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Yk;//deltaY[tn0, toh0, tow0, Yk]

	//prepare for threadIdx
	const int GIdx = ((tx & 1) + ((tx >> 3) << 1)) << 4;//4
	const int DIdx = ((tx & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ty & 1) << 2);
	const int xj0 = bj0 + ((DIdx + (ty >> 1)) << 1), xj1 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr4, IWr4); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IWr4, IWr4); xiw1 += iw_index;
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//X00 = xj0 * IC + xic0;
	const int X10 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;//X10 = X00 + IC * 8;

	//======[compute area1: local]======================================================
	const int Di = (Yi + (Yk >> 1 << 3)) & 31;//avoid bank conflict (1 / 4)
	float g0[8], g1[8], g2[8], g3[8], y[10], d0[8], d1[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 4 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 7 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		float4 w6 = *(float4*)(W + W0);
		float4 w5 = *(float4*)(W + W1);
		float4 w4 = *(float4*)(W + W2);
		float4 w3 = *(float4*)(W + W3);
		float4 w2 = *(float4*)(W + W4);
		float4 w1 = *(float4*)(W + W5);
		float4 w0 = *(float4*)(W + W6);

		//load 2 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=   0) && (tow0      < OW);//opw = -3
		bool ly1 = lh0 && (tow0 >=  -1) && (tow0 +  1 < OW);//opw = -2
		bool ly2 = lh0 && (tow0 >=  -2) && (tow0 +  2 < OW);//opw = -1
		bool ly7 = lh0 && (tow0 >=  -7) && (tow0 +  7 < OW);//opw = +1
		bool ly8 = lh0 && (tow0 >=  -8) && (tow0 +  8 < OW);//opw = +2
		bool ly9 = lh0 && (tow0 >=  -9) && (tow0 +  9 < OW);//opw = +3
		int tY0 = Y0 + fh * OW * OC, tY1 = tY0 + OC;
		int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
		int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
		int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC * 7;
		int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC * 9;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1); tY2 = IF_int(ly2, tY2, -1);
		tY3 = IF_int(lh0, tY3, -1); tY4 = IF_int(lh0, tY4, -1); 
		tY5 = IF_int(lh0, tY5, -1); tY6 = IF_int(lh0, tY6, -1); 
		tY7 = IF_int(ly7, tY7, -1); tY8 = IF_int(ly8, tY8, -1); tY9 = IF_int(ly9, tY9, -1);
		y[0x0] = tex1Dfetch<float>(deltaY, tY0); y[0x1] = tex1Dfetch<float>(deltaY, tY1);
		y[0x2] = tex1Dfetch<float>(deltaY, tY2); y[0x3] = tex1Dfetch<float>(deltaY, tY3);
		y[0x4] = tex1Dfetch<float>(deltaY, tY4); y[0x5] = tex1Dfetch<float>(deltaY, tY5);
		y[0x6] = tex1Dfetch<float>(deltaY, tY6); y[0x7] = tex1Dfetch<float>(deltaY, tY7);
		y[0x8] = tex1Dfetch<float>(deltaY, tY8); y[0x9] = tex1Dfetch<float>(deltaY, tY9);

		//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
		winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
		winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
		winograd_f2x7_G(g2, w0.z, w1.z, w2.z, w3.z, w4.z, w5.z, w6.z);
		winograd_f2x7_G(g3, w0.w, w1.w, w2.w, w3.w, w4.w, w5.w, w6.w);
		winograd_f2x7_D(d0, y);//x[0 - 7]
		winograd_f2x7_D_oft(d1, y, 2);//x[2 - 9]

		//write to shread memory
		*(float2*)(&Ds[buf][Yk][0][Di]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Yk][1][Di]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Yk][2][Di]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Yk][3][Di]) = { d0[3], d1[3] };
		*(float2*)(&Ds[buf][Yk][4][Di]) = { d0[4], d1[4] };
		*(float2*)(&Ds[buf][Yk][5][Di]) = { d0[5], d1[5] };
		*(float2*)(&Ds[buf][Yk][6][Di]) = { d0[6], d1[6] };
		*(float2*)(&Ds[buf][Yk][7][Di]) = { d0[7], d1[7] };

		*(float4*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0], g2[0], g3[0] };
		*(float4*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1], g2[1], g3[1] };
		*(float4*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2], g2[2], g3[2] };
		*(float4*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3], g2[3], g3[3] };
		*(float4*)(&Gs[buf][Gk][4][Gi]) = { g0[4], g1[4], g2[4], g3[4] };
		*(float4*)(&Gs[buf][Gk][5][Gi]) = { g0[5], g1[5], g2[5], g3[5] };
		*(float4*)(&Gs[buf][Gk][6][Gi]) = { g0[6], g1[6], g2[6], g3[6] };
		*(float4*)(&Gs[buf][Gk][7][Gi]) = { g0[7], g1[7], g2[7], g3[7] };
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 b0 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31)    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ty][((DIdx + (ik >> 1 << 3)) & 31) + 4]);

				//=======================================================
				float4 a0 = *(float4*)(&Gs[buf][ik][ty][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ty][GIdx + 4]);

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
				float4 a2 = *(float4*)(&Gs[buf][ik][ty][GIdx +  8]);
				float4 a3 = *(float4*)(&Gs[buf][ik][ty][GIdx + 12]);
	
				//ic[8 - 11]             ic[12 - 15]
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

			//load 4 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 7 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			float4 w6 = *(float4*)(W + W0);
			float4 w5 = *(float4*)(W + W1);
			float4 w4 = *(float4*)(W + W2);
			float4 w3 = *(float4*)(W + W3);
			float4 w2 = *(float4*)(W + W4);
			float4 w1 = *(float4*)(W + W5);
			float4 w0 = *(float4*)(W + W6);

			//load 2 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc, tY1 = tY0 + OC;
			int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
			int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
			int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC * 7;
			int tY8 = tY0 + (OC << 3), tY9 = tY0 + OC * 9;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1); tY2 = IF_int(ly2, tY2, -1);
			tY3 = IF_int(lh0, tY3, -1); tY4 = IF_int(lh0, tY4, -1); 
			tY5 = IF_int(lh0, tY5, -1); tY6 = IF_int(lh0, tY6, -1);
			tY7 = IF_int(ly7, tY7, -1); tY8 = IF_int(ly8, tY8, -1); tY9 = IF_int(ly9, tY9, -1);
			y[0x0] = tex1Dfetch<float>(deltaY, tY0); y[0x1] = tex1Dfetch<float>(deltaY, tY1);
			y[0x2] = tex1Dfetch<float>(deltaY, tY2); y[0x3] = tex1Dfetch<float>(deltaY, tY3);
			y[0x4] = tex1Dfetch<float>(deltaY, tY4); y[0x5] = tex1Dfetch<float>(deltaY, tY5);
			y[0x6] = tex1Dfetch<float>(deltaY, tY6); y[0x7] = tex1Dfetch<float>(deltaY, tY7);
			y[0x8] = tex1Dfetch<float>(deltaY, tY8); y[0x9] = tex1Dfetch<float>(deltaY, tY9);

			//Winograd transform: W(7) -> G(8); Y(8) -> D(8)
			winograd_f2x7_G(g0, w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x);
			winograd_f2x7_G(g1, w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y);
			winograd_f2x7_G(g2, w0.z, w1.z, w2.z, w3.z, w4.z, w5.z, w6.z);
			winograd_f2x7_G(g3, w0.w, w1.w, w2.w, w3.w, w4.w, w5.w, w6.w);
			winograd_f2x7_D(d0, y);//x[0 - 7]
			winograd_f2x7_D_oft(d1, y, 2);//x[2 - 9]

			//write to shread memory
			*(float2*)(&Ds[buf][Yk][0][Di]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Yk][1][Di]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Yk][2][Di]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Yk][3][Di]) = { d0[3], d1[3] };
			*(float2*)(&Ds[buf][Yk][4][Di]) = { d0[4], d1[4] };
			*(float2*)(&Ds[buf][Yk][5][Di]) = { d0[5], d1[5] };
			*(float2*)(&Ds[buf][Yk][6][Di]) = { d0[6], d1[6] };
			*(float2*)(&Ds[buf][Yk][7][Di]) = { d0[7], d1[7] };

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
			float4 a2 = *(float4*)(&Gs[buf][ik][ty][GIdx +  8]);
			float4 a3 = *(float4*)(&Gs[buf][ik][ty][GIdx + 12]);

			//ic[8 - 11]             ic[12 - 15]
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

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Xs)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float x0[2], x1[2], x2[2], x3[2];
	__syncthreads();

	//------write-read-turn0: {ic0 - ic7} x {j0 - j3}------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v0.x, v0.y, v1.x, v1.y };
	*(float4*)(&Xs[ty][tx][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Xs[ty][tx][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Xs[ty][tx][6]) = { v6.x, v6.y, v7.x, v7.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v0.z, v0.w, v1.z, v1.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic0 - ic7} x {j4 - j7}-------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = {  v8.x,  v8.y,  v9.x,  v9.y };
	*(float4*)(&Xs[ty][tx][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Xs[ty][tx][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Xs[ty][tx][6]) = { v14.x, v14.y, v15.x, v15.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = {  v8.z,  v8.w,  v9.z,  v9.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();

	//------write-read-turn0: {ic8 - ic15} x {j0 - j3}-----------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v16.x, v16.y, v17.x, v17.y };
	*(float4*)(&Xs[ty][tx][2]) = { v18.x, v18.y, v19.x, v19.y };
	*(float4*)(&Xs[ty][tx][4]) = { v20.x, v20.y, v21.x, v21.y };
	*(float4*)(&Xs[ty][tx][6]) = { v22.x, v22.y, v23.x, v23.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v16.z, v16.w, v17.z, v17.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v18.z, v18.w, v19.z, v19.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v20.z, v20.w, v21.z, v21.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v22.z, v22.w, v23.z, v23.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00          + 8) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC * 1 + 8) = { x0[1], x1[1], x2[1], x3[1] };
	__syncthreads();
	
	//------write-read-turn0: {ic8 - ic15} x {j4 - j7}-------------------------------------
	*(float4*)(&Xs[ty][tx][0]) = { v24.x, v24.y, v25.x, v25.y };
	*(float4*)(&Xs[ty][tx][2]) = { v26.x, v26.y, v27.x, v27.y };
	*(float4*)(&Xs[ty][tx][4]) = { v28.x, v28.y, v29.x, v29.y };
	*(float4*)(&Xs[ty][tx][6]) = { v30.x, v30.y, v31.x, v31.y };

	*(float4*)(&Xs[ty][tx + 16][0]) = { v24.z, v24.w, v25.z, v25.w };
	*(float4*)(&Xs[ty][tx + 16][2]) = { v26.z, v26.w, v27.z, v27.w };
	*(float4*)(&Xs[ty][tx + 16][4]) = { v28.z, v28.w, v29.z, v29.w };
	*(float4*)(&Xs[ty][tx + 16][6]) = { v30.z, v30.w, v31.z, v31.w };
	__syncthreads();

	a[0] = Xs[0][tx][ty]; a[1] = Xs[1][tx][ty];
	a[2] = Xs[2][tx][ty]; a[3] = Xs[3][tx][ty];
	a[4] = Xs[4][tx][ty]; a[5] = Xs[5][tx][ty];
	a[6] = Xs[6][tx][ty]; a[7] = Xs[7][tx][ty];
	winograd_f2x7_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f2x7_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }

	a[0] = Xs[0][tx + 16][ty]; a[1] = Xs[1][tx + 16][ty];
	a[2] = Xs[2][tx + 16][ty]; a[3] = Xs[3][tx + 16][ty];
	a[4] = Xs[4][tx + 16][ty]; a[5] = Xs[5][tx + 16][ty];
	a[6] = Xs[6][tx + 16][ty]; a[7] = Xs[7][tx + 16][ty];
	winograd_f2x7_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f2x7_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10          + 8) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC * 1 + 8) = { x0[1], x1[1], x2[1], x3[1] };
}

#endif


#endif
