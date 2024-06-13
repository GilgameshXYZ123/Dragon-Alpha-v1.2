#pragma once

#ifndef DECONV_3D_DX_WINOGRAD_S4_F2X3R_H
#define DECONV_3D_DX_WINOGRAD_S4_F2X3R_H

//(1) sh = sw = 1
//(2) FW = 3
//(3) IW % 2: group = 2 elements
#ifndef DECONV_3D_DX_WINOGRAD_S4_F2X3R_CALL
#define DECONV_3D_DX_WINOGRAD_S4_F2X3R_CALL

//================[Cooperate: 64 (OC) * 128 (GM)]=============================
//Cooperate: GM = N * IH * ((IW - ow_index) / 2 * 2)

//IWr % 2 == 0
#define winograd_f2x3_k64x128C_tex(stream, iw_index, deltaY, OH, OW, W, FH, deltaX, IH, IW, IC, OC, ph, pw, IWr) \
	winograd_f2x3_kernel_64x128C_tex<FH>\
		<<< dim3(IC>>6, ((N*IH)>>6)*(IWr>>1)), dim3(16, 16), 0, stream >>>\
			(deltaY,OH,OW, W+((FH-1)*3*IC), deltaX,IH,IW, IC,OC, (FH-1-ph),(2-pw), iw_index)

//IWr % 4 == 0
#define winograd_f2x3_k64x128x4C_tex(stream, iw_index, deltaY, OH, OW, W, FH, deltaX, IH, IW, IC, OC, ph, pw, IWr) \
	winograd_f2x3_kernel_64x128x4C_tex<FH>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IWr>>2)), dim3(16, 16), 0, stream >>>\
			(deltaY,OH,OW, W+((FH-1)*3*IC), deltaX,IH,IW, IC,OC, (FH-1-ph),(2-pw), iw_index)

//Split FW: IWr % 2 == 0, FW % 3 == 0
#define winograd_SFW_f2x3_k64x128C_tex(stream, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, IWr) \
	winograd_SFW_f2x3_kernel_64x128C_tex<FW>\
		<<< dim3(IC>>6, ((N*IH)>>6)*(IWr>>1)), dim3(16, 16), 0, stream >>>\
			(deltaY,OH,OW, (W+((FH-1)*FW+(FW-3))*IC),FH, deltaX,IH,IW, IC,OC, (FH-1-ph),(FW-1-pw), iw_index)

//Split FW: IWr % 4 == 0, FW % 3 == 0
#define winograd_SFW_f2x3_k64x128x4C_tex(stream, iw_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, IWr) \
	winograd_SFW_f2x3_kernel_64x128x4C_tex<FW>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IWr>>2)), dim3(16, 16), 0, stream >>>\
			(deltaY,OH,OW, (W+((FH-1)*FW+(FW-3))*IC),FH, deltaX,IH,IW, IC,OC, (FH-1-ph),(FW-1-pw), iw_index)

#endif


//================[Cooperate: 64(IC) * 128(GM)]===============================
//LB = 4, OC % 8 == 0, IWr % 2 == 0
#ifndef DECONV_3D_WINOGRAD_F2X3_KERNEL_64X128C_TEXTURE
#define DECONV_3D_WINOGRAD_F2X3_KERNEL_64X128C_TEXTURE

//for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 18, Time = 3.083 msec, Performace = 12538 GFlop/s
//for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 18, Time = 3.01 msec, Performace = 12842.1 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 18, Time = 2.795 msec, Performace = 13829.9 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 18, Time = 2.666 msec, Performace = 14499.1 GFlop/s

template<int FH>
__global__ void winograd_f2x3_kernel_64x128C_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,//sh = sw = 1
	int iw_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][g(ic): 64]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][d( j): 64]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << 6); 
	const int Gk = (ty & 7), Gi = (tx << 2) + ((ty > 7) << 1);
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 3 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << 7);
	const int Dk = (tx & 7), Di = (ty << 2) + ((tx > 7) << 1);
	const int tj0 = bj0 + (Di << 1), tj4 = tj0 + 2;
	const int IWr2 = (IW - iw_index) >> 1 << 1, IH_IWr2 = IH * IWr2;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr2, IWr2); tiw0 += iw_index;//IWr % 2
	get_n_ih_iw_Temp(tj4, tn4, tih4, tiw4, IH_IWr2, IWr2); tiw4 += iw_index;
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int toh4 = tih4 - oph, tow4 = tiw4 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Dk;//deltaY[tn0, toh0, tow0, Dk]
	const int Y4 = ((tn4*OH + toh4)*OW + tow4)*OC + Dk;//deltaY[tn0, toh0, tow0, Dk]

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
	const int GIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int DIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//======[compute area1: local]======================================================
	float w[6], y[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 3 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		*(float2*)(w + 4) = *(float2*)(W + W0);//fw = 2
		*(float2*)(w + 2) = *(float2*)(W + W1);//fw = 1
		*(float2*)(w    ) = *(float2*)(W + W2);//fw = 0

		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);//group0
		bool ly0 = lh0 && (tow0 >=  0) && (tow0     < OW);
		bool ly1 = lh0 && (tow0 >= -1) && (tow0 + 1 < OW);
		bool ly2 = lh0 && (tow0 >= -2) && (tow0 + 2 < OW);
		bool ly3 = lh0 && (tow0 >= -3) && (tow0 + 3 < OW);
		int tY0 = Y0 + fh * OW * OC;
		int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
		tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
		y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
		y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
	
		bool lh4 = (toh4 >= -fh) && (toh4 + fh < OH);//group1
		bool ly4 = lh4 && (tow4 >=  0) && (tow4     < OW);
		bool ly5 = lh4 && (tow4 >= -1) && (tow4 + 1 < OW);
		bool ly6 = lh4 && (tow4 >= -2) && (tow4 + 2 < OW);
		bool ly7 = lh4 && (tow4 >= -3) && (tow4 + 3 < OW);
		int tY4 = Y4 + fh * OW * OC;
		int tY5 = tY4 + OC, tY6 = tY4 + (OC << 1), tY7 = tY4 + OC * 3;
		tY4 = IF_int(ly4, tY4, -1); tY5 = IF_int(ly5, tY5, -1); 
		tY6 = IF_int(ly6, tY6, -1); tY7 = IF_int(ly7, tY7, -1);
		y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5); 
		y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

		//Winograd transform: W(3) -> G(4); Y(4) -> D(4)
		float gst0 = 0.5f*(w[0] + w[2] + w[4]), gst1 = gst0 - w[2];
		float gst2 = 0.5f*(w[1] + w[3] + w[5]), gst3 = gst2 - w[3];
		float g0[4]{ w[0], gst0, gst1, w[4] };
		float g1[4]{ w[1], gst2, gst3, w[5] };

		float d0[4]{ y[0] - y[2], y[1] + y[2], y[2] - y[1], y[1] - y[3] };
		float d1[4]{ y[4] - y[6], y[5] + y[6], y[6] - y[5], y[5] - y[7] };

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };

		*(float2*)(&Ds[buf][Dk][0][Di]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Dk][1][Di]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Dk][2][Di]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Dk][3][Di]) = { d0[3], d1[3] };
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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

			//load 2 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 3 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			*(float2*)(w + 4) = *(float2*)(W + W0);//fw = 2
			*(float2*)(w + 2) = *(float2*)(W + W1);//fw = 1
			*(float2*)(w    ) = *(float2*)(W + W2);//fw = 0

			//load 1 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc;//group0
			int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
			tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
			y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
			y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);

			int tY4 = Y4 + fh * OW * OC + ooc;//group1
			int tY5 = tY4 + OC, tY6 = tY4 + (OC << 1), tY7 = tY4 + OC * 3;
			tY4 = IF_int(ly4, tY4, -1); tY5 = IF_int(ly5, tY5, -1);
			tY6 = IF_int(ly6, tY6, -1); tY7 = IF_int(ly7, tY7, -1);
			y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
			y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

			//Winograd transform: W(3) -> G(4); Y(4) -> D(4)
			float gst0 = 0.5f*(w[0] + w[2] + w[4]), gst1 = gst0 - w[2];
			float gst2 = 0.5f*(w[1] + w[3] + w[5]), gst3 = gst2 - w[3];
			float g0[4]{ w[0], gst0, gst1, w[4] };
			float g1[4]{ w[1], gst2, gst3, w[5] };

			float d0[4]{ y[0] - y[2], y[1] + y[2], y[2] - y[1], y[1] - y[3] };
			float d1[4]{ y[4] - y[6], y[5] + y[6], y[6] - y[5], y[5] - y[7] };

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };

			*(float2*)(&Ds[buf][Dk][0][Di]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Dk][1][Di]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Dk][2][Di]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Dk][3][Di]) = { d0[3], d1[3] };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx    ]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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
	__syncthreads();

	//======[compute area12: block]======================================================
	//prepare for Y[N, OH, OW, OC]
	const int xic0 = bic0 + GIdx;
	const int xj0 = bj0 + ((DIdx + ux) << 1), xj2 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr2, IWr2); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj2, xn2, xih2, xiw2, IH_IWr2, IWr2); xiw2 += iw_index;
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//ux: j0 -> j3
	const int X20 = ((xn2*IH + xih2)*IW + xiw2)*IC + xic0;//ux: j4 -> j7

	float4 (* __restrict__ Xs0)[65][5] = (float4(*)[65][5])(Gs);//4 * 64 * 16
	float4 (* __restrict__ Xs1)[65][5] = (float4(*)[65][5])(Ds);

	float4 a0, a1, a2, a3, a4, a5, a6, a7;
	float4 k0, k1, k2, k3, k4, k5, k6, k7;
	float2 x0, x1, x2, x3, x4, x5, x6, x7;

	//write-turn[0, 1]: j0, j1, j2, j3, [ic0 - ic3], [ic4 - ic7]
	Xs0[ux][uy][0] = v0; Xs1[ux][uy][0] = v1; 
	Xs0[ux][uy][1] = v2; Xs1[ux][uy][1] = v3;
	Xs0[ux][uy][2] = v4; Xs1[ux][uy][2] = v5;
	Xs0[ux][uy][3] = v6; Xs1[ux][uy][3] = v7;
	__syncthreads();

	//read-turn[0, 1]: j0, j1, j2, j3, [ic0 - ic3], [ic4 - ic7]
	a0 = Xs0[0][uy][ux]; a4 = Xs1[0][uy][ux];
	a1 = Xs0[1][uy][ux]; a5 = Xs1[1][uy][ux];
	a2 = Xs0[2][uy][ux]; a6 = Xs1[2][uy][ux];
	a3 = Xs0[3][uy][ux]; a7 = Xs1[3][uy][ux];

	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f2x3_y_f32_64(x0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f2x3_y_f32_64(x1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f2x3_y_f32_64(x2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f2x3_y_f32_64(x3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f2x3_y_f32_64(x4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f2x3_y_f32_64(x5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f2x3_y_f32_64(x6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f2x3_y_f32_64(x7, k7);

	*(float4*)(deltaX + X00     ) = { x0.x, x1.x, x2.x, x3.x }; *(float4*)(deltaX + X00      + 4) = { x4.x, x5.x, x6.x, x7.x };
	*(float4*)(deltaX + X00 + IC) = { x0.y, x1.y, x2.y, x3.y }; *(float4*)(deltaX + X00 + IC + 4) = { x4.y, x5.y, x6.y, x7.y };
	__syncthreads();

	//write-turn2: j4, j5, j6, j7, [ic0 - ic3], [ic4 - ic7]
	Xs0[ux][uy][0] =  v8; Xs1[ux][uy][0] =  v9;
	Xs0[ux][uy][1] = v10; Xs1[ux][uy][1] = v11;
	Xs0[ux][uy][2] = v12; Xs1[ux][uy][2] = v13;
	Xs0[ux][uy][3] = v14; Xs1[ux][uy][3] = v15;
	__syncthreads();

	//write-turn[2, 3]: j4, j5, j6, j7, [ic0 - ic3], [ic4 - c7]
	a0 = Xs0[0][uy][ux]; a4 = Xs1[0][uy][ux];
	a1 = Xs0[1][uy][ux]; a5 = Xs1[1][uy][ux];
	a2 = Xs0[2][uy][ux]; a6 = Xs1[2][uy][ux];
	a3 = Xs0[3][uy][ux]; a7 = Xs1[3][uy][ux];

	//read-turn[2, 3]: j4, j5, j6, j7, [ic0 - ic3], [ic4 - ic7]
	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f2x3_y_f32_64(x0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f2x3_y_f32_64(x1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f2x3_y_f32_64(x2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f2x3_y_f32_64(x3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f2x3_y_f32_64(x4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f2x3_y_f32_64(x5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f2x3_y_f32_64(x6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f2x3_y_f32_64(x7, k7);

	*(float4*)(deltaX + X20     ) = { x0.x, x1.x, x2.x, x3.x }; *(float4*)(deltaX + X20      + 4) = { x4.x, x5.x, x6.x, x7.x };
	*(float4*)(deltaX + X20 + IC) = { x0.y, x1.y, x2.y, x3.y }; *(float4*)(deltaX + X20 + IC + 4) = { x4.y, x5.y, x6.y, x7.y };
}

#endif


//LB = 4, OC % 8 == 0, IWr % 4 == 0
#ifndef DECONV_3D_WINOGRAD_F2X3_KERNEL_64X128X4C_TEXTURE
#define DECONV_3D_WINOGRAD_F2X3_KERNEL_64X128X4C_TEXTURE

//for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 18, Time = 2.988 msec, Performace = 12936.6 GFlop/s
//for: Feature = (64, 64), [N, IC, OC] = [128, 64, 64]
//LB = 4: Size = 18, Time = 2.978 msec, Performace = 12980.1 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 18, Time = 2.748 msec, Performace = 14066.5 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 18, Time = 2.592 msec, Performace = 14913.1 GFlop/s

template<int FH>
__global__ void winograd_f2x3_kernel_64x128x4C_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,//sh = sw = 1
	int iw_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][g(ic): 64]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][d( j): 64]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << 6); 
	const int Gk = (ty & 7), Gi = (tx << 2) + ((ty > 7) << 1);
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * 3 * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << 7);
	const int Dk = (tx & 7), Di = (ty << 2) + ((tx > 7) << 1);
	const int tj0 = bj0 + (Di << 1);
	const int IWr4 = (IW - iw_index) >> 2 << 2, IH_IWr4 = IH * IWr4;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr4, IWr4); tiw0 += iw_index;
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Dk;//deltaY[tn0, toh0, tow0, Dk]

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
	const int GIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int DIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//prepare for Y[N, OH, OW, OC]
	const int xic0 = bic0 + GIdx;
	const int xj0 = bj0 + ((DIdx + ux) << 1), xj2 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr4, IWr4); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj2, xn2, xih2, xiw2, IH_IWr4, IWr4); xiw2 += iw_index;
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//ux: j0 -> j3
	const int X20 = ((xn2*IH + xih2)*IW + xiw2)*IC + xic0;//ux: j4 -> j7

	//======[compute area1: local]======================================================
	float w[6], y[6];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = -fh * 3 * IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		*(float2*)(w + 4) = *(float2*)(W + W0);//fw = 2
		*(float2*)(w + 2) = *(float2*)(W + W1);//fw = 1
		*(float2*)(w    ) = *(float2*)(W + W2);//fw = 0

		//load 2 groups from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=  0) && (tow0     < OW);
		bool ly1 = lh0 && (tow0 >= -1) && (tow0 + 1 < OW);
		bool ly2 = lh0 && (tow0 >= -2) && (tow0 + 2 < OW);
		bool ly3 = lh0 && (tow0 >= -3) && (tow0 + 3 < OW);
		bool ly4 = lh0 && (tow0 >= -4) && (tow0 + 4 < OW);
		bool ly5 = lh0 && (tow0 >= -5) && (tow0 + 5 < OW);
		int tY0 = Y0 + fh * OW * OC;
		int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1);
		int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
		tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
		tY4 = IF_int(ly4, tY4, -1); tY5 = IF_int(ly5, tY5, -1);
		y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
		y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
		y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);

		//Winograd transform: W(3) -> G(4); Y(4) -> D(4)
		float gst0 = 0.5f*(w[0] + w[2] + w[4]), gst1 = gst0 - w[2];
		float gst2 = 0.5f*(w[1] + w[3] + w[5]), gst3 = gst2 - w[3];
		float g0[4]{ w[0], gst0, gst1, w[4] };
		float g1[4]{ w[1], gst2, gst3, w[5] };

		float d0[4]{ y[0] - y[2], y[1] + y[2], y[2] - y[1], y[1] - y[3] };
		float d1[4]{ y[2] - y[4], y[3] + y[4], y[4] - y[3], y[3] - y[5] };

		//write to shread memory
		*(float2*)(&Ds[buf][Dk][0][Di]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Dk][1][Di]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Dk][2][Di]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Dk][3][Di]) = { d0[3], d1[3] };

		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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

			//load 2 group from W[OC, FH, FW, IC]
			const int W0 = (ooc*FH - fh) * 3 * IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			*(float2*)(w + 4) = *(float2*)(W + W0);//fw = 2
			*(float2*)(w + 2) = *(float2*)(W + W1);//fw = 1
			*(float2*)(w    ) = *(float2*)(W + W2);//fw = 0

			//load 2 groups from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc;
			int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1);
			int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
			tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
			tY4 = IF_int(ly4, tY4, -1); tY5 = IF_int(ly5, tY5, -1);
			y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
			y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
			y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float gst0 = 0.5f*(w[0] + w[2] + w[4]), gst1 = gst0 - w[2];
			float gst2 = 0.5f*(w[1] + w[3] + w[5]), gst3 = gst2 - w[3];
			float g0[4]{ w[0], gst0, gst1, w[4] };
			float g1[4]{ w[1], gst2, gst3, w[5] };

			float d0[4]{ y[0] - y[2], y[1] + y[2], y[2] - y[1], y[1] - y[3] };
			float d1[4]{ y[2] - y[4], y[3] + y[4], y[4] - y[3], y[3] - y[5] };

			//write to shread memory
			*(float2*)(&Ds[buf][Dk][0][Di]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Dk][1][Di]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Dk][2][Di]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Dk][3][Di]) = { d0[3], d1[3] };

			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
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
	float4 (* __restrict__ Xs0)[65][5] = (float4(*)[65][5])(Gs);//4 * 64 * 16
	float4 (* __restrict__ Xs1)[65][5] = (float4(*)[65][5])(Ds);

	float4 a0, a1, a2, a3, a4, a5, a6, a7;
	float4 k0, k1, k2, k3, k4, k5, k6, k7;
	float2 x0, x1, x2, x3, x4, x5, x6, x7;

	//write-turn[0, 1]: j0, j1, j2, j3, [ic0 - ic3], [ic4 - ic7]
	Xs0[ux][uy][0] = v0; Xs1[ux][uy][0] = v1; 
	Xs0[ux][uy][1] = v2; Xs1[ux][uy][1] = v3;
	Xs0[ux][uy][2] = v4; Xs1[ux][uy][2] = v5;
	Xs0[ux][uy][3] = v6; Xs1[ux][uy][3] = v7;
	__syncthreads();

	//read-turn[0, 1]: j0, j1, j2, j3, [ic0 - ic3], [ic4 - ic7]
	a0 = Xs0[0][uy][ux]; a4 = Xs1[0][uy][ux];
	a1 = Xs0[1][uy][ux]; a5 = Xs1[1][uy][ux];
	a2 = Xs0[2][uy][ux]; a6 = Xs1[2][uy][ux];
	a3 = Xs0[3][uy][ux]; a7 = Xs1[3][uy][ux];

	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f2x3_y_f32_64(x0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f2x3_y_f32_64(x1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f2x3_y_f32_64(x2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f2x3_y_f32_64(x3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f2x3_y_f32_64(x4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f2x3_y_f32_64(x5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f2x3_y_f32_64(x6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f2x3_y_f32_64(x7, k7);

	*(float4*)(deltaX + X00     ) = { x0.x, x1.x, x2.x, x3.x }; *(float4*)(deltaX + X00      + 4) = { x4.x, x5.x, x6.x, x7.x };
	*(float4*)(deltaX + X00 + IC) = { x0.y, x1.y, x2.y, x3.y }; *(float4*)(deltaX + X00 + IC + 4) = { x4.y, x5.y, x6.y, x7.y };
	__syncthreads();

	//write-turn2: j4, j5, j6, j7, [ic0 - ic3], [ic4 - ic7]
	Xs0[ux][uy][0] =  v8; Xs1[ux][uy][0] =  v9;
	Xs0[ux][uy][1] = v10; Xs1[ux][uy][1] = v11;
	Xs0[ux][uy][2] = v12; Xs1[ux][uy][2] = v13;
	Xs0[ux][uy][3] = v14; Xs1[ux][uy][3] = v15;
	__syncthreads();

	//write-turn[2, 3]: j4, j5, j6, j7, [ic0 - ic3], [ic4 - c7]
	a0 = Xs0[0][uy][ux]; a4 = Xs1[0][uy][ux];
	a1 = Xs0[1][uy][ux]; a5 = Xs1[1][uy][ux];
	a2 = Xs0[2][uy][ux]; a6 = Xs1[2][uy][ux];
	a3 = Xs0[3][uy][ux]; a7 = Xs1[3][uy][ux];

	//read-turn[2, 3]: j4, j5, j6, j7, [ic0 - ic3], [ic4 - ic7]
	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f2x3_y_f32_64(x0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f2x3_y_f32_64(x1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f2x3_y_f32_64(x2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f2x3_y_f32_64(x3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f2x3_y_f32_64(x4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f2x3_y_f32_64(x5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f2x3_y_f32_64(x6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f2x3_y_f32_64(x7, k7);

	*(float4*)(deltaX + X20     ) = { x0.x, x1.x, x2.x, x3.x }; *(float4*)(deltaX + X20      + 4) = { x4.x, x5.x, x6.x, x7.x };
	*(float4*)(deltaX + X20 + IC) = { x0.y, x1.y, x2.y, x3.y }; *(float4*)(deltaX + X20 + IC + 4) = { x4.y, x5.y, x6.y, x7.y };
}

#endif


//Split FW: LB = 4, OC % 8 == 0, IWr % 2 == 0
#ifndef DECONV_3D_WINOGRAD_SFW_F2X3_KERNEL_64X128C_TEXTURE
#define DECONV_3D_WINOGRAD_SFW_F2X3_KERNEL_64X128C_TEXTURE

//for: [FH, FW] = 6, Feature = (24, 24), [N, IC, OC] = [64, 256, 256],
//LB = 4: Size = 81, Time = 12.427 msec, Performace = 13997.4 GFlop/s

template<int FW>
__global__ void winograd_SFW_f2x3_kernel_64x128C_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W, int FH,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,//sh = sw = 1
	int iw_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][g(ic): 64]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][d( j): 64]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << 6); 
	const int Gk = (ty & 7), Gi = (tx << 2) + ((ty > 7) << 1);
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * FW * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << 7);
	const int Dk = (tx & 7), Di = (ty << 2) + ((tx > 7) << 1);
	const int tj0 = bj0 + (Di << 1), tj4 = tj0 + 2;
	const int IWr2 = (IW - iw_index) >> 1 << 1, IH_IWr2 = IH * IWr2;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr2, IWr2); tiw0 += iw_index;//IWr % 2
	get_n_ih_iw_Temp(tj4, tn4, tih4, tiw4, IH_IWr2, IWr2); tiw4 += iw_index;
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int toh4 = tih4 - oph, tow4 = tiw4 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Dk;//deltaY[tn0, toh0, tow0, Dk]
	const int Y4 = ((tn4*OH + toh4)*OW + tow4)*OC + Dk;//deltaY[tn0, toh0, tow0, Dk]

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
	const int GIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int DIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//======[compute area1: local]======================================================
	float w[6], y[8]; const int FHW = FH * FW;
	for (int fhw = 0; fhw < FHW; fhw += 3) {
		const int fh = fhw / FW, fw = fhw - fh * FW;

		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = (-fh*FW - fw)*IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		*(float2*)(w + 4) = *(float2*)(W + W0);//fw = 2
		*(float2*)(w + 2) = *(float2*)(W + W1);//fw = 1
		*(float2*)(w    ) = *(float2*)(W + W2);//fw = 0

		//load 2 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);//group0
		bool ly0 = lh0 && (tow0 + fw >=  0) && (tow0 + fw     < OW);
		bool ly1 = lh0 && (tow0 + fw >= -1) && (tow0 + fw + 1 < OW);
		bool ly2 = lh0 && (tow0 + fw >= -2) && (tow0 + fw + 2 < OW);
		bool ly3 = lh0 && (tow0 + fw >= -3) && (tow0 + fw + 3 < OW);
		int tY0 = Y0 + (fh*OW + fw)*OC;
		int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
		tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
		y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
		y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
	
		bool lh4 = (toh4 >= -fh) && (toh4 + fh < OH);//group1
		bool ly4 = lh4 && (tow4 + fw >=  0) && (tow4 + fw     < OW);
		bool ly5 = lh4 && (tow4 + fw >= -1) && (tow4 + fw + 1 < OW);
		bool ly6 = lh4 && (tow4 + fw >= -2) && (tow4 + fw + 2 < OW);
		bool ly7 = lh4 && (tow4 + fw >= -3) && (tow4 + fw + 3 < OW);
		int tY4 = Y4 + (fh*OW + fw)*OC;
		int tY5 = tY4 + OC, tY6 = tY4 + (OC << 1), tY7 = tY4 + OC * 3;
		tY4 = IF_int(ly4, tY4, -1); tY5 = IF_int(ly5, tY5, -1); 
		tY6 = IF_int(ly6, tY6, -1); tY7 = IF_int(ly7, tY7, -1);
		y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5); 
		y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

		//Winograd transform: W(3) -> G(4); Y(4) -> D(4)
		float gst0 = 0.5f*(w[0] + w[2] + w[4]), gst1 = gst0 - w[2];
		float gst2 = 0.5f*(w[1] + w[3] + w[5]), gst3 = gst2 - w[3];
		float g0[4]{ w[0], gst0, gst1, w[4] };
		float g1[4]{ w[1], gst2, gst3, w[5] };

		float d0[4]{ y[0] - y[2], y[1] + y[2], y[2] - y[1], y[1] - y[3] };
		float d1[4]{ y[4] - y[6], y[5] + y[6], y[6] - y[5], y[5] - y[7] };

		//write to shread memory
		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };

		*(float2*)(&Ds[buf][Dk][0][Di]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Dk][1][Di]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Dk][2][Di]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Dk][3][Di]) = { d0[3], d1[3] };
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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

			//load 2 groups from W[OC, FH, FW, IC]
			const int W0 = ((ooc*FH - fh)*FW - fw)*IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			*(float2*)(w + 4) = *(float2*)(W + W0);//fw = 2
			*(float2*)(w + 2) = *(float2*)(W + W1);//fw = 1
			*(float2*)(w    ) = *(float2*)(W + W2);//fw = 0

			//load 2 groups from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + (fh*OW + fw)*OC + ooc;//group0
			int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
			tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
			y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
			y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);

			int tY4 = Y4 + (fh*OW + fw)*OC + ooc;//group1
			int tY5 = tY4 + OC, tY6 = tY4 + (OC << 1), tY7 = tY4 + OC * 3;
			tY4 = IF_int(ly4, tY4, -1); tY5 = IF_int(ly5, tY5, -1);
			tY6 = IF_int(ly6, tY6, -1); tY7 = IF_int(ly7, tY7, -1);
			y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
			y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

			//Winograd transform: W(3) -> G(4); Y(4) -> D(4)
			float gst0 = 0.5f*(w[0] + w[2] + w[4]), gst1 = gst0 - w[2];
			float gst2 = 0.5f*(w[1] + w[3] + w[5]), gst3 = gst2 - w[3];
			float g0[4]{ w[0], gst0, gst1, w[4] };
			float g1[4]{ w[1], gst2, gst3, w[5] };

			float d0[4]{ y[0] - y[2], y[1] + y[2], y[2] - y[1], y[1] - y[3] };
			float d1[4]{ y[4] - y[6], y[5] + y[6], y[6] - y[5], y[5] - y[7] };

			//write to shread memory
			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };

			*(float2*)(&Ds[buf][Dk][0][Di]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Dk][1][Di]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Dk][2][Di]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Dk][3][Di]) = { d0[3], d1[3] };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
			float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx    ]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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
	__syncthreads();

	//======[compute area12: block]======================================================
	//prepare for Y[N, OH, OW, OC]
	const int xic0 = bic0 + GIdx;
	const int xj0 = bj0 + ((DIdx + ux) << 1), xj2 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr2, IWr2); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj2, xn2, xih2, xiw2, IH_IWr2, IWr2); xiw2 += iw_index;
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//ux: j0 -> j3
	const int X20 = ((xn2*IH + xih2)*IW + xiw2)*IC + xic0;//ux: j4 -> j7

	float4 (* __restrict__ Xs0)[65][5] = (float4(*)[65][5])(Gs);//4 * 64 * 16
	float4 (* __restrict__ Xs1)[65][5] = (float4(*)[65][5])(Ds);

	float4 a0, a1, a2, a3, a4, a5, a6, a7;
	float4 k0, k1, k2, k3, k4, k5, k6, k7;
	float2 x0, x1, x2, x3, x4, x5, x6, x7;

	//write-turn[0, 1]: j0, j1, j2, j3, [ic0 - ic3], [ic4 - ic7]
	Xs0[ux][uy][0] = v0; Xs1[ux][uy][0] = v1; 
	Xs0[ux][uy][1] = v2; Xs1[ux][uy][1] = v3;
	Xs0[ux][uy][2] = v4; Xs1[ux][uy][2] = v5;
	Xs0[ux][uy][3] = v6; Xs1[ux][uy][3] = v7;
	__syncthreads();

	//read-turn[0, 1]: j0, j1, j2, j3, [ic0 - ic3], [ic4 - ic7]
	a0 = Xs0[0][uy][ux]; a4 = Xs1[0][uy][ux];
	a1 = Xs0[1][uy][ux]; a5 = Xs1[1][uy][ux];
	a2 = Xs0[2][uy][ux]; a6 = Xs1[2][uy][ux];
	a3 = Xs0[3][uy][ux]; a7 = Xs1[3][uy][ux];

	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f2x3_y_f32_64(x0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f2x3_y_f32_64(x1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f2x3_y_f32_64(x2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f2x3_y_f32_64(x3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f2x3_y_f32_64(x4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f2x3_y_f32_64(x5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f2x3_y_f32_64(x6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f2x3_y_f32_64(x7, k7);

	*(float4*)(deltaX + X00     ) = { x0.x, x1.x, x2.x, x3.x }; *(float4*)(deltaX + X00      + 4) = { x4.x, x5.x, x6.x, x7.x };
	*(float4*)(deltaX + X00 + IC) = { x0.y, x1.y, x2.y, x3.y }; *(float4*)(deltaX + X00 + IC + 4) = { x4.y, x5.y, x6.y, x7.y };
	__syncthreads();

	//write-turn2: j4, j5, j6, j7, [ic0 - ic3], [ic4 - ic7]
	Xs0[ux][uy][0] =  v8; Xs1[ux][uy][0] =  v9;
	Xs0[ux][uy][1] = v10; Xs1[ux][uy][1] = v11;
	Xs0[ux][uy][2] = v12; Xs1[ux][uy][2] = v13;
	Xs0[ux][uy][3] = v14; Xs1[ux][uy][3] = v15;
	__syncthreads();

	//write-turn[2, 3]: j4, j5, j6, j7, [ic0 - ic3], [ic4 - c7]
	a0 = Xs0[0][uy][ux]; a4 = Xs1[0][uy][ux];
	a1 = Xs0[1][uy][ux]; a5 = Xs1[1][uy][ux];
	a2 = Xs0[2][uy][ux]; a6 = Xs1[2][uy][ux];
	a3 = Xs0[3][uy][ux]; a7 = Xs1[3][uy][ux];

	//read-turn[2, 3]: j4, j5, j6, j7, [ic0 - ic3], [ic4 - ic7]
	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f2x3_y_f32_64(x0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f2x3_y_f32_64(x1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f2x3_y_f32_64(x2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f2x3_y_f32_64(x3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f2x3_y_f32_64(x4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f2x3_y_f32_64(x5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f2x3_y_f32_64(x6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f2x3_y_f32_64(x7, k7);

	*(float4*)(deltaX + X20     ) = { x0.x, x1.x, x2.x, x3.x }; *(float4*)(deltaX + X20      + 4) = { x4.x, x5.x, x6.x, x7.x };
	*(float4*)(deltaX + X20 + IC) = { x0.y, x1.y, x2.y, x3.y }; *(float4*)(deltaX + X20 + IC + 4) = { x4.y, x5.y, x6.y, x7.y };
}

#endif


//Split FW: LB = 4, OC % 8 == 0, IWr % 4 == 0
#ifndef DECONV_3D_WINOGRAD_SFW_F2X3_KERNEL_64X128X4C_TEXTURE
#define DECONV_3D_WINOGRAD_SFW_F2X3_KERNEL_64X128X4C_TEXTURE

//for: [FH, FW] = 6, Feature = (24, 24), [N, IC, OC] = [64, 256, 256],
//LB = 4: Size = 81, Time = 11.318 msec, Performace = 15369 GFlop/s

template<int FW>
__global__ void winograd_SFW_f2x3_kernel_64x128x4C_tex(
	cudaTextureObject_t       deltaY, int OH, int OW,
	const float* __restrict__      W, int FH,
	      float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int oph, int opw,//sh = sw = 1
	int iw_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][g(ic): 64]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][d( j): 64]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << 6); 
	const int Gk = (ty & 7), Gi = (tx << 2) + ((ty > 7) << 1);
	const int tic0 = bic0 + Gi;
	W += (Gk * FH * FW * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << 7);
	const int Dk = (tx & 7), Di = (ty << 2) + ((tx > 7) << 1);
	const int tj0 = bj0 + (Di << 1);
	const int IWr4 = (IW - iw_index) >> 2 << 2, IH_IWr4 = IH * IWr4;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IWr4, IWr4); tiw0 += iw_index;
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Dk;//deltaY[tn0, toh0, tow0, Dk]

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
	const int GIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int DIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//prepare for Y[N, OH, OW, OC]
	const int xic0 = bic0 + GIdx;
	const int xj0 = bj0 + ((DIdx + ux) << 1), xj2 = xj0 + 8;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IWr4, IWr4); xiw0 += iw_index;
	get_n_ih_iw_Temp(xj2, xn2, xih2, xiw2, IH_IWr4, IWr4); xiw2 += iw_index;
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//ux: j0 -> j3
	const int X20 = ((xn2*IH + xih2)*IW + xiw2)*IC + xic0;//ux: j4 -> j7

	//======[compute area1: local]======================================================
	float w[6], y[6]; const int FHW = FH * FW;
	for (int fhw = 0; fhw < FHW; fhw += 3) {
		const int fh = fhw / FW, fw = fhw - fh * FW;

		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = (-fh * FW - fw)*IC;//rotate180
		const int W1 = W0 + IC, W2 = W0 + (IC << 1);
		*(float2*)(w + 4) = *(float2*)(W + W0);//fw = 2
		*(float2*)(w + 2) = *(float2*)(W + W1);//fw = 1
		*(float2*)(w    ) = *(float2*)(W + W2);//fw = 0

		//load 2 groups from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 + fw >=  0) && (tow0 + fw     < OW);
		bool ly1 = lh0 && (tow0 + fw >= -1) && (tow0 + fw + 1 < OW);
		bool ly2 = lh0 && (tow0 + fw >= -2) && (tow0 + fw + 2 < OW);
		bool ly3 = lh0 && (tow0 + fw >= -3) && (tow0 + fw + 3 < OW);
		bool ly4 = lh0 && (tow0 + fw >= -4) && (tow0 + fw + 4 < OW);
		bool ly5 = lh0 && (tow0 + fw >= -5) && (tow0 + fw + 5 < OW);
		int tY0 = Y0 + (fh*OW + fw)*OC;
		int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1);
		int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
		tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
		tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
		tY4 = IF_int(ly4, tY4, -1); tY5 = IF_int(ly5, tY5, -1);
		y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
		y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
		y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);

		//Winograd transform: W(3) -> G(4); Y(4) -> D(4)
		float gst0 = 0.5f*(w[0] + w[2] + w[4]), gst1 = gst0 - w[2];
		float gst2 = 0.5f*(w[1] + w[3] + w[5]), gst3 = gst2 - w[3];
		float g0[4]{ w[0], gst0, gst1, w[4] };
		float g1[4]{ w[1], gst2, gst3, w[5] };

		float d0[4]{ y[0] - y[2], y[1] + y[2], y[2] - y[1], y[1] - y[3] };
		float d1[4]{ y[2] - y[4], y[3] + y[4], y[4] - y[3], y[3] - y[5] };

		//write to shread memory
		*(float2*)(&Ds[buf][Dk][0][Di]) = { d0[0], d1[0] };
		*(float2*)(&Ds[buf][Dk][1][Di]) = { d0[1], d1[1] };
		*(float2*)(&Ds[buf][Dk][2][Di]) = { d0[2], d1[2] };
		*(float2*)(&Ds[buf][Dk][3][Di]) = { d0[3], d1[3] };

		*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
		__syncthreads();

		for (int ooc = 8; ooc < OC; ooc += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++) {
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx    ]);
				float4 a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx    ]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

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

			//load 2 group from W[OC, FH, FW, IC]
			const int W0 = ((ooc*FH - fh)*FW - fw)*IC;//rotate180
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			*(float2*)(w + 4) = *(float2*)(W + W0);//fw = 2
			*(float2*)(w + 2) = *(float2*)(W + W1);//fw = 1
			*(float2*)(w    ) = *(float2*)(W + W2);//fw = 0

			//load 2 groups from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + (fh*OW + fw)*OC + ooc;//group0
			int tY1 = tY0 + OC, tY2 = tY0 + (OC << 1);
			int tY3 = tY0 + OC * 3, tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
			tY0 = IF_int(ly0, tY0, -1); tY1 = IF_int(ly1, tY1, -1);
			tY2 = IF_int(ly2, tY2, -1); tY3 = IF_int(ly3, tY3, -1);
			tY4 = IF_int(ly4, tY4, -1); tY5 = IF_int(ly5, tY5, -1);
			y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
			y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
			y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float gst0 = 0.5f*(w[0] + w[2] + w[4]), gst1 = gst0 - w[2];
			float gst2 = 0.5f*(w[1] + w[3] + w[5]), gst3 = gst2 - w[3];
			float g0[4]{ w[0], gst0, gst1, w[4] };
			float g1[4]{ w[1], gst2, gst3, w[5] };

			float d0[4]{ y[0] - y[2], y[1] + y[2], y[2] - y[1], y[1] - y[3] };
			float d1[4]{ y[2] - y[4], y[3] + y[4], y[4] - y[3], y[3] - y[5] };

			//write to shread memory
			*(float2*)(&Ds[buf][Dk][0][Di]) = { d0[0], d1[0] };
			*(float2*)(&Ds[buf][Dk][1][Di]) = { d0[1], d1[1] };
			*(float2*)(&Ds[buf][Dk][2][Di]) = { d0[2], d1[2] };
			*(float2*)(&Ds[buf][Dk][3][Di]) = { d0[3], d1[3] };

			*(float2*)(&Gs[buf][Gk][0][Gi]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gk][1][Gi]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gk][2][Gi]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gk][3][Gi]) = { g0[3], g1[3] };
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
	float4 (* __restrict__ Xs0)[65][5] = (float4(*)[65][5])(Gs);//4 * 64 * 16
	float4 (* __restrict__ Xs1)[65][5] = (float4(*)[65][5])(Ds);

	float4 a0, a1, a2, a3, a4, a5, a6, a7;
	float4 k0, k1, k2, k3, k4, k5, k6, k7;
	float2 x0, x1, x2, x3, x4, x5, x6, x7;

	//write-turn[0, 1]: j0, j1, j2, j3, [ic0 - ic3], [ic4 - ic7]
	Xs0[ux][uy][0] = v0; Xs1[ux][uy][0] = v1; 
	Xs0[ux][uy][1] = v2; Xs1[ux][uy][1] = v3;
	Xs0[ux][uy][2] = v4; Xs1[ux][uy][2] = v5;
	Xs0[ux][uy][3] = v6; Xs1[ux][uy][3] = v7;
	__syncthreads();

	//read-turn[0, 1]: j0, j1, j2, j3, [ic0 - ic3], [ic4 - ic7]
	a0 = Xs0[0][uy][ux]; a4 = Xs1[0][uy][ux];
	a1 = Xs0[1][uy][ux]; a5 = Xs1[1][uy][ux];
	a2 = Xs0[2][uy][ux]; a6 = Xs1[2][uy][ux];
	a3 = Xs0[3][uy][ux]; a7 = Xs1[3][uy][ux];

	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f2x3_y_f32_64(x0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f2x3_y_f32_64(x1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f2x3_y_f32_64(x2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f2x3_y_f32_64(x3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f2x3_y_f32_64(x4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f2x3_y_f32_64(x5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f2x3_y_f32_64(x6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f2x3_y_f32_64(x7, k7);

	*(float4*)(deltaX + X00     ) = { x0.x, x1.x, x2.x, x3.x }; *(float4*)(deltaX + X00      + 4) = { x4.x, x5.x, x6.x, x7.x };
	*(float4*)(deltaX + X00 + IC) = { x0.y, x1.y, x2.y, x3.y }; *(float4*)(deltaX + X00 + IC + 4) = { x4.y, x5.y, x6.y, x7.y };
	__syncthreads();

	//write-turn2: j4, j5, j6, j7, [ic0 - ic3], [ic4 - ic7]
	Xs0[ux][uy][0] =  v8; Xs1[ux][uy][0] =  v9;
	Xs0[ux][uy][1] = v10; Xs1[ux][uy][1] = v11;
	Xs0[ux][uy][2] = v12; Xs1[ux][uy][2] = v13;
	Xs0[ux][uy][3] = v14; Xs1[ux][uy][3] = v15;
	__syncthreads();

	//write-turn[2, 3]: j4, j5, j6, j7, [ic0 - ic3], [ic4 - c7]
	a0 = Xs0[0][uy][ux]; a4 = Xs1[0][uy][ux];
	a1 = Xs0[1][uy][ux]; a5 = Xs1[1][uy][ux];
	a2 = Xs0[2][uy][ux]; a6 = Xs1[2][uy][ux];
	a3 = Xs0[3][uy][ux]; a7 = Xs1[3][uy][ux];

	//read-turn[2, 3]: j4, j5, j6, j7, [ic0 - ic3], [ic4 - ic7]
	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f2x3_y_f32_64(x0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f2x3_y_f32_64(x1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f2x3_y_f32_64(x2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f2x3_y_f32_64(x3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f2x3_y_f32_64(x4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f2x3_y_f32_64(x5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f2x3_y_f32_64(x6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f2x3_y_f32_64(x7, k7);

	*(float4*)(deltaX + X20     ) = { x0.x, x1.x, x2.x, x3.x }; *(float4*)(deltaX + X20      + 4) = { x4.x, x5.x, x6.x, x7.x };
	*(float4*)(deltaX + X20 + IC) = { x0.y, x1.y, x2.y, x3.y }; *(float4*)(deltaX + X20 + IC + 4) = { x4.y, x5.y, x6.y, x7.y };
}

#endif


#endif