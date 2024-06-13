#pragma once

#ifndef DECONV_3D_DX_WINOGRAD_S8_F7X2_H
#define DECONV_3D_DX_WINOGRAD_S8_F7X2_H

//(1) sh = sw = 1
//(2) FW = 2
//(3) IW % 7: group = 7 elements
#ifndef DECONV_3D_DX_WINOGRAD_F7X2_CALL
#define DECONV_3D_DX_WINOGRAD_F7X2_CALL

//<1> (opw <= 1) -> (FW - 1 - pw <= 1) -> (pw >= 0)
//<2> (IW - OW - opw + FW - 1 <= 2) -> (IW - OW + pw <= 2) -> (OW - IW - pw + 2 >= 0)

//================[Standard: 64(IC) * 224(GM)]===============================
//IW % 7 == 0
#define winograd_f7x2_k64x224_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f7x2_kernel_64x224_tex<FH>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IW/7)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*2*IC), deltaX,IH,IW, IC,OC, (FH-1-ph),(1-pw))

//IW % 7 == 0, (opw = 1 - pw <= 1) -> (pw >= 0)
#define winograd_f7x2_k64x224_p0_tex(stream, deltaY, OH, OW, W, FH, deltaX, IH, IW, N, IC, OC, ph, pw) \
	winograd_f7x2_kernel_64x224_p0_tex<FH>\
		<<< dim3(IC>>6, ((N*IH)>>5)*(IW/7)), dim3(16, 16), 0, stream>>>\
			(deltaY,OH,OW, W+((FH-1)*2*IC), deltaX,IH,IW, IC,OC, (FH-1-ph),(1-pw))

#endif


//================[Standard: 64(IC) * 224(GM)]===============================
//LB = 4, OC % 8 == 0, IW % 7 == 0
#ifndef DECONV_3D_DX_WINOGRAD_F7X2_KERNEL_64X224_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F7X2_KERNEL_64X224_TEXTURE

//for: Feature = (56, 56), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 24.5, Time = 4.25 msec, Performace = 12379.6 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 4.315 msec, Performance = 12193.1 GFlop/s
//for: Feature = (28, 28), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 24.5, Time = 3.826 msec, Performace = 13751.5 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 3.672 msec, Performance = 14328.3 GFlop/s
//for: Feature = (14, 14), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 24.5, Time = 3.623 msec, Performace = 14522 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 3.315 msec, Performance = 15871.3 GFlop/s

template<int FH>
__global__ void winograd_f7x2_kernel_64x224_tex(
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
	W += (Gk * (FH << 1) * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y * 224);//32 * 7 = 224
	const int Yk = (tx & 7), Yi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + Yi * 7;
	const int IW7 = (IW / 7) * 7, IH_IW7 = IH * IW7;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW7, IW7);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Yk;//deltaY[tn0, toh0, tow0, Yk]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//4 * 8
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);
	const int xj0 = bj0 + (DIdx + (ux >> 1)) * 7, xj1 = xj0 + 28;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW7, IW7);
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IW7, IW7);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//X00 = xj0 * IC + xic0;
	const int X10 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;//X10 = X00 + IC * 28;

	//======[compute area1: local]======================================================
	const int Di = (Yi + (Yk >> 1 << 3)) & 31;//avoid bank conflict (1/8)
	float g0[8], g1[8], y[8], d[8]; char ly[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from W[OC, FH, 2, IC]
		const int W0 = - (fh << 1) * IC, W1 = W0 + IC;//rotate180
		float2 w1 = *(float2*)(W + W0);
		float2 w0 = *(float2*)(W + W1);
		
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

		//Winograd transform: W(2) -> G(8); Y(8) -> D(8)
		winograd_f7x2_G(g0, w0.x, w1.x);
		winograd_f7x2_G(g1, w0.y, w1.y);
		winograd_f7x2_D(d, y);

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

			//load 2 group from W[OC, FH, 2, IC]
			const int W0 = ((ooc*FH - fh) << 1) * IC, W1 = W0 + IC;//rotate180
			float2 w1 = *(float2*)(W + W0);
			float2 w0 = *(float2*)(W + W1);

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

			//Winograd transform: W(2) -> G(8); Y(8) -> D(8)
			winograd_f7x2_G(g0, w0.x, w1.x);
			winograd_f7x2_G(g1, w0.y, w1.y);
			winograd_f7x2_D(d, y);

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
	float2 a[16]; float x0[7], x1[7], x2[7], x3[7];
	__syncthreads();

	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j0 - j3}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v0.x, v0.y, v1.x, v1.y };//{ic0, ic1}, {ic4, ic5}
	*(float4*)(&Xs[ux][uy][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Xs[ux][uy][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Xs[ux][uy][6]) = { v6.x, v6.y, v7.x, v7.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f7x2_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f7x2_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v0.z, v0.w, v1.z, v1.w };//{ic2, ic3}, {ic6, ic7}
	*(float4*)(&Xs[ux][uy][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Xs[ux][uy][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Xs[ux][uy][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f7x2_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f7x2_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC    ) = { x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2) = { x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3) = { x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4) = { x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5) = { x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6) = { x0[6], x1[6], x2[6], x3[6] };
	__syncthreads();
	
	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j4 - j7}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v8.x,  v8.y,  v9.x,  v9.y };//{ic0, ic1}, {ic4, ic5}
	*(float4*)(&Xs[ux][uy][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Xs[ux][uy][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Xs[ux][uy][6]) = { v14.x, v14.y, v15.x, v15.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f7x2_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f7x2_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v8.z,  v8.w,  v9.z,  v9.w };//{ic2, ic3}, {ic6, ic7}
	*(float4*)(&Xs[ux][uy][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Xs[ux][uy][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Xs[ux][uy][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f7x2_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f7x2_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC    ) = { x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X10 + IC * 2) = { x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X10 + IC * 3) = { x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X10 + IC * 4) = { x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X10 + IC * 5) = { x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X10 + IC * 6) = { x0[6], x1[6], x2[6], x3[6] };
}

#endif


//LB = 4, OC % 8 == 0, IW % 7 == 0, (opw = 1 - pw <= 1) -> (pw >= 0)
#ifndef DECONV_3D_DX_WINOGRAD_F7X2_KERNEL_64X224_P0_TEXTURE
#define DECONV_3D_DX_WINOGRAD_F7X2_KERNEL_64X224_P0_TEXTURE

//for: Feature = (56, 56), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 24.5, Time = 4.179 msec, Performace = 12589.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 4.315 msec, Performance = 12193.1 GFlop/s
//for: Feature = (28, 28), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 24.5, Time = 3.75 msec, Performace = 14030.2 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 3.672 msec, Performance = 14328.3 GFlop/s
//for: Feature = (14, 14), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 24.5, Time = 3.529 msec, Performace = 14908.9 GFlop/s
//cuDNN-NHWC-implicit-prec: Size = 24.5, Time = 3.315 msec, Performance = 15871.3 GFlop/s

template<int FH>
__global__ void winograd_f7x2_kernel_64x224_p0_tex(
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
	W += (Gk * (FH << 1) * IC) + tic0;//W[Gk, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y * 224);//32 * 7 = 224
	const int Yk = (tx & 7), Yi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + Yi * 7;
	const int IW7 = (IW / 7) * 7, IH_IW7 = IH * IW7;
	get_n_ih_iw_Temp(tj0, tn0, tih0, tiw0, IH_IW7, IW7);
	const int toh0 = tih0 - oph, tow0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + toh0)*OW + tow0)*OC + Yk;//deltaY[tn0, toh0, tow0, Yk]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//4 * 8
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for deltaX[N, IH, IW, IC]
	const int xic0 = bic0 + GIdx + ((ux & 1) << 2);
	const int xj0 = bj0 + (DIdx + (ux >> 1)) * 7, xj1 = xj0 + 28;
	get_n_ih_iw_Temp(xj0, xn0, xih0, xiw0, IH_IW7, IW7);
	get_n_ih_iw_Temp(xj1, xn1, xih1, xiw1, IH_IW7, IW7);
	const int X00 = ((xn0*IH + xih0)*IW + xiw0)*IC + xic0;//X00 = xj0 * IC + xic0;
	const int X10 = ((xn1*IH + xih1)*IW + xiw1)*IC + xic0;//X10 = X00 + IC * 28;

	//======[compute area1: local]======================================================
	const int Di = (Yi + (Yk << 2)) & 31;//avoid bank conflict (1/8)
	float g0[8], g1[8], y[8], d[8]; char ly[8];
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from W[OC, FH, FW, IC]
		const int W0 = - (fh << 1) * IC, W1 = W0 + IC;//rotate180
		float2 w1 = *(float2*)(W + W0);
		float2 w0 = *(float2*)(W + W1);
		
		//load 1 group from deltaY[N, OH, OW, OC]
		bool lh0 = (toh0 >= -fh) && (toh0 + fh < OH);
		bool ly0 = lh0 && (tow0 >=  0) && (tow0     < OW);//pw = -1
		bool ly6 = lh0 && (tow0 >= -6) && (tow0 + 6 < OW);//pw = +1
		bool ly7 = lh0 && (tow0 >= -7) && (tow0 + 7 < OW);//pw = +2
		int tY0 = Y0 + fh * OW * OC, tY1 = tY0 + OC;
		int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
		int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
		int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC * 7;
		tY0 = IF_int(ly0, tY0, -1); 
		tY1 = IF_int(lh0, tY1, -1); tY2 = IF_int(lh0, tY2, -1); 
		tY3 = IF_int(lh0, tY3, -1); tY4 = IF_int(lh0, tY4, -1); 
		tY5 = IF_int(lh0, tY5, -1); 
		tY6 = IF_int(ly6, tY6, -1); tY7 = IF_int(ly7, tY7, -1);
		y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
		y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
		y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
		y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

		//Winograd transform: W(2) -> G(8); Y(8) -> D(8)
		winograd_f7x2_G(g0, w0.x, w1.x);
		winograd_f7x2_G(g1, w0.y, w1.y);
		winograd_f7x2_D(d, y);

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
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][(DIdx + (ik << 2)    ) & 31]);
				float4 b1 = *(float4*)(&Ds[buf][ik][ux][(DIdx + (ik << 2) + 4) & 31]);

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

			//load 2 group from W[OC, FH, 2, IC]
			const int W0 = ((ooc*FH - fh) << 1) * IC, W1 = W0 + IC;//rotate180
			float2 w1 = *(float2*)(W + W0);
			float2 w0 = *(float2*)(W + W1);

			//load 1 group from deltaY[N, OH, OW, OC]
			int tY0 = Y0 + fh * OW * OC + ooc, tY1 = tY0 + OC;
			int tY2 = tY0 + (OC << 1), tY3 = tY0 + OC * 3;
			int tY4 = tY0 + (OC << 2), tY5 = tY0 + OC * 5;
			int tY6 = tY0 + OC * 6   , tY7 = tY0 + OC * 7;
			tY0 = IF_int(ly0, tY0, -1);
			tY1 = IF_int(lh0, tY1, -1); tY2 = IF_int(lh0, tY2, -1);
			tY3 = IF_int(lh0, tY3, -1); tY4 = IF_int(lh0, tY4, -1);
			tY5 = IF_int(lh0, tY5, -1); 
			tY6 = IF_int(ly6, tY6, -1); tY7 = IF_int(ly7, tY7, -1);
			y[0] = tex1Dfetch<float>(deltaY, tY0); y[1] = tex1Dfetch<float>(deltaY, tY1);
			y[2] = tex1Dfetch<float>(deltaY, tY2); y[3] = tex1Dfetch<float>(deltaY, tY3);
			y[4] = tex1Dfetch<float>(deltaY, tY4); y[5] = tex1Dfetch<float>(deltaY, tY5);
			y[6] = tex1Dfetch<float>(deltaY, tY6); y[7] = tex1Dfetch<float>(deltaY, tY7);

			//Winograd transform: W(2) -> G(8); Y(8) -> D(8)
			winograd_f7x2_G(g0, w0.x, w1.x);
			winograd_f7x2_G(g1, w0.y, w1.y);
			winograd_f7x2_D(d, y);

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
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][(DIdx + (ik << 2)    ) & 31]);
			float4 b1 = *(float4*)(&Ds[buf][ik][ux][(DIdx + (ik << 2) + 4) & 31]);

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
	float2 a[16]; float x0[7], x1[7], x2[7], x3[7];
	__syncthreads();

	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j0 - j3}------------------------
	*(float4*)(&Xs[ux][uy][0]) = { v0.x, v0.y, v1.x, v1.y };//{ic0, ic1}, {ic4, ic5}
	*(float4*)(&Xs[ux][uy][2]) = { v2.x, v2.y, v3.x, v3.y };
	*(float4*)(&Xs[ux][uy][4]) = { v4.x, v4.y, v5.x, v5.y };
	*(float4*)(&Xs[ux][uy][6]) = { v6.x, v6.y, v7.x, v7.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f7x2_Y_vec(x0, a, x);//{ic0 / ic4} * { j0 - j3 }
	winograd_f7x2_Y_vec(x1, a, y);//{ic1 / ic5} * { j0 - j3 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v0.z, v0.w, v1.z, v1.w };//{ic2, ic3}, {ic6, ic7}
	*(float4*)(&Xs[ux][uy][2]) = { v2.z, v2.w, v3.z, v3.w };
	*(float4*)(&Xs[ux][uy][4]) = { v4.z, v4.w, v5.z, v5.w };
	*(float4*)(&Xs[ux][uy][6]) = { v6.z, v6.w, v7.z, v7.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f7x2_Y_vec(x2, a, x);//{ic2 / ic6} * { j0 - j3 }
	winograd_f7x2_Y_vec(x3, a, y);//{ic3 / ic7} * { j0 - j3 }
	
	*(float4*)(deltaX + X00         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X00 + IC    ) = { x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X00 + IC * 2) = { x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X00 + IC * 3) = { x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X00 + IC * 4) = { x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X00 + IC * 5) = { x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X00 + IC * 6) = { x0[6], x1[6], x2[6], x3[6] };
	__syncthreads();
	
	//------write-read-turn0: {ic0 - ic3 | ic4 - ic7} x {j4 - j7}------------------------
	*(float4*)(&Xs[ux][uy][0]) = {  v8.x,  v8.y,  v9.x,  v9.y };//{ic0, ic1}, {ic4, ic5}
	*(float4*)(&Xs[ux][uy][2]) = { v10.x, v10.y, v11.x, v11.y };
	*(float4*)(&Xs[ux][uy][4]) = { v12.x, v12.y, v13.x, v13.y };
	*(float4*)(&Xs[ux][uy][6]) = { v14.x, v14.y, v15.x, v15.y };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f7x2_Y_vec(x0, a, x);//{ic0 / ic4} * { j4 - j7 }
	winograd_f7x2_Y_vec(x1, a, y);//{ic1 / ic5} * { j4 - j7 }
	__syncthreads();

	*(float4*)(&Xs[ux][uy][0]) = { v8.z,  v8.w,  v9.z,  v9.w };//{ic2, ic3}, {ic6, ic7}
	*(float4*)(&Xs[ux][uy][2]) = { v10.z, v10.w, v11.z, v11.w };
	*(float4*)(&Xs[ux][uy][4]) = { v12.z, v12.w, v13.z, v13.w };
	*(float4*)(&Xs[ux][uy][6]) = { v14.z, v14.w, v15.z, v15.w };
	__syncthreads();

	a[0] = Xs[0][uy][ux]; a[1] = Xs[1][uy][ux];
	a[2] = Xs[2][uy][ux]; a[3] = Xs[3][uy][ux];
	a[4] = Xs[4][uy][ux]; a[5] = Xs[5][uy][ux];
	a[6] = Xs[6][uy][ux]; a[7] = Xs[7][uy][ux];
	winograd_f7x2_Y_vec(x2, a, x);//{ic2 / ic6} * { j4 - j7 }
	winograd_f7x2_Y_vec(x3, a, y);//{ic3 / ic7} * { j4 - j7 }

	*(float4*)(deltaX + X10         ) = { x0[0], x1[0], x2[0], x3[0] };
	*(float4*)(deltaX + X10 + IC    ) = { x0[1], x1[1], x2[1], x3[1] };
	*(float4*)(deltaX + X10 + IC * 2) = { x0[2], x1[2], x2[2], x3[2] };
	*(float4*)(deltaX + X10 + IC * 3) = { x0[3], x1[3], x2[3], x3[3] };
	*(float4*)(deltaX + X10 + IC * 4) = { x0[4], x1[4], x2[4], x3[4] };
	*(float4*)(deltaX + X10 + IC * 5) = { x0[5], x1[5], x2[5], x3[5] };
	*(float4*)(deltaX + X10 + IC * 6) = { x0[6], x1[6], x2[6], x3[6] };
}

#endif


#endif

