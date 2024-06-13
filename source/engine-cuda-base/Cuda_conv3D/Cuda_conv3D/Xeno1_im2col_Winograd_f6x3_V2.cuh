

//Select: Size = 20.25, Time = 2.67964 msec, Performace = 16228.5 GFlop/s
#ifndef WG_2X6_A1
#define WG_2X6_A1

#define wg2x6_A1(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel_A1<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
//Size = 81, Time = 10.7723 msec, Performace = 16147.6 GFlop/s
//Size = 81, Time = 10.479 msec, Performace = 16599.6 GFlop/s
template<int FH>
__global__ void Winograd2x6_kernel_A1(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
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
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int toc0 = boc0 + Gs_i;
	CW += Gs_k *OC + toc0;//CW[0, 0, Gs_k, toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int Ds_k = (tx & 7), Ds_i = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + Ds_i * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Ds_k;//X[tn0, tih0, tiw0, Ds_k]

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx / 32,  uy = idx % 32;//8 * 32
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	float g0[8], g1[8], x[8], d[8];

//#pragma unroll
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + IC_OC;
		const int W2 = W0 + (IC_OC << 1);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);//padding = 8
		int tX0 = X0 + fh * IW * IC;//[n, tih0 + fh, tiw0 + fw, ic]
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6, tX7 = tX0 + IC * 7;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
		tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
		x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5);
		x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
		winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
		winograd_f6x3_D(d, x);

		//write to shread memory
		*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gs_k][4][Gs_i]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gs_k][5][Gs_i]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gs_k][6][Gs_i]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gs_k][7][Gs_i]) = { g0[7], g1[7] };

		Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++)
			{
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

				//oc[0 - 3]             oc[4 - 7]
				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);//j0
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);//j1
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);//j2
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);//j3
				simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}
			buf ^= 1;

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC;
			const int W2 = W0 + (IC_OC << 1);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6, tX7 = tX0 + IC * 7;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
			tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
			x[0] = tex1Dfetch<float>(X, tX0); x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2); x[3] = tex1Dfetch<float>(X, tX3);
			x[4] = tex1Dfetch<float>(X, tX4); x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6); x[7] = tex1Dfetch<float>(X, tX7);

			//Winograd transform: W(3) -> G(8); X(4) -> D(4)
			winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
			winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
			winograd_f6x3_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gs_k][4][Gs_i]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gs_k][5][Gs_i]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gs_k][6][Gs_i]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gs_k][7][Gs_i]) = { g0[7], g1[7] };

			Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
			Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
			Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
			Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++)
		{
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

			//oc[0 - 3]             oc[4 - 7]
			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);//j0
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);//j1
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);//j2
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);//j3
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);//j4
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
		}
		buf ^= 1;
	}

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Ys)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float y0[6], y1[6], y2[6], y3[6];
	
	const int Y10 = Y00 + 24 * OC;//ux: j4 -> j7
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
	winograd_f6x3_Y_vec(y0, a, x);//{oc0 / oc4} * { j0 - j3 }
	winograd_f6x3_Y_vec(y1, a, y);//{oc1 / oc5} * { j0 - j3 }
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
	winograd_f6x3_Y_vec(y2, a, x);//{oc2 / oc6} * { j0 - j3 }
	winograd_f6x3_Y_vec(y3, a, y);//{oc3 / oc7} * { j0 - j3 }
	
	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
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

	winograd_f6x3_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f6x3_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
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

	winograd_f6x3_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f6x3_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
}

#endif



#ifndef WG_2X6_A2
#define WG_2X6_A2

//__constant__ int SIDX_S[64] = {
//    //0     1     2     3     4     5     6     7
//	2,1,  2,0,  0,1,  0,0,  3,1,  3,0,  1,1,  1,0,
//	//8     9    10    11    12    13    14    15
//	2,3,  2,2,  0,3,  0,2,  3,3,  3,2,  1,3,  1,2,
//	//16   17    18    19    20    21    22    23
//	2,5,  2,4,  0,5,  0,4,  3,5,  3,4,  1,5,  1,4,
//	//24   25    26    27    28    29    30    31
//	2,7,  2,6,  0,7,  0,6,  3,7,  3,6,  1,7,  1,6
//};

//readAs = ((tid128 >> 4) | ((tid >> 1) & 7)) << 4;
//readBs = (((tid & 112) >> 3) | (tid & 1)) << 4;

//const int warp_id = tid / 32;//[0- 7]
//const int lane_id = tid % 32;//[0-31]

//const int tile_index_b = (warp_id % 4) * 16 + (lane_id / 16) * 8 + (lane_id % 2) * 4;
//const int tile_index_a = (warp_id / 4) * 32 + ((lane_id % 16) / 2) * 4;

//[0 - 16]
//const int tile_index_b = (warp_id % 4) * 4 + (lane_id / 16) * 2 + (lane_id % 2);
//[0 - 16]
//const int tile_index_a = (warp_id / 4) * 8 + ((lane_id % 16) / 2);

__constant__ int SIDX_S[64] = {
    //0     1     2     3     4     5     6     7
	2,1,  2,0,  0,1,  0,0,  3,1,  3,0,  1,1,  1,0,
	//8     9    10    11    12    13    14    15
	2,3,  2,2,  0,3,  0,2,  3,3,  3,2,  1,3,  1,2,
	//16   17    18    19    20    21    22    23
	2,5,  2,4,  0,5,  0,4,  3,5,  3,4,  1,5,  1,4,
	//24   25    26    27    28    29    30    31
	2,7,  2,6,  0,7,  0,6,  3,7,  3,6,  1,7,  1,6
};

#define wg2x6_A2(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel_A2<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 20.25, Time = 2.52857 msec, Performace = 17198.1 GFlop/s

//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
//Size = 81, Time = 10.7723 msec, Performace = 16147.6 GFlop/s
//Size = 81, Time = 10.479 msec, Performace = 16599.6 GFlop/s
template<int FH>
__global__ void Winograd2x6_kernel_A2(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
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
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int toc0 = boc0 + Gs_i;
	CW += Gs_k *OC + toc0;//CW[0, 0, Gs_k, toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int Ds_k = (tx & 7), Ds_i = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + Ds_i * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Ds_k;//X[tn0, tih0, tiw0, Ds_k]

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx >> 5, uy = idx & 31;//8 * 32

	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	float g0[8], g1[8], x[8], d[8]; char ly[8];

	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + IC_OC;
		const int W2 = W0 + (IC_OC << 1);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		
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

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
		winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
		winograd_f6x3_D(d, x);

		//write to shread memory
		*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gs_k][4][Gs_i]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gs_k][5][Gs_i]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gs_k][6][Gs_i]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gs_k][7][Gs_i]) = { g0[7], g1[7] };

		Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++)
			{
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

				//oc[0 - 3]             oc[4 - 7]
				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);//j0
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);//j1
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);//j2
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);//j3
				simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}
			buf ^= 1;

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC;
			const int W2 = W0 + (IC_OC << 1);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);

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

			//Winograd transform: W(3) -> G(8); X(4) -> D(4)
			winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
			winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
			winograd_f6x3_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gs_k][4][Gs_i]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gs_k][5][Gs_i]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gs_k][6][Gs_i]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gs_k][7][Gs_i]) = { g0[7], g1[7] };

			Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
			Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
			Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
			Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++)
		{
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

			//oc[0 - 3]             oc[4 - 7]
			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);//j0
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);//j1
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);//j2
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);//j3
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);//j4
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
		}
		buf ^= 1;
	}

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Ys)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float y0[6], y1[6], y2[6], y3[6];
	
	const int Y10 = Y00 + 24 * OC;//ux: j4 -> j7
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
	winograd_f6x3_Y_vec(y0, a, x);//{oc0 / oc4} * { j0 - j3 }
	winograd_f6x3_Y_vec(y1, a, y);//{oc1 / oc5} * { j0 - j3 }
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
	winograd_f6x3_Y_vec(y2, a, x);//{oc2 / oc6} * { j0 - j3 }
	winograd_f6x3_Y_vec(y3, a, y);//{oc3 / oc7} * { j0 - j3 }
	
	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
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

	winograd_f6x3_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f6x3_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
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

	winograd_f6x3_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f6x3_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
}

#endif


//Standard_Version: Size = 81, Time = 10.4777 msec, Performace = 16601.6 GFlop/s
#ifndef WG_2X6_A3
#define WG_2X6_A3

#define wg2x6_A3(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel_A3<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 20.25, Time = 2.52857 msec, Performace = 17198.1 GFlop/s

//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
//Size = 81, Time = 10.7723 msec, Performace = 16147.6 GFlop/s
//Size = 81, Time = 10.479 msec, Performace = 16599.6 GFlop/s
template<int FH>
__global__ void Winograd2x6_kernel_A3(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
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
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int toc0 = boc0 + Gs_i;
	CW += Gs_k *OC + toc0;//CW[0, 0, Gs_k, toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int Ds_k = (tx & 7), Ds_i = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + Ds_i * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Ds_k;//X[tn0, tih0, tiw0, Ds_k]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//4 * 8
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	float g0[8], g1[8], x[8], d[8]; char ly[8];

	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 3 * IC_OC;//with the same tx
		const int W1 = W0 + IC_OC;
		const int W2 = W0 + (IC_OC << 1);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		
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

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
		winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
		winograd_f6x3_D(d, x);

		//write to shread memory
		*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gs_k][4][Gs_i]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gs_k][5][Gs_i]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gs_k][6][Gs_i]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gs_k][7][Gs_i]) = { g0[7], g1[7] };

		Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++)
			{
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

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = fh*3*IC_OC + oic * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC;
			const int W2 = W0 + (IC_OC << 1);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);

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

			//Winograd transform: W(3) -> G(8); X(4) -> D(4)
			winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
			winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
			winograd_f6x3_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gs_k][4][Gs_i]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gs_k][5][Gs_i]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gs_k][6][Gs_i]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gs_k][7][Gs_i]) = { g0[7], g1[7] };

			Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
			Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
			Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
			Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++)
		{
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

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Ys)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float y0[6], y1[6], y2[6], y3[6];
	
	const int Y10 = Y00 + 24 * OC;//ux: j4 -> j7
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
	winograd_f6x3_Y_vec(y0, a, x);//{oc0 / oc4} * { j0 - j3 }
	winograd_f6x3_Y_vec(y1, a, y);//{oc1 / oc5} * { j0 - j3 }
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
	winograd_f6x3_Y_vec(y2, a, x);//{oc2 / oc6} * { j0 - j3 }
	winograd_f6x3_Y_vec(y3, a, y);//{oc3 / oc7} * { j0 - j3 }
	
	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
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

	winograd_f6x3_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f6x3_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
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

	winograd_f6x3_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f6x3_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
}

#endif


//CW[FH, FW, IC, OC] is fine
//Size = 81, Time = 10.493 msec, Performace = 16577.4 GFlop/s
#ifndef WG_2X6_A4
#define WG_2X6_A4

#define wg2x6_A4(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel_A4<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 20.25, Time = 2.52857 msec, Performace = 17198.1 GFlop/s

//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
//Size = 81, Time = 10.7723 msec, Performace = 16147.6 GFlop/s
//Size = 81, Time = 10.479 msec, Performace = 16599.6 GFlop/s

template<int FH>
__global__ void Winograd2x6_kernel_A4(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
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

	//prepare for CW[FH, FW, IC, OC] -> [FH, IC, 3, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int toc0 = boc0 + Gs_i;
	CW += Gs_k * OC * 3 + toc0;//CW[0, Gs_k, 0, toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int Ds_k = (tx & 7), Ds_i = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + Ds_i * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Ds_k;//X[tn0, tih0, tiw0, Ds_k]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//4 * 8
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float g0[8], g1[8], x[8], d[8]; char ly[8];

	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from CW[FH, FW, IC, OC] -> [FH, IC, 3, OC]
		const int W0 = fh * 3 * IC * OC;//with the same tx
		const int W1 = W0 + OC;
		const int W2 = W0 + (OC << 1);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		
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

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
		winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
		winograd_f6x3_D(d, x);

		//write to shread memory
		*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gs_k][4][Gs_i]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gs_k][5][Gs_i]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gs_k][6][Gs_i]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gs_k][7][Gs_i]) = { g0[7], g1[7] };

		Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++)
			{
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

			//load 2 group from CW[FH, FW, IC, OC] -> [FH, IC, 3, OC]
			const int W0 = (fh*IC + oic) * 3 * OC;//fh, ic, fw, oc
			const int W1 = W0 + OC;
			const int W2 = W0 + (OC << 1);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);

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

			//Winograd transform: W(3) -> G(8); X(4) -> D(4)
			winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
			winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
			winograd_f6x3_D(d, x);

			//write to shread memory
			*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gs_k][4][Gs_i]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gs_k][5][Gs_i]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gs_k][6][Gs_i]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gs_k][7][Gs_i]) = { g0[7], g1[7] };

			Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
			Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
			Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
			Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++)
		{
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

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Ys)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float y0[6], y1[6], y2[6], y3[6];
	
	const int Y10 = Y00 + 24 * OC;//ux: j4 -> j7
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
	winograd_f6x3_Y_vec(y0, a, x);//{oc0 / oc4} * { j0 - j3 }
	winograd_f6x3_Y_vec(y1, a, y);//{oc1 / oc5} * { j0 - j3 }
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
	winograd_f6x3_Y_vec(y2, a, x);//{oc2 / oc6} * { j0 - j3 }
	winograd_f6x3_Y_vec(y3, a, y);//{oc3 / oc7} * { j0 - j3 }
	
	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
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

	winograd_f6x3_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f6x3_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
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

	winograd_f6x3_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f6x3_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
}

#endif


#ifndef WG_2X6_A5
#define WG_2X6_A5

#define wg2x6_A5(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel_A5<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 20.25, Time = 2.52857 msec, Performace = 17198.1 GFlop/s

//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
//Size = 81, Time = 10.7723 msec, Performace = 16147.6 GFlop/s
//Size = 81, Time = 10.479 msec, Performace = 16599.6 GFlop/s
template<int FH, int IC = 512>
__global__ void Winograd2x6_kernel_A5(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
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
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	const int toc0 = boc0 + Gs_i;
	CW += Gs_k *OC + toc0;//CW[0, 0, Gs_k, toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int Ds_k = (tx & 7), Ds_i = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + Ds_i * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Ds_k;//X[tn0, tih0, tiw0, Ds_k]

	//prepare for threadIdx
	const int ux = (ty >> 1), uy = tx + ((ty & 1) << 4);//4 * 8
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
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
		const int W0 = fh * 3 * IC_OC;//with the same tx
		const int W1 = W0 + IC_OC;
		const int W2 = W0 + (IC_OC << 1);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
		winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
		winograd_f6x3_D(d, x);
		
		Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];

		//write to shread memory
		*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = { g0[0], g1[0] };
		*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = { g0[1], g1[1] };
		*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = { g0[2], g1[2] };
		*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = { g0[3], g1[3] };
		*(float2*)(&Gs[buf][Gs_k][4][Gs_i]) = { g0[4], g1[4] };
		*(float2*)(&Gs[buf][Gs_k][5][Gs_i]) = { g0[5], g1[5] };
		*(float2*)(&Gs[buf][Gs_k][6][Gs_i]) = { g0[6], g1[6] };
		*(float2*)(&Gs[buf][Gs_k][7][Gs_i]) = { g0[7], g1[7] };
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++)
			{
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

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = fh * 3 * IC_OC + oic * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC;
			const int W2 = W0 + (IC_OC << 1);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);

			//Winograd transform: W(3) -> G(8); X(4) -> D(4)
			winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
			winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
			winograd_f6x3_D(d, x);
			
			Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
			Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
			Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
			Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];

			//write to shread memory
			*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = { g0[0], g1[0] };
			*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = { g0[1], g1[1] };
			*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = { g0[2], g1[2] };
			*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = { g0[3], g1[3] };
			*(float2*)(&Gs[buf][Gs_k][4][Gs_i]) = { g0[4], g1[4] };
			*(float2*)(&Gs[buf][Gs_k][5][Gs_i]) = { g0[5], g1[5] };
			*(float2*)(&Gs[buf][Gs_k][6][Gs_i]) = { g0[6], g1[6] };
			*(float2*)(&Gs[buf][Gs_k][7][Gs_i]) = { g0[7], g1[7] };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++)
		{
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

	//======[compute area12: block]======================================================
	float2 (* __restrict__ Ys)[33][10] = (float2(*)[33][10])(Gs);//8 * 32 * 16[elem]
	float2 a[16]; float y0[6], y1[6], y2[6], y3[6];
	
	const int Y10 = Y00 + 24 * OC;//ux: j4 -> j7
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
	winograd_f6x3_Y_vec(y0, a, x);//{oc0 / oc4} * { j0 - j3 }
	winograd_f6x3_Y_vec(y1, a, y);//{oc1 / oc5} * { j0 - j3 }
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
	winograd_f6x3_Y_vec(y2, a, x);//{oc2 / oc6} * { j0 - j3 }
	winograd_f6x3_Y_vec(y3, a, y);//{oc3 / oc7} * { j0 - j3 }
	
	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
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

	winograd_f6x3_Y_vec(y0, a, x);//{oc0 / oc4} * { j4 - j7 }
	winograd_f6x3_Y_vec(y1, a, y);//{oc1 / oc5} * { j4 - j7 }
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

	winograd_f6x3_Y_vec(y2, a, x);//{oc2 / oc6} * { j4 - j7 }
	winograd_f6x3_Y_vec(y3, a, y);//{oc3 / oc7} * { j4 - j7 }

	*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
}

#endif



