



#ifndef WINOGRAD_F8X9_K1
#define WINOGRAD_F8X9_K1

//32 * 32 group
#define WG_f8x9_K1(stream, oc_index, j_index, X, IH, IW, W, FH, Y, OH, OW, IC, OC) \
	WG_f8x9_Kernel1<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW>>3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC)

//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 20.25, Time = 2.52857 msec, Performace = 17198.1 GFlop/s

template<int FH>
__global__ void WG_f8x9_Kernel1(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC)//ph = pw = 1, sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 2][16][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 2][16][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 5);
	const int Gs_k = (ty & 7), Gs_i = (tx << 1) + (ty > 7);//[8, 64]
	const int toc0 = boc0 + Gs_i;
	CW += Gs_k *OC + toc0;//CW[0, 0, Gs_k, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 8);//32 * 8 = 256
	const int Ds_k = (tx & 7), Ds_i = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0  + Ds_i * 8;
	const int OW8 = OW / 8 * 8, OH_OW8 = OH * OW8;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW8; int jr = tj0 - tn0 * OH_OW8; 
		toh0 = jr / OW8; tow0 = jr - toh0 * OW8;
	}
	const int tih0 = toh0 - 1, tiw0 = tow0 - 1;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Ds_k;//X[tn0, tih0, tiw0, Ds_k]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 8;
	const int y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16]; char ly[16];

	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0     < IW);//pw = -4
		bool ly1 = lh0 && (tiw0 >= 1) && (tiw0 + 1 < IW);//pw = -3
		bool ly2 = lh0 && (tiw0 >= 2) && (tiw0 + 2 < IW);//pw = -2
		bool ly3 = lh0 && (tiw0 >= 3) && (tiw0 + 3 < IW);//pw = -1
		bool lyC = lh0 && (tiw0 >= -12) && (tiw0 + 12 < IW);//pw = +1
		bool lyD = lh0 && (tiw0 >= -13) && (tiw0 + 13 < IW);//pw = +2
		bool lyE = lh0 && (tiw0 >= -14) && (tiw0 + 14 < IW);//pw = +3
		bool lyF = lh0 && (tiw0 >= -15) && (tiw0 + 15 < IW);//pw = +4
		int tX0  = X0 + fh * IW * IC;
		int tX1  = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3  = tX0 + IC *  3, tX4  = tX0 + (IC << 2);
		int tX5  = tX0 + IC *  5, tX6  = tX0 + IC * 6;
		int tX7  = tX0 + IC *  7, tX8  = tX0 + (IC << 3);
		int tX9  = tX0 + IC *  9, tX10 = tX0 + IC * 10;
		int tX11 = tX0 + IC * 11, tX12 = tX0 + IC * 12;
		int tX13 = tX0 + IC * 13, tX14 = tX0 + IC * 14;
		int tX15 = tX0 + IC * 15;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);

		tX4  = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
		tX6  = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
		tX8  = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
		tX10 = IF_int(lh0, tX10, -1); tX11 = IF_int(lh0, tX11, -1);

		tX12 = IF_int(lyC, tX12, -1); tX13 = IF_int(lyD, tX13, -1); 
		tX14 = IF_int(lyE, tX14, -1); tX15 = IF_int(lyF, tX15, -1);

		x[ 0] = tex1Dfetch<float>(X,  tX0); x[ 1] = tex1Dfetch<float>(X,  tX1);
		x[ 2] = tex1Dfetch<float>(X,  tX2); x[ 3] = tex1Dfetch<float>(X,  tX3);
		x[ 4] = tex1Dfetch<float>(X,  tX4); x[ 5] = tex1Dfetch<float>(X,  tX5);
		x[ 6] = tex1Dfetch<float>(X,  tX6); x[ 7] = tex1Dfetch<float>(X,  tX7);
		x[ 8] = tex1Dfetch<float>(X,  tX8); x[ 9] = tex1Dfetch<float>(X,  tX9);
		x[10] = tex1Dfetch<float>(X, tX10); x[11] = tex1Dfetch<float>(X, tX11);
		x[12] = tex1Dfetch<float>(X, tX12); x[13] = tex1Dfetch<float>(X, tX13);
		x[14] = tex1Dfetch<float>(X, tX14); x[15] = tex1Dfetch<float>(X, tX15);

		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 9 * IC * OC;//with the same tx
		const int W1 = W0 + IC * OC;
		const int W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + (IC * OC << 1);
		const int W4 = W0 + (IC * OC << 1);
		const int W5 = W0 + (IC * OC << 1);
		const int W6 = W0 + (IC * OC << 1);
		const int W7 = W0 + (IC * OC << 1);
		const int W8 = W0 + (IC * OC << 1);
		w[0] = CW[W0];
		w[1] = CW[W1];
		w[2] = CW[W2];
		w[3] = CW[W3];
		w[4] = CW[W4];
		w[5] = CW[W5];
		w[6] = CW[W6];
		w[7] = CW[W7];
		w[8] = CW[W8];


		for (int oic = 8; oic < IC; oic += 8) {
			winograd_f14x3_G(g, w[0], w[1], w[2]);
			winograd_f14x3_D(d, x);

			//write to shread memory
			Gs[Gs_k][ 0][Gs_i] = g[ 0]; Gs[Gs_k][ 1][Gs_i] = g[ 1];
			Gs[Gs_k][ 2][Gs_i] = g[ 2]; Gs[Gs_k][ 3][Gs_i] = g[ 3];
			Gs[Gs_k][ 4][Gs_i] = g[ 4]; Gs[Gs_k][ 5][Gs_i] = g[ 5];
			Gs[Gs_k][ 6][Gs_i] = g[ 6]; Gs[Gs_k][ 7][Gs_i] = g[ 7];
			Gs[Gs_k][ 8][Gs_i] = g[ 8]; Gs[Gs_k][ 9][Gs_i] = g[ 9];
			Gs[Gs_k][10][Gs_i] = g[10]; Gs[Gs_k][11][Gs_i] = g[11];
			Gs[Gs_k][12][Gs_i] = g[12]; Gs[Gs_k][13][Gs_i] = g[13];
			Gs[Gs_k][14][Gs_i] = g[14]; Gs[Gs_k][15][Gs_i] = g[15];
			
			Ds[Ds_k][ 0][Ds_i] = d[ 0]; Ds[Ds_k][ 1][Ds_i] = d[ 1];
			Ds[Ds_k][ 2][Ds_i] = d[ 2]; Ds[Ds_k][ 3][Ds_i] = d[ 3];
			Ds[Ds_k][ 4][Ds_i] = d[ 4]; Ds[Ds_k][ 5][Ds_i] = d[ 5];
			Ds[Ds_k][ 6][Ds_i] = d[ 6]; Ds[Ds_k][ 7][Ds_i] = d[ 7];
			Ds[Ds_k][ 8][Ds_i] = d[ 8]; Ds[Ds_k][ 9][Ds_i] = d[ 9];
			Ds[Ds_k][10][Ds_i] = d[10]; Ds[Ds_k][11][Ds_i] = d[11];
			Ds[Ds_k][12][Ds_i] = d[12]; Ds[Ds_k][13][Ds_i] = d[13];
			Ds[Ds_k][14][Ds_i] = d[14]; Ds[Ds_k][15][Ds_i] = d[15];
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

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1  = tX0 + IC, tX2 = tX0 + (IC << 1);
			int tX3  = tX0 + IC *  3, tX4  = tX0 + (IC << 2);
			int tX5  = tX0 + IC *  5, tX6  = tX0 + IC * 6;
			int tX7  = tX0 + IC *  7, tX8  = tX0 + (IC << 3);
			int tX9  = tX0 + IC *  9, tX10 = tX0 + IC * 10;
			int tX11 = tX0 + IC * 11, tX12 = tX0 + IC * 12;
			int tX13 = tX0 + IC * 13, tX14 = tX0 + IC * 14;
			int tX15 = tX0 + IC * 15;
			
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);

			tX4 = IF_int(lh0, tX4, -1); tX5 = IF_int(lh0, tX5, -1);
			tX6 = IF_int(lh0, tX6, -1); tX7 = IF_int(lh0, tX7, -1);
			tX8 = IF_int(lh0, tX8, -1); tX9 = IF_int(lh0, tX9, -1);
			tX10 = IF_int(lh0, tX10, -1); tX11 = IF_int(lh0, tX11, -1);

			tX12 = IF_int(lyC, tX12, -1); tX13 = IF_int(lyD, tX13, -1);
			tX14 = IF_int(lyE, tX14, -1); tX15 = IF_int(lyF, tX15, -1);

			x[ 0] = tex1Dfetch<float>(X,  tX0); x[ 1] = tex1Dfetch<float>(X,  tX1);
			x[ 2] = tex1Dfetch<float>(X,  tX2); x[ 3] = tex1Dfetch<float>(X,  tX3);
			x[ 4] = tex1Dfetch<float>(X,  tX4); x[ 5] = tex1Dfetch<float>(X,  tX5);
			x[ 6] = tex1Dfetch<float>(X,  tX6); x[ 7] = tex1Dfetch<float>(X,  tX7);
			x[ 8] = tex1Dfetch<float>(X,  tX8); x[ 9] = tex1Dfetch<float>(X,  tX9);
			x[10] = tex1Dfetch<float>(X, tX10); x[11] = tex1Dfetch<float>(X, tX11);
			x[12] = tex1Dfetch<float>(X, tX12); x[13] = tex1Dfetch<float>(X, tX13);
			x[14] = tex1Dfetch<float>(X, tX14); x[15] = tex1Dfetch<float>(X, tX15);

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh * 9 * IC + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC * OC;
			const int W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + (IC * OC << 1);
			const int W4 = W0 + (IC * OC << 1);
			const int W5 = W0 + (IC * OC << 1);
			const int W6 = W0 + (IC * OC << 1);
			const int W7 = W0 + (IC * OC << 1);
			const int W8 = W0 + (IC * OC << 1);
			w[0] = CW[W0];
			w[1] = CW[W1];
			w[2] = CW[W2];
			w[3] = CW[W3];
			w[4] = CW[W4];
			w[5] = CW[W5];
			w[6] = CW[W6];
			w[7] = CW[W7];
			w[8] = CW[W8];
			__syncthreads();
		}
		{
			winograd_f14x3_G(g, w[0], w[1], w[2]);
			winograd_f14x3_D(d, x);

			//write to shread memory
			Gs[Gs_k][ 0][Gs_i] = g[ 0]; Gs[Gs_k][ 1][Gs_i] = g[ 1];
			Gs[Gs_k][ 2][Gs_i] = g[ 2]; Gs[Gs_k][ 3][Gs_i] = g[ 3];
			Gs[Gs_k][ 4][Gs_i] = g[ 4]; Gs[Gs_k][ 5][Gs_i] = g[ 5];
			Gs[Gs_k][ 6][Gs_i] = g[ 6]; Gs[Gs_k][ 7][Gs_i] = g[ 7];
			Gs[Gs_k][ 8][Gs_i] = g[ 8]; Gs[Gs_k][ 9][Gs_i] = g[ 9];
			Gs[Gs_k][10][Gs_i] = g[10]; Gs[Gs_k][11][Gs_i] = g[11];
			Gs[Gs_k][12][Gs_i] = g[12]; Gs[Gs_k][13][Gs_i] = g[13];
			Gs[Gs_k][14][Gs_i] = g[14]; Gs[Gs_k][15][Gs_i] = g[15];
			
			Ds[Ds_k][ 0][Ds_i] = d[ 0]; Ds[Ds_k][ 1][Ds_i] = d[ 1];
			Ds[Ds_k][ 2][Ds_i] = d[ 2]; Ds[Ds_k][ 3][Ds_i] = d[ 3];
			Ds[Ds_k][ 4][Ds_i] = d[ 4]; Ds[Ds_k][ 5][Ds_i] = d[ 5];
			Ds[Ds_k][ 6][Ds_i] = d[ 6]; Ds[Ds_k][ 7][Ds_i] = d[ 7];
			Ds[Ds_k][ 8][Ds_i] = d[ 8]; Ds[Ds_k][ 9][Ds_i] = d[ 9];
			Ds[Ds_k][10][Ds_i] = d[10]; Ds[Ds_k][11][Ds_i] = d[11];
			Ds[Ds_k][12][Ds_i] = d[12]; Ds[Ds_k][13][Ds_i] = d[13];
			Ds[Ds_k][14][Ds_i] = d[14]; Ds[Ds_k][15][Ds_i] = d[15];
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
	}

	//======[compute area12: block]======================================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Ds[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0-----------------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{ v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{ v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//write-turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{ v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{ v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	////read-turn0: x, [oc0, oc4]
	a[0] = Ys0[Yrd]; a[1] = Ys0[Yrd + 340]; a[2] = Ys0[Yrd + 680]; a[3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[4] = Ys0[Yrd + 1360]; a[5] = Ys0[Yrd + 1700]; a[6] = Ys0[Yrd + 2040]; a[7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[8] = Ys0[Yrd + 2720]; a[9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	//read-turn1: x, [oc0, oc4]
	a[0] = Ys1[Yrd]; a[1] = Ys1[Yrd + 340]; a[2] = Ys1[Yrd + 680]; a[3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[4] = Ys1[Yrd + 1360]; a[5] = Ys1[Yrd + 1700]; a[6] = Ys1[Yrd + 2040]; a[7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[8] = Ys1[Yrd + 2720]; a[9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt + 4) = float4{ v4.z,  v5.z,  v6.z,  v7.z };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt + 8) = float4{ v8.z,  v9.z, v10.z, v11.z };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };//(ux, uy, 12 - 15)

	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt + 4) = float4{ v4.w,  v5.w,  v6.w,  v7.w };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt + 8) = float4{ v8.w,  v9.w, v10.w, v11.w };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };//(ux, uy, 12 - 15)
	__syncthreads();

	//read-turn2: z, [oc0, oc4]
	a[0] = Ys0[Yrd]; a[1] = Ys0[Yrd + 340]; a[2] = Ys0[Yrd + 680]; a[3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[4] = Ys0[Yrd + 1360]; a[5] = Ys0[Yrd + 1700]; a[6] = Ys0[Yrd + 2040]; a[7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[8] = Ys0[Yrd + 2720]; a[9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	//read-turn3: z, [oc0, oc4]
	a[0] = Ys1[Yrd]; a[1] = Ys1[Yrd + 340]; a[2] = Ys1[Yrd + 680]; a[3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[4] = Ys1[Yrd + 1360]; a[5] = Ys1[Yrd + 1700]; a[6] = Ys1[Yrd + 2040]; a[7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[8] = Ys1[Yrd + 2720]; a[9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y3);

	//write to Y[N, OH, OW, OC]
	const int y01 = y00 + OC;
	const int y10 = y00 + OW * OC, y11 = y10 + OC;

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif



#ifndef WINOGRAD_F8X9_K2
#define WINOGRAD_F8X9_K2

//32 * 32 group
#define WG_f8x9_K2(stream, oc_index, j_index, X, IH, IW, W, FH, Y, OH, OW, IC, OC) \
	WG_f8x9_Kernel2<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW>>3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC)

template<int FH>
__global__ void WG_f8x9_Kernel2(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC)//ph = pw = 4, sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 2][16][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 2][16][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 4);//OW % 8 == 0
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - 4, tiw0 = tow0 - 4;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 8;
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);

		bool ly0 = lh0 && (tiw0 >=   0) && (tiw0      < IW);//pw = -4
		bool ly1 = lh0 && (tiw0 >=  -1) && (tiw0 +  1 < IW);//pw = -3
		bool ly2 = lh0 && (tiw0 >=  -2) && (tiw0 +  2 < IW);//pw = -2
		bool ly3 = lh0 && (tiw0 >=  -3) && (tiw0 +  3 < IW);//pw = -1

		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);//pw = -4
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);//pw = -3
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);//pw = -2
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);//pw = -1

		bool lyC = lh0 && (tiw0 >= -12) && (tiw0 + 12 < IW);//pw = +1
		bool lyD = lh0 && (tiw0 >= -13) && (tiw0 + 13 < IW);//pw = +2
		bool lyE = lh0 && (tiw0 >= -14) && (tiw0 + 14 < IW);//pw = +3
		bool lyF = lh0 && (tiw0 >= -15) && (tiw0 + 15 < IW);//pw = +4

		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC     , tX2 = tX0 + IC *  2;
		int tX3 = tX0 + IC *  3, tX4 = tX0 + IC *  4;
		int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
		int tX7 = tX0 + IC *  7, tX8 = tX0 + IC *  8;
		int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
		int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
		int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
		int tXF = tX0 + IC * 15;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);

		tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
		tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);

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

		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 9 * IC * OC;//with the same tx
		const int W1 = W0 + IC * OC,     W2 = W0 + IC * OC * 2;
		const int W3 = W0 + IC * OC * 3, W4 = W0 + IC * OC * 4;
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7, W8 = W0 + IC * OC * 8;
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

		for (int oic = 8; oic < IC; oic += 8) {
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

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC     , tX2 = tX0 + IC *  2;
			int tX3 = tX0 + IC *  3, tX4 = tX0 + IC *  4;
			int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
			int tX7 = tX0 + IC *  7, tX8 = tX0 + IC *  8;
			int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
			int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
			int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
			int tXF = tX0 + IC * 15;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);

			tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
			tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);

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
			
			//load 1 group from CW[FH, 9, IC, OC]
			const int W0 = (fh * 9 * IC + oic) * OC;
			const int W1 = W0 + IC * OC,     W2 = W0 + IC * OC * 2;
			const int W3 = W0 + IC * OC * 3, W4 = W0 + IC * OC * 4;
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7, W8 = W0 + IC * OC * 8;
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];
			winograd_f8x9_G(g, w);
			winograd_f8x9_D(d, x);
			__syncthreads();
		}
		{
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
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[8], y1[8], y2[8], y3[8];//oc0, oc1, oc2, oc3
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
	winograd_f8x9_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y1, a);
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
	winograd_f8x9_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
}

#endif


//Size = 162, Time = 20.0877 msec, Performace = 17318.7 GFlop/s
#ifndef WINOGRAD_F8X9_K3
#define WINOGRAD_F8X9_K3

//32 * 32 group
#define WG_f8x9_K3(stream, oc_index, j_index, X, IH, IW, W, FH, Y, OH, OW, IC, OC, ph, pw) \
	WG_f8x9_Kernel3<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW>>3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC, ph, pw)

template<int FH>
__global__ void WG_f8x9_Kernel3(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw)//sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 2][16][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 2][16][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 3);//OW % 8 == 0
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 8;
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);

		bool ly0 = lh0 && (tiw0 >=   0) && (tiw0      < IW);//pw = -4
		bool ly1 = lh0 && (tiw0 >=  -1) && (tiw0 +  1 < IW);//pw = -3
		bool ly2 = lh0 && (tiw0 >=  -2) && (tiw0 +  2 < IW);//pw = -2
		bool ly3 = lh0 && (tiw0 >=  -3) && (tiw0 +  3 < IW);//pw = -1

		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);//pw = -4
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);//pw = -3
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);//pw = -2
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);//pw = -1

		bool ly8 = lh0 && (tiw0 >=  -8) && (tiw0 +  8 < IW);//pw = -4
		bool ly9 = lh0 && (tiw0 >=  -9) && (tiw0 +  9 < IW);//pw = -3
		bool lyA = lh0 && (tiw0 >= -10) && (tiw0 + 10 < IW);//pw = -2
		bool lyB = lh0 && (tiw0 >= -11) && (tiw0 + 11 < IW);//pw = -1

		bool lyC = lh0 && (tiw0 >= -12) && (tiw0 + 12 < IW);//pw = +1
		bool lyD = lh0 && (tiw0 >= -13) && (tiw0 + 13 < IW);//pw = +2
		bool lyE = lh0 && (tiw0 >= -14) && (tiw0 + 14 < IW);//pw = +3
		bool lyF = lh0 && (tiw0 >= -15) && (tiw0 + 15 < IW);//pw = +4

		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC     , tX2 = tX0 + IC *  2;
		int tX3 = tX0 + IC *  3, tX4 = tX0 + IC *  4;
		int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
		int tX7 = tX0 + IC *  7, tX8 = tX0 + IC *  8;
		int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
		int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
		int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
		int tXF = tX0 + IC * 15;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
		tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
		tX8 = IF_int(ly8, tX8, -1); tX9 = IF_int(ly9, tX9, -1);
		tXA = IF_int(lyA, tXA, -1); tXB = IF_int(lyB, tXB, -1);
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

		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 9 * IC * OC;//with the same tx
		const int W1 = W0 + IC * OC,     W2 = W0 + IC * OC * 2;
		const int W3 = W0 + IC * OC * 3, W4 = W0 + IC * OC * 4;
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7, W8 = W0 + IC * OC * 8;
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

		for (int oic = 8; oic < IC; oic += 8) {
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

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC     , tX2 = tX0 + IC *  2;
			int tX3 = tX0 + IC *  3, tX4 = tX0 + IC *  4;
			int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
			int tX7 = tX0 + IC *  7, tX8 = tX0 + IC *  8;
			int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
			int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
			int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
			int tXF = tX0 + IC * 15;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);

			tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
			tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);

			tX8 = IF_int(ly8, tX8, -1); tX9 = IF_int(ly9, tX9, -1);
			tXA = IF_int(lyA, tXA, -1); tXB = IF_int(lyB, tXB, -1);

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
			
			//load 1 group from CW[FH, 9, IC, OC]
			const int W0 = (fh * 9 * IC + oic) * OC;
			const int W1 = W0 + IC * OC,     W2 = W0 + IC * OC * 2;
			const int W3 = W0 + IC * OC * 3, W4 = W0 + IC * OC * 4;
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7, W8 = W0 + IC * OC * 8;
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];
			__syncthreads();
		}
		{
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
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[8], y1[8], y2[8], y3[8];//oc0, oc1, oc2, oc3
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
	winograd_f8x9_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y1, a);
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
	winograd_f8x9_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
}

#endif



//LB = 4: Size = 162, Time = 17.7006 msec, Performace = 19654.3 GFlop/s
//WB = 4: Size = 162, Time = 17.0855 msec, Performace = 20361.8 GFlop/s
#ifndef WINOGRAD_F8X9_K4
#define WINOGRAD_F8X9_K4

//32 * 32 group
#define WG_f8x9_K4(stream, oc_index, j_index, X, IH, IW, W, FH, Y, OH, OW, IC, OC) \
	WG_f8x9_Kernel4<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW>>3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC)

template<int FH>
__global__ void WG_f8x9_Kernel4(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC)//ph = pw = 4, sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 3);//OW % 8 == 0
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - 4, tiw0 = tow0 - 4;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 9 * IC * OC;
		const int W1 = W0 + IC * OC,     W2 = W0 + IC * OC * 2;
		const int W3 = W0 + IC * OC * 3, W4 = W0 + IC * OC * 4;
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7, W8 = W0 + IC * OC * 8;
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

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
		int tX1 = tX0 + IC     , tX2 = tX0 + IC *  2;
		int tX3 = tX0 + IC *  3, tX4 = tX0 + IC *  4;
		int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
		int tX7 = tX0 + IC *  7, tX8 = tX0 + IC *  8;
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

		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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

			//load 1 group from CW[FH, 9, IC, OC]
			const int W0 = (fh * 9 * IC + oic) * OC;
			const int W1 = W0 + IC * OC    , W2 = W0 + IC * OC * 2;
			const int W3 = W0 + IC * OC * 3, W4 = W0 + IC * OC * 4;
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7, W8 = W0 + IC * OC * 8;
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC     , tX2 = tX0 + IC *  2;
			int tX3 = tX0 + IC *  3, tX4 = tX0 + IC *  4;
			int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
			int tX7 = tX0 + IC *  7, tX8 = tX0 + IC *  8;
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
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[8], y1[8], y2[8], y3[8];//oc0, oc1, oc2, oc3
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
	winograd_f8x9_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y1, a);
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
	winograd_f8x9_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
}

#endif


//Standard(Common), ph = pw = 4
#ifndef WINOGRAD_F8X9_K5
#define WINOGRAD_F8X9_K5

//32 * 32 group
#define WG_f8x9_K5(stream, oc_index, j_index, X, IH, IW, W, FH, Y, OH, OW, IC, OC) \
	WG_f8x9_Kernel5<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW>>3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC)

template<int FH>
__global__ void WG_f8x9_Kernel5(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC)//ph = pw = 4, sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 3);//OW % 8 == 0
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - 4, tiw0 = tow0 - 4;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 9 * IC * OC;
		const int W1 = W0 + IC * OC,     W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7, W8 = W0 + (IC * OC << 3);
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

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
		
		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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

			//load 1 group from CW[FH, 9, IC, OC]
			const int W0 = (fh * 9 * IC + oic) * OC;
			const int W1 = W0 + IC * OC    , W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7, W8 = W0 + (IC * OC << 3);
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

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
			x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
			x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
			x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
			x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
			x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
			x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
			x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
			x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[8], y1[8], y2[8], y3[8];//oc0, oc1, oc2, oc3
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
	winograd_f8x9_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y1, a);
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
	winograd_f8x9_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
}

#endif


//template(IC), ph = pw = 4
#ifndef WINOGRAD_F8X9_K6
#define WINOGRAD_F8X9_K6

//32 * 32 group
#define WG_f8x9_K6(stream, oc_index, j_index, X, IH, IW, W, FH, Y, OH, OW, IC, OC) \
	WG_f8x9_Kernel6<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW>>3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, W, Y, OH, OW, OC)

//Size = 162, Time = 15.544 msec, Performace = 22381.1 GFlop/s
template<int FH, int IC = 256>
__global__ void WG_f8x9_Kernel6(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int OC)//ph = pw = 4, sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 3);//OW % 8 == 0
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - 4, tiw0 = tow0 - 4;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 9 * IC * OC;
		const int W1 = W0 + IC * OC,     W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7, W8 = W0 + (IC * OC << 3);
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

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
		
		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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

			//load 1 group from CW[FH, 9, IC, OC]
			const int W0 = (fh * 9 * IC + oic) * OC;
			const int W1 = W0 + IC * OC    , W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7, W8 = W0 + (IC * OC << 3);
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

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
			x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
			x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
			x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
			x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
			x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
			x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
			x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
			x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[8], y1[8], y2[8], y3[8];//oc0, oc1, oc2, oc3
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
	winograd_f8x9_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y1, a);
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
	winograd_f8x9_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
}

#endif


//template(IC), ph = pw = 4
#ifndef WINOGRAD_F8X9_K7
#define WINOGRAD_F8X9_K7

//32 * 32 group
#define WG_f8x9_K7(stream, oc_index, j_index, X, IH, IW, W, FH, Y, OH, OW, IC, OC) \
	WG_f8x9_Kernel7<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW>>3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, W, Y, OH, OW)

//Size = 162, Time = 14.8187 msec, Performace = 23476.6 GFlop/s
//Size = 162, Time = 15.544  msec, Performace = 22381.1 GFlop/s
template<int FH, int IC = 256, int OC = 256>
__global__ void WG_f8x9_Kernel7(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW)//ph = pw = 4, sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 3);//OW % 8 == 0
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - 4, tiw0 = tow0 - 4;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 9 * IC * OC;
		const int W1 = W0 + IC * OC,     W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7, W8 = W0 + (IC * OC << 3);
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

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
		
		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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

			//load 1 group from CW[FH, 9, IC, OC]
			const int W0 = (fh * 9 * IC + oic) * OC;
			const int W1 = W0 + IC * OC    , W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7, W8 = W0 + (IC * OC << 3);
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

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
			x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
			x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
			x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
			x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
			x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
			x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
			x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
			x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[8], y1[8], y2[8], y3[8];//oc0, oc1, oc2, oc3
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
	winograd_f8x9_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y1, a);
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
	winograd_f8x9_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
}

#endif


//Swicczle
//Standard(Common), ph = pw = 4
#ifndef WINOGRAD_F8X9_K8
#define WINOGRAD_F8X9_K8

//32 * 32 group
#define WG_f8x9_K8(stream, oc_index, j_index, X, IH, IW, W, FH, Y, OH, OW, IC, OC) \
	WG_f8x9_Kernel8<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW>>3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC)

//Size = 162, Time = 16.3678 msec, Performace = 21254.6 GFlop/s
//

template<int FH>
__global__ void WG_f8x9_Kernel8(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC)//ph = pw = 4, sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	//const int bx = blockIdx.x, by = blockIdx.y;

	const int log_tile = 1;
	const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
	const int by = ((bidx / (gridDim.x << log_tile)) << log_tile) + (bidx & ((1 << log_tile) - 1));
	const int bx = (bidx % (gridDim.x << log_tile)) >> log_tile;

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (by << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 3);//OW % 8 == 0
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - 4, tiw0 = tow0 - 4;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 9 * IC * OC;
		const int W1 = W0 + IC * OC,     W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7, W8 = W0 + (IC * OC << 3);
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

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
		
		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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

			//load 1 group from CW[FH, 9, IC, OC]
			const int W0 = (fh * 9 * IC + oic) * OC;
			const int W1 = W0 + IC * OC    , W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7, W8 = W0 + (IC * OC << 3);
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

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
			x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
			x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
			x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
			x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
			x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
			x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
			x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
			x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[8], y1[8], y2[8], y3[8];//oc0, oc1, oc2, oc3
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
	winograd_f8x9_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y1, a);
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
	winograd_f8x9_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
}

#endif


//[OC, FH, FW, IC] has a very low L2 hit ratio
//Standard(Common), ph = pw = 4
#ifndef WINOGRAD_F8X9_K9
#define WINOGRAD_F8X9_K9

//32 * 32 group
#define WG_f8x9_K9(stream, oc_index, j_index, X, IH, IW, W, FH, Y, OH, OW, IC, OC) \
	WG_f8x9_Kernel9<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW>>3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC)

//Size = 162, Time = 16.3678 msec, Performace = 21254.6 GFlop/s

template<int FH>
__global__ void WG_f8x9_Kernel9(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ W,//[ OC, FH, FW, IC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC)//ph = pw = 4, sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;
	//const int bx = blockIdx.x, by = blockIdx.y;

	const int log_tile = 1;
	const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
	/*const int by = ((bidx / (gridDim.x << log_tile)) << log_tile) + (bidx & ((1 << log_tile) - 1));
	const int bx = (bidx % (gridDim.x << log_tile)) >> log_tile;
*/
	const int bx = ((bidx / (gridDim.y << log_tile)) << log_tile) + (bidx & ((1 << log_tile) - 1));
	const int by = (bidx % (gridDim.y << log_tile)) >> log_tile;


	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int toc0 = boc0 + Gi;
	W += toc0 * FH * 9 * IC + Gk;//W[toc0, 0, 0, Gk]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (by << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 3);//OW % 8 == 0
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - 4, tiw0 = tow0 - 4;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 9 * IC;
		const int W1 = W0 + IC,     W2 = W0 + (IC << 1);
		const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
		const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
		const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
		w[0] = W[W0]; w[1] = W[W1]; w[2] = W[W2];
		w[3] = W[W3]; w[4] = W[W4]; w[5] = W[W5];
		w[6] = W[W6]; w[7] = W[W7]; w[8] = W[W8];

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
		
		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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

			//load 1 group from CW[FH, 9, IC, OC]
			const int W0 = fh * 9 * IC + oic;
			const int W1 = W0 + IC, W2 = W0 + (IC << 1);
			const int W3 = W0 + IC * 3, W4 = W0 + (IC << 2);
			const int W5 = W0 + IC * 5, W6 = W0 + IC * 6;
			const int W7 = W0 + IC * 7, W8 = W0 + (IC << 3);
			w[0] = W[W0]; w[1] = W[W1]; w[2] = W[W2];
			w[3] = W[W3]; w[4] = W[W4]; w[5] = W[W5];
			w[6] = W[W6]; w[7] = W[W7]; w[8] = W[W8];

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
			x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
			x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
			x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
			x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
			x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
			x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
			x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
			x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[8], y1[8], y2[8], y3[8];//oc0, oc1, oc2, oc3
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
	winograd_f8x9_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y1, a);
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
	winograd_f8x9_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
}

#endif


//for texture memory, the conditional statements has no effects
#ifndef WINOGRAD_F8X9_K10
#define WINOGRAD_F8X9_K10

//32 * 32 group
#define WG_f8x9_K10(stream, oc_index, j_index, X, IH, IW, W, FH, Y, OH, OW, IC, OC) \
	WG_f8x9_Kernel10<FH>\
		<<< dim3(OC>>5, ((N*OH)>>5) * (OW>>3)), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC)

//Size = 162, Time = 16.3678 msec, Performace = 21254.6 GFlop/s

template<int FH>
__global__ void WG_f8x9_Kernel10(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC)//ph = pw = 4, sh = sw = 1
{
	const int tx = threadIdx.x, ty = threadIdx.y;
	const int bx = blockIdx.x, by = blockIdx.y;

	/*const int log_tile = 1;
	const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
	const int by = ((bidx / (gridDim.x << log_tile)) << log_tile) + (bidx & ((1 << log_tile) - 1));
	const int bx = (bidx % (gridDim.x << log_tile)) >> log_tile;*/

	__shared__ float Gs[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][g(oc): 32]
	__shared__ float Ds[8 + 1][16 + 1][32 + 4];//[buf][ik: STEP][elem: 16][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (bx << 5);
	const int Gk = (ty & 7), Gi = (tx << 1) + (ty > 7);//[8, 64]
	const int toc0 = boc0 + Gi;
	CW += Gk * OC + toc0;//CW[0, 0, Gk, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (by << 8);//32 * 8 = 256
	const int Xk = (tx & 7), Xi = (ty << 1) + (tx > 7);//[8, 32]
	const int tj0 = bj0 + (Xi << 3);//OW % 8 == 0
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - 4, tiw0 = tow0 - 4;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + Xk;//X[tn0, tih0, tiw0, Xk]

	//prepare for threadIdx
	const int ux = ty, uy = tx;
	const int DIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//4
	const int GIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);
	const int yj0 = bj0 + ((DIdx + (ux >> 1)) << 3);
	const int Y00 = yj0 * OC + yoc0;

	//======[compute area1: local]======================================================
	float w[9], g[16], x[16], d[16];
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from CW[FH, FW, IC, OC]
		const int W0 = fh * 9 * IC * OC;
		const int W1 = W0 + IC * OC,     W2 = W0 + (IC * OC << 1);
		const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
		const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
		const int W7 = W0 + IC * OC * 7, W8 = W0 + (IC * OC << 3);
		w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
		w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
		w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

		//load 1 group from X[N, IH, IW, IC]
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC     , tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC *  3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC *  5, tX6 = tX0 + IC *  6;
		int tX7 = tX0 + IC *  7, tX8 = tX0 + (IC << 3);
		int tX9 = tX0 + IC *  9, tXA = tX0 + IC * 10;
		int tXB = tX0 + IC * 11, tXC = tX0 + IC * 12;
		int tXD = tX0 + IC * 13, tXE = tX0 + IC * 14;
		int tXF = tX0 + IC * 15;
		x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
		x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
		x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
		x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
		x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
		x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
		x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
		x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);
		
		for (int oic = 8; oic < IC; oic += 8) {
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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

			//load 1 group from CW[FH, 9, IC, OC]
			const int W0 = (fh * 9 * IC + oic) * OC;
			const int W1 = W0 + IC * OC    , W2 = W0 + (IC * OC << 1);
			const int W3 = W0 + IC * OC * 3, W4 = W0 + (IC * OC << 2);
			const int W5 = W0 + IC * OC * 5, W6 = W0 + IC * OC * 6;
			const int W7 = W0 + IC * OC * 7, W8 = W0 + (IC * OC << 3);
			w[0] = CW[W0]; w[1] = CW[W1]; w[2] = CW[W2];
			w[3] = CW[W3]; w[4] = CW[W4]; w[5] = CW[W5];
			w[6] = CW[W6]; w[7] = CW[W7]; w[8] = CW[W8];

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
			x[ 0] = tex1Dfetch<float>(X, tX0); x[ 1] = tex1Dfetch<float>(X, tX1);
			x[ 2] = tex1Dfetch<float>(X, tX2); x[ 3] = tex1Dfetch<float>(X, tX3);
			x[ 4] = tex1Dfetch<float>(X, tX4); x[ 5] = tex1Dfetch<float>(X, tX5);
			x[ 6] = tex1Dfetch<float>(X, tX6); x[ 7] = tex1Dfetch<float>(X, tX7);
			x[ 8] = tex1Dfetch<float>(X, tX8); x[ 9] = tex1Dfetch<float>(X, tX9);
			x[10] = tex1Dfetch<float>(X, tXA); x[11] = tex1Dfetch<float>(X, tXB);
			x[12] = tex1Dfetch<float>(X, tXC); x[13] = tex1Dfetch<float>(X, tXD);
			x[14] = tex1Dfetch<float>(X, tXE); x[15] = tex1Dfetch<float>(X, tXF);
			__syncthreads();
		}
		{
			//Winograd Transform: W(9) -> G(16), X(16) -> D(16)
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
	}

	//======[compute area12: block]======================================================
	float(* __restrict__ Ys0)[17][20] = (float(*)[17][20])(Gs);
	float(* __restrict__ Ys1)[17][20] = (float(*)[17][20])(Ds);
	float a[16], y0[8], y1[8], y2[8], y3[8];//oc0, oc1, oc2, oc3
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
	winograd_f8x9_Y(y0, a);

	//read-turn1: y, [oc1, oc5]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y1, a);
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
	winograd_f8x9_Y(y2, a);

	//read-turn3: w, [oc3, oc7]
	a[ 0] = Ys1[ 0][uy][ux]; a[ 1] = Ys1[ 1][uy][ux]; a[ 2] = Ys1[ 2][uy][ux]; a[ 3] = Ys1[ 3][uy][ux];
	a[ 4] = Ys1[ 4][uy][ux]; a[ 5] = Ys1[ 5][uy][ux]; a[ 6] = Ys1[ 6][uy][ux]; a[ 7] = Ys1[ 7][uy][ux];
	a[ 8] = Ys1[ 8][uy][ux]; a[ 9] = Ys1[ 9][uy][ux]; a[10] = Ys1[10][uy][ux]; a[11] = Ys1[11][uy][ux];
	a[12] = Ys1[12][uy][ux]; a[13] = Ys1[13][uy][ux]; a[14] = Ys1[14][uy][ux]; a[15] = Ys1[15][uy][ux];
	winograd_f8x9_Y(y3, a);

	//write to Y[N, OH, OW, OC]
	*(float4*)(Y + Y00         ) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + Y00 + OC    ) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = float4{ y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = float4{ y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = float4{ y0[5], y1[5], y2[5], y3[5] };
	*(float4*)(Y + Y00 + OC * 6) = float4{ y0[6], y1[6], y2[6], y3[6] };
	*(float4*)(Y + Y00 + OC * 7) = float4{ y0[7], y1[7], y2[7], y3[7] };
}

#endif



