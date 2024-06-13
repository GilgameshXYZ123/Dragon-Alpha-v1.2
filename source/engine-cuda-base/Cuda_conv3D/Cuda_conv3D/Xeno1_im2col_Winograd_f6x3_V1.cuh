


//4*4 = 16, 4*9=36, 36/16 = 2.25
//8*3 = 24, 6*9=54, 54/24 = 2.25


//================[standard: 64 (OC) x 128 (GM)]============================
#ifndef WG_2X6_K1
#define WG_2X6_K1

//GM = N*OH*OW

//32 * 32 per thread
//(2 * 2) * 6 elements per thread
//OW -> OW6 padding tp 6x

#define wg2x6_k1(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel1<FH>\
		<<< dim3(OC>>5, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

template<int FH>
__global__ void Winograd2x6_kernel1(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//2 * 8 * 8 * 32
	__shared__ float Gs[2][8][8 + 1][32 + 4];//[buf][ik: STEP][elem: 8][g(oc): 32]
	__shared__ float Ds[2][8][8 + 1][32 + 4];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 5) + oc_index;//32 out_channel
	//G_group_id = (tx << 1) + (ty > 7) [0-31]
	const int toc0 = boc0 + (tx << 1) + (ty > 7);
	CW += (ty & 7)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y << 5) * 6 + j_index;
	//X_group_id = (ty << 1) + (tx > 7) [0-31]
	const int tj0 = bj0 + ((ty << 1) + (tx > 7)) * 6;
	const int OW6 = (OW + 5) / 6 * 6;
	const int OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx >> 5;//8(elem) 
	const int ridx = idx & 31;// { 2(k) * 4(oc) * 4(j) }
	const int uk = (ridx & 1) << 2;//-> tx & 1
	const int uy = ridx >> 1;
	const int GIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int DIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//======[compute area1: local]======================================================
	float w[3], x[8];
	float g[8], d[8];

	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i = (ty << 1) + (tx > 7);//[8, 32]
	const int Gs_k = (ty & 7), Gs_i = (tx << 1) + (ty > 7);//[8, 32]

#pragma once 
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC;
		int tX2 = tX0 + IC * 2;
		int tX3 = tX0 + IC * 3;
		int tX4 = tX0 + IC * 4;
		int tX5 = tX0 + IC * 5;
		int tX6 = tX0 + IC * 6;
		int tX7 = tX0 + IC * 7;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
		tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
		x[0] = tex1Dfetch<float>(X, tX0);
		x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2);
		x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4);
		x[5] = tex1Dfetch<float>(X, tX5);
		x[6] = tex1Dfetch<float>(X, tX6);
		x[7] = tex1Dfetch<float>(X, tX7);
		
		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + IC_OC;
		const int W2 = W0 + (IC_OC << 1);
		w[0] = CW[W0];//fw = 0
		w[1] = CW[W1];//fw = 1
		w[2] = CW[W2];//fw = 2

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_d(x, d);
		winograd_f6x3_g(w, g);

		//write to shread memory
		Ds[buf][Ds_k][0][Ds_i] = d[0];
		Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2];
		Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4];
		Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6];
		Ds[buf][Ds_k][7][Ds_i] = d[7];

		Gs[buf][Gs_k][0][Gs_i] = g[0];
		Gs[buf][Gs_k][1][Gs_i] = g[1];
		Gs[buf][Gs_k][2][Gs_i] = g[2];
		Gs[buf][Gs_k][3][Gs_i] = g[3];
		Gs[buf][Gs_k][4][Gs_i] = g[4];
		Gs[buf][Gs_k][5][Gs_i] = g[5];
		Gs[buf][Gs_k][6][Gs_i] = g[6];
		Gs[buf][Gs_k][7][Gs_i] = g[7];
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 4; ik++)
			{
				float4 a0 = *(float4*)(&Gs[buf][ik + uk][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik + uk][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik + uk][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik + uk][ux][DIdx + 4]);

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

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC;
			int tX2 = tX0 + IC * 2;
			int tX3 = tX0 + IC * 3;
			int tX4 = tX0 + IC * 4;
			int tX5 = tX0 + IC * 5;
			int tX6 = tX0 + IC * 6;
			int tX7 = tX0 + IC * 7;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
			tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
			x[0] = tex1Dfetch<float>(X, tX0);
			x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2);
			x[3] = tex1Dfetch<float>(X, tX3);
			x[4] = tex1Dfetch<float>(X, tX4);
			x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6);
			x[7] = tex1Dfetch<float>(X, tX7);

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC;
			const int W2 = W0 + (IC_OC << 1);
			w[0] = CW[W0];//fw = 0
			w[1] = CW[W1];//fw = 1
			w[2] = CW[W2];//fw = 2

			//Winograd transform: W(3) -> G(8); X(4) -> D(4)
			winograd_f6x3_d(x, d);
			winograd_f6x3_g(w, g);

			//write to shread memory
			Ds[buf][Ds_k][0][Ds_i] = d[0]; 
			Ds[buf][Ds_k][1][Ds_i] = d[1];
			Ds[buf][Ds_k][2][Ds_i] = d[2];
			Ds[buf][Ds_k][3][Ds_i] = d[3];
			Ds[buf][Ds_k][4][Ds_i] = d[4];
			Ds[buf][Ds_k][5][Ds_i] = d[5];
			Ds[buf][Ds_k][6][Ds_i] = d[6];
			Ds[buf][Ds_k][7][Ds_i] = d[7]; 

			Gs[buf][Gs_k][0][Gs_i] = g[0];
			Gs[buf][Gs_k][1][Gs_i] = g[1];
			Gs[buf][Gs_k][2][Gs_i] = g[2]; 
			Gs[buf][Gs_k][3][Gs_i] = g[3];
			Gs[buf][Gs_k][4][Gs_i] = g[4];
			Gs[buf][Gs_k][5][Gs_i] = g[5];
			Gs[buf][Gs_k][6][Gs_i] = g[6]; 
			Gs[buf][Gs_k][7][Gs_i] = g[7];
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 4; ik++)
		{
			float4 a0 = *(float4*)(&Gs[buf][ik + uk][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik + uk][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik + uk][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik + uk][ux][DIdx + 4]);

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
	__syncthreads();

	//======[compute area12: block]======================================================
	//2*8*8*32 = 16[uy] * 8[ux] * 32[elem]
	//16[uy] * 16[ux * uk] * 16[elem]

	float *Ys0 = &Gs[0][0][0][0];//16 * 16 * 16[elem]
	float *Ys1 = &Ds[0][0][0][0];//16 * 16 * 16[elem]
	float a[16];
	float y0[6], y1[6], y2[6], y3[6];

	int elem = uk * 2;

	//write-turn[0, 1]: 16[uy] * 16[ux * uk] * 16[elem]
	{
		*(float4*)(&get3d(Ys0, ux + elem, uy,  0, 16, 16)) = {  v0.x,  v1.x,  v2.x,  v3.x };//oc0, oc4
		*(float4*)(&get3d(Ys0, ux + elem, uy,  4, 16, 16)) = {  v4.x,  v5.x,  v6.x,  v7.x };
		*(float4*)(&get3d(Ys0, ux + elem, uy,  8, 16, 16)) = {  v8.x,  v9.x, v10.x, v11.x };
		*(float4*)(&get3d(Ys0, ux + elem, uy, 12, 16, 16)) = { v10.x, v12.x, v13.x, v15.x };

		*(float4*)(&get3d(Ys1, ux + elem, uy,  0, 16, 16)) = {  v0.y,  v1.y,  v2.y,  v3.y };//oc0, oc4
		*(float4*)(&get3d(Ys1, ux + elem, uy,  4, 16, 16)) = {  v4.y,  v5.y,  v6.y,  v7.y };
		*(float4*)(&get3d(Ys1, ux + elem, uy,  8, 16, 16)) = {  v8.y,  v9.y, v10.y, v11.y };
		*(float4*)(&get3d(Ys1, ux + elem, uy, 12, 16, 16)) = { v10.y, v12.y, v13.y, v15.y };
	}
	__syncthreads();

	//read-turn[0]: 16[uy] * 16[elem] * 16[ux]
	{
		a[ 0] = get3d(Ys0,  0, uy, ux, 16, 16);
		a[ 1] = get3d(Ys0,  1, uy, ux, 16, 16);
		a[ 2] = get3d(Ys0,  2, uy, ux, 16, 16);
		a[ 3] = get3d(Ys0,  3, uy, ux, 16, 16);
		a[ 4] = get3d(Ys0,  4, uy, ux, 16, 16);
		a[ 5] = get3d(Ys0,  5, uy, ux, 16, 16);
		a[ 6] = get3d(Ys0,  6, uy, ux, 16, 16);
		a[ 7] = get3d(Ys0,  7, uy, ux, 16, 16);
		a[ 8] = get3d(Ys0,  8, uy, ux, 16, 16);
		a[ 9] = get3d(Ys0,  9, uy, ux, 16, 16);
		a[10] = get3d(Ys0, 10, uy, ux, 16, 16);
		a[11] = get3d(Ys0, 11, uy, ux, 16, 16);
		a[12] = get3d(Ys0, 12, uy, ux, 16, 16);
		a[13] = get3d(Ys0, 13, uy, ux, 16, 16);
		a[14] = get3d(Ys0, 14, uy, ux, 16, 16);
		a[15] = get3d(Ys0, 15, uy, ux, 16, 16);

		a[0] += a[ 8]; a[1] += a[ 9]; a[2] += a[10]; a[3] += a[11];
		a[4] += a[12]; a[5] += a[13]; a[6] += a[14]; a[7] += a[15];
		winograd_f6x3_y(y0, a);
	}

	//read-turn[1]: 16[uy] * 16[elem] * 16[ux]
	{
		a[ 0] = get3d(Ys1,  0, uy, ux, 16, 16);
		a[ 1] = get3d(Ys1,  1, uy, ux, 16, 16);
		a[ 2] = get3d(Ys1,  2, uy, ux, 16, 16);
		a[ 3] = get3d(Ys1,  3, uy, ux, 16, 16);
		a[ 4] = get3d(Ys1,  4, uy, ux, 16, 16);
		a[ 5] = get3d(Ys1,  5, uy, ux, 16, 16);
		a[ 6] = get3d(Ys1,  6, uy, ux, 16, 16);
		a[ 7] = get3d(Ys1,  7, uy, ux, 16, 16);
		a[ 8] = get3d(Ys1,  8, uy, ux, 16, 16);
		a[ 9] = get3d(Ys1,  9, uy, ux, 16, 16);
		a[10] = get3d(Ys1, 10, uy, ux, 16, 16);
		a[11] = get3d(Ys1, 11, uy, ux, 16, 16);
		a[12] = get3d(Ys1, 12, uy, ux, 16, 16);
		a[13] = get3d(Ys1, 13, uy, ux, 16, 16);
		a[14] = get3d(Ys1, 14, uy, ux, 16, 16);
		a[15] = get3d(Ys1, 15, uy, ux, 16, 16);

		a[0] += a[ 8]; a[1] += a[ 9]; a[2] += a[10]; a[3] += a[11];
		a[4] += a[12]; a[5] += a[13]; a[6] += a[14]; a[7] += a[15];
		winograd_f6x3_y(y1, a);
	}

	//write-turn[0, 1]: 16[uy] * 16[ux * uk] * 16[elem]
	{
		*(float4*)(&get3d(Ys0, ux + elem, uy,  0, 16, 16)) = v8;//oc0 - oc3
		*(float4*)(&get3d(Ys0, ux + elem, uy,  4, 16, 16)) = v10;
		*(float4*)(&get3d(Ys0, ux + elem, uy,  8, 16, 16)) = v12;
		*(float4*)(&get3d(Ys0, ux + elem, uy, 12, 16, 16)) = v14;

		*(float4*)(&get3d(Ys1, ux + elem, uy,  0, 16, 16)) = v9;//oc4 - oc7
		*(float4*)(&get3d(Ys1, ux + elem, uy,  4, 16, 16)) = v11;
		*(float4*)(&get3d(Ys1, ux + elem, uy,  8, 16, 16)) = v13;
		*(float4*)(&get3d(Ys1, ux + elem, uy, 12, 16, 16)) = v15;
	}
	__syncthreads();

	//read-turn[0]: 16[uy] * 16[elem] * 16[ux]
	{
		a[ 0] = get3d(Ys0,  0, uy, ux, 16, 16);
		a[ 1] = get3d(Ys0,  1, uy, ux, 16, 16);
		a[ 2] = get3d(Ys0,  2, uy, ux, 16, 16);
		a[ 3] = get3d(Ys0,  3, uy, ux, 16, 16);
		a[ 4] = get3d(Ys0,  4, uy, ux, 16, 16);
		a[ 5] = get3d(Ys0,  5, uy, ux, 16, 16);
		a[ 6] = get3d(Ys0,  6, uy, ux, 16, 16);
		a[ 7] = get3d(Ys0,  7, uy, ux, 16, 16);
		a[ 8] = get3d(Ys0,  8, uy, ux, 16, 16);
		a[ 9] = get3d(Ys0,  9, uy, ux, 16, 16);
		a[10] = get3d(Ys0, 10, uy, ux, 16, 16);
		a[11] = get3d(Ys0, 11, uy, ux, 16, 16);
		a[12] = get3d(Ys0, 12, uy, ux, 16, 16);
		a[13] = get3d(Ys0, 13, uy, ux, 16, 16);
		a[14] = get3d(Ys0, 14, uy, ux, 16, 16);
		a[15] = get3d(Ys0, 15, uy, ux, 16, 16);

		a[0] += a[ 8]; a[1] += a[ 9]; a[2] += a[10]; a[3] += a[11];
		a[4] += a[12]; a[5] += a[13]; a[6] += a[14]; a[7] += a[15];
		winograd_f6x3_y(y2, a);
	}

	//read-turn[1]: 16[uy] * 16[elem] * 16[ux]
	{
		a[ 0] = get3d(Ys1,  0, uy, ux, 16, 16);
		a[ 1] = get3d(Ys1,  1, uy, ux, 16, 16);
		a[ 2] = get3d(Ys1,  2, uy, ux, 16, 16);
		a[ 3] = get3d(Ys1,  3, uy, ux, 16, 16);
		a[ 4] = get3d(Ys1,  4, uy, ux, 16, 16);
		a[ 5] = get3d(Ys1,  5, uy, ux, 16, 16);
		a[ 6] = get3d(Ys1,  6, uy, ux, 16, 16);
		a[ 7] = get3d(Ys1,  7, uy, ux, 16, 16);
		a[ 8] = get3d(Ys1,  8, uy, ux, 16, 16);
		a[ 9] = get3d(Ys1,  9, uy, ux, 16, 16);
		a[10] = get3d(Ys1, 10, uy, ux, 16, 16);
		a[11] = get3d(Ys1, 11, uy, ux, 16, 16);
		a[12] = get3d(Ys1, 12, uy, ux, 16, 16);
		a[13] = get3d(Ys1, 13, uy, ux, 16, 16);
		a[14] = get3d(Ys1, 14, uy, ux, 16, 16);
		a[15] = get3d(Ys1, 15, uy, ux, 16, 16);

		a[0] += a[ 8]; a[1] += a[ 9]; a[2] += a[10]; a[3] += a[11];
		a[4] += a[12]; a[5] += a[13]; a[6] += a[14]; a[7] += a[15];
		winograd_f6x3_y(y3, a);
	}

	//prepare for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx;
	const int yj0  = bj0 + DIdx * 6;//6 elements per group
	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3

	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0]};
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
}

#endif



#ifndef WG_2X6_K2
#define WG_2X6_K2

//GM = N*OH*OW

//32 * 32 per thread
//(2 * 2) * 6 elements per thread
//OW -> OW6 padding tp 6x

//64 * 32

#define wg2x6_k2(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel2<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

template<int FH>
__global__ void Winograd2x6_kernel2(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//2 * 8 * 8 * 32
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 32]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;//32 out_channel
	const int toc0 = boc0 + ((tx << 1) + (ty > 7)) * 2;//G_group_id = (tx << 1) + (ty > 7) [0-31]
	CW += (ty & 7)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y << 5) * 6 + j_index;//32 group of Y
	const int tj0 = bj0  + ((ty << 1) + (tx > 7)) * 6;//X_group_id = (ty << 1) + (tx > 7) [0-31]
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	//const int OH_OW = OH * OW;
	//get_n_oh_ow(tj0, tn0, toh0, tow0);

	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx >> 5, uy = idx & 31;//8 * 32
	const int DIdx = (uy / 8) << 3;
	const int GIdx = (uy & 7) << 3;

	//const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//4
	//const int GIdx = ((uy & 15) >> 1) << 3;//8

	//======[compute area1: local]======================================================
	float  w[3], x[8];
	float g0[8], g1[8], d[8];

	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i =  (ty << 1) + (tx > 7);      //[8, 32]
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]

#pragma once 
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC;
		int tX2 = tX0 + IC * 2;
		int tX3 = tX0 + IC * 3;
		int tX4 = tX0 + IC * 4;
		int tX5 = tX0 + IC * 5;
		int tX6 = tX0 + IC * 6;
		int tX7 = tX0 + IC * 7;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
		tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
		x[0] = tex1Dfetch<float>(X, tX0);
		x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2);
		x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4);
		x[5] = tex1Dfetch<float>(X, tX5);
		x[6] = tex1Dfetch<float>(X, tX6);
		x[7] = tex1Dfetch<float>(X, tX7);
		
		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + IC_OC;
		const int W2 = W0 + (IC_OC << 1);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_d(x, d);
		w[0] = w0.x; w[1] = w1.x; w[2] = w2.x; winograd_f6x3_g(w, g0);
		w[0] = w0.y; w[1] = w1.y; w[2] = w2.y; winograd_f6x3_g(w, g1);

		//write to shread memory
		Ds[buf][Ds_k][0][Ds_i] = d[0];
		Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2];
		Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4];
		Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6];
		Ds[buf][Ds_k][7][Ds_i] = d[7];

		Gs[buf][Gs_k][0][Gs_i] = g0[0]; Gs[buf][Gs_k][0][Gs_i + 1] = g1[0];
		Gs[buf][Gs_k][1][Gs_i] = g0[1]; Gs[buf][Gs_k][1][Gs_i + 1] = g1[1];
		Gs[buf][Gs_k][2][Gs_i] = g0[2]; Gs[buf][Gs_k][2][Gs_i + 1] = g1[2];
		Gs[buf][Gs_k][3][Gs_i] = g0[3]; Gs[buf][Gs_k][3][Gs_i + 1] = g1[3];
		Gs[buf][Gs_k][4][Gs_i] = g0[4]; Gs[buf][Gs_k][4][Gs_i + 1] = g1[4];
		Gs[buf][Gs_k][5][Gs_i] = g0[5]; Gs[buf][Gs_k][5][Gs_i + 1] = g1[5];
		Gs[buf][Gs_k][6][Gs_i] = g0[6]; Gs[buf][Gs_k][6][Gs_i + 1] = g1[6];
		Gs[buf][Gs_k][7][Gs_i] = g0[7]; Gs[buf][Gs_k][7][Gs_i + 1] = g1[7];
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

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC;
			int tX2 = tX0 + IC * 2;
			int tX3 = tX0 + IC * 3;
			int tX4 = tX0 + IC * 4;
			int tX5 = tX0 + IC * 5;
			int tX6 = tX0 + IC * 6;
			int tX7 = tX0 + IC * 7;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
			tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
			x[0] = tex1Dfetch<float>(X, tX0);
			x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2);
			x[3] = tex1Dfetch<float>(X, tX3);
			x[4] = tex1Dfetch<float>(X, tX4);
			x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6);
			x[7] = tex1Dfetch<float>(X, tX7);

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC;
			const int W2 = W0 + (IC_OC << 1);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);

			//Winograd transform: W(3) -> G(8); X(4) -> D(4)
			winograd_f6x3_d(x, d);
			w[0] = w0.x; w[1] = w1.x; w[2] = w2.x; winograd_f6x3_g(w, g0);
			w[0] = w0.y; w[1] = w1.y; w[2] = w2.y; winograd_f6x3_g(w, g1);

			//write to shread memory
			Ds[buf][Ds_k][0][Ds_i] = d[0]; 
			Ds[buf][Ds_k][1][Ds_i] = d[1];
			Ds[buf][Ds_k][2][Ds_i] = d[2];
			Ds[buf][Ds_k][3][Ds_i] = d[3];
			Ds[buf][Ds_k][4][Ds_i] = d[4];
			Ds[buf][Ds_k][5][Ds_i] = d[5];
			Ds[buf][Ds_k][6][Ds_i] = d[6];
			Ds[buf][Ds_k][7][Ds_i] = d[7]; 

			Gs[buf][Gs_k][0][Gs_i] = g0[0]; Gs[buf][Gs_k][0][Gs_i + 1] = g1[0];
			Gs[buf][Gs_k][1][Gs_i] = g0[1]; Gs[buf][Gs_k][1][Gs_i + 1] = g1[1];
			Gs[buf][Gs_k][2][Gs_i] = g0[2]; Gs[buf][Gs_k][2][Gs_i + 1] = g1[2];
			Gs[buf][Gs_k][3][Gs_i] = g0[3]; Gs[buf][Gs_k][3][Gs_i + 1] = g1[3];
			Gs[buf][Gs_k][4][Gs_i] = g0[4]; Gs[buf][Gs_k][4][Gs_i + 1] = g1[4];
			Gs[buf][Gs_k][5][Gs_i] = g0[5]; Gs[buf][Gs_k][5][Gs_i + 1] = g1[5];
			Gs[buf][Gs_k][6][Gs_i] = g0[6]; Gs[buf][Gs_k][6][Gs_i + 1] = g1[6];
			Gs[buf][Gs_k][7][Gs_i] = g0[7]; Gs[buf][Gs_k][7][Gs_i + 1] = g1[7];
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
	__syncthreads();

	//======[compute area12: block]======================================================
	//2*8*8*32 = 16[uy] * 8[ux] * 32[elem]
	//16[uy] * 16[ux * uk] * 16[elem]

	float *Ys0 = &Gs[0][0][0][0];//16 * 16 * 16[elem]
	float *Ys1 = &Ds[0][0][0][0];//16 * 16 * 16[elem]
	float a[16];
	float y0[6], y1[6], y2[6], y3[6];

	//write-turn[0, 1]: 16[uy] * 32[ux] * 16[elem]
	{
		*(float4*)(&get3d(Ys0, ux, uy,  0, 16, 32)) = {  v0.x,  v1.x,  v2.x,  v3.x };//oc0, oc4
		*(float4*)(&get3d(Ys0, ux, uy,  4, 16, 32)) = {  v4.x,  v5.x,  v6.x,  v7.x };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 16, 32)) = {  v8.x,  v9.x, v10.x, v11.x };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 16, 32)) = { v10.x, v12.x, v13.x, v15.x };
		__syncthreads();

		a[ 0] = get3d(Ys0,  0, uy, ux, 16, 32);
		a[ 1] = get3d(Ys0,  1, uy, ux, 16, 32);
		a[ 2] = get3d(Ys0,  2, uy, ux, 16, 32);
		a[ 3] = get3d(Ys0,  3, uy, ux, 16, 32);
		a[ 4] = get3d(Ys0,  4, uy, ux, 16, 32);
		a[ 5] = get3d(Ys0,  5, uy, ux, 16, 32);
		a[ 6] = get3d(Ys0,  6, uy, ux, 16, 32);
		a[ 7] = get3d(Ys0,  7, uy, ux, 16, 32);
		a[ 8] = get3d(Ys0,  8, uy, ux, 16, 32);
		__syncthreads();
		winograd_f6x3_y(y0, a);
	}

	//write-turn[0, 1]: 16[uy] * 16[ux * uk] * 16[elem]
	{
		*(float4*)(&get3d(Ys0, ux, uy,  0, 16, 32)) = {  v0.y,  v1.y,  v2.y,  v3.y };//oc0, oc4
		*(float4*)(&get3d(Ys0, ux, uy,  4, 16, 32)) = {  v4.y,  v5.y,  v6.y,  v7.y };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 16, 32)) = {  v8.y,  v9.y, v10.y, v11.y };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 16, 32)) = { v10.y, v12.y, v13.y, v15.y };
		__syncthreads();

		a[ 0] = get3d(Ys0,  0, uy, ux, 16, 32);
		a[ 1] = get3d(Ys0,  1, uy, ux, 16, 32);
		a[ 2] = get3d(Ys0,  2, uy, ux, 16, 32);
		a[ 3] = get3d(Ys0,  3, uy, ux, 16, 32);
		a[ 4] = get3d(Ys0,  4, uy, ux, 16, 32);
		a[ 5] = get3d(Ys0,  5, uy, ux, 16, 32);
		a[ 6] = get3d(Ys0,  6, uy, ux, 16, 32);
		a[ 7] = get3d(Ys0,  7, uy, ux, 16, 32);
		a[ 8] = get3d(Ys0,  8, uy, ux, 16, 32);
		__syncthreads();
		winograd_f6x3_y(y1, a);
	}
	
	//write-turn[0, 1]: 16[uy] * 16[ux * uk] * 16[elem]
	{
		*(float4*)(&get3d(Ys0, ux, uy,  0, 16, 32)) = {  v0.z,  v1.z,  v2.z,  v3.z };//oc0, oc4
		*(float4*)(&get3d(Ys0, ux, uy,  4, 16, 32)) = {  v4.z,  v5.z,  v6.z,  v7.z };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 16, 32)) = {  v8.z,  v9.z, v10.z, v11.z };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 16, 32)) = { v10.z, v12.z, v13.z, v15.z };
		__syncthreads();

		a[ 0] = get3d(Ys0,  0, uy, ux, 16, 32);
		a[ 1] = get3d(Ys0,  1, uy, ux, 16, 32);
		a[ 2] = get3d(Ys0,  2, uy, ux, 16, 32);
		a[ 3] = get3d(Ys0,  3, uy, ux, 16, 32);
		a[ 4] = get3d(Ys0,  4, uy, ux, 16, 32);
		a[ 5] = get3d(Ys0,  5, uy, ux, 16, 32);
		a[ 6] = get3d(Ys0,  6, uy, ux, 16, 32);
		a[ 7] = get3d(Ys0,  7, uy, ux, 16, 32);
		a[ 8] = get3d(Ys0,  8, uy, ux, 16, 32);
		__syncthreads();
		winograd_f6x3_y(y2, a);
	}

	//write-turn[0, 1]: 16[uy] * 16[ux * uk] * 16[elem]
	{
		*(float4*)(&get3d(Ys0, ux, uy,  0, 16, 32)) = {  v0.w,  v1.w,  v2.w,  v3.w };//oc0, oc4
		*(float4*)(&get3d(Ys0, ux, uy,  4, 16, 32)) = {  v4.w,  v5.w,  v6.w,  v7.w };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 16, 32)) = {  v8.w,  v9.w, v10.w, v11.w };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 16, 32)) = { v10.w, v12.w, v13.w, v15.w };
		__syncthreads();

		a[ 0] = get3d(Ys0,  0, uy, ux, 16, 32);
		a[ 1] = get3d(Ys0,  1, uy, ux, 16, 32);
		a[ 2] = get3d(Ys0,  2, uy, ux, 16, 32);
		a[ 3] = get3d(Ys0,  3, uy, ux, 16, 32);
		a[ 4] = get3d(Ys0,  4, uy, ux, 16, 32);
		a[ 5] = get3d(Ys0,  5, uy, ux, 16, 32);
		a[ 6] = get3d(Ys0,  6, uy, ux, 16, 32);
		a[ 7] = get3d(Ys0,  7, uy, ux, 16, 32);
		a[ 8] = get3d(Ys0,  8, uy, ux, 16, 32);
		winograd_f6x3_y(y3, a);
	}

	//const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//4
	//prepare for Y[N, OH, OW, OC]

	//const int DIdx = (uy / 8) << 3;
	//const int GIdx = (uy & 7) << 3;

	const int didx = DIdx >> 3;
	const int gidx = GIdx >> 3;
	const int yoc0 = boc0 + (GIdx) + ((ux & 1) << 2);
	const int yj0  = bj0 + (DIdx + (ux >> 1)) * 6;

	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3
	
	*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0]};
	*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
	*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
	*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
}

#endif


#ifndef WG_2X6_K3
#define WG_2X6_K3

//GM = N*OH*OW

//32 * 32 per thread
//(2 * 2) * 6 elements per thread
//OW -> OW6 padding tp 6x

//64 * 32
//(N*OH*((OW + 5)/6))>>5
//GM / 192

#define wg2x6_k3(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel3<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

template<int FH>
__global__ void Winograd2x6_kernel3(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//2 * 8 * 8 * 32
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 32]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;//oc0 - oc4
	const int toc0 = boc0 + ((tx << 1) + (ty > 7)) * 2;//
	CW += (ty & 7)*OC + toc0;//CW[0, 0, (ty & 7), toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int tj0 = bj0  + ((ty << 1) + (tx > 7)) * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	//const int OH_OW = OH * OW;
	//get_n_oh_ow(tj0, tn0, toh0, tow0);

	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);//X[tn0, tih0, tiw0, tx&7]

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx / 32,  uy = idx % 32;//8 * 32
	
	//const int DIdx = (uy / 8) << 3;//[0 - 4]
	//const int GIdx = (uy % 8) << 3;//[0 - 8]

	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//4
	const int GIdx = ((uy & 15) >> 1) << 3;//8

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i =  (ty << 1) + (tx > 7);      //[8, 32]
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	float w[3], g0[8], g1[8];
	float x[8], d[8];

#pragma once 
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC;
		int tX2 = tX0 + IC * 2;
		int tX3 = tX0 + IC * 3;
		int tX4 = tX0 + IC * 4;
		int tX5 = tX0 + IC * 5;
		int tX6 = tX0 + IC * 6;
		int tX7 = tX0 + IC * 7;
		tX0 = IF_int(ly0, tX0, -1);
		tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1);
		tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(ly4, tX4, -1); 
		tX5 = IF_int(ly5, tX5, -1);
		tX6 = IF_int(ly6, tX6, -1);
		tX7 = IF_int(ly7, tX7, -1);
		x[0] = tex1Dfetch<float>(X, tX0);
		x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2);
		x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4);
		x[5] = tex1Dfetch<float>(X, tX5);
		x[6] = tex1Dfetch<float>(X, tX6);
		x[7] = tex1Dfetch<float>(X, tX7);
		
		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + IC_OC;
		const int W2 = W0 + (IC_OC << 1);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_d(x, d);
		w[0] = w0.x; w[1] = w1.x; w[2] = w2.x; winograd_f6x3_g(w, g0);
		w[0] = w0.y; w[1] = w1.y; w[2] = w2.y; winograd_f6x3_g(w, g1);

		//write to shread memory
		Ds[buf][Ds_k][0][Ds_i] = d[0];
		Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2];
		Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4];
		Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6];
		Ds[buf][Ds_k][7][Ds_i] = d[7];

		//const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
		Gs[buf][Gs_k][0][Gs_i] = g0[0]; Gs[buf][Gs_k][0][Gs_i + 1] = g1[0];
		Gs[buf][Gs_k][1][Gs_i] = g0[1]; Gs[buf][Gs_k][1][Gs_i + 1] = g1[1];
		Gs[buf][Gs_k][2][Gs_i] = g0[2]; Gs[buf][Gs_k][2][Gs_i + 1] = g1[2];
		Gs[buf][Gs_k][3][Gs_i] = g0[3]; Gs[buf][Gs_k][3][Gs_i + 1] = g1[3];
		Gs[buf][Gs_k][4][Gs_i] = g0[4]; Gs[buf][Gs_k][4][Gs_i + 1] = g1[4];
		Gs[buf][Gs_k][5][Gs_i] = g0[5]; Gs[buf][Gs_k][5][Gs_i + 1] = g1[5];
		Gs[buf][Gs_k][6][Gs_i] = g0[6]; Gs[buf][Gs_k][6][Gs_i + 1] = g1[6];
		Gs[buf][Gs_k][7][Gs_i] = g0[7]; Gs[buf][Gs_k][7][Gs_i + 1] = g1[7];
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

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC;
			int tX2 = tX0 + IC * 2;
			int tX3 = tX0 + IC * 3;
			int tX4 = tX0 + IC * 4;
			int tX5 = tX0 + IC * 5;
			int tX6 = tX0 + IC * 6;
			int tX7 = tX0 + IC * 7;
			tX0 = IF_int(ly0, tX0, -1); 
			tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); 
			tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(ly4, tX4, -1);
			tX5 = IF_int(ly5, tX5, -1);
			tX6 = IF_int(ly6, tX6, -1);
			tX7 = IF_int(ly7, tX7, -1);
			x[0] = tex1Dfetch<float>(X, tX0);
			x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2);
			x[3] = tex1Dfetch<float>(X, tX3);
			x[4] = tex1Dfetch<float>(X, tX4);
			x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6);
			x[7] = tex1Dfetch<float>(X, tX7);

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC;
			const int W2 = W0 + (IC_OC << 1);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);

			//Winograd transform: W(3) -> G(8); X(4) -> D(4)
			winograd_f6x3_d(x, d);
			w[0] = w0.x; w[1] = w1.x; w[2] = w2.x; winograd_f6x3_g(w, g0);
			w[0] = w0.y; w[1] = w1.y; w[2] = w2.y; winograd_f6x3_g(w, g1);

			//write to shread memory
			Ds[buf][Ds_k][0][Ds_i] = d[0]; 
			Ds[buf][Ds_k][1][Ds_i] = d[1];
			Ds[buf][Ds_k][2][Ds_i] = d[2];
			Ds[buf][Ds_k][3][Ds_i] = d[3];
			Ds[buf][Ds_k][4][Ds_i] = d[4];
			Ds[buf][Ds_k][5][Ds_i] = d[5];
			Ds[buf][Ds_k][6][Ds_i] = d[6];
			Ds[buf][Ds_k][7][Ds_i] = d[7]; 

			Gs[buf][Gs_k][0][Gs_i] = g0[0]; Gs[buf][Gs_k][0][Gs_i + 1] = g1[0];
			Gs[buf][Gs_k][1][Gs_i] = g0[1]; Gs[buf][Gs_k][1][Gs_i + 1] = g1[1];
			Gs[buf][Gs_k][2][Gs_i] = g0[2]; Gs[buf][Gs_k][2][Gs_i + 1] = g1[2];
			Gs[buf][Gs_k][3][Gs_i] = g0[3]; Gs[buf][Gs_k][3][Gs_i + 1] = g1[3];
			Gs[buf][Gs_k][4][Gs_i] = g0[4]; Gs[buf][Gs_k][4][Gs_i + 1] = g1[4];
			Gs[buf][Gs_k][5][Gs_i] = g0[5]; Gs[buf][Gs_k][5][Gs_i + 1] = g1[5];
			Gs[buf][Gs_k][6][Gs_i] = g0[6]; Gs[buf][Gs_k][6][Gs_i + 1] = g1[6];
			Gs[buf][Gs_k][7][Gs_i] = g0[7]; Gs[buf][Gs_k][7][Gs_i + 1] = g1[7];
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
	__syncthreads();

	//======[compute area12: block]======================================================
	//2*8*8*32 = 16[uy] * 8[ux] * 32[elem]
	//16[uy] * 16[ux * uk] * 16[elem]

	float *Ys0 = &Gs[0][0][0][0];//16 * 16 * 16[elem]
	float *Ys1 = &Ds[0][0][0][0];//16 * 16 * 16[elem]
	float a[8];
	float y0[6], y1[6], y2[6], y3[6];

	//const int DIdx = (uy / 8) << 3;//[0 - 4]
	//const int GIdx = (uy % 8) << 3;//[0 - 8]

	//ux [0 - 7]
	//const int ux = idx / 32, uy = idx % 32;//8 * 32

	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);//0 - 64
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int yj1 = bj0 + (DIdx + (ux >> 1) + 4) * 6;

	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3
	const int Y10 = yj1 * OC + yoc0;//ux: j4 -> j7

	//2*8*8*64 = 16 * 8 * 64 -> 8[ux: accu] * 32[ux] * 32[elem]

	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 32[uy (32)] * 16[elem]
	{
		//write 32 elements to shared memory
		*(float4*)(&get3d(Ys0, ux, uy,  0, 32, 16)) = { v0.x, v0.y, v1.x, v1.y };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 32, 16)) = { v2.x, v2.y, v3.x, v3.y };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 32, 16)) = { v4.x, v4.y, v5.x, v5.y };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 32, 16)) = { v6.x, v6.y, v7.x, v7.y };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 32, 16);//
		a[1] = get3d(Ys0, 1, uy, ux * 2, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 32, 16);
		a[8] = get3d(Ys0, 8, uy, ux * 2, 32, 16);
		winograd_f6x3_y(y0, a);//{oc0 / oc4} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, ux * 2 + 1, 32, 16);
		a[1] = get3d(Ys0, 1, uy, ux * 2 + 1, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2 + 1, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2 + 1, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2 + 1, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2 + 1, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2 + 1, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2 + 1, 32, 16);
		a[8] = get3d(Ys0, 8, uy, ux * 2 + 1, 32, 16);
		winograd_f6x3_y(y1, a);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 32 elements to shared memory
		*(float4*)(&get3d(Ys0, ux, uy,  0, 32, 16)) = { v0.z, v0.w, v1.x, v1.y };//{oc2, oc3}, {oc6, oc7}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 32, 16)) = { v2.z, v2.w, v3.w, v3.w };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 32, 16)) = { v4.z, v4.w, v5.w, v5.w };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 32, 16)) = { v6.z, v6.w, v7.w, v7.w };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 32, 16);//{oc2, oc3}
		a[1] = get3d(Ys0, 1, uy, ux * 2, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 32, 16);
		a[8] = get3d(Ys0, 8, uy, ux * 2, 32, 16);
		winograd_f6x3_y(y2, a);//{oc2 / oc6} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, ux * 2 + 1, 32, 16);//{oc1, oc5}
		a[1] = get3d(Ys0, 1, uy, ux * 2 + 1, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2 + 1, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2 + 1, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2 + 1, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2 + 1, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2 + 1, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2 + 1, 32, 16);
		a[8] = get3d(Ys0, 8, uy, ux * 2 + 1, 32, 16);
		winograd_f6x3_y(y3, a);//{oc3 / oc7} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y
	{
		*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };//83
		*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };//
		*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
	
	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 32 elements to shared memory
		*(float4*)(&get3d(Ys0, ux, uy,  0, 32, 16)) = {  v8.x,  v8.y,  v9.x,  v9.y };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 32, 16)) = { v10.x, v10.y, v11.x, v11.y };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 32, 16)) = { v12.x, v12.y, v13.x, v13.y };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 32, 16)) = { v14.x, v14.y, v15.x, v15.y };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 32, 16);//
		a[1] = get3d(Ys0, 1, uy, ux * 2, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 32, 16);
		a[8] = get3d(Ys0, 8, uy, ux * 2, 32, 16);
		winograd_f6x3_y(y0, a);//{oc0 / oc4} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, ux * 2 + 1, 32, 16);
		a[1] = get3d(Ys0, 1, uy, ux * 2 + 1, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2 + 1, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2 + 1, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2 + 1, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2 + 1, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2 + 1, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2 + 1, 32, 16);
		a[8] = get3d(Ys0, 8, uy, ux * 2 + 1, 32, 16);
		winograd_f6x3_y(y1, a);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 32 elements to shared memory
		*(float4*)(&get3d(Ys0, ux, uy,  0, 32, 16)) = {  v8.z,  v8.w,  v9.z,  v9.w };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 32, 16)) = { v10.z, v10.w, v11.z, v11.w };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 32, 16)) = { v12.z, v12.w, v13.z, v13.w };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 32, 16)) = { v14.z, v14.w, v15.z, v15.w };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 32, 16);//{oc2, oc3}
		a[1] = get3d(Ys0, 1, uy, ux * 2, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 32, 16);
		a[8] = get3d(Ys0, 8, uy, ux * 2, 32, 16);
		winograd_f6x3_y(y2, a);//{oc2 / oc6} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, ux * 2 + 1, 32, 16);//{oc1, oc5}
		a[1] = get3d(Ys0, 1, uy, ux * 2 + 1, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2 + 1, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2 + 1, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2 + 1, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2 + 1, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2 + 1, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2 + 1, 32, 16);
		a[8] = get3d(Ys0, 8, uy, ux * 2 + 1, 32, 16);
		winograd_f6x3_y(y3, a);//{oc3 / oc7} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y 
	{
		*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
		*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
		*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
}

#endif


#ifndef WG_2X6_K4
#define WG_2X6_K4

//GM = N*OH*OW

//32 * 32 per thread
//(2 * 2) * 6 elements per thread
//OW -> OW6 padding tp 6x

//64 * 32
//(N*OH*((OW + 5)/6))>>5
//GM / 192

#define wg2x6_k4(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel4<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.89154 msec, Performace = 15039.2 GFlop/s
template<int FH>
__global__ void Winograd2x6_kernel4(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//2 * 8 * 8 * 32
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 32]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;//oc0 - oc4
	const int toc0 = boc0 + ((tx << 1) + (ty > 7)) * 2;//
	CW += (ty & 7)*OC + toc0;//CW[0, 0, (ty & 7), toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int tj0 = bj0  + ((ty << 1) + (tx > 7)) * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);//X[tn0, tih0, tiw0, tx&7]

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx / 32,  uy = idx % 32;//8 * 32
	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//4
	const int GIdx = ((uy & 15) >> 1) << 3;//8

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i =  (ty << 1) + (tx > 7);      //[8, 32]
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	float w[3], g0[8], g1[8];
	float x[8], d[8];

#pragma once 
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC;
		int tX2 = tX0 + IC * 2;
		int tX3 = tX0 + IC * 3;
		int tX4 = tX0 + IC * 4;
		int tX5 = tX0 + IC * 5;
		int tX6 = tX0 + IC * 6;
		int tX7 = tX0 + IC * 7;
		tX0 = IF_int(ly0, tX0, -1);
		tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1);
		tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(ly4, tX4, -1); 
		tX5 = IF_int(ly5, tX5, -1);
		tX6 = IF_int(ly6, tX6, -1);
		tX7 = IF_int(ly7, tX7, -1);
		x[0] = tex1Dfetch<float>(X, tX0);
		x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2);
		x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4);
		x[5] = tex1Dfetch<float>(X, tX5);
		x[6] = tex1Dfetch<float>(X, tX6);
		x[7] = tex1Dfetch<float>(X, tX7);
		
		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + IC_OC;
		const int W2 = W0 + (IC_OC << 1);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_d(d, x);

		w[0] = w0.x; w[1] = w1.x; w[2] = w2.x; winograd_f6x3_g(g0, w);
		w[0] = w0.y; w[1] = w1.y; w[2] = w2.y; winograd_f6x3_g(g1, w);

		//write to shread memory
		Ds[buf][Ds_k][0][Ds_i] = d[0];
		Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2];
		Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4];
		Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6];
		Ds[buf][Ds_k][7][Ds_i] = d[7];

		//const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
		Gs[buf][Gs_k][0][Gs_i] = g0[0]; Gs[buf][Gs_k][0][Gs_i + 1] = g1[0];
		Gs[buf][Gs_k][1][Gs_i] = g0[1]; Gs[buf][Gs_k][1][Gs_i + 1] = g1[1];
		Gs[buf][Gs_k][2][Gs_i] = g0[2]; Gs[buf][Gs_k][2][Gs_i + 1] = g1[2];
		Gs[buf][Gs_k][3][Gs_i] = g0[3]; Gs[buf][Gs_k][3][Gs_i + 1] = g1[3];
		Gs[buf][Gs_k][4][Gs_i] = g0[4]; Gs[buf][Gs_k][4][Gs_i + 1] = g1[4];
		Gs[buf][Gs_k][5][Gs_i] = g0[5]; Gs[buf][Gs_k][5][Gs_i + 1] = g1[5];
		Gs[buf][Gs_k][6][Gs_i] = g0[6]; Gs[buf][Gs_k][6][Gs_i + 1] = g1[6];
		Gs[buf][Gs_k][7][Gs_i] = g0[7]; Gs[buf][Gs_k][7][Gs_i + 1] = g1[7];
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

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC;
			int tX2 = tX0 + IC * 2;
			int tX3 = tX0 + IC * 3;
			int tX4 = tX0 + IC * 4;
			int tX5 = tX0 + IC * 5;
			int tX6 = tX0 + IC * 6;
			int tX7 = tX0 + IC * 7;
			tX0 = IF_int(ly0, tX0, -1); 
			tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); 
			tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(ly4, tX4, -1);
			tX5 = IF_int(ly5, tX5, -1);
			tX6 = IF_int(ly6, tX6, -1);
			tX7 = IF_int(ly7, tX7, -1);
			x[0] = tex1Dfetch<float>(X, tX0);
			x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2);
			x[3] = tex1Dfetch<float>(X, tX3);
			x[4] = tex1Dfetch<float>(X, tX4);
			x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6);
			x[7] = tex1Dfetch<float>(X, tX7);

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC;
			const int W2 = W0 + (IC_OC << 1);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);

			//Winograd transform: W(3) -> G(8); X(4) -> D(4)
			winograd_f6x3_d(d, x);
			w[0] = w0.x; w[1] = w1.x; w[2] = w2.x; winograd_f6x3_g(g0, w);
			w[0] = w0.y; w[1] = w1.y; w[2] = w2.y; winograd_f6x3_g(g1, w);

			//write to shread memory
			Ds[buf][Ds_k][0][Ds_i] = d[0]; 
			Ds[buf][Ds_k][1][Ds_i] = d[1];
			Ds[buf][Ds_k][2][Ds_i] = d[2];
			Ds[buf][Ds_k][3][Ds_i] = d[3];
			Ds[buf][Ds_k][4][Ds_i] = d[4];
			Ds[buf][Ds_k][5][Ds_i] = d[5];
			Ds[buf][Ds_k][6][Ds_i] = d[6];
			Ds[buf][Ds_k][7][Ds_i] = d[7]; 

			Gs[buf][Gs_k][0][Gs_i] = g0[0]; Gs[buf][Gs_k][0][Gs_i + 1] = g1[0];
			Gs[buf][Gs_k][1][Gs_i] = g0[1]; Gs[buf][Gs_k][1][Gs_i + 1] = g1[1];
			Gs[buf][Gs_k][2][Gs_i] = g0[2]; Gs[buf][Gs_k][2][Gs_i + 1] = g1[2];
			Gs[buf][Gs_k][3][Gs_i] = g0[3]; Gs[buf][Gs_k][3][Gs_i + 1] = g1[3];
			Gs[buf][Gs_k][4][Gs_i] = g0[4]; Gs[buf][Gs_k][4][Gs_i + 1] = g1[4];
			Gs[buf][Gs_k][5][Gs_i] = g0[5]; Gs[buf][Gs_k][5][Gs_i + 1] = g1[5];
			Gs[buf][Gs_k][6][Gs_i] = g0[6]; Gs[buf][Gs_k][6][Gs_i + 1] = g1[6];
			Gs[buf][Gs_k][7][Gs_i] = g0[7]; Gs[buf][Gs_k][7][Gs_i + 1] = g1[7];
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
	__syncthreads();

	//======[compute area12: block]======================================================
	//2*8*8*32 = 16[uy] * 8[ux] * 32[elem]
	//16[uy] * 16[ux * uk] * 16[elem]

	float *Ys0 = &Gs[0][0][0][0];//16 * 16 * 16[elem]
	float a[8], y0[6], y1[6], y2[6], y3[6];

	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);//0 - 64
	const int yj0 = bj0 + (DIdx + (ux >> 1)    ) * 6;
	const int yj1 = bj0 + (DIdx + (ux >> 1) + 4) * 6;
	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3
	const int Y10 = yj1 * OC + yoc0;//ux: j4 -> j7

	//2*8*8*64 = 16 * 8 * 64 -> 8[ux: accu] * 32[ux] * 32[elem]

	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 32[uy (32)] * 16[elem]
	{
		*(float4*)(&get3d(Ys0, ux, uy,  0, 32, 16)) = { v0.x, v0.y, v1.x, v1.y };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 32, 16)) = { v2.x, v2.y, v3.x, v3.y };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 32, 16)) = { v4.x, v4.y, v5.x, v5.y };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 32, 16)) = { v6.x, v6.y, v7.x, v7.y };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 32, 16);//
		a[1] = get3d(Ys0, 1, uy, ux * 2, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 32, 16);
		winograd_f6x3_y(y0, a);//{oc0 / oc4} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, ux * 2 + 1, 32, 16);
		a[1] = get3d(Ys0, 1, uy, ux * 2 + 1, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2 + 1, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2 + 1, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2 + 1, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2 + 1, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2 + 1, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2 + 1, 32, 16);
		winograd_f6x3_y(y1, a);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 32 elements to shared memory
		*(float4*)(&get3d(Ys0, ux, uy,  0, 32, 16)) = { v0.z, v0.w, v1.z, v1.w };//{oc2, oc3}, {oc6, oc7}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 32, 16)) = { v2.z, v2.w, v3.z, v3.w };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 32, 16)) = { v4.z, v4.w, v5.z, v5.w };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 32, 16)) = { v6.z, v6.w, v7.z, v7.w };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 32, 16);//{oc2, oc3}
		a[1] = get3d(Ys0, 1, uy, ux * 2, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 32, 16);
		winograd_f6x3_y(y2, a);//{oc2 / oc6} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, (ux * 2 + 1), 32, 16);//{oc1, oc5}
		a[1] = get3d(Ys0, 1, uy, (ux * 2 + 1), 32, 16);
		a[2] = get3d(Ys0, 2, uy, (ux * 2 + 1), 32, 16);
		a[3] = get3d(Ys0, 3, uy, (ux * 2 + 1), 32, 16);
		a[4] = get3d(Ys0, 4, uy, (ux * 2 + 1), 32, 16);
		a[5] = get3d(Ys0, 5, uy, (ux * 2 + 1), 32, 16);
		a[6] = get3d(Ys0, 6, uy, (ux * 2 + 1), 32, 16);
		a[7] = get3d(Ys0, 7, uy, (ux * 2 + 1), 32, 16);
		winograd_f6x3_y(y3, a);//{oc3 / oc7} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y
	{
		*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };//83
		*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };//
		*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
	
	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 32 elements to shared memory
		*(float4*)(&get3d(Ys0, ux, uy,  0, 32, 16)) = {  v8.x,  v8.y,  v9.x,  v9.y };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 32, 16)) = { v10.x, v10.y, v11.x, v11.y };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 32, 16)) = { v12.x, v12.y, v13.x, v13.y };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 32, 16)) = { v14.x, v14.y, v15.x, v15.y };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 32, 16);//
		a[1] = get3d(Ys0, 1, uy, ux * 2, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 32, 16);
		winograd_f6x3_y(y0, a);//{oc0 / oc4} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, ux * 2 + 1, 32, 16);
		a[1] = get3d(Ys0, 1, uy, ux * 2 + 1, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2 + 1, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2 + 1, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2 + 1, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2 + 1, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2 + 1, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2 + 1, 32, 16);
		winograd_f6x3_y(y1, a);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 32 elements to shared memory
		*(float4*)(&get3d(Ys0, ux, uy,  0, 32, 16)) = {  v8.z,  v8.w,  v9.z,  v9.w };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 32, 16)) = { v10.z, v10.w, v11.z, v11.w };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 32, 16)) = { v12.z, v12.w, v13.z, v13.w };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 32, 16)) = { v14.z, v14.w, v15.z, v15.w };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 32, 16);//{oc2, oc3}
		a[1] = get3d(Ys0, 1, uy, ux * 2, 32, 16);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 32, 16);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 32, 16);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 32, 16);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 32, 16);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 32, 16);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 32, 16);
		winograd_f6x3_y(y2, a);//{oc2 / oc6} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, (ux * 2 + 1), 32, 16);//{oc1, oc5}
		a[1] = get3d(Ys0, 1, uy, (ux * 2 + 1), 32, 16);
		a[2] = get3d(Ys0, 2, uy, (ux * 2 + 1), 32, 16);
		a[3] = get3d(Ys0, 3, uy, (ux * 2 + 1), 32, 16);
		a[4] = get3d(Ys0, 4, uy, (ux * 2 + 1), 32, 16);
		a[5] = get3d(Ys0, 5, uy, (ux * 2 + 1), 32, 16);
		a[6] = get3d(Ys0, 6, uy, (ux * 2 + 1), 32, 16);
		a[7] = get3d(Ys0, 7, uy, (ux * 2 + 1), 32, 16);
		winograd_f6x3_y(y3, a);//{oc3 / oc7} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y 
	{
		*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
		*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
		*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
}

#endif


#ifndef WG_2X6_K5
#define WG_2X6_K5

//GM = N*OH*OW

//32 * 32 per thread
//(2 * 2) * 6 elements per thread
//OW -> OW6 padding tp 6x

//64 * 32
//(N*OH*((OW + 5)/6))>>5
//GM / 192

#define wg2x6_k5(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel5<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.74787 msec, Performace = 15825.5 GFlop/s
//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
template<int FH>
__global__ void Winograd2x6_kernel5(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//2 * 8 * 8 * 32
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 32]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;//oc0 - oc4
	const int toc0 = boc0 + ((tx << 1) + (ty > 7)) * 2;//
	CW += (ty & 7)*OC + toc0;//CW[0, 0, (ty & 7), toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int tj0 = bj0  + ((ty << 1) + (tx > 7)) * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);//X[tn0, tih0, tiw0, tx&7]

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx / 32,  uy = idx % 32;//8 * 32

	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i =  (ty << 1) + (tx > 7);      //[8, 32]
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	float g0[8], g1[8];
	float x[8], d[8];

#pragma once 
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
		int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6, tX7 = tX0 + IC * 7;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
		tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
		x[0] = tex1Dfetch<float>(X, tX0);
		x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2);
		x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4);
		x[5] = tex1Dfetch<float>(X, tX5);
		x[6] = tex1Dfetch<float>(X, tX6);
		x[7] = tex1Dfetch<float>(X, tX7);
		
		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + IC_OC;
		const int W2 = W0 + (IC_OC << 1);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_D(d, x);
		winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
		winograd_f6x3_G(g1, w0.y, w1.y, w2.y);

		//write to shread memory
		Ds[buf][Ds_k][0][Ds_i] = d[0];
		Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2];
		Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4];
		Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6];
		Ds[buf][Ds_k][7][Ds_i] = d[7];

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

			//load 2 group from X[N, IH, IW, IC]
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2);
			int tX5 = tX0 + IC * 5, tX6 = tX0 + IC * 6, tX7 = tX0 + IC * 7;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
			tX6 = IF_int(ly6, tX6, -1); tX7 = IF_int(ly7, tX7, -1);
			x[0] = tex1Dfetch<float>(X, tX0);
			x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2);
			x[3] = tex1Dfetch<float>(X, tX3);
			x[4] = tex1Dfetch<float>(X, tX4);
			x[5] = tex1Dfetch<float>(X, tX5);
			x[6] = tex1Dfetch<float>(X, tX6);
			x[7] = tex1Dfetch<float>(X, tX7);

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC;
			const int W2 = W0 + (IC_OC << 1);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);

			//Winograd transform: W(3) -> G(8); X(4) -> D(4)
			winograd_f6x3_D(d, x);
			winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
			winograd_f6x3_G(g1, w0.y, w1.y, w2.y);

			//write to shread memory
			Ds[buf][Ds_k][0][Ds_i] = d[0]; 
			Ds[buf][Ds_k][1][Ds_i] = d[1];
			Ds[buf][Ds_k][2][Ds_i] = d[2];
			Ds[buf][Ds_k][3][Ds_i] = d[3];
			Ds[buf][Ds_k][4][Ds_i] = d[4];
			Ds[buf][Ds_k][5][Ds_i] = d[5];
			Ds[buf][Ds_k][6][Ds_i] = d[6];
			Ds[buf][Ds_k][7][Ds_i] = d[7]; 

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
	__syncthreads();

	//======[compute area12: block]======================================================
	//2*8*8*32 = 16[uy] * 8[ux] * 32[elem]
	//16[uy] * 16[ux * uk] * 16[elem]

	float *Ys0 = &Gs[0][0][0][0];//16 * 16 * 16[elem]
	float a[8], y0[6], y1[6], y2[6], y3[6];

	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);//0 - 64
	const int yj0 = bj0 + (DIdx + (ux >> 1)    ) * 6;
	const int yj1 = bj0 + (DIdx + (ux >> 1) + 4) * 6;
	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3
	const int Y10 = yj1 * OC + yoc0;//ux: j4 -> j7

	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 32[uy (32)] * 16[elem]
	{
		*(float4*)(&get3d(Ys0, ux, uy,  0, 33, 20)) = { v0.x, v0.y, v1.x, v1.y };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 33, 20)) = { v2.x, v2.y, v3.x, v3.y };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 33, 20)) = { v4.x, v4.y, v5.x, v5.y };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 33, 20)) = { v6.x, v6.y, v7.x, v7.y };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 33, 20);//
		a[1] = get3d(Ys0, 1, uy, ux * 2, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 33, 20);
		winograd_f6x3_Y(y0, a);//{oc0 / oc4} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, ux * 2 + 1, 33, 20);
		a[1] = get3d(Ys0, 1, uy, ux * 2 + 1, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2 + 1, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2 + 1, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2 + 1, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2 + 1, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2 + 1, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2 + 1, 33, 20);
		winograd_f6x3_Y(y1, a);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 33 elements to shared memory
		*(float4*)(&get3d(Ys0, ux, uy,  0, 33, 20)) = { v0.z, v0.w, v1.z, v1.w };//{oc2, oc3}, {oc6, oc7}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 33, 20)) = { v2.z, v2.w, v3.z, v3.w };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 33, 20)) = { v4.z, v4.w, v5.z, v5.w };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 33, 20)) = { v6.z, v6.w, v7.z, v7.w };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 33, 20);//{oc2, oc3}
		a[1] = get3d(Ys0, 1, uy, ux * 2, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 33, 20);
		winograd_f6x3_Y(y2, a);//{oc2 / oc6} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, (ux * 2 + 1), 33, 20);//{oc1, oc5}
		a[1] = get3d(Ys0, 1, uy, (ux * 2 + 1), 33, 20);
		a[2] = get3d(Ys0, 2, uy, (ux * 2 + 1), 33, 20);
		a[3] = get3d(Ys0, 3, uy, (ux * 2 + 1), 33, 20);
		a[4] = get3d(Ys0, 4, uy, (ux * 2 + 1), 33, 20);
		a[5] = get3d(Ys0, 5, uy, (ux * 2 + 1), 33, 20);
		a[6] = get3d(Ys0, 6, uy, (ux * 2 + 1), 33, 20);
		a[7] = get3d(Ys0, 7, uy, (ux * 2 + 1), 33, 20);
		winograd_f6x3_Y(y3, a);//{oc3 / oc7} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y
	{
		*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };//83
		*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };//
		*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
	
	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 33 elements to shared memory
		*(float4*)(&get3d(Ys0, ux, uy,  0, 33, 20)) = {  v8.x,  v8.y,  v9.x,  v9.y };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 33, 20)) = { v10.x, v10.y, v11.x, v11.y };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 33, 20)) = { v12.x, v12.y, v13.x, v13.y };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 33, 20)) = { v14.x, v14.y, v15.x, v15.y };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 33, 20);//
		a[1] = get3d(Ys0, 1, uy, ux * 2, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 33, 20);
		winograd_f6x3_Y(y0, a);//{oc0 / oc4} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, ux * 2 + 1, 33, 20);
		a[1] = get3d(Ys0, 1, uy, ux * 2 + 1, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2 + 1, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2 + 1, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2 + 1, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2 + 1, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2 + 1, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2 + 1, 33, 20);
		winograd_f6x3_Y(y1, a);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 32 elements to shared memory
		*(float4*)(&get3d(Ys0, ux, uy,  0, 33, 20)) = {  v8.z,  v8.w,  v9.z,  v9.w };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(&get3d(Ys0, ux, uy,  4, 33, 20)) = { v10.z, v10.w, v11.z, v11.w };
		*(float4*)(&get3d(Ys0, ux, uy,  8, 33, 20)) = { v12.z, v12.w, v13.z, v13.w };
		*(float4*)(&get3d(Ys0, ux, uy, 12, 33, 20)) = { v14.z, v14.w, v15.z, v15.w };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 33, 20);//{oc2, oc3}
		a[1] = get3d(Ys0, 1, uy, ux * 2, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 33, 20);
		winograd_f6x3_Y(y2, a);//{oc2 / oc6} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, (ux * 2 + 1), 33, 20);//{oc1, oc5}
		a[1] = get3d(Ys0, 1, uy, (ux * 2 + 1), 33, 20);
		a[2] = get3d(Ys0, 2, uy, (ux * 2 + 1), 33, 20);
		a[3] = get3d(Ys0, 3, uy, (ux * 2 + 1), 33, 20);
		a[4] = get3d(Ys0, 4, uy, (ux * 2 + 1), 33, 20);
		a[5] = get3d(Ys0, 5, uy, (ux * 2 + 1), 33, 20);
		a[6] = get3d(Ys0, 6, uy, (ux * 2 + 1), 33, 20);
		a[7] = get3d(Ys0, 7, uy, (ux * 2 + 1), 33, 20);
		winograd_f6x3_Y(y3, a);//{oc3 / oc7} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y 
	{
		*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
		*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
		*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
}

#endif


#ifndef WG_2X6_K6
#define WG_2X6_K6

#define wg2x6_k6(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel6<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)


//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
//Size = 81, Time = 10.7723 msec, Performace = 16147.6 GFlop/s
//Size = 81, Time = 10.479 msec, Performace = 16599.6 GFlop/s
template<int FH>
__global__ void Winograd2x6_kernel6(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//2 * 8 * 8 * 32
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 32]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;//oc0 - oc4
	const int toc0 = boc0 + ((tx << 1) + (ty > 7)) * 2;//
	CW += (ty & 7)*OC + toc0;//CW[0, 0, (ty & 7), toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int tj0 = bj0  + ((ty << 1) + (tx > 7)) * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);//X[tn0, tih0, tiw0, tx&7]

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx / 32,  uy = idx % 32;//8 * 32

	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);//0 - 64
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i =  (ty << 1) + (tx > 7);      //[8, 32]
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	float g0[8], g1[8];
	float x[8], d[8];

#pragma once 
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
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;
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
	float* __restrict__ Ys0 = &Gs[0][0][0][0];//16 * 16 * 16[elem]
	float a[8], y0[6], y1[6], y2[6], y3[6];
	const int Y10 = Y00 + 24 * OC;//ux: j4 -> j7
	const int Ywt = (ux * 33 + uy) * 20;
	__syncthreads();

	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		*(float4*)(Ys0 + Ywt     ) = { v0.x, v0.y, v1.x, v1.y };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(Ys0 + Ywt +  4) = { v2.x, v2.y, v3.x, v3.y };
		*(float4*)(Ys0 + Ywt +  8) = { v4.x, v4.y, v5.x, v5.y };
		*(float4*)(Ys0 + Ywt + 12) = { v6.x, v6.y, v7.x, v7.y };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 33, 20);//
		a[1] = get3d(Ys0, 1, uy, ux * 2, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 33, 20);
		winograd_f6x3_Y(y0, a);//{oc0 / oc4} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, ux * 2 + 1, 33, 20);
		a[1] = get3d(Ys0, 1, uy, ux * 2 + 1, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2 + 1, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2 + 1, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2 + 1, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2 + 1, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2 + 1, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2 + 1, 33, 20);
		winograd_f6x3_Y(y1, a);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 33 elements to shared memory
		*(float4*)(Ys0 + Ywt     ) = { v0.z, v0.w, v1.z, v1.w };//{oc2, oc3}, {oc6, oc7}}
		*(float4*)(Ys0 + Ywt +  4) = { v2.z, v2.w, v3.z, v3.w };
		*(float4*)(Ys0 + Ywt +  8) = { v4.z, v4.w, v5.z, v5.w };
		*(float4*)(Ys0 + Ywt + 12) = { v6.z, v6.w, v7.z, v7.w };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 33, 20);//{oc2, oc3}
		a[1] = get3d(Ys0, 1, uy, ux * 2, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 33, 20);
		winograd_f6x3_Y(y2, a);//{oc2 / oc6} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, (ux * 2 + 1), 33, 20);//{oc1, oc5}
		a[1] = get3d(Ys0, 1, uy, (ux * 2 + 1), 33, 20);
		a[2] = get3d(Ys0, 2, uy, (ux * 2 + 1), 33, 20);
		a[3] = get3d(Ys0, 3, uy, (ux * 2 + 1), 33, 20);
		a[4] = get3d(Ys0, 4, uy, (ux * 2 + 1), 33, 20);
		a[5] = get3d(Ys0, 5, uy, (ux * 2 + 1), 33, 20);
		a[6] = get3d(Ys0, 6, uy, (ux * 2 + 1), 33, 20);
		a[7] = get3d(Ys0, 7, uy, (ux * 2 + 1), 33, 20);
		winograd_f6x3_Y(y3, a);//{oc3 / oc7} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y
	{
		*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };//83
		*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };//
		*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
	
	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 33 elements to shared memory
		*(float4*)(Ys0 + Ywt     ) = {  v8.x,  v8.y,  v9.x,  v9.y };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(Ys0 + Ywt +  4) = { v10.x, v10.y, v11.x, v11.y };
		*(float4*)(Ys0 + Ywt +  8) = { v12.x, v12.y, v13.x, v13.y };
		*(float4*)(Ys0 + Ywt + 12) = { v14.x, v14.y, v15.x, v15.y };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 33, 20);
		a[1] = get3d(Ys0, 1, uy, ux * 2, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 33, 20);
		winograd_f6x3_Y(y0, a);//{oc0 / oc4} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, ux * 2 + 1, 33, 20);
		a[1] = get3d(Ys0, 1, uy, ux * 2 + 1, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2 + 1, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2 + 1, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2 + 1, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2 + 1, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2 + 1, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2 + 1, 33, 20);
		winograd_f6x3_Y(y1, a);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		//write 32 elements to shared memory
		*(float4*)(Ys0 + Ywt     ) = {  v8.z,  v8.w,  v9.z,  v9.w };//{oc0, oc1}, {oc4, oc5}}
		*(float4*)(Ys0 + Ywt +  4) = { v10.z, v10.w, v11.z, v11.w };
		*(float4*)(Ys0 + Ywt +  8) = { v12.z, v12.w, v13.z, v13.w };
		*(float4*)(Ys0 + Ywt + 12) = { v14.z, v14.w, v15.z, v15.w };
		__syncthreads();

		a[0] = get3d(Ys0, 0, uy, ux * 2, 33, 20);//{oc2, oc3}
		a[1] = get3d(Ys0, 1, uy, ux * 2, 33, 20);
		a[2] = get3d(Ys0, 2, uy, ux * 2, 33, 20);
		a[3] = get3d(Ys0, 3, uy, ux * 2, 33, 20);
		a[4] = get3d(Ys0, 4, uy, ux * 2, 33, 20);
		a[5] = get3d(Ys0, 5, uy, ux * 2, 33, 20);
		a[6] = get3d(Ys0, 6, uy, ux * 2, 33, 20);
		a[7] = get3d(Ys0, 7, uy, ux * 2, 33, 20);
		winograd_f6x3_Y(y2, a);//{oc2 / oc6} * { j0 - j3 }

		a[0] = get3d(Ys0, 0, uy, (ux * 2 + 1), 33, 20);//{oc1, oc5}
		a[1] = get3d(Ys0, 1, uy, (ux * 2 + 1), 33, 20);
		a[2] = get3d(Ys0, 2, uy, (ux * 2 + 1), 33, 20);
		a[3] = get3d(Ys0, 3, uy, (ux * 2 + 1), 33, 20);
		a[4] = get3d(Ys0, 4, uy, (ux * 2 + 1), 33, 20);
		a[5] = get3d(Ys0, 5, uy, (ux * 2 + 1), 33, 20);
		a[6] = get3d(Ys0, 6, uy, (ux * 2 + 1), 33, 20);
		a[7] = get3d(Ys0, 7, uy, (ux * 2 + 1), 33, 20);
		winograd_f6x3_Y(y3, a);//{oc3 / oc7} * { j0 - j3 }
	}

	//write to Y 
	{
		*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
		*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
		*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
}

#endif
 

//No
#ifndef WG_2X6_K7
#define WG_2X6_K7

#define wg2x6_k7(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel7<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
template<int FH>
__global__ void Winograd2x6_kernel7(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//2 * 8 * 8 * 32
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 32]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;//oc0 - oc4
	const int toc0 = boc0 + ((tx << 1) + (ty > 7)) * 2;//
	CW += (ty & 7)*OC + toc0;//CW[0, 0, (ty & 7), toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int tj0 = bj0  + ((ty << 1) + (tx > 7)) * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);//X[tn0, tih0, tiw0, tx&7]

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx / 32,  uy = idx % 32;//8 * 32

	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);//0 - 64
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3
	const int Y10 = Y00 + 24 * OC;//ux: j4 -> j7

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i =  (ty << 1) + (tx > 7);      //[8, 32]
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	float g0[8], g1[8];
	float x[8], d[8];

#pragma once 
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
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;
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
	//float (Ys*)[8][33][28] = (float(*)[8][33][28])(&Gs[0])

	float *Ys0 = &Gs[0][0][0][0];//8 * 32 * 24[elem] {x, y, z}
	float *Ys1 = &Ds[0][0][0][0];//8 * 32 *  8[elem] {w}

	float a0[8], a1[8], a2[8], a3[8];
	float y0[6], y1[6], y2[6], y3[6];//{oc0/4, oc1/5, oc2/6, oc3/7}
	__syncthreads();

	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		const int Ywt0 = (ux * 33 + uy) * 28;
		const int Ywt1 = (ux * 33 + uy) * 12;

		*(float4*)(Ys1 + Ywt1     ) = { v0.w, v1.w, v2.w, v3.w };//{oc3, oc8}
		*(float4*)(Ys1 + Ywt1 +  4) = { v4.w, v5.w, v6.w, v7.w };

		*(float4*)(Ys0 + Ywt0     ) = { v0.x, v1.x, v2.x, v3.x };//{oc0, oc4}
		*(float4*)(Ys0 + Ywt0 +  4) = { v4.x, v5.x, v6.x, v7.x };

		*(float4*)(Ys0 + Ywt0 +  8) = { v0.y, v1.y, v2.y, v3.y };//{oc1, oc5}
		*(float4*)(Ys0 + Ywt0 + 12) = { v4.y, v5.y, v6.y, v7.y };
		
		*(float4*)(Ys0 + Ywt0 + 16) = { v0.z, v1.z, v2.z, v3.z };//{oc2, oc7}
		*(float4*)(Ys0 + Ywt0 + 20) = { v4.z, v5.z, v6.z, v7.z };
		__syncthreads();

		a0[0] = get3d(Ys0, 0, uy, ux, 33, 28);//
		a0[1] = get3d(Ys0, 1, uy, ux, 33, 28);
		a0[2] = get3d(Ys0, 2, uy, ux, 33, 28);
		a0[3] = get3d(Ys0, 3, uy, ux, 33, 28);
		a0[4] = get3d(Ys0, 4, uy, ux, 33, 28);
		a0[5] = get3d(Ys0, 5, uy, ux, 33, 28);
		a0[6] = get3d(Ys0, 6, uy, ux, 33, 28);
		a0[7] = get3d(Ys0, 7, uy, ux, 33, 28);

		a1[0] = get3d(Ys0, 0, uy, ux + 8, 33, 28);
		a1[1] = get3d(Ys0, 1, uy, ux + 8, 33, 28);
		a1[2] = get3d(Ys0, 2, uy, ux + 8, 33, 28);
		a1[3] = get3d(Ys0, 3, uy, ux + 8, 33, 28);
		a1[4] = get3d(Ys0, 4, uy, ux + 8, 33, 28);
		a1[5] = get3d(Ys0, 5, uy, ux + 8, 33, 28);
		a1[6] = get3d(Ys0, 6, uy, ux + 8, 33, 28);
		a1[7] = get3d(Ys0, 7, uy, ux + 8, 33, 28);

		a2[0] = get3d(Ys0, 0, uy, ux + 16, 33, 28);//
		a2[1] = get3d(Ys0, 1, uy, ux + 16, 33, 28);
		a2[2] = get3d(Ys0, 2, uy, ux + 16, 33, 28);
		a2[3] = get3d(Ys0, 3, uy, ux + 16, 33, 28);
		a2[4] = get3d(Ys0, 4, uy, ux + 16, 33, 28);
		a2[5] = get3d(Ys0, 5, uy, ux + 16, 33, 28);
		a2[6] = get3d(Ys0, 6, uy, ux + 16, 33, 28);
		a2[7] = get3d(Ys0, 7, uy, ux + 16, 33, 28);

		a3[0] = get3d(Ys1, 0, uy, ux, 33, 12);
		a3[1] = get3d(Ys1, 1, uy, ux, 33, 12);
		a3[2] = get3d(Ys1, 2, uy, ux, 33, 12);
		a3[3] = get3d(Ys1, 3, uy, ux, 33, 12);
		a3[4] = get3d(Ys1, 4, uy, ux, 33, 12);
		a3[5] = get3d(Ys1, 5, uy, ux, 33, 12);
		a3[6] = get3d(Ys1, 6, uy, ux, 33, 12);
		a3[7] = get3d(Ys1, 7, uy, ux, 33, 12);

		winograd_f6x3_Y(y0, a0);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y1, a1);//{oc1 / oc5} * { j0 - j3 }
		winograd_f6x3_Y(y2, a2);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y3, a3);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y
	{
		*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };//83
		*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };//
		*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
	
	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		const int Ywt0 = (ux * 33 + uy) * 28;
		const int Ywt1 = (ux * 33 + uy) * 12;

		*(float4*)(Ys0 + Ywt0     ) = {  v8.x,  v9.x, v10.x, v11.x };//{oc0, oc4}
		*(float4*)(Ys0 + Ywt0 +  4) = { v12.x, v13.x, v14.x, v15.x };

		*(float4*)(Ys0 + Ywt0 +  8) = {  v8.y,  v9.y, v10.y, v11.y };//{oc1, oc5}
		*(float4*)(Ys0 + Ywt0 + 12) = { v12.y, v13.y, v14.y, v15.y };

		*(float4*)(Ys0 + Ywt0 + 16) = {  v8.z,  v9.z, v10.z, v11.z };//{oc2, oc7}
		*(float4*)(Ys0 + Ywt0 + 20) = { v12.z, v13.z, v14.z, v15.z };

		*(float4*)(Ys1 + Ywt1     ) = {  v8.w,  v9.w, v10.w, v11.w };//{oc3, oc8}
		*(float4*)(Ys1 + Ywt1 +  4) = { v12.w, v13.w, v14.w, v15.w };
		__syncthreads();

		a0[0] = get3d(Ys0, 0, uy, ux, 33, 28);//
		a0[1] = get3d(Ys0, 1, uy, ux, 33, 28);
		a0[2] = get3d(Ys0, 2, uy, ux, 33, 28);
		a0[3] = get3d(Ys0, 3, uy, ux, 33, 28);
		a0[4] = get3d(Ys0, 4, uy, ux, 33, 28);
		a0[5] = get3d(Ys0, 5, uy, ux, 33, 28);
		a0[6] = get3d(Ys0, 6, uy, ux, 33, 28);
		a0[7] = get3d(Ys0, 7, uy, ux, 33, 28);

		a1[0] = get3d(Ys0, 0, uy, ux + 8, 33, 28);
		a1[1] = get3d(Ys0, 1, uy, ux + 8, 33, 28);
		a1[2] = get3d(Ys0, 2, uy, ux + 8, 33, 28);
		a1[3] = get3d(Ys0, 3, uy, ux + 8, 33, 28);
		a1[4] = get3d(Ys0, 4, uy, ux + 8, 33, 28);
		a1[5] = get3d(Ys0, 5, uy, ux + 8, 33, 28);
		a1[6] = get3d(Ys0, 6, uy, ux + 8, 33, 28);
		a1[7] = get3d(Ys0, 7, uy, ux + 8, 33, 28);

		a2[0] = get3d(Ys0, 0, uy, ux + 16, 33, 28);//
		a2[1] = get3d(Ys0, 1, uy, ux + 16, 33, 28);
		a2[2] = get3d(Ys0, 2, uy, ux + 16, 33, 28);
		a2[3] = get3d(Ys0, 3, uy, ux + 16, 33, 28);
		a2[4] = get3d(Ys0, 4, uy, ux + 16, 33, 28);
		a2[5] = get3d(Ys0, 5, uy, ux + 16, 33, 28);
		a2[6] = get3d(Ys0, 6, uy, ux + 16, 33, 28);
		a2[7] = get3d(Ys0, 7, uy, ux + 16, 33, 28);

		a3[0] = get3d(Ys1, 0, uy, ux, 33, 12);
		a3[1] = get3d(Ys1, 1, uy, ux, 33, 12);
		a3[2] = get3d(Ys1, 2, uy, ux, 33, 12);
		a3[3] = get3d(Ys1, 3, uy, ux, 33, 12);
		a3[4] = get3d(Ys1, 4, uy, ux, 33, 12);
		a3[5] = get3d(Ys1, 5, uy, ux, 33, 12);
		a3[6] = get3d(Ys1, 6, uy, ux, 33, 12);
		a3[7] = get3d(Ys1, 7, uy, ux, 33, 12);

		winograd_f6x3_Y(y0, a0);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y1, a1);//{oc1 / oc5} * { j0 - j3 }
		winograd_f6x3_Y(y2, a2);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y3, a3);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y 
	{
		*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
		*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
		*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
}

#endif


//No
#ifndef WG_2X6_K8
#define WG_2X6_K8

#define wg2x6_k8(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel8<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
template<int FH>
__global__ void Winograd2x6_kernel8(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//2 * 8 * 8 * 32
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 32]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;//oc0 - oc4
	const int toc0 = boc0 + ((tx << 1) + (ty > 7)) * 2;//
	CW += (ty & 7)*OC + toc0;//CW[0, 0, (ty & 7), toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int tj0 = bj0  + ((ty << 1) + (tx > 7)) * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);//X[tn0, tih0, tiw0, tx&7]

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx / 32,  uy = idx % 32;//8 * 32

	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);//0 - 64
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3
	const int Y10 = Y00 + 24 * OC;//ux: j4 -> j7

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i =  (ty << 1) + (tx > 7);      //[8, 32]
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	float g0[8], g1[8];
	float x[8], d[8];

#pragma once 
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
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;
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
	float (* __restrict__ Ys0)[33][28] = (float(*)[33][28])(&Gs[0][0][0][0]);//[8, 33. 28]
	float (* __restrict__ Ys1)[33][12] = (float(*)[33][12])(&Ds[0][0][0][0]);

	float a0[8], a1[8], a2[8], a3[8];//{oc0/4, oc1/5, oc2/6, oc3/7}
	float y0[6], y1[6], y2[6], y3[6];//{oc0/4, oc1/5, oc2/6, oc3/7}
	__syncthreads();

	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		*(float4*)(&Ys0[ux][uy][ 0]) = { v0.x, v1.x, v2.x, v3.x };//{oc0, oc4}
		*(float4*)(&Ys0[ux][uy][ 4]) = { v4.x, v5.x, v6.x, v7.x };
		*(float4*)(&Ys0[ux][uy][ 8]) = { v0.y, v1.y, v2.y, v3.y };//{oc1, oc5}
		*(float4*)(&Ys0[ux][uy][12]) = { v4.y, v5.y, v6.y, v7.y };
		*(float4*)(&Ys0[ux][uy][16]) = { v0.z, v1.z, v2.z, v3.z };//{oc1, oc5}
		*(float4*)(&Ys0[ux][uy][20]) = { v4.z, v5.z, v6.z, v7.z };
		*(float4*)(&Ys1[ux][uy][ 0]) = { v0.w, v1.w, v2.w, v3.w };//{oc3, oc8}
		*(float4*)(&Ys1[ux][uy][ 4]) = { v4.w, v5.w, v6.w, v7.w };
		__syncthreads();

		a0[0] = Ys0[0][uy][ux];
		a0[1] = Ys0[1][uy][ux];
		a0[2] = Ys0[2][uy][ux];
		a0[3] = Ys0[3][uy][ux];
		a0[4] = Ys0[4][uy][ux];
		a0[5] = Ys0[5][uy][ux];
		a0[6] = Ys0[6][uy][ux];
		a0[7] = Ys0[7][uy][ux];

		a1[0] = Ys0[0][uy][ux + 8];
		a1[1] = Ys0[1][uy][ux + 8];
		a1[2] = Ys0[2][uy][ux + 8];
		a1[3] = Ys0[3][uy][ux + 8];
		a1[4] = Ys0[4][uy][ux + 8];
		a1[5] = Ys0[5][uy][ux + 8];
		a1[6] = Ys0[6][uy][ux + 8];
		a1[7] = Ys0[7][uy][ux + 8];

		a2[0] = Ys0[0][uy][ux + 16];
		a2[1] = Ys0[1][uy][ux + 16];
		a2[2] = Ys0[2][uy][ux + 16];
		a2[3] = Ys0[3][uy][ux + 16];
		a2[4] = Ys0[4][uy][ux + 16];
		a2[5] = Ys0[5][uy][ux + 16];
		a2[6] = Ys0[6][uy][ux + 16];
		a2[7] = Ys0[7][uy][ux + 16];

		a3[0] = Ys1[0][uy][ux];
		a3[1] = Ys1[1][uy][ux];
		a3[2] = Ys1[2][uy][ux];
		a3[3] = Ys1[3][uy][ux];
		a3[4] = Ys1[4][uy][ux];
		a3[5] = Ys1[5][uy][ux];
		a3[6] = Ys1[6][uy][ux];
		a3[7] = Ys1[7][uy][ux];

		winograd_f6x3_Y(y0, a0);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y1, a1);//{oc1 / oc5} * { j0 - j3 }
		winograd_f6x3_Y(y2, a2);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y3, a3);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y
	{
		*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };//83
		*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };//
		*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
	
	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		*(float4*)(&Ys0[ux][uy][ 0]) = {  v8.x,  v9.x, v10.x, v11.x };//{oc0, oc4}
		*(float4*)(&Ys0[ux][uy][ 4]) = { v12.x, v13.x, v14.x, v15.x };
		*(float4*)(&Ys0[ux][uy][ 8]) = {  v8.y,  v9.y, v10.y, v11.y };//{oc1, oc5}
		*(float4*)(&Ys0[ux][uy][12]) = { v12.y, v13.y, v14.y, v15.y };
		*(float4*)(&Ys0[ux][uy][16]) = {  v8.z,  v9.z, v10.z, v11.z };//{oc2, oc7}
		*(float4*)(&Ys0[ux][uy][20]) = { v12.z, v13.z, v14.z, v15.z };
		*(float4*)(&Ys1[ux][uy][ 0]) = {  v8.w,  v9.w, v10.w, v11.w };//{oc3, oc8}
		*(float4*)(&Ys1[ux][uy][ 4]) = { v12.w, v13.w, v14.w, v15.w };
		__syncthreads();

		a0[0] = Ys0[0][uy][ux];
		a0[1] = Ys0[1][uy][ux];
		a0[2] = Ys0[2][uy][ux];
		a0[3] = Ys0[3][uy][ux];
		a0[4] = Ys0[4][uy][ux];
		a0[5] = Ys0[5][uy][ux];
		a0[6] = Ys0[6][uy][ux];
		a0[7] = Ys0[7][uy][ux];

		a1[0] = Ys0[0][uy][ux + 8];
		a1[1] = Ys0[1][uy][ux + 8];
		a1[2] = Ys0[2][uy][ux + 8];
		a1[3] = Ys0[3][uy][ux + 8];
		a1[4] = Ys0[4][uy][ux + 8];
		a1[5] = Ys0[5][uy][ux + 8];
		a1[6] = Ys0[6][uy][ux + 8];
		a1[7] = Ys0[7][uy][ux + 8];

		a2[0] = Ys0[0][uy][ux + 16];
		a2[1] = Ys0[1][uy][ux + 16];
		a2[2] = Ys0[2][uy][ux + 16];
		a2[3] = Ys0[3][uy][ux + 16];
		a2[4] = Ys0[4][uy][ux + 16];
		a2[5] = Ys0[5][uy][ux + 16];
		a2[6] = Ys0[6][uy][ux + 16];
		a2[7] = Ys0[7][uy][ux + 16];

		a3[0] = Ys1[0][uy][ux];
		a3[1] = Ys1[1][uy][ux];
		a3[2] = Ys1[2][uy][ux];
		a3[3] = Ys1[3][uy][ux];
		a3[4] = Ys1[4][uy][ux];
		a3[5] = Ys1[5][uy][ux];
		a3[6] = Ys1[6][uy][ux];
		a3[7] = Ys1[7][uy][ux];

		winograd_f6x3_Y(y0, a0);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y1, a1);//{oc1 / oc5} * { j0 - j3 }
		winograd_f6x3_Y(y2, a2);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y3, a3);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y 
	{
		*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
		*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
		*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
}

#endif


//No
#ifndef WG_2X6_K9
#define WG_2X6_K9

#define wg2x6_k9(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel9<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
template<int FH>
__global__ void Winograd2x6_kernel9(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//2 * 8 * 8 * 32
	__shared__ float SMEM[2 * 8 * 8 * 96];
	float(*__restrict__ Gs)[8][8][64] = (float(*)[8][8][64])(SMEM);
	float(*__restrict__ Ds)[8][8][32] = (float(*)[8][8][32])(SMEM + 2 * 8 * 8 * 64);

	//__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 32]
	//__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;//oc0 - oc4
	const int toc0 = boc0 + ((tx << 1) + (ty > 7)) * 2;//
	CW += (ty & 7)*OC + toc0;//CW[0, 0, (ty & 7), toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int tj0 = bj0  + ((ty << 1) + (tx > 7)) * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);//X[tn0, tih0, tiw0, tx&7]

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx / 32,  uy = idx % 32;//8 * 32

	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);//0 - 64
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3
	const int Y10 = Y00 + 24 * OC;//ux: j4 -> j7

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i =  (ty << 1) + (tx > 7);      //[8, 32]
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	float g0[8], g1[8];
	float x[8], d[8];

#pragma once 
	for (int fh = 0; fh < FH; fh++) {
		//load 1 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		bool ly6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;
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

		//load 2 group from CW[FH, FW, IC, OC]
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + IC_OC;
		const int W2 = W0 + (IC_OC << 1);
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		winograd_f6x3_G(g0, w0.x, w1.x, w2.x);
		winograd_f6x3_G(g1, w0.y, w1.y, w2.y);
		winograd_f6x3_D(d, x);

		//write to shread memory
		Ds[buf][Ds_k][0][Ds_i] = d[0]; Ds[buf][Ds_k][1][Ds_i] = d[1];
		Ds[buf][Ds_k][2][Ds_i] = d[2]; Ds[buf][Ds_k][3][Ds_i] = d[3];
		Ds[buf][Ds_k][4][Ds_i] = d[4]; Ds[buf][Ds_k][5][Ds_i] = d[5];
		Ds[buf][Ds_k][6][Ds_i] = d[6]; Ds[buf][Ds_k][7][Ds_i] = d[7];

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

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC;
			const int W2 = W0 + (IC_OC << 1);
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);

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
	float (* __restrict__ Ys)[33][36] = (float(*)[33][36])(SMEM);

	float a0[8], a1[8], a2[8], a3[8];//{oc0/4, oc1/5, oc2/6, oc3/7}
	float y0[6], y1[6], y2[6], y3[6];//{oc0/4, oc1/5, oc2/6, oc3/7}
	__syncthreads();

	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		*(float4*)(&Ys[ux][uy][ 0]) = { v0.x, v1.x, v2.x, v3.x };//{oc0, oc4}
		*(float4*)(&Ys[ux][uy][ 4]) = { v4.x, v5.x, v6.x, v7.x };
		*(float4*)(&Ys[ux][uy][ 8]) = { v0.y, v1.y, v2.y, v3.y };//{oc1, oc5}
		*(float4*)(&Ys[ux][uy][12]) = { v4.y, v5.y, v6.y, v7.y };
		*(float4*)(&Ys[ux][uy][16]) = { v0.z, v1.z, v2.z, v3.z };//{oc1, oc5}
		*(float4*)(&Ys[ux][uy][20]) = { v4.z, v5.z, v6.z, v7.z };
		*(float4*)(&Ys[ux][uy][24]) = { v0.w, v1.w, v2.w, v3.w };//{oc3, oc8}
		*(float4*)(&Ys[ux][uy][28]) = { v4.w, v5.w, v6.w, v7.w };
		__syncthreads();

		a0[0] = Ys[0][uy][ux];
		a0[1] = Ys[1][uy][ux];
		a0[2] = Ys[2][uy][ux];
		a0[3] = Ys[3][uy][ux];
		a0[4] = Ys[4][uy][ux];
		a0[5] = Ys[5][uy][ux];
		a0[6] = Ys[6][uy][ux];
		a0[7] = Ys[7][uy][ux];

		a1[0] = Ys[0][uy][ux + 8];
		a1[1] = Ys[1][uy][ux + 8];
		a1[2] = Ys[2][uy][ux + 8];
		a1[3] = Ys[3][uy][ux + 8];
		a1[4] = Ys[4][uy][ux + 8];
		a1[5] = Ys[5][uy][ux + 8];
		a1[6] = Ys[6][uy][ux + 8];
		a1[7] = Ys[7][uy][ux + 8];

		a2[0] = Ys[0][uy][ux + 16];
		a2[1] = Ys[1][uy][ux + 16];
		a2[2] = Ys[2][uy][ux + 16];
		a2[3] = Ys[3][uy][ux + 16];
		a2[4] = Ys[4][uy][ux + 16];
		a2[5] = Ys[5][uy][ux + 16];
		a2[6] = Ys[6][uy][ux + 16];
		a2[7] = Ys[7][uy][ux + 16];

		a3[0] = Ys[0][uy][ux + 24];
		a3[1] = Ys[1][uy][ux + 24];
		a3[2] = Ys[2][uy][ux + 24];
		a3[3] = Ys[3][uy][ux + 24];
		a3[4] = Ys[4][uy][ux + 24];
		a3[5] = Ys[5][uy][ux + 24];
		a3[6] = Ys[6][uy][ux + 24];
		a3[7] = Ys[7][uy][ux + 24];

		winograd_f6x3_Y(y0, a0);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y1, a1);//{oc1 / oc5} * { j0 - j3 }
		winograd_f6x3_Y(y2, a2);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y3, a3);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y
	{
		*(float4*)(Y + Y00         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y00 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };//83
		*(float4*)(Y + Y00 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };//
		*(float4*)(Y + Y00 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y00 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y00 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
	
	//=========================================================================
	//write-turn[0, 1]: 8[ux: accu] * 16[uy (32)] * 32[elem]
	{
		*(float4*)(&Ys[ux][uy][ 0]) = {  v8.x,  v9.x, v10.x, v11.x };//{oc0, oc4}
		*(float4*)(&Ys[ux][uy][ 4]) = { v12.x, v13.x, v14.x, v15.x };
		*(float4*)(&Ys[ux][uy][ 8]) = {  v8.y,  v9.y, v10.y, v11.y };//{oc1, oc5}
		*(float4*)(&Ys[ux][uy][12]) = { v12.y, v13.y, v14.y, v15.y };
		*(float4*)(&Ys[ux][uy][16]) = {  v8.z,  v9.z, v10.z, v11.z };//{oc2, oc7}
		*(float4*)(&Ys[ux][uy][20]) = { v12.z, v13.z, v14.z, v15.z };
		*(float4*)(&Ys[ux][uy][24]) = {  v8.w,  v9.w, v10.w, v11.w };//{oc3, oc8}
		*(float4*)(&Ys[ux][uy][28]) = { v12.w, v13.w, v14.w, v15.w };
		__syncthreads();

		a0[0] = Ys[0][uy][ux];
		a0[1] = Ys[1][uy][ux];
		a0[2] = Ys[2][uy][ux];
		a0[3] = Ys[3][uy][ux];
		a0[4] = Ys[4][uy][ux];
		a0[5] = Ys[5][uy][ux];
		a0[6] = Ys[6][uy][ux];
		a0[7] = Ys[7][uy][ux];

		a1[0] = Ys[0][uy][ux + 8];
		a1[1] = Ys[1][uy][ux + 8];
		a1[2] = Ys[2][uy][ux + 8];
		a1[3] = Ys[3][uy][ux + 8];
		a1[4] = Ys[4][uy][ux + 8];
		a1[5] = Ys[5][uy][ux + 8];
		a1[6] = Ys[6][uy][ux + 8];
		a1[7] = Ys[7][uy][ux + 8];

		a2[0] = Ys[0][uy][ux + 16];
		a2[1] = Ys[1][uy][ux + 16];
		a2[2] = Ys[2][uy][ux + 16];
		a2[3] = Ys[3][uy][ux + 16];
		a2[4] = Ys[4][uy][ux + 16];
		a2[5] = Ys[5][uy][ux + 16];
		a2[6] = Ys[6][uy][ux + 16];
		a2[7] = Ys[7][uy][ux + 16];

		a3[0] = Ys[0][uy][ux + 24];
		a3[1] = Ys[1][uy][ux + 24];
		a3[2] = Ys[2][uy][ux + 24];
		a3[3] = Ys[3][uy][ux + 24];
		a3[4] = Ys[4][uy][ux + 24];
		a3[5] = Ys[5][uy][ux + 24];
		a3[6] = Ys[6][uy][ux + 24];
		a3[7] = Ys[7][uy][ux + 24];

		winograd_f6x3_Y(y0, a0);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y1, a1);//{oc1 / oc5} * { j0 - j3 }
		winograd_f6x3_Y(y2, a2);//{oc0 / oc4} * { j0 - j3 }
		winograd_f6x3_Y(y3, a3);//{oc1 / oc5} * { j0 - j3 }
		__syncthreads();
	}

	//write to Y 
	{
		*(float4*)(Y + Y10         ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y10 + OC * 1) = { y0[1], y1[1], y2[1], y3[1] };
		*(float4*)(Y + Y10 + OC * 2) = { y0[2], y1[2], y2[2], y3[2] };
		*(float4*)(Y + Y10 + OC * 3) = { y0[3], y1[3], y2[3], y3[3] };
		*(float4*)(Y + Y10 + OC * 4) = { y0[4], y1[4], y2[4], y3[4] };
		*(float4*)(Y + Y10 + OC * 5) = { y0[5], y1[5], y2[5], y3[5] };
	}
}

#endif


//Select: Size = 20.25, Time = 2.67964 msec, Performace = 16228.5 GFlop/s
#ifndef WG_2X6_K10
#define WG_2X6_K10

#define wg2x6_k10(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw) \
	Winograd2x6_kernel10<FH>\
		<<< dim3(OC>>6, (N*OH*((OW + 5)/6))>>5), dim3(16, 16), 0, stream>>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 20.25, Time = 2.67103 msec, Performace = 16280.8 GFlop/s
//Size = 81, Time = 10.7292 msec, Performace = 16212.4 GFlop/s
//Size = 81, Time = 10.7723 msec, Performace = 16147.6 GFlop/s
//Size = 81, Time = 10.479 msec, Performace = 16599.6 GFlop/s
template<int FH>
__global__ void Winograd2x6_kernel10(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//2 * 8 * 8 * 32
	__shared__ float Gs[2][8][8][64];//[buf][ik: STEP][elem: 8][g(oc): 32]
	__shared__ float Ds[2][8][8][32];//[buf][ik: STEP][elem: 8][d( j): 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;//oc0 - oc4
	const int toc0 = boc0 + ((tx << 1) + (ty > 7)) * 2;//
	CW += (ty & 7)*OC + toc0;//CW[0, 0, (ty & 7), toc0]

	//prepare for X[N, IH, IW, IC], 32 * 6 = 192
	const int bj0 = (blockIdx.y * 192) + j_index;//32 group of Y
	const int tj0 = bj0  + ((ty << 1) + (tx > 7)) * 6;
	const int OW6 = (OW + 5) / 6 * 6, OH_OW6 = OH * OW6;
	int tn0, toh0, tow0; {
		tn0 = tj0 / OH_OW6; int jr = tj0 - tn0 * OH_OW6; 
		toh0 = jr / OW6; tow0 = jr - toh0 * OW6;
	}
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);//X[tn0, tih0, tiw0, tx&7]

	//prepare for threadIdx: 8(elem) * { 2(k) * 4(oc) * 4(j) }= 256
	const int idx = (ty << 4) + tx;
	const int ux = idx / 32,  uy = idx % 32;//8 * 32

	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;//8
	const int DIdx = ((uy & 7) >> 1) << 3;//4

	//preapre for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx + ((ux & 1) << 2);//0 - 64
	const int yj0 = bj0 + (DIdx + (ux >> 1)) * 6;
	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3

	//======[compute area1: local]======================================================
	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i =  (ty << 1) + (tx > 7);      //[8, 32]
	const int Gs_k = (ty & 7), Gs_i = ((tx << 1) + (ty > 7)) << 1;//[8, 64]
	float g0[8], g1[8], x[8], d[8];

#pragma once 
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
		bool ly7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		int tX0 = X0 + fh * IW * IC;
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
