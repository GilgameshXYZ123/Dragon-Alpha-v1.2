

//CW[FH, IC, 3, OC] -> faster
//X[N, IH, IW, IC] -> faster
#ifndef WG_F2X3_KERNEL1
#define WG_F2X3_KERNEL1

#define wg_f2x3_k1(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	wg_f2x3_kernel1<FH>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, N, IC, OC, ph, pw, oc_index, j_index)

//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 18, Time = 2.73621 msec, Performace = 14127.1 GFlop/s
//LB = 4:Size = 18, Time = 2.51127 msec, Performace = 15392.5 GFlop/s

template<int FH>
__global__ void wg_f2x3_kernel1(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
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
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty > 7) << 1);
	CW += (ty & 7)*OC*3 + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx > 7) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = (tx & 7) * N * IH * IW + ((tn0*IH + tih0)*IW + tiw0);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
	const int gIdx = uy >> 3, dIdx = uy & 7;
	const int GIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int DIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//prepare for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + ((DIdx + ux) << 1);
	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3

	//======[compute area1: local]======================================================
	float w[6], x[6];
	const int IC_OC = IC * OC;
	const int Ds_k = (tx & 7), Ds_i = (ty << 2) + ((tx > 7) << 1);
	const int Gs_k = (ty & 7), Gs_i = (tx << 2) + ((ty > 7) << 1);

#pragma once 
	for (int fh = 0; fh < FH; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		int tX0 = X0 + fh * IW;
		int tX1 = tX0 + 1, tX2 = tX0 + 2;
		int tX3 = tX0 + 3, tX4 = tX0 + 4, tX5 = tX0 + 5;
		tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
		tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
		tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
		x[0] = tex1Dfetch<float>(X, tX0);
		x[1] = tex1Dfetch<float>(X, tX1);
		x[2] = tex1Dfetch<float>(X, tX2);
		x[3] = tex1Dfetch<float>(X, tX3);
		x[4] = tex1Dfetch<float>(X, tX4);
		x[5] = tex1Dfetch<float>(X, tX5);

		//load 2 group from CW[FH, 3, IC, OC] -> [FH, IC, 3, OC]
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + OC, W2 = W0 + (OC << 1);
		*(float2*)(w    ) = *(float2*)(CW + W0);//fw = 0
		*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1
		*(float2*)(w + 4) = *(float2*)(CW + W2);//fw = 2

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x[0] - x[2], x[1] + x[2], x[2] - x[1], x[1] - x[3] };
		float4 d1 = float4{ x[2] - x[4], x[3] + x[4], x[4] - x[3], x[3] - x[5] };

		float gst0 = 0.5f*(w[0] + w[2] + w[4]), gst1 = gst0 - w[2];
		float gst2 = 0.5f*(w[1] + w[3] + w[5]), gst3 = gst2 - w[3];
		float4 g0 = float4{ w[0], gst0, gst1, w[4] };
		float4 g1 = float4{ w[1], gst2, gst3, w[5] };

		//write to shread memory
		Ds[buf][Ds_k][0][Ds_i] = d0.x; Ds[buf][Ds_k][0][Ds_i + 1] = d1.x;
		Ds[buf][Ds_k][1][Ds_i] = d0.y; Ds[buf][Ds_k][1][Ds_i + 1] = d1.y;
		Ds[buf][Ds_k][2][Ds_i] = d0.z; Ds[buf][Ds_k][2][Ds_i + 1] = d1.z;
		Ds[buf][Ds_k][3][Ds_i] = d0.w; Ds[buf][Ds_k][3][Ds_i + 1] = d1.w;

		Gs[buf][Gs_k][0][Gs_i] = g0.x; Gs[buf][Gs_k][0][Gs_i + 1] = g1.x;
		Gs[buf][Gs_k][1][Gs_i] = g0.y; Gs[buf][Gs_k][1][Gs_i + 1] = g1.y;
		Gs[buf][Gs_k][2][Gs_i] = g0.z; Gs[buf][Gs_k][2][Gs_i + 1] = g1.z;
		Gs[buf][Gs_k][3][Gs_i] = g0.w; Gs[buf][Gs_k][3][Gs_i + 1] = g1.w;
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
			int tX0 = X0 + fh * IW + oic * N * IH * IW;
			int tX1 = tX0 + 1, tX2 = tX0 + 2;
			int tX3 = tX0 + 3, tX4 = tX0 + 4, tX5 = tX0 + 5;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
			x[0] = tex1Dfetch<float>(X, tX0);
			x[1] = tex1Dfetch<float>(X, tX1);
			x[2] = tex1Dfetch<float>(X, tX2);
			x[3] = tex1Dfetch<float>(X, tX3);
			x[4] = tex1Dfetch<float>(X, tX4);
			x[5] = tex1Dfetch<float>(X, tX5);

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic * 3) * OC;//fh, ic, fw, oc
			const int W1 = W0 + OC, W2 = W0 + (OC << 1);
			*(float2*)(w    ) = *(float2*)(CW + W0);//fw = 0
			*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1
			*(float2*)(w + 4) = *(float2*)(CW + W2);//fw = 2

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x[0] - x[2], x[1] + x[2], x[2] - x[1], x[1] - x[3] };
			float4 d1 = float4{ x[2] - x[4], x[3] + x[4], x[4] - x[3], x[3] - x[5] };

			float gst0 = 0.5f*(w[0] + w[2] + w[4]), gst1 = gst0 - w[2];
			float gst2 = 0.5f*(w[1] + w[3] + w[5]), gst3 = gst2 - w[3];
			float4 g0 = float4{ w[0], gst0, gst1, w[4] };
			float4 g1 = float4{ w[1], gst2, gst3, w[5] };

			//write to shread memory
			Ds[buf][Ds_k][0][Ds_i] = d0.x; Ds[buf][Ds_k][0][Ds_i + 1] = d1.x;
			Ds[buf][Ds_k][1][Ds_i] = d0.y; Ds[buf][Ds_k][1][Ds_i + 1] = d1.y;
			Ds[buf][Ds_k][2][Ds_i] = d0.z; Ds[buf][Ds_k][2][Ds_i + 1] = d1.z;
			Ds[buf][Ds_k][3][Ds_i] = d0.w; Ds[buf][Ds_k][3][Ds_i + 1] = d1.w; 

			Gs[buf][Gs_k][0][Gs_i] = g0.x; Gs[buf][Gs_k][0][Gs_i + 1] = g1.x;
			Gs[buf][Gs_k][1][Gs_i] = g0.y; Gs[buf][Gs_k][1][Gs_i + 1] = g1.y;
			Gs[buf][Gs_k][2][Gs_i] = g0.z; Gs[buf][Gs_k][2][Gs_i + 1] = g1.z;
			Gs[buf][Gs_k][3][Gs_i] = g0.w; Gs[buf][Gs_k][3][Gs_i + 1] = g1.w;
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
	float *Ys0 = &Gs[0][0][0][0];//Ys[elem: 4][64: (1 << LB << LB >> 2)][16: elem]
	float *Ys1 = &Ds[0][0][0][0];
	const int Ywt = (ux * 65 + uy) * 20;
	const int Yrd = uy * 20 + (ux << 2);

	float4 a0, a1, a2, a3, a4, a5, a6, a7;
	float4 k0, k1, k2, k3, k4, k5, k6, k7;
	float2 y0, y1, y2, y3, y4, y5, y6, y7;

	const int Y01 = Y00 + OC;
	const int Y10 = Y00 + 8 * OC, Y11 = Y10 + OC;

	//write-turn[0, 1]: j0, j1, j2, j3, [oc0 - oc3], [oc4 - oc7]
	*(float4*)(Ys0 + Ywt     ) = v0; *(float4*)(Ys1 + Ywt     ) = v1;
	*(float4*)(Ys0 + Ywt +  4) = v2; *(float4*)(Ys1 + Ywt +  4) = v3;
	*(float4*)(Ys0 + Ywt +  8) = v4; *(float4*)(Ys1 + Ywt +  8) = v5;
	*(float4*)(Ys0 + Ywt + 12) = v6; *(float4*)(Ys1 + Ywt + 12) = v7;
	__syncthreads();

	//read-turn[0, 1]: j0, j1, j2, j3, [oc0 - oc3], [oc4 - oc7]
	a0 = *(float4*)(Ys0 +        Yrd); a4 = *(float4*)(Ys1 +        Yrd);
	a1 = *(float4*)(Ys0 + 1300 + Yrd); a5 = *(float4*)(Ys1 + 1300 + Yrd);
	a2 = *(float4*)(Ys0 + 2600 + Yrd); a6 = *(float4*)(Ys1 + 2600 + Yrd);
	a3 = *(float4*)(Ys0 + 3900 + Yrd); a7 = *(float4*)(Ys1 + 3900 + Yrd);

	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f2x3_y_f32_64(y0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f2x3_y_f32_64(y1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f2x3_y_f32_64(y2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f2x3_y_f32_64(y3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f2x3_y_f32_64(y4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f2x3_y_f32_64(y5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f2x3_y_f32_64(y6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f2x3_y_f32_64(y7, k7);

	*(float4*)(Y + Y00) = { y0.x, y1.x, y2.x, y3.x }; *(float4*)(Y + Y00 + 4) = { y4.x, y5.x, y6.x, y7.x };
	*(float4*)(Y + Y01) = { y0.y, y1.y, y2.y, y3.y }; *(float4*)(Y + Y01 + 4) = { y4.y, y5.y, y6.y, y7.y };
	__syncthreads();

	//write-turn2: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	*(float4*)(Ys0 + Ywt     ) =  v8; *(float4*)(Ys1 + Ywt     ) =  v9;
	*(float4*)(Ys0 + Ywt +  4) = v10; *(float4*)(Ys1 + Ywt +  4) = v11;
	*(float4*)(Ys0 + Ywt +  8) = v12; *(float4*)(Ys1 + Ywt +  8) = v13;
	*(float4*)(Ys0 + Ywt + 12) = v14; *(float4*)(Ys1 + Ywt + 12) = v15;
	__syncthreads();

	//write-turn[2, 3]: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	a0 = *(float4*)(Ys0 +        Yrd); a4 = *(float4*)(Ys1 +        Yrd);
	a1 = *(float4*)(Ys0 + 1300 + Yrd); a5 = *(float4*)(Ys1 + 1300 + Yrd);
	a2 = *(float4*)(Ys0 + 2600 + Yrd); a6 = *(float4*)(Ys1 + 2600 + Yrd);
	a3 = *(float4*)(Ys0 + 3900 + Yrd); a7 = *(float4*)(Ys1 + 3900 + Yrd);

	//read-turn[2, 3]: j4, j5, j6, j7, [oc0 - oc3], [oc4 - oc7]
	k0 = float4{ a0.x, a1.x, a2.x, a3.x }; winograd_f2x3_y_f32_64(y0, k0);
	k1 = float4{ a0.y, a1.y, a2.y, a3.y }; winograd_f2x3_y_f32_64(y1, k1);
	k2 = float4{ a0.z, a1.z, a2.z, a3.z }; winograd_f2x3_y_f32_64(y2, k2);
	k3 = float4{ a0.w, a1.w, a2.w, a3.w }; winograd_f2x3_y_f32_64(y3, k3);

	k4 = float4{ a4.x, a5.x, a6.x, a7.x }; winograd_f2x3_y_f32_64(y4, k4);
	k5 = float4{ a4.y, a5.y, a6.y, a7.y }; winograd_f2x3_y_f32_64(y5, k5);
	k6 = float4{ a4.z, a5.z, a6.z, a7.z }; winograd_f2x3_y_f32_64(y6, k6);
	k7 = float4{ a4.w, a5.w, a6.w, a7.w }; winograd_f2x3_y_f32_64(y7, k7);

	*(float4*)(Y + Y10) = { y0.x, y1.x, y2.x, y3.x }; *(float4*)(Y + Y10 + 4) = { y4.x, y5.x, y6.x, y7.x };
	*(float4*)(Y + Y11) = { y0.y, y1.y, y2.y, y3.y }; *(float4*)(Y + Y11 + 4) = { y4.y, y5.y, y6.y, y7.y };
}

#endif

