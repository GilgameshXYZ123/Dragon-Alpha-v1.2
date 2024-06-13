


//half shared memory(Try to realse)
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_B1
#define XENO1_WINOGRAD_F4X3_B1

#define xeno_winograd_f4x3_b1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_b1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 1.94215 msec, Performace = 9951.52 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_b1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	__shared__ float4 Gs0[1 << LB][(2 << LB) + 1];
	__shared__ float2 Gs1[1 << LB][(2 << LB) + 1];
	__shared__ float4 Ds0[1 << LB][(2 << LB) + 1];
	__shared__ float2 Ds1[1 << LB][(2 << LB) + 1];

	float4  m0 = F32_4_0, m1 = F32_4_0, m2 = F32_4_0;//0-3
	float4  m3 = F32_4_0, m4 = F32_4_0, m5 = F32_4_0;

	float4  m6 = F32_4_0, m7 = F32_4_0, m8 = F32_4_0;//4-8
	float4  m9 = F32_4_0, m10 = F32_4_0, m11 = F32_4_0;

	float4 m12 = F32_4_0, m13 = F32_4_0, m14 = F32_4_0;//8-11
	float4 m15 = F32_4_0, m16 = F32_4_0, m17 = F32_4_0;

	float4 m18 = F32_4_0, m19 = F32_4_0, m20 = F32_4_0;//12-15
	float4 m21 = F32_4_0, m22 = F32_4_0, m23 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 4) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 3);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool lx0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool lx1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool lx2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool lx3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool lx4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool lx5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		bool lx6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		bool lx7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		bool lx8 = lh0 && (tiw0 >= -8) && (tiw0 + 8 < IW);
		bool lx9 = lh0 && (tiw0 >= -9) && (tiw0 + 9 < IW);

		for (int oic = 0; oic < IC; oic += STEP)
		{
			//load 2 group from X
			const int xic = oic + (tx & STEP_m1);//with the same ty
			const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);//0
			float x1 = (lx1 ? X[xoffset + IC] : 0);//1
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float x6 = (lx6 ? X[xoffset + IC * 6] : 0);//6
			float x7 = (lx7 ? X[xoffset + IC * 7] : 0);//7
			float x8 = (lx8 ? X[xoffset + (IC << 3)] : 0);//8
			float x9 = (lx9 ? X[xoffset + IC * 9] : 0);
			float2 d0, d1, d2, d3, d4, d5;

			winograd_f4x3_d(d0.x, d1.x, d2.x, d3.x, d4.x, d5.x, x0, x1, x2, x3, x4, x5);
			Ds0[tx][ty] = float4{ d0.x, d1.x, d2.x, d3.x };
			Ds1[tx][ty] = float2{ d4.x, d5.x };

			winograd_f4x3_d(d0.y, d1.y, d2.y, d3.y, d4.y, d5.y, x4, x5, x6, x7, x8, x9);
			Ds0[tx][ty + STEP2] = float4{ d0.y, d1.y, d2.y, d3.y };
			Ds1[tx][ty + STEP2] = float2{ d4.y, d5.y };

			//load 2 group from W
			const int wic = oic + (ty & STEP_m1);//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float2 g0, g1, g2, g3, g4, g5;

			winograd_f4x3_g(g0.x, g1.x, g2.x, g3.x, g4.x, g5.x, w0.x, w1.x, w2.x);
			Gs0[ty][tx] = float4{ g0.x, g1.x, g2.x, g3.x };
			Gs1[ty][tx] = float2{ g4.x, g5.x };

			winograd_f4x3_g(g0.y, g1.y, g2.y, g3.y, g4.y, g5.y, w0.y, w1.y, w2.y);
			Gs0[ty][tx + STEP2] = float4{ g0.y, g1.y, g2.y, g3.y };
			Gs1[ty][tx + STEP2] = float2{ g4.y, g5.y };
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs0[ik][tx]; float2 g1 = Gs1[ik][tx];
				float4 g2 = Gs0[ik][tx + STEP2]; float2 g3 = Gs1[ik][tx + STEP2];
				float4 g4 = Gs0[ik + STEP][tx]; float2 g5 = Gs1[ik + STEP][tx];
				float4 g6 = Gs0[ik + STEP][tx + STEP2]; float2 g7 = Gs1[ik + STEP][tx + STEP2];

				float4 d0 = Ds0[ik][ty]; float2 d1 = Ds1[ik][ty];
				float4 d2 = Ds0[ik][ty + STEP2]; float2 d3 = Ds1[ik][ty + STEP2];
				float4 d4 = Ds0[ik + STEP][ty]; float2 d5 = Ds1[ik + STEP][ty];
				float4 d6 = Ds0[ik + STEP][ty + STEP2]; float2 d7 = Ds1[ik + STEP][ty + STEP2];

				m0.x += g0.x * d0.x; m0.y += g2.x * d0.x; m0.z += g4.x * d0.x; m0.w += g6.x * d0.x;
				m1.x += g0.y * d0.y; m1.y += g2.y * d0.y; m1.z += g4.y * d0.y; m1.w += g6.y * d0.y;
				m2.x += g0.z * d0.z; m2.y += g2.z * d0.z; m2.z += g4.z * d0.z; m2.w += g6.z * d0.z;
				m3.x += g0.w * d0.w; m3.y += g2.w * d0.w; m3.z += g4.w * d0.w; m3.w += g6.w * d0.w;
				m4.x += g1.x * d1.x; m4.y += g3.x * d1.x; m4.z += g5.x * d1.x; m4.w += g7.x * d1.x;
				m5.x += g1.y * d1.y; m5.y += g3.y * d1.y; m5.z += g5.y * d1.y; m5.w += g7.y * d1.y;

				m6.x += g0.x * d2.x; m6.y += g2.x * d2.x; m6.z += g4.x * d2.x; m6.w += g6.x * d2.x;
				m7.x += g0.y * d2.y; m7.y += g2.y * d2.y; m7.z += g4.y * d2.y; m7.w += g6.y * d2.y;
				m8.x += g0.z * d2.z; m8.y += g2.z * d2.z; m8.z += g4.z * d2.z; m8.w += g6.z * d2.z;
				m9.x += g0.w * d2.w; m9.y += g2.w * d2.w; m9.z += g4.w * d2.w; m9.w += g6.w * d2.w;
				m10.x += g1.x * d3.x; m10.y += g3.x * d3.x; m10.z += g5.x * d3.x; m10.w += g7.x * d3.x;
				m11.x += g1.y * d3.y; m11.y += g3.y * d3.y; m11.z += g5.y * d3.y; m11.w += g7.y * d3.y;

				m12.x += g0.x * d4.x; m12.y += g2.x * d4.x; m12.z += g4.x * d4.x; m12.w += g6.x * d4.x;
				m13.x += g0.y * d4.y; m13.y += g2.y * d4.y; m13.z += g4.y * d4.y; m13.w += g6.y * d4.y;
				m14.x += g0.z * d4.z; m14.y += g2.z * d4.z; m14.z += g4.z * d4.z; m14.w += g6.z * d4.z;
				m15.x += g0.w * d4.w; m15.y += g2.w * d4.w; m15.z += g4.w * d4.w; m15.w += g6.w * d4.w;
				m16.x += g1.x * d5.x; m16.y += g3.x * d5.x; m16.z += g5.x * d5.x; m16.w += g7.x * d5.x;
				m17.x += g1.y * d5.y; m17.y += g3.y * d5.y; m17.z += g5.y * d5.y; m17.w += g7.y * d5.y;

				m18.x += g0.x * d6.x; m18.y += g2.x * d6.x; m18.z += g4.x * d6.x; m18.w += g6.x * d6.x;
				m19.x += g0.y * d6.y; m19.y += g2.y * d6.y; m19.z += g4.y * d6.y; m19.w += g6.y * d6.y;
				m20.x += g0.z * d6.z; m20.y += g2.z * d6.z; m20.z += g4.z * d6.z; m20.w += g6.z * d6.z;
				m21.x += g0.w * d6.w; m21.y += g2.w * d6.w; m21.z += g4.w * d6.w; m21.w += g6.w * d6.w;
				m22.x += g1.x * d7.x; m22.y += g3.x * d7.x; m22.z += g5.x * d7.x; m22.w += g7.x * d7.x;
				m23.x += g1.y * d7.y; m23.y += g3.y * d7.y; m23.z += g5.y * d7.y; m23.w += g7.y * d7.y;
			}
			__syncthreads();
		}
	}

	float4 v0, v1, v2, v3;
	winograd_f4x3_v(v0.x, v1.x, v2.x, v3.x, m0.x, m1.x, m2.x, m3.x, m4.x, m5.x);
	winograd_f4x3_v(v0.y, v1.y, v2.y, v3.y, m0.y, m1.y, m2.y, m3.y, m4.y, m5.y);
	winograd_f4x3_v(v0.z, v1.z, v2.z, v3.z, m0.z, m1.z, m2.z, m3.z, m4.z, m5.z);
	winograd_f4x3_v(v0.w, v1.w, v2.w, v3.w, m0.w, m1.w, m2.w, m3.w, m4.w, m5.w);

	float4 v4, v5, v6, v7;
	winograd_f4x3_v(v4.x, v5.x, v6.x, v7.x, m6.x, m7.x, m8.x, m9.x, m10.x, m11.x);
	winograd_f4x3_v(v4.y, v5.y, v6.y, v7.y, m6.y, m7.y, m8.y, m9.y, m10.y, m11.y);
	winograd_f4x3_v(v4.z, v5.z, v6.z, v7.z, m6.z, m7.z, m8.z, m9.z, m10.z, m11.z);
	winograd_f4x3_v(v4.w, v5.w, v6.w, v7.w, m6.w, m7.w, m8.w, m9.w, m10.w, m11.w);

	float4 v8, v9, v10, v11;
	winograd_f4x3_v(v8.x, v9.x, v10.x, v11.x, m12.x, m13.x, m14.x, m15.x, m16.x, m17.x);
	winograd_f4x3_v(v8.y, v9.y, v10.y, v11.y, m12.y, m13.y, m14.y, m15.y, m16.y, m17.y);
	winograd_f4x3_v(v8.z, v9.z, v10.z, v11.z, m12.z, m13.z, m14.z, m15.z, m16.z, m17.z);
	winograd_f4x3_v(v8.w, v9.w, v10.w, v11.w, m12.w, m13.w, m14.w, m15.w, m16.w, m17.w);

	float4 v12, v13, v14, v15;
	winograd_f4x3_v(v12.x, v13.x, v14.x, v15.x, m18.x, m19.x, m20.x, m21.x, m22.x, m23.x);
	winograd_f4x3_v(v12.y, v13.y, v14.y, v15.y, m18.y, m19.y, m20.y, m21.y, m22.y, m23.y);
	winograd_f4x3_v(v12.z, v13.z, v14.z, v15.z, m18.z, m19.z, m20.z, m21.z, m22.z, m23.z);
	winograd_f4x3_v(v12.w, v13.w, v14.w, v15.w, m18.w, m19.w, m20.w, m21.w, m22.w, m23.w);

	const int  j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int  j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	const int  j8 = j7 + OC, j9 = j8 + OC, j10 = j9 + OC, j11 = j10 + OC;
	const int j12 = j11 + OC, j13 = j12 + OC, j14 = j13 + OC, j15 = j14 + OC;

	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;

	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;

	*(float4*)(Y + j8) = v8; *(float4*)(Y + j9) = v9;
	*(float4*)(Y + j10) = v10; *(float4*)(Y + j11) = v11;

	*(float4*)(Y + j12) = v12; *(float4*)(Y + j13) = v13;
	*(float4*)(Y + j14) = v14; *(float4*)(Y + j15) = v15;
}

#endif



//quarter shared memory
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_B2
#define XENO1_WINOGRAD_F4X3_B2

#define xeno_winograd_f4x3_b2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_b2<LB, (1<<LB>>2),(2<<LB>>2),(3<<LB>>2), (1<<LB>>2)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 2.40782 msec, Performace = 8026.9 GFlop/s
template<int LB, int STEP, int STEP2, int STEP3, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_b2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	__shared__ float4 Gs0[1 << LB][(1 << LB) + 1];
	__shared__ float2 Gs1[1 << LB][(1 << LB) + 1];
	__shared__ float4 Ds0[1 << LB][(1 << LB) + 1];
	__shared__ float2 Ds1[1 << LB][(1 << LB) + 1];

	float4  m0 = F32_4_0, m1 = F32_4_0, m2 = F32_4_0;//0-3
	float4  m3 = F32_4_0, m4 = F32_4_0, m5 = F32_4_0;

	float4  m6 = F32_4_0, m7 = F32_4_0, m8 = F32_4_0;//4-8
	float4  m9 = F32_4_0, m10 = F32_4_0, m11 = F32_4_0;

	float4 m12 = F32_4_0, m13 = F32_4_0, m14 = F32_4_0;//8-11
	float4 m15 = F32_4_0, m16 = F32_4_0, m17 = F32_4_0;

	float4 m18 = F32_4_0, m19 = F32_4_0, m20 = F32_4_0;//12-15
	float4 m21 = F32_4_0, m22 = F32_4_0, m23 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + (ty / STEP);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 4) + j_index;
	int tj0 = j0 + ((tx / STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool lx0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool lx1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool lx2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool lx3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool lx4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool lx5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		for (int oic = 0; oic < IC; oic += STEP)
		{
			//load 2 group from X
			const int xic = oic + (tx & STEP_m1);//with the same ty
			const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);//0
			float x1 = (lx1 ? X[xoffset + IC] : 0);//1
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
			Ds0[tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[tx][ty] = float2{ d4, d5 };

			//load 2 group from W
			const int wic = oic + (ty & STEP_m1);//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float w0 = CW[woffset];
			float w1 = CW[woffset + Wstride];
			float w2 = CW[woffset + (Wstride << 1)];
			float g0, g1, g2, g3, g4, g5;
			winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
			Gs0[ty][tx] = float4{ g0, g1, g2, g3 };
			Gs1[ty][tx] = float2{ g4, g5 };
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs0[ik][tx]; float2 g1 = Gs1[ik][tx];
				float4 g2 = Gs0[ik + STEP][tx]; float2 g3 = Gs1[ik + STEP][tx];
				float4 g4 = Gs0[ik + STEP2][tx]; float2 g5 = Gs1[ik + STEP2][tx];
				float4 g6 = Gs0[ik + STEP3][tx]; float2 g7 = Gs1[ik + STEP3][tx];

				float4 d0 = Ds0[ik][ty]; float2 d1 = Ds1[ik][ty];
				float4 d2 = Ds0[ik + STEP][ty]; float2 d3 = Ds1[ik + STEP][ty];
				float4 d4 = Ds0[ik + STEP2][ty]; float2 d5 = Ds1[ik + STEP2][ty];
				float4 d6 = Ds0[ik + STEP3][ty]; float2 d7 = Ds1[ik + STEP3][ty];

				m0.x += g0.x * d0.x; m0.y += g2.x * d0.x; m0.z += g4.x * d0.x; m0.w += g6.x * d0.x;
				m1.x += g0.y * d0.y; m1.y += g2.y * d0.y; m1.z += g4.y * d0.y; m1.w += g6.y * d0.y;
				m2.x += g0.z * d0.z; m2.y += g2.z * d0.z; m2.z += g4.z * d0.z; m2.w += g6.z * d0.z;
				m3.x += g0.w * d0.w; m3.y += g2.w * d0.w; m3.z += g4.w * d0.w; m3.w += g6.w * d0.w;
				m4.x += g1.x * d1.x; m4.y += g3.x * d1.x; m4.z += g5.x * d1.x; m4.w += g7.x * d1.x;
				m5.x += g1.y * d1.y; m5.y += g3.y * d1.y; m5.z += g5.y * d1.y; m5.w += g7.y * d1.y;

				m6.x += g0.x * d2.x; m6.y += g2.x * d2.x; m6.z += g4.x * d2.x; m6.w += g6.x * d2.x;
				m7.x += g0.y * d2.y; m7.y += g2.y * d2.y; m7.z += g4.y * d2.y; m7.w += g6.y * d2.y;
				m8.x += g0.z * d2.z; m8.y += g2.z * d2.z; m8.z += g4.z * d2.z; m8.w += g6.z * d2.z;
				m9.x += g0.w * d2.w; m9.y += g2.w * d2.w; m9.z += g4.w * d2.w; m9.w += g6.w * d2.w;
				m10.x += g1.x * d3.x; m10.y += g3.x * d3.x; m10.z += g5.x * d3.x; m10.w += g7.x * d3.x;
				m11.x += g1.y * d3.y; m11.y += g3.y * d3.y; m11.z += g5.y * d3.y; m11.w += g7.y * d3.y;

				m12.x += g0.x * d4.x; m12.y += g2.x * d4.x; m12.z += g4.x * d4.x; m12.w += g6.x * d4.x;
				m13.x += g0.y * d4.y; m13.y += g2.y * d4.y; m13.z += g4.y * d4.y; m13.w += g6.y * d4.y;
				m14.x += g0.z * d4.z; m14.y += g2.z * d4.z; m14.z += g4.z * d4.z; m14.w += g6.z * d4.z;
				m15.x += g0.w * d4.w; m15.y += g2.w * d4.w; m15.z += g4.w * d4.w; m15.w += g6.w * d4.w;
				m16.x += g1.x * d5.x; m16.y += g3.x * d5.x; m16.z += g5.x * d5.x; m16.w += g7.x * d5.x;
				m17.x += g1.y * d5.y; m17.y += g3.y * d5.y; m17.z += g5.y * d5.y; m17.w += g7.y * d5.y;

				m18.x += g0.x * d6.x; m18.y += g2.x * d6.x; m18.z += g4.x * d6.x; m18.w += g6.x * d6.x;
				m19.x += g0.y * d6.y; m19.y += g2.y * d6.y; m19.z += g4.y * d6.y; m19.w += g6.y * d6.y;
				m20.x += g0.z * d6.z; m20.y += g2.z * d6.z; m20.z += g4.z * d6.z; m20.w += g6.z * d6.z;
				m21.x += g0.w * d6.w; m21.y += g2.w * d6.w; m21.z += g4.w * d6.w; m21.w += g6.w * d6.w;
				m22.x += g1.x * d7.x; m22.y += g3.x * d7.x; m22.z += g5.x * d7.x; m22.w += g7.x * d7.x;
				m23.x += g1.y * d7.y; m23.y += g3.y * d7.y; m23.z += g5.y * d7.y; m23.w += g7.y * d7.y;
			}
			__syncthreads();
		}
	}

	float4 v0, v1, v2, v3;
	winograd_f4x3_v(v0.x, v1.x, v2.x, v3.x, m0.x, m1.x, m2.x, m3.x, m4.x, m5.x);
	winograd_f4x3_v(v0.y, v1.y, v2.y, v3.y, m0.y, m1.y, m2.y, m3.y, m4.y, m5.y);
	winograd_f4x3_v(v0.z, v1.z, v2.z, v3.z, m0.z, m1.z, m2.z, m3.z, m4.z, m5.z);
	winograd_f4x3_v(v0.w, v1.w, v2.w, v3.w, m0.w, m1.w, m2.w, m3.w, m4.w, m5.w);

	float4 v4, v5, v6, v7;
	winograd_f4x3_v(v4.x, v5.x, v6.x, v7.x, m6.x, m7.x, m8.x, m9.x, m10.x, m11.x);
	winograd_f4x3_v(v4.y, v5.y, v6.y, v7.y, m6.y, m7.y, m8.y, m9.y, m10.y, m11.y);
	winograd_f4x3_v(v4.z, v5.z, v6.z, v7.z, m6.z, m7.z, m8.z, m9.z, m10.z, m11.z);
	winograd_f4x3_v(v4.w, v5.w, v6.w, v7.w, m6.w, m7.w, m8.w, m9.w, m10.w, m11.w);

	float4 v8, v9, v10, v11;
	winograd_f4x3_v(v8.x, v9.x, v10.x, v11.x, m12.x, m13.x, m14.x, m15.x, m16.x, m17.x);
	winograd_f4x3_v(v8.y, v9.y, v10.y, v11.y, m12.y, m13.y, m14.y, m15.y, m16.y, m17.y);
	winograd_f4x3_v(v8.z, v9.z, v10.z, v11.z, m12.z, m13.z, m14.z, m15.z, m16.z, m17.z);
	winograd_f4x3_v(v8.w, v9.w, v10.w, v11.w, m12.w, m13.w, m14.w, m15.w, m16.w, m17.w);

	float4 v12, v13, v14, v15;
	winograd_f4x3_v(v12.x, v13.x, v14.x, v15.x, m18.x, m19.x, m20.x, m21.x, m22.x, m23.x);
	winograd_f4x3_v(v12.y, v13.y, v14.y, v15.y, m18.y, m19.y, m20.y, m21.y, m22.y, m23.y);
	winograd_f4x3_v(v12.z, v13.z, v14.z, v15.z, m18.z, m19.z, m20.z, m21.z, m22.z, m23.z);
	winograd_f4x3_v(v12.w, v13.w, v14.w, v15.w, m18.w, m19.w, m20.w, m21.w, m22.w, m23.w);

	const int  j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int  j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	const int  j8 = j7 + OC, j9 = j8 + OC, j10 = j9 + OC, j11 = j10 + OC;
	const int j12 = j11 + OC, j13 = j12 + OC, j14 = j13 + OC, j15 = j14 + OC;

	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;

	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;

	*(float4*)(Y + j8) = v8; *(float4*)(Y + j9) = v9;
	*(float4*)(Y + j10) = v10; *(float4*)(Y + j11) = v11;

	*(float4*)(Y + j12) = v12; *(float4*)(Y + j13) = v13;
	*(float4*)(Y + j14) = v14; *(float4*)(Y + j15) = v15;
}

#endif


//double buffer
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_B3
#define XENO1_WINOGRAD_F4X3_B3

#define xeno_winograd_f4x3_b3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_b3<LB, (1<<LB>>2),(2<<LB>>2),(3<<LB>>2), (1<<LB>>2)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 1.93712 msec, Performace = 9977.34 GFlop/s
template<int LB, int STEP, int STEP2, int STEP3, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_b3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Gs0[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Gs1[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ds0[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ds1[2][1 << LB][(1 << LB) + 1];

	float4  m0 = F32_4_0, m1 = F32_4_0, m2 = F32_4_0;//0-3
	float4  m3 = F32_4_0, m4 = F32_4_0, m5 = F32_4_0;

	float4  m6 = F32_4_0, m7 = F32_4_0, m8 = F32_4_0;//4-8
	float4  m9 = F32_4_0, m10 = F32_4_0, m11 = F32_4_0;

	float4 m12 = F32_4_0, m13 = F32_4_0, m14 = F32_4_0;//8-11
	float4 m15 = F32_4_0, m16 = F32_4_0, m17 = F32_4_0;

	float4 m18 = F32_4_0, m19 = F32_4_0, m20 = F32_4_0;//12-15
	float4 m21 = F32_4_0, m22 = F32_4_0, m23 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + (ty / STEP);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 4) + j_index;
	int tj0 = j0 + ((tx / STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool lx0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool lx1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool lx2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool lx3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool lx4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool lx5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		//load 2 group from X
		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
		float x0 = (lx0 ? X[xoffset] : 0);//0
		float x1 = (lx1 ? X[xoffset + IC] : 0);//1
		float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
		float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
		float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
		float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
		float d0, d1, d2, d3, d4, d5;
		winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
		Ds0[buf][tx][ty] = float4{ d0, d1, d2, d3 };
		Ds1[buf][tx][ty] = float2{ d4, d5 };

		//load 2 group from W
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
		float w0 = CW[woffset];
		float w1 = CW[woffset + Wstride];
		float w2 = CW[woffset + (Wstride << 1)];
		float g0, g1, g2, g3, g4, g5;
		winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
		Gs0[buf][ty][tx] = float4{ g0, g1, g2, g3 };
		Gs1[buf][ty][tx] = float2{ g4, g5 };
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs0[buf][ik        ][tx]; float2 g1 = Gs1[buf][ik        ][tx];
				float4 g2 = Gs0[buf][ik + STEP ][tx]; float2 g3 = Gs1[buf][ik + STEP ][tx];
				float4 g4 = Gs0[buf][ik + STEP2][tx]; float2 g5 = Gs1[buf][ik + STEP2][tx];
				float4 g6 = Gs0[buf][ik + STEP3][tx]; float2 g7 = Gs1[buf][ik + STEP3][tx];

				float4 d0 = Ds0[buf][ik        ][ty]; float2 d1 = Ds1[buf][ik        ][ty];
				float4 d2 = Ds0[buf][ik + STEP ][ty]; float2 d3 = Ds1[buf][ik + STEP ][ty];
				float4 d4 = Ds0[buf][ik + STEP2][ty]; float2 d5 = Ds1[buf][ik + STEP2][ty];
				float4 d6 = Ds0[buf][ik + STEP3][ty]; float2 d7 = Ds1[buf][ik + STEP3][ty];

				m0.x += g0.x * d0.x; m0.y += g2.x * d0.x; m0.z += g4.x * d0.x; m0.w += g6.x * d0.x;
				m1.x += g0.y * d0.y; m1.y += g2.y * d0.y; m1.z += g4.y * d0.y; m1.w += g6.y * d0.y;
				m2.x += g0.z * d0.z; m2.y += g2.z * d0.z; m2.z += g4.z * d0.z; m2.w += g6.z * d0.z;
				m3.x += g0.w * d0.w; m3.y += g2.w * d0.w; m3.z += g4.w * d0.w; m3.w += g6.w * d0.w;
				m4.x += g1.x * d1.x; m4.y += g3.x * d1.x; m4.z += g5.x * d1.x; m4.w += g7.x * d1.x;
				m5.x += g1.y * d1.y; m5.y += g3.y * d1.y; m5.z += g5.y * d1.y; m5.w += g7.y * d1.y;

				m6.x += g0.x * d2.x; m6.y += g2.x * d2.x; m6.z += g4.x * d2.x; m6.w += g6.x * d2.x;
				m7.x += g0.y * d2.y; m7.y += g2.y * d2.y; m7.z += g4.y * d2.y; m7.w += g6.y * d2.y;
				m8.x += g0.z * d2.z; m8.y += g2.z * d2.z; m8.z += g4.z * d2.z; m8.w += g6.z * d2.z;
				m9.x += g0.w * d2.w; m9.y += g2.w * d2.w; m9.z += g4.w * d2.w; m9.w += g6.w * d2.w;
				m10.x += g1.x * d3.x; m10.y += g3.x * d3.x; m10.z += g5.x * d3.x; m10.w += g7.x * d3.x;
				m11.x += g1.y * d3.y; m11.y += g3.y * d3.y; m11.z += g5.y * d3.y; m11.w += g7.y * d3.y;

				m12.x += g0.x * d4.x; m12.y += g2.x * d4.x; m12.z += g4.x * d4.x; m12.w += g6.x * d4.x;
				m13.x += g0.y * d4.y; m13.y += g2.y * d4.y; m13.z += g4.y * d4.y; m13.w += g6.y * d4.y;
				m14.x += g0.z * d4.z; m14.y += g2.z * d4.z; m14.z += g4.z * d4.z; m14.w += g6.z * d4.z;
				m15.x += g0.w * d4.w; m15.y += g2.w * d4.w; m15.z += g4.w * d4.w; m15.w += g6.w * d4.w;
				m16.x += g1.x * d5.x; m16.y += g3.x * d5.x; m16.z += g5.x * d5.x; m16.w += g7.x * d5.x;
				m17.x += g1.y * d5.y; m17.y += g3.y * d5.y; m17.z += g5.y * d5.y; m17.w += g7.y * d5.y;

				m18.x += g0.x * d6.x; m18.y += g2.x * d6.x; m18.z += g4.x * d6.x; m18.w += g6.x * d6.x;
				m19.x += g0.y * d6.y; m19.y += g2.y * d6.y; m19.z += g4.y * d6.y; m19.w += g6.y * d6.y;
				m20.x += g0.z * d6.z; m20.y += g2.z * d6.z; m20.z += g4.z * d6.z; m20.w += g6.z * d6.z;
				m21.x += g0.w * d6.w; m21.y += g2.w * d6.w; m21.z += g4.w * d6.w; m21.w += g6.w * d6.w;
				m22.x += g1.x * d7.x; m22.y += g3.x * d7.x; m22.z += g5.x * d7.x; m22.w += g7.x * d7.x;
				m23.x += g1.y * d7.y; m23.y += g3.y * d7.y; m23.z += g5.y * d7.y; m23.w += g7.y * d7.y;
			}
			buf ^= 1;

			//load 2 group from X
			const int xic = oic + (tx & STEP_m1);//with the same ty
			const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);//0
			float x1 = (lx1 ? X[xoffset + IC] : 0);//1
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
			Ds0[buf][tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[buf][tx][ty] = float2{ d4, d5 };

			//load 2 group from W
			const int wic = oic + (ty & STEP_m1);//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float w0 = CW[woffset];
			float w1 = CW[woffset + Wstride];
			float w2 = CW[woffset + (Wstride << 1)];
			float g0, g1, g2, g3, g4, g5;
			winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
			Gs0[buf][ty][tx] = float4{ g0, g1, g2, g3 };
			Gs1[buf][ty][tx] = float2{ g4, g5 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs0[buf][ik][tx]; float2 g1 = Gs1[buf][ik][tx];
			float4 g2 = Gs0[buf][ik + STEP][tx]; float2 g3 = Gs1[buf][ik + STEP][tx];
			float4 g4 = Gs0[buf][ik + STEP2][tx]; float2 g5 = Gs1[buf][ik + STEP2][tx];
			float4 g6 = Gs0[buf][ik + STEP3][tx]; float2 g7 = Gs1[buf][ik + STEP3][tx];

			float4 d0 = Ds0[buf][ik][ty]; float2 d1 = Ds1[buf][ik][ty];
			float4 d2 = Ds0[buf][ik + STEP][ty]; float2 d3 = Ds1[buf][ik + STEP][ty];
			float4 d4 = Ds0[buf][ik + STEP2][ty]; float2 d5 = Ds1[buf][ik + STEP2][ty];
			float4 d6 = Ds0[buf][ik + STEP3][ty]; float2 d7 = Ds1[buf][ik + STEP3][ty];

			m0.x += g0.x * d0.x; m0.y += g2.x * d0.x; m0.z += g4.x * d0.x; m0.w += g6.x * d0.x;
			m1.x += g0.y * d0.y; m1.y += g2.y * d0.y; m1.z += g4.y * d0.y; m1.w += g6.y * d0.y;
			m2.x += g0.z * d0.z; m2.y += g2.z * d0.z; m2.z += g4.z * d0.z; m2.w += g6.z * d0.z;
			m3.x += g0.w * d0.w; m3.y += g2.w * d0.w; m3.z += g4.w * d0.w; m3.w += g6.w * d0.w;
			m4.x += g1.x * d1.x; m4.y += g3.x * d1.x; m4.z += g5.x * d1.x; m4.w += g7.x * d1.x;
			m5.x += g1.y * d1.y; m5.y += g3.y * d1.y; m5.z += g5.y * d1.y; m5.w += g7.y * d1.y;

			m6.x += g0.x * d2.x; m6.y += g2.x * d2.x; m6.z += g4.x * d2.x; m6.w += g6.x * d2.x;
			m7.x += g0.y * d2.y; m7.y += g2.y * d2.y; m7.z += g4.y * d2.y; m7.w += g6.y * d2.y;
			m8.x += g0.z * d2.z; m8.y += g2.z * d2.z; m8.z += g4.z * d2.z; m8.w += g6.z * d2.z;
			m9.x += g0.w * d2.w; m9.y += g2.w * d2.w; m9.z += g4.w * d2.w; m9.w += g6.w * d2.w;
			m10.x += g1.x * d3.x; m10.y += g3.x * d3.x; m10.z += g5.x * d3.x; m10.w += g7.x * d3.x;
			m11.x += g1.y * d3.y; m11.y += g3.y * d3.y; m11.z += g5.y * d3.y; m11.w += g7.y * d3.y;

			m12.x += g0.x * d4.x; m12.y += g2.x * d4.x; m12.z += g4.x * d4.x; m12.w += g6.x * d4.x;
			m13.x += g0.y * d4.y; m13.y += g2.y * d4.y; m13.z += g4.y * d4.y; m13.w += g6.y * d4.y;
			m14.x += g0.z * d4.z; m14.y += g2.z * d4.z; m14.z += g4.z * d4.z; m14.w += g6.z * d4.z;
			m15.x += g0.w * d4.w; m15.y += g2.w * d4.w; m15.z += g4.w * d4.w; m15.w += g6.w * d4.w;
			m16.x += g1.x * d5.x; m16.y += g3.x * d5.x; m16.z += g5.x * d5.x; m16.w += g7.x * d5.x;
			m17.x += g1.y * d5.y; m17.y += g3.y * d5.y; m17.z += g5.y * d5.y; m17.w += g7.y * d5.y;

			m18.x += g0.x * d6.x; m18.y += g2.x * d6.x; m18.z += g4.x * d6.x; m18.w += g6.x * d6.x;
			m19.x += g0.y * d6.y; m19.y += g2.y * d6.y; m19.z += g4.y * d6.y; m19.w += g6.y * d6.y;
			m20.x += g0.z * d6.z; m20.y += g2.z * d6.z; m20.z += g4.z * d6.z; m20.w += g6.z * d6.z;
			m21.x += g0.w * d6.w; m21.y += g2.w * d6.w; m21.z += g4.w * d6.w; m21.w += g6.w * d6.w;
			m22.x += g1.x * d7.x; m22.y += g3.x * d7.x; m22.z += g5.x * d7.x; m22.w += g7.x * d7.x;
			m23.x += g1.y * d7.y; m23.y += g3.y * d7.y; m23.z += g5.y * d7.y; m23.w += g7.y * d7.y;
		}
		buf ^= 1;
	}

	float4 v0, v1, v2, v3;
	winograd_f4x3_v(v0.x, v1.x, v2.x, v3.x, m0.x, m1.x, m2.x, m3.x, m4.x, m5.x);
	winograd_f4x3_v(v0.y, v1.y, v2.y, v3.y, m0.y, m1.y, m2.y, m3.y, m4.y, m5.y);
	winograd_f4x3_v(v0.z, v1.z, v2.z, v3.z, m0.z, m1.z, m2.z, m3.z, m4.z, m5.z);
	winograd_f4x3_v(v0.w, v1.w, v2.w, v3.w, m0.w, m1.w, m2.w, m3.w, m4.w, m5.w);

	float4 v4, v5, v6, v7;
	winograd_f4x3_v(v4.x, v5.x, v6.x, v7.x, m6.x, m7.x, m8.x, m9.x, m10.x, m11.x);
	winograd_f4x3_v(v4.y, v5.y, v6.y, v7.y, m6.y, m7.y, m8.y, m9.y, m10.y, m11.y);
	winograd_f4x3_v(v4.z, v5.z, v6.z, v7.z, m6.z, m7.z, m8.z, m9.z, m10.z, m11.z);
	winograd_f4x3_v(v4.w, v5.w, v6.w, v7.w, m6.w, m7.w, m8.w, m9.w, m10.w, m11.w);

	float4 v8, v9, v10, v11;
	winograd_f4x3_v(v8.x, v9.x, v10.x, v11.x, m12.x, m13.x, m14.x, m15.x, m16.x, m17.x);
	winograd_f4x3_v(v8.y, v9.y, v10.y, v11.y, m12.y, m13.y, m14.y, m15.y, m16.y, m17.y);
	winograd_f4x3_v(v8.z, v9.z, v10.z, v11.z, m12.z, m13.z, m14.z, m15.z, m16.z, m17.z);
	winograd_f4x3_v(v8.w, v9.w, v10.w, v11.w, m12.w, m13.w, m14.w, m15.w, m16.w, m17.w);

	float4 v12, v13, v14, v15;
	winograd_f4x3_v(v12.x, v13.x, v14.x, v15.x, m18.x, m19.x, m20.x, m21.x, m22.x, m23.x);
	winograd_f4x3_v(v12.y, v13.y, v14.y, v15.y, m18.y, m19.y, m20.y, m21.y, m22.y, m23.y);
	winograd_f4x3_v(v12.z, v13.z, v14.z, v15.z, m18.z, m19.z, m20.z, m21.z, m22.z, m23.z);
	winograd_f4x3_v(v12.w, v13.w, v14.w, v15.w, m18.w, m19.w, m20.w, m21.w, m22.w, m23.w);

	const int  j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int  j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	const int  j8 = j7 + OC, j9 = j8 + OC, j10 = j9 + OC, j11 = j10 + OC;
	const int j12 = j11 + OC, j13 = j12 + OC, j14 = j13 + OC, j15 = j14 + OC;

	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;

	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;

	*(float4*)(Y + j8) = v8; *(float4*)(Y + j9) = v9;
	*(float4*)(Y + j10) = v10; *(float4*)(Y + j11) = v11;

	*(float4*)(Y + j12) = v12; *(float4*)(Y + j13) = v13;
	*(float4*)(Y + j14) = v14; *(float4*)(Y + j15) = v15;
}

#endif


//double buffer
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_B4
#define XENO1_WINOGRAD_F4X3_B4

#define xeno_winograd_f4x3_b4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_b4<LB, (1<<LB>>2),(2<<LB>>2),(3<<LB>>2), (1<<LB>>2)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 1.92068 msec, Performace = 10062.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP3, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_b4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Gs0[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Gs1[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ds0[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ds1[2][1 << LB][(1 << LB) + 1];

	float4  v0 = F32_4_0, m1 = F32_4_0, m2 = F32_4_0;//0-3
	float4  m3 = F32_4_0, m4 = F32_4_0, m5 = F32_4_0;

	float4  v4 = F32_4_0, m7 = F32_4_0, m8 = F32_4_0;//4-8
	float4  m9 = F32_4_0, m10 = F32_4_0, m11 = F32_4_0;

	float4  v8 = F32_4_0, m13 = F32_4_0, m14 = F32_4_0;//8-11
	float4 m15 = F32_4_0, m16 = F32_4_0, m17 = F32_4_0;

	float4 v12 = F32_4_0, m19 = F32_4_0, m20 = F32_4_0;//12-15
	float4 m21 = F32_4_0, m22 = F32_4_0, m23 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + (ty / STEP);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 4) + j_index;
	int tj0 = j0 + ((tx / STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool lx0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool lx1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool lx2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool lx3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool lx4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool lx5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		//load 2 group from X
		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
		float x0 = (lx0 ? X[xoffset] : 0);//0
		float x1 = (lx1 ? X[xoffset + IC] : 0);//1
		float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
		float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
		float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
		float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
		float d0, d1, d2, d3, d4, d5;
		winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
		Ds0[buf][tx][ty] = float4{ d0, d1, d2, d3 };
		Ds1[buf][tx][ty] = float2{ d4, d5 };

		//load 2 group from W
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
		float w0 = CW[woffset];
		float w1 = CW[woffset + Wstride];
		float w2 = CW[woffset + (Wstride << 1)];
		float g0, g1, g2, g3, g4, g5;
		winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
		Gs0[buf][ty][tx] = float4{ g0, g1, g2, g3 };
		Gs1[buf][ty][tx] = float2{ g4, g5 };
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs0[buf][ik][tx]; float2 g1 = Gs1[buf][ik][tx];
				float4 g2 = Gs0[buf][ik + STEP][tx]; float2 g3 = Gs1[buf][ik + STEP][tx];

				float4 g4 = Gs0[buf][ik + STEP2][tx]; float2 g5 = Gs1[buf][ik + STEP2][tx];
				float4 g6 = Gs0[buf][ik + STEP3][tx]; float2 g7 = Gs1[buf][ik + STEP3][tx];

				float4 d0 = Ds0[buf][ik][ty]; float2 d1 = Ds1[buf][ik][ty];
				float4 d2 = Ds0[buf][ik + STEP][ty]; float2 d3 = Ds1[buf][ik + STEP][ty];
				float4 d4 = Ds0[buf][ik + STEP2][ty]; float2 d5 = Ds1[buf][ik + STEP2][ty];
				float4 d6 = Ds0[buf][ik + STEP3][ty]; float2 d7 = Ds1[buf][ik + STEP3][ty];

				v0.x += g0.x * d0.x; v0.y += g2.x * d0.x; v0.z += g4.x * d0.x; v0.w += g6.x * d0.x;
				m1.x += g0.y * d0.y; m1.y += g2.y * d0.y; m1.z += g4.y * d0.y; m1.w += g6.y * d0.y;
				m2.x += g0.z * d0.z; m2.y += g2.z * d0.z; m2.z += g4.z * d0.z; m2.w += g6.z * d0.z;
				m3.x += g0.w * d0.w; m3.y += g2.w * d0.w; m3.z += g4.w * d0.w; m3.w += g6.w * d0.w;
				m4.x += g1.x * d1.x; m4.y += g3.x * d1.x; m4.z += g5.x * d1.x; m4.w += g7.x * d1.x;
				m5.x += g1.y * d1.y; m5.y += g3.y * d1.y; m5.z += g5.y * d1.y; m5.w += g7.y * d1.y;

				v4.x += g0.x * d2.x; v4.y += g2.x * d2.x; v4.z += g4.x * d2.x; v4.w += g6.x * d2.x;
				m7.x += g0.y * d2.y; m7.y += g2.y * d2.y; m7.z += g4.y * d2.y; m7.w += g6.y * d2.y;
				m8.x += g0.z * d2.z; m8.y += g2.z * d2.z; m8.z += g4.z * d2.z; m8.w += g6.z * d2.z;
				m9.x += g0.w * d2.w; m9.y += g2.w * d2.w; m9.z += g4.w * d2.w; m9.w += g6.w * d2.w;
				m10.x += g1.x * d3.x; m10.y += g3.x * d3.x; m10.z += g5.x * d3.x; m10.w += g7.x * d3.x;
				m11.x += g1.y * d3.y; m11.y += g3.y * d3.y; m11.z += g5.y * d3.y; m11.w += g7.y * d3.y;

				v8.x += g0.x * d4.x; v8.y += g2.x * d4.x; v8.z += g4.x * d4.x; v8.w += g6.x * d4.x;
				m13.x += g0.y * d4.y; m13.y += g2.y * d4.y; m13.z += g4.y * d4.y; m13.w += g6.y * d4.y;
				m14.x += g0.z * d4.z; m14.y += g2.z * d4.z; m14.z += g4.z * d4.z; m14.w += g6.z * d4.z;
				m15.x += g0.w * d4.w; m15.y += g2.w * d4.w; m15.z += g4.w * d4.w; m15.w += g6.w * d4.w;
				m16.x += g1.x * d5.x; m16.y += g3.x * d5.x; m16.z += g5.x * d5.x; m16.w += g7.x * d5.x;
				m17.x += g1.y * d5.y; m17.y += g3.y * d5.y; m17.z += g5.y * d5.y; m17.w += g7.y * d5.y;

				v12.x += g0.x * d6.x; v12.y += g2.x * d6.x; v12.z += g4.x * d6.x; v12.w += g6.x * d6.x;
				m19.x += g0.y * d6.y; m19.y += g2.y * d6.y; m19.z += g4.y * d6.y; m19.w += g6.y * d6.y;
				m20.x += g0.z * d6.z; m20.y += g2.z * d6.z; m20.z += g4.z * d6.z; m20.w += g6.z * d6.z;
				m21.x += g0.w * d6.w; m21.y += g2.w * d6.w; m21.z += g4.w * d6.w; m21.w += g6.w * d6.w;
				m22.x += g1.x * d7.x; m22.y += g3.x * d7.x; m22.z += g5.x * d7.x; m22.w += g7.x * d7.x;
				m23.x += g1.y * d7.y; m23.y += g3.y * d7.y; m23.z += g5.y * d7.y; m23.w += g7.y * d7.y;
			}
			buf ^= 1;

			//load 2 group from X
			const int xic = oic + (tx & STEP_m1);//with the same ty
			const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);//0
			float x1 = (lx1 ? X[xoffset + IC] : 0);//1
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
			Ds0[buf][tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[buf][tx][ty] = float2{ d4, d5 };

			//load 2 group from W
			const int wic = oic + (ty & STEP_m1);//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float w0 = CW[woffset];
			float w1 = CW[woffset + Wstride];
			float w2 = CW[woffset + (Wstride << 1)];
			float g0, g1, g2, g3, g4, g5;
			winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
			Gs0[buf][ty][tx] = float4{ g0, g1, g2, g3 };
			Gs1[buf][ty][tx] = float2{ g4, g5 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs0[buf][ik][tx]; float2 g1 = Gs1[buf][ik][tx];
			float4 g2 = Gs0[buf][ik + STEP][tx]; float2 g3 = Gs1[buf][ik + STEP][tx];
			float4 g4 = Gs0[buf][ik + STEP2][tx]; float2 g5 = Gs1[buf][ik + STEP2][tx];
			float4 g6 = Gs0[buf][ik + STEP3][tx]; float2 g7 = Gs1[buf][ik + STEP3][tx];

			float4 d0 = Ds0[buf][ik][ty]; float2 d1 = Ds1[buf][ik][ty];
			float4 d2 = Ds0[buf][ik + STEP][ty]; float2 d3 = Ds1[buf][ik + STEP][ty];
			float4 d4 = Ds0[buf][ik + STEP2][ty]; float2 d5 = Ds1[buf][ik + STEP2][ty];
			float4 d6 = Ds0[buf][ik + STEP3][ty]; float2 d7 = Ds1[buf][ik + STEP3][ty];

			v0.x += g0.x * d0.x; v0.y += g2.x * d0.x; v0.z += g4.x * d0.x; v0.w += g6.x * d0.x;
			m1.x += g0.y * d0.y; m1.y += g2.y * d0.y; m1.z += g4.y * d0.y; m1.w += g6.y * d0.y;
			m2.x += g0.z * d0.z; m2.y += g2.z * d0.z; m2.z += g4.z * d0.z; m2.w += g6.z * d0.z;
			m3.x += g0.w * d0.w; m3.y += g2.w * d0.w; m3.z += g4.w * d0.w; m3.w += g6.w * d0.w;
			m4.x += g1.x * d1.x; m4.y += g3.x * d1.x; m4.z += g5.x * d1.x; m4.w += g7.x * d1.x;
			m5.x += g1.y * d1.y; m5.y += g3.y * d1.y; m5.z += g5.y * d1.y; m5.w += g7.y * d1.y;

			v4.x += g0.x * d2.x; v4.y += g2.x * d2.x; v4.z += g4.x * d2.x; v4.w += g6.x * d2.x;
			m7.x += g0.y * d2.y; m7.y += g2.y * d2.y; m7.z += g4.y * d2.y; m7.w += g6.y * d2.y;
			m8.x += g0.z * d2.z; m8.y += g2.z * d2.z; m8.z += g4.z * d2.z; m8.w += g6.z * d2.z;
			m9.x += g0.w * d2.w; m9.y += g2.w * d2.w; m9.z += g4.w * d2.w; m9.w += g6.w * d2.w;
			m10.x += g1.x * d3.x; m10.y += g3.x * d3.x; m10.z += g5.x * d3.x; m10.w += g7.x * d3.x;
			m11.x += g1.y * d3.y; m11.y += g3.y * d3.y; m11.z += g5.y * d3.y; m11.w += g7.y * d3.y;

			v8.x += g0.x * d4.x; v8.y += g2.x * d4.x; v8.z += g4.x * d4.x; v8.w += g6.x * d4.x;
			m13.x += g0.y * d4.y; m13.y += g2.y * d4.y; m13.z += g4.y * d4.y; m13.w += g6.y * d4.y;
			m14.x += g0.z * d4.z; m14.y += g2.z * d4.z; m14.z += g4.z * d4.z; m14.w += g6.z * d4.z;
			m15.x += g0.w * d4.w; m15.y += g2.w * d4.w; m15.z += g4.w * d4.w; m15.w += g6.w * d4.w;
			m16.x += g1.x * d5.x; m16.y += g3.x * d5.x; m16.z += g5.x * d5.x; m16.w += g7.x * d5.x;
			m17.x += g1.y * d5.y; m17.y += g3.y * d5.y; m17.z += g5.y * d5.y; m17.w += g7.y * d5.y;

			v12.x += g0.x * d6.x; v12.y += g2.x * d6.x; v12.z += g4.x * d6.x; v12.w += g6.x * d6.x;
			m19.x += g0.y * d6.y; m19.y += g2.y * d6.y; m19.z += g4.y * d6.y; m19.w += g6.y * d6.y;
			m20.x += g0.z * d6.z; m20.y += g2.z * d6.z; m20.z += g4.z * d6.z; m20.w += g6.z * d6.z;
			m21.x += g0.w * d6.w; m21.y += g2.w * d6.w; m21.z += g4.w * d6.w; m21.w += g6.w * d6.w;
			m22.x += g1.x * d7.x; m22.y += g3.x * d7.x; m22.z += g5.x * d7.x; m22.w += g7.x * d7.x;
			m23.x += g1.y * d7.y; m23.y += g3.y * d7.y; m23.z += g5.y * d7.y; m23.w += g7.y * d7.y;
		}
		buf ^= 1;
	}

	float4 v1, v2, v3;
	winograd_f4x3_v2(v0.x, v1.x, v2.x, v3.x, m1.x, m2.x, m3.x, m4.x, m5.x);
	winograd_f4x3_v2(v0.y, v1.y, v2.y, v3.y, m1.y, m2.y, m3.y, m4.y, m5.y);
	winograd_f4x3_v2(v0.z, v1.z, v2.z, v3.z, m1.z, m2.z, m3.z, m4.z, m5.z);
	winograd_f4x3_v2(v0.w, v1.w, v2.w, v3.w, m1.w, m2.w, m3.w, m4.w, m5.w);

	float4 v5, v6, v7;
	winograd_f4x3_v2(v4.x, v5.x, v6.x, v7.x, m7.x, m8.x, m9.x, m10.x, m11.x);
	winograd_f4x3_v2(v4.y, v5.y, v6.y, v7.y, m7.y, m8.y, m9.y, m10.y, m11.y);
	winograd_f4x3_v2(v4.z, v5.z, v6.z, v7.z, m7.z, m8.z, m9.z, m10.z, m11.z);
	winograd_f4x3_v2(v4.w, v5.w, v6.w, v7.w, m7.w, m8.w, m9.w, m10.w, m11.w);

	float4 v9, v10, v11;
	winograd_f4x3_v2(v8.x, v9.x, v10.x, v11.x, m13.x, m14.x, m15.x, m16.x, m17.x);
	winograd_f4x3_v2(v8.y, v9.y, v10.y, v11.y, m13.y, m14.y, m15.y, m16.y, m17.y);
	winograd_f4x3_v2(v8.z, v9.z, v10.z, v11.z, m13.z, m14.z, m15.z, m16.z, m17.z);
	winograd_f4x3_v2(v8.w, v9.w, v10.w, v11.w, m13.w, m14.w, m15.w, m16.w, m17.w);

	float4 v13, v14, v15;
	winograd_f4x3_v2(v12.x, v13.x, v14.x, v15.x, m19.x, m20.x, m21.x, m22.x, m23.x);
	winograd_f4x3_v2(v12.y, v13.y, v14.y, v15.y, m19.y, m20.y, m21.y, m22.y, m23.y);
	winograd_f4x3_v2(v12.z, v13.z, v14.z, v15.z, m19.z, m20.z, m21.z, m22.z, m23.z);
	winograd_f4x3_v2(v12.w, v13.w, v14.w, v15.w, m19.w, m20.w, m21.w, m22.w, m23.w);

	const int  j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int  j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	const int  j8 = j7 + OC, j9 = j8 + OC, j10 = j9 + OC, j11 = j10 + OC;
	const int j12 = j11 + OC, j13 = j12 + OC, j14 = j13 + OC, j15 = j14 + OC;

	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;

	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;

	*(float4*)(Y + j8) = v8; *(float4*)(Y + j9) = v9;
	*(float4*)(Y + j10) = v10; *(float4*)(Y + j11) = v11;

	*(float4*)(Y + j12) = v12; *(float4*)(Y + j13) = v13;
	*(float4*)(Y + j14) = v14; *(float4*)(Y + j15) = v15;
}

#endif


//double buffer
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_B5
#define XENO1_WINOGRAD_F4X3_B5

#define xeno_winograd_f4x3_b5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_b5<LB, (1<<LB>>2),(2<<LB>>2),(3<<LB>>2), (1<<LB>>2)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 1.92068 msec, Performace = 10062.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP3, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_b5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Gs0[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Gs1[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ds0[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ds1[2][1 << LB][(1 << LB) + 1];

	float4  v0 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v15 = F32_4_0;

	float4  m1 = F32_4_0,  m2 = F32_4_0,  m3 = F32_4_0,  m4 = F32_4_0;
	float4  m7 = F32_4_0,  m8 = F32_4_0,  m9 = F32_4_0, m10 = F32_4_0;
	float4 m13 = F32_4_0, m14 = F32_4_0, m15 = F32_4_0, m16 = F32_4_0;
	float4 m19 = F32_4_0, m20 = F32_4_0, m21 = F32_4_0, m22 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + (ty / STEP);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 4) + j_index;
	int tj0 = j0 + ((tx / STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		

		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool lx0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool lx1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool lx2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool lx3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool lx4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool lx5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		//load 2 group from X
		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
		float x0 = (lx0 ? X[xoffset] : 0);//0
		float x1 = (lx1 ? X[xoffset + IC] : 0);//1
		float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
		float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
		float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
		float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
		float d0, d1, d2, d3, d4, d5;
		winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
		Ds0[buf][tx][ty] = float4{ d0, d1, d2, d3 };
		Ds1[buf][tx][ty] = float2{ d4, d5 };
		
		//load 2 group from W
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
		float w0 = CW[woffset];
		float w1 = CW[woffset + Wstride];
		float w2 = CW[woffset + (Wstride << 1)];
		float g0, g1, g2, g3, g4, g5;
		winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
		Gs0[buf][ty][tx] = float4{ g0, g1, g2, g3 };
		Gs1[buf][ty][tx] = float2{ g4, g5 };
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs0[buf][ik][tx]; float2 g1 = Gs1[buf][ik][tx];
				float4 g2 = Gs0[buf][ik + STEP][tx]; float2 g3 = Gs1[buf][ik + STEP][tx];
				float4 g4 = Gs0[buf][ik + STEP2][tx]; float2 g5 = Gs1[buf][ik + STEP2][tx];
				float4 g6 = Gs0[buf][ik + STEP3][tx]; float2 g7 = Gs1[buf][ik + STEP3][tx];

				float4 d0 = Ds0[buf][ik][ty]; float2 d1 = Ds1[buf][ik][ty];
				float4 d2 = Ds0[buf][ik + STEP][ty]; float2 d3 = Ds1[buf][ik + STEP][ty];
				v0.x += g0.x * d0.x; v0.y += g2.x * d0.x; v0.z += g4.x * d0.x; v0.w += g6.x * d0.x;
				m1.x += g0.y * d0.y; m1.y += g2.y * d0.y; m1.z += g4.y * d0.y; m1.w += g6.y * d0.y;
				m2.x += g0.z * d0.z; m2.y += g2.z * d0.z; m2.z += g4.z * d0.z; m2.w += g6.z * d0.z;
				m3.x += g0.w * d0.w; m3.y += g2.w * d0.w; m3.z += g4.w * d0.w; m3.w += g6.w * d0.w;
				m4.x += g1.x * d1.x; m4.y += g3.x * d1.x; m4.z += g5.x * d1.x; m4.w += g7.x * d1.x;
				v3.x += g1.y * d1.y; v3.y += g3.y * d1.y; v3.z += g5.y * d1.y; v3.w += g7.y * d1.y;

				v4.x += g0.x * d2.x; v4.y += g2.x * d2.x; v4.z += g4.x * d2.x; v4.w += g6.x * d2.x;
				m7.x += g0.y * d2.y; m7.y += g2.y * d2.y; m7.z += g4.y * d2.y; m7.w += g6.y * d2.y;
				m8.x += g0.z * d2.z; m8.y += g2.z * d2.z; m8.z += g4.z * d2.z; m8.w += g6.z * d2.z;
				m9.x += g0.w * d2.w; m9.y += g2.w * d2.w; m9.z += g4.w * d2.w; m9.w += g6.w * d2.w;
				m10.x += g1.x * d3.x; m10.y += g3.x * d3.x; m10.z += g5.x * d3.x; m10.w += g7.x * d3.x;
				v7.x += g1.y * d3.y; v7.y += g3.y * d3.y; v7.z += g5.y * d3.y; v7.w += g7.y * d3.y;

				float4 d4 = Ds0[buf][ik + STEP2][ty]; float2 d5 = Ds1[buf][ik + STEP2][ty];
				float4 d6 = Ds0[buf][ik + STEP3][ty]; float2 d7 = Ds1[buf][ik + STEP3][ty];
				v8.x += g0.x * d4.x; v8.y += g2.x * d4.x; v8.z += g4.x * d4.x; v8.w += g6.x * d4.x;
				m13.x += g0.y * d4.y; m13.y += g2.y * d4.y; m13.z += g4.y * d4.y; m13.w += g6.y * d4.y;
				m14.x += g0.z * d4.z; m14.y += g2.z * d4.z; m14.z += g4.z * d4.z; m14.w += g6.z * d4.z;
				m15.x += g0.w * d4.w; m15.y += g2.w * d4.w; m15.z += g4.w * d4.w; m15.w += g6.w * d4.w;
				m16.x += g1.x * d5.x; m16.y += g3.x * d5.x; m16.z += g5.x * d5.x; m16.w += g7.x * d5.x;
				v11.x += g1.y * d5.y; v11.y += g3.y * d5.y; v11.z += g5.y * d5.y; v11.w += g7.y * d5.y;

				v12.x += g0.x * d6.x; v12.y += g2.x * d6.x; v12.z += g4.x * d6.x; v12.w += g6.x * d6.x;
				m19.x += g0.y * d6.y; m19.y += g2.y * d6.y; m19.z += g4.y * d6.y; m19.w += g6.y * d6.y;
				m20.x += g0.z * d6.z; m20.y += g2.z * d6.z; m20.z += g4.z * d6.z; m20.w += g6.z * d6.z;
				m21.x += g0.w * d6.w; m21.y += g2.w * d6.w; m21.z += g4.w * d6.w; m21.w += g6.w * d6.w;
				m22.x += g1.x * d7.x; m22.y += g3.x * d7.x; m22.z += g5.x * d7.x; m22.w += g7.x * d7.x;
				v15.x += g1.y * d7.y; v15.y += g3.y * d7.y; v15.z += g5.y * d7.y; v15.w += g7.y * d7.y;
			}
			buf ^= 1;

			//load 2 group from X
			const int xic = oic + (tx & STEP_m1);//with the same ty
			const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);//0
			float x1 = (lx1 ? X[xoffset + IC] : 0);//1
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
			Ds0[buf][tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[buf][tx][ty] = float2{ d4, d5 };
			
			//load 2 group from W
			const int wic = oic + (ty & STEP_m1);//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float w0 = CW[woffset];
			float w1 = CW[woffset + Wstride];
			float w2 = CW[woffset + (Wstride << 1)];
			float g0, g1, g2, g3, g4, g5;
			winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
			Gs0[buf][ty][tx] = float4{ g0, g1, g2, g3 };
			Gs1[buf][ty][tx] = float2{ g4, g5 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs0[buf][ik][tx]; float2 g1 = Gs1[buf][ik][tx];
			float4 g2 = Gs0[buf][ik + STEP][tx]; float2 g3 = Gs1[buf][ik + STEP][tx];
			float4 g4 = Gs0[buf][ik + STEP2][tx]; float2 g5 = Gs1[buf][ik + STEP2][tx];
			float4 g6 = Gs0[buf][ik + STEP3][tx]; float2 g7 = Gs1[buf][ik + STEP3][tx];

			float4 d0 = Ds0[buf][ik][ty]; float2 d1 = Ds1[buf][ik][ty];
			float4 d2 = Ds0[buf][ik + STEP][ty]; float2 d3 = Ds1[buf][ik + STEP][ty];
			float4 d4 = Ds0[buf][ik + STEP2][ty]; float2 d5 = Ds1[buf][ik + STEP2][ty];
			float4 d6 = Ds0[buf][ik + STEP3][ty]; float2 d7 = Ds1[buf][ik + STEP3][ty];

			v0.x += g0.x * d0.x; v0.y += g2.x * d0.x; v0.z += g4.x * d0.x; v0.w += g6.x * d0.x;
			m1.x += g0.y * d0.y; m1.y += g2.y * d0.y; m1.z += g4.y * d0.y; m1.w += g6.y * d0.y;
			m2.x += g0.z * d0.z; m2.y += g2.z * d0.z; m2.z += g4.z * d0.z; m2.w += g6.z * d0.z;
			m3.x += g0.w * d0.w; m3.y += g2.w * d0.w; m3.z += g4.w * d0.w; m3.w += g6.w * d0.w;
			m4.x += g1.x * d1.x; m4.y += g3.x * d1.x; m4.z += g5.x * d1.x; m4.w += g7.x * d1.x;
			v3.x += g1.y * d1.y; v3.y += g3.y * d1.y; v3.z += g5.y * d1.y; v3.w += g7.y * d1.y;

			v4.x += g0.x * d2.x; v4.y += g2.x * d2.x; v4.z += g4.x * d2.x; v4.w += g6.x * d2.x;
			m7.x += g0.y * d2.y; m7.y += g2.y * d2.y; m7.z += g4.y * d2.y; m7.w += g6.y * d2.y;
			m8.x += g0.z * d2.z; m8.y += g2.z * d2.z; m8.z += g4.z * d2.z; m8.w += g6.z * d2.z;
			m9.x += g0.w * d2.w; m9.y += g2.w * d2.w; m9.z += g4.w * d2.w; m9.w += g6.w * d2.w;
			m10.x += g1.x * d3.x; m10.y += g3.x * d3.x; m10.z += g5.x * d3.x; m10.w += g7.x * d3.x;
			v7.x += g1.y * d3.y; v7.y += g3.y * d3.y; v7.z += g5.y * d3.y; v7.w += g7.y * d3.y;

			v8.x += g0.x * d4.x; v8.y += g2.x * d4.x; v8.z += g4.x * d4.x; v8.w += g6.x * d4.x;
			m13.x += g0.y * d4.y; m13.y += g2.y * d4.y; m13.z += g4.y * d4.y; m13.w += g6.y * d4.y;
			m14.x += g0.z * d4.z; m14.y += g2.z * d4.z; m14.z += g4.z * d4.z; m14.w += g6.z * d4.z;
			m15.x += g0.w * d4.w; m15.y += g2.w * d4.w; m15.z += g4.w * d4.w; m15.w += g6.w * d4.w;
			m16.x += g1.x * d5.x; m16.y += g3.x * d5.x; m16.z += g5.x * d5.x; m16.w += g7.x * d5.x;
			v11.x += g1.y * d5.y; v11.y += g3.y * d5.y; v11.z += g5.y * d5.y; v11.w += g7.y * d5.y;

			v12.x += g0.x * d6.x; v12.y += g2.x * d6.x; v12.z += g4.x * d6.x; v12.w += g6.x * d6.x;
			m19.x += g0.y * d6.y; m19.y += g2.y * d6.y; m19.z += g4.y * d6.y; m19.w += g6.y * d6.y;
			m20.x += g0.z * d6.z; m20.y += g2.z * d6.z; m20.z += g4.z * d6.z; m20.w += g6.z * d6.z;
			m21.x += g0.w * d6.w; m21.y += g2.w * d6.w; m21.z += g4.w * d6.w; m21.w += g6.w * d6.w;
			m22.x += g1.x * d7.x; m22.y += g3.x * d7.x; m22.z += g5.x * d7.x; m22.w += g7.x * d7.x;
			v15.x += g1.y * d7.y; v15.y += g3.y * d7.y; v15.z += g5.y * d7.y; v15.w += g7.y * d7.y;
		}
		buf ^= 1;
	}

	float4 v1, v2;
	winograd_f4x3_v3(v0.x, v1.x, v2.x, v3.x, m1.x, m2.x, m3.x, m4.x);
	winograd_f4x3_v3(v0.y, v1.y, v2.y, v3.y, m1.y, m2.y, m3.y, m4.y);
	winograd_f4x3_v3(v0.z, v1.z, v2.z, v3.z, m1.z, m2.z, m3.z, m4.z);
	winograd_f4x3_v3(v0.w, v1.w, v2.w, v3.w, m1.w, m2.w, m3.w, m4.w);

	float4 v5, v6;
	winograd_f4x3_v3(v4.x, v5.x, v6.x, v7.x, m7.x, m8.x, m9.x, m10.x, m11.x);
	winograd_f4x3_v3(v4.y, v5.y, v6.y, v7.y, m7.y, m8.y, m9.y, m10.y, m11.y);
	winograd_f4x3_v3(v4.z, v5.z, v6.z, v7.z, m7.z, m8.z, m9.z, m10.z, m11.z);
	winograd_f4x3_v3(v4.w, v5.w, v6.w, v7.w, m7.w, m8.w, m9.w, m10.w, m11.w);

	float4 v9, v10;
	winograd_f4x3_v3(v8.x, v9.x, v10.x, v11.x, m13.x, m14.x, m15.x, m16.x);
	winograd_f4x3_v3(v8.y, v9.y, v10.y, v11.y, m13.y, m14.y, m15.y, m16.y);
	winograd_f4x3_v3(v8.z, v9.z, v10.z, v11.z, m13.z, m14.z, m15.z, m16.z);
	winograd_f4x3_v3(v8.w, v9.w, v10.w, v11.w, m13.w, m14.w, m15.w, m16.w);

	float4 v13, v14;
	winograd_f4x3_v3(v12.x, v13.x, v14.x, v15.x, m19.x, m20.x, m21.x, m22.x);
	winograd_f4x3_v3(v12.y, v13.y, v14.y, v15.y, m19.y, m20.y, m21.y, m22.y);
	winograd_f4x3_v3(v12.z, v13.z, v14.z, v15.z, m19.z, m20.z, m21.z, m22.z);
	winograd_f4x3_v3(v12.w, v13.w, v14.w, v15.w, m19.w, m20.w, m21.w, m22.w);

	const int  j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int  j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	const int  j8 = j7 + OC, j9 = j8 + OC, j10 = j9 + OC, j11 = j10 + OC;
	const int j12 = j11 + OC, j13 = j12 + OC, j14 = j13 + OC, j15 = j14 + OC;

	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;

	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;

	*(float4*)(Y + j8) = v8; *(float4*)(Y + j9) = v9;
	*(float4*)(Y + j10) = v10; *(float4*)(Y + j11) = v11;

	*(float4*)(Y + j12) = v12; *(float4*)(Y + j13) = v13;
	*(float4*)(Y + j14) = v14; *(float4*)(Y + j15) = v15;
}

#endif


//double buffer: Split G for m, v
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_B6
#define XENO1_WINOGRAD_F4X3_B6

#define xeno_winograd_f4x3_b6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_b6<LB, (1<<LB>>2),(2<<LB>>2),(3<<LB>>2), (1<<LB>>2)-1,\
			IC, IC*2, IC*3, IC*4, IC*5>\
		<<< dim3(GN>>LB>>2, GM>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 1.92068 msec, Performace = 10062.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP3, int STEP_m1, 
	int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f4x3_kernel_b6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Gsm[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Gsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ds0[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ds1[2][1 << LB][(1 << LB) + 1];

	float4  v0 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v15 = F32_4_0;

	float4  m1 = F32_4_0, m2 = F32_4_0, m3 = F32_4_0, m4 = F32_4_0;
	float4  m7 = F32_4_0, m8 = F32_4_0, m9 = F32_4_0, m10 = F32_4_0;
	float4 m13 = F32_4_0, m14 = F32_4_0, m15 = F32_4_0, m16 = F32_4_0;
	float4 m19 = F32_4_0, m20 = F32_4_0, m21 = F32_4_0, m22 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + (ty / STEP);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 4) + j_index;
	int tj0 = j0 + ((tx / STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool lx0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool lx1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool lx2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool lx3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool lx4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool lx5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		//load 2 group from X
		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
		float x0 = (lx0 ? X[xoffset] : 0);//0
		float x1 = (lx1 ? X[xoffset + IC] : 0);//1
		float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
		float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
		float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
		float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
		float d0, d1, d2, d3, d4, d5;
		winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
		Ds0[buf][tx][ty] = float4{ d0, d1, d2, d3 };
		Ds1[buf][tx][ty] = float2{ d4, d5 };

		//load 2 group from W
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
		float w0 = CW[woffset];
		float w1 = CW[woffset + Wstride];
		float w2 = CW[woffset + (Wstride << 1)];
		float g0, g1, g2, g3, g4, g5;
		winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
		Gsm[buf][ty][tx] = float4{ g1, g2, g3, g4 };
		Gsv[buf][ty][tx] = float2{ g0, g5 };
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 gm0 = Gsm[buf][ik        ][tx]; float2 gv0 = Gsv[buf][ik][tx];
				float4 gm1 = Gsm[buf][ik + STEP ][tx]; float2 gv1 = Gsv[buf][ik + STEP][tx];
				float4 gm2 = Gsm[buf][ik + STEP2][tx]; float2 gv2 = Gsv[buf][ik + STEP2][tx];
				float4 gm3 = Gsm[buf][ik + STEP3][tx]; float2 gv3 = Gsv[buf][ik + STEP3][tx];

				float4 d0 = Ds0[buf][ik][ty]; float2 d1 = Ds1[buf][ik][ty];
				float4 d2 = Ds0[buf][ik + STEP][ty]; float2 d3 = Ds1[buf][ik + STEP][ty];
				float4 d4 = Ds0[buf][ik + STEP2][ty]; float2 d5 = Ds1[buf][ik + STEP2][ty];
				float4 d6 = Ds0[buf][ik + STEP3][ty]; float2 d7 = Ds1[buf][ik + STEP3][ty];

				v0.x  += gv0.x * d0.x; v0.y  += gv1.x * d0.x; v0.z  += gv2.x * d0.x; v0.w  += gv3.x * d0.x;
				v3.x  += gv0.y * d1.y; v3.y  += gv1.y * d1.y; v3.z  += gv2.y * d1.y; v3.w  += gv3.y * d1.y;
				v4.x  += gv0.x * d2.x; v4.y  += gv1.x * d2.x; v4.z  += gv2.x * d2.x; v4.w  += gv3.x * d2.x;
				v7.x  += gv0.y * d3.y; v7.y  += gv1.y * d3.y; v7.z  += gv2.y * d3.y; v7.w  += gv3.y * d3.y;
				v8.x  += gv0.x * d4.x; v8.y  += gv1.x * d4.x; v8.z  += gv2.x * d4.x; v8.w  += gv3.x * d4.x;
				v11.x += gv0.y * d5.y; v11.y += gv1.y * d5.y; v11.z += gv2.y * d5.y; v11.w += gv3.y * d5.y;
				v12.x += gv0.x * d6.x; v12.y += gv1.x * d6.x; v12.z += gv2.x * d6.x; v12.w += gv3.x * d6.x;
				v15.x += gv0.y * d7.y; v15.y += gv1.y * d7.y; v15.z += gv2.y * d7.y; v15.w += gv3.y * d7.y;

				m1.x  += gm0.x * d0.y; m1.y  += gm1.x * d0.y; m1.z  += gm2.x * d0.y; m1.w  += gm3.x * d0.y;
				m2.x  += gm0.y * d0.z; m2.y  += gm1.y * d0.z; m2.z  += gm2.y * d0.z; m2.w  += gm3.y * d0.z;
				m3.x  += gm0.z * d0.w; m3.y  += gm1.z * d0.w; m3.z  += gm2.z * d0.w; m3.w  += gm3.z * d0.w;
				m4.x  += gm0.w * d1.x; m4.y  += gm1.w * d1.x; m4.z  += gm2.w * d1.x; m4.w  += gm3.w * d1.x;
				m7.x  += gm0.x * d2.y; m7.y  += gm1.x * d2.y; m7.z  += gm2.x * d2.y; m7.w  += gm3.x * d2.y;
				m8.x  += gm0.y * d2.z; m8.y  += gm1.y * d2.z; m8.z  += gm2.y * d2.z; m8.w  += gm3.y * d2.z;
				m9.x  += gm0.z * d2.w; m9.y  += gm1.z * d2.w; m9.z  += gm2.z * d2.w; m9.w  += gm3.z * d2.w;
				m10.x += gm0.w * d3.x; m10.y += gm1.w * d3.x; m10.z += gm2.w * d3.x; m10.w += gm3.w * d3.x;
				m13.x += gm0.x * d4.y; m13.y += gm1.x * d4.y; m13.z += gm2.x * d4.y; m13.w += gm3.x * d4.y;
				m14.x += gm0.y * d4.z; m14.y += gm1.y * d4.z; m14.z += gm2.y * d4.z; m14.w += gm3.y * d4.z;
				m15.x += gm0.z * d4.w; m15.y += gm1.z * d4.w; m15.z += gm2.z * d4.w; m15.w += gm3.z * d4.w;
				m16.x += gm0.w * d5.x; m16.y += gm1.w * d5.x; m16.z += gm2.w * d5.x; m16.w += gm3.w * d5.x;
				m19.x += gm0.x * d6.y; m19.y += gm1.x * d6.y; m19.z += gm2.x * d6.y; m19.w += gm3.x * d6.y;
				m20.x += gm0.y * d6.z; m20.y += gm1.y * d6.z; m20.z += gm2.y * d6.z; m20.w += gm3.y * d6.z;
				m21.x += gm0.z * d6.w; m21.y += gm1.z * d6.w; m21.z += gm2.z * d6.w; m21.w += gm3.z * d6.w;
				m22.x += gm0.w * d7.x; m22.y += gm1.w * d7.x; m22.z += gm2.w * d7.x; m22.w += gm3.w * d7.x;
				
			}
			buf ^= 1;

			//load 2 group from X
			const int xic = oic + (tx & STEP_m1);//with the same ty
			const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);//0
			float x1 = (lx1 ? X[xoffset + IC] : 0);//1
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
			Ds0[buf][tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[buf][tx][ty] = float2{ d4, d5 };

			//load 2 group from W
			const int wic = oic + (ty & STEP_m1);//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float w0 = CW[woffset];
			float w1 = CW[woffset + Wstride];
			float w2 = CW[woffset + (Wstride << 1)];
			float g0, g1, g2, g3, g4, g5;
			winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
			Gsm[buf][ty][tx] = float4{ g1, g2, g3, g4 };
			Gsv[buf][ty][tx] = float2{ g0, g5 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 gm0 = Gsm[buf][ik][tx]; float2 gv0 = Gsv[buf][ik][tx];
			float4 gm1 = Gsm[buf][ik + STEP][tx]; float2 gv1 = Gsv[buf][ik + STEP][tx];
			float4 gm2 = Gsm[buf][ik + STEP2][tx]; float2 gv2 = Gsv[buf][ik + STEP2][tx];
			float4 gm3 = Gsm[buf][ik + STEP3][tx]; float2 gv3 = Gsv[buf][ik + STEP3][tx];

			float4 d0 = Ds0[buf][ik][ty]; float2 d1 = Ds1[buf][ik][ty];
			float4 d2 = Ds0[buf][ik + STEP][ty]; float2 d3 = Ds1[buf][ik + STEP][ty];
			float4 d4 = Ds0[buf][ik + STEP2][ty]; float2 d5 = Ds1[buf][ik + STEP2][ty];
			float4 d6 = Ds0[buf][ik + STEP3][ty]; float2 d7 = Ds1[buf][ik + STEP3][ty];

			v0.x += gv0.x * d0.x; v0.y += gv1.x * d0.x; v0.z += gv2.x * d0.x; v0.w += gv3.x * d0.x;
			v3.x += gv0.y * d1.y; v3.y += gv1.y * d1.y; v3.z += gv2.y * d1.y; v3.w += gv3.y * d1.y;
			v4.x += gv0.x * d2.x; v4.y += gv1.x * d2.x; v4.z += gv2.x * d2.x; v4.w += gv3.x * d2.x;
			v7.x += gv0.y * d3.y; v7.y += gv1.y * d3.y; v7.z += gv2.y * d3.y; v7.w += gv3.y * d3.y;
			v8.x += gv0.x * d4.x; v8.y += gv1.x * d4.x; v8.z += gv2.x * d4.x; v8.w += gv3.x * d4.x;
			v11.x += gv0.y * d5.y; v11.y += gv1.y * d5.y; v11.z += gv2.y * d5.y; v11.w += gv3.y * d5.y;
			v12.x += gv0.x * d6.x; v12.y += gv1.x * d6.x; v12.z += gv2.x * d6.x; v12.w += gv3.x * d6.x;
			v15.x += gv0.y * d7.y; v15.y += gv1.y * d7.y; v15.z += gv2.y * d7.y; v15.w += gv3.y * d7.y;

			m1.x += gm0.x * d0.y; m1.y += gm1.x * d0.y; m1.z += gm2.x * d0.y; m1.w += gm3.x * d0.y;
			m2.x += gm0.y * d0.z; m2.y += gm1.y * d0.z; m2.z += gm2.y * d0.z; m2.w += gm3.y * d0.z;
			m3.x += gm0.z * d0.w; m3.y += gm1.z * d0.w; m3.z += gm2.z * d0.w; m3.w += gm3.z * d0.w;
			m4.x += gm0.w * d1.x; m4.y += gm1.w * d1.x; m4.z += gm2.w * d1.x; m4.w += gm3.w * d1.x;
			m7.x += gm0.x * d2.y; m7.y += gm1.x * d2.y; m7.z += gm2.x * d2.y; m7.w += gm3.x * d2.y;
			m8.x += gm0.y * d2.z; m8.y += gm1.y * d2.z; m8.z += gm2.y * d2.z; m8.w += gm3.y * d2.z;
			m9.x += gm0.z * d2.w; m9.y += gm1.z * d2.w; m9.z += gm2.z * d2.w; m9.w += gm3.z * d2.w;
			m10.x += gm0.w * d3.x; m10.y += gm1.w * d3.x; m10.z += gm2.w * d3.x; m10.w += gm3.w * d3.x;
			m13.x += gm0.x * d4.y; m13.y += gm1.x * d4.y; m13.z += gm2.x * d4.y; m13.w += gm3.x * d4.y;
			m14.x += gm0.y * d4.z; m14.y += gm1.y * d4.z; m14.z += gm2.y * d4.z; m14.w += gm3.y * d4.z;
			m15.x += gm0.z * d4.w; m15.y += gm1.z * d4.w; m15.z += gm2.z * d4.w; m15.w += gm3.z * d4.w;
			m16.x += gm0.w * d5.x; m16.y += gm1.w * d5.x; m16.z += gm2.w * d5.x; m16.w += gm3.w * d5.x;
			m19.x += gm0.x * d6.y; m19.y += gm1.x * d6.y; m19.z += gm2.x * d6.y; m19.w += gm3.x * d6.y;
			m20.x += gm0.y * d6.z; m20.y += gm1.y * d6.z; m20.z += gm2.y * d6.z; m20.w += gm3.y * d6.z;
			m21.x += gm0.z * d6.w; m21.y += gm1.z * d6.w; m21.z += gm2.z * d6.w; m21.w += gm3.z * d6.w;
			m22.x += gm0.w * d7.x; m22.y += gm1.w * d7.x; m22.z += gm2.w * d7.x; m22.w += gm3.w * d7.x;
		}
		buf ^= 1;
	}

	float4 v1, v2;
	winograd_f4x3_v3(v0.x, v1.x, v2.x, v3.x, m1.x, m2.x, m3.x, m4.x);
	winograd_f4x3_v3(v0.y, v1.y, v2.y, v3.y, m1.y, m2.y, m3.y, m4.y);
	winograd_f4x3_v3(v0.z, v1.z, v2.z, v3.z, m1.z, m2.z, m3.z, m4.z);
	winograd_f4x3_v3(v0.w, v1.w, v2.w, v3.w, m1.w, m2.w, m3.w, m4.w);

	float4 v5, v6;
	winograd_f4x3_v3(v4.x, v5.x, v6.x, v7.x, m7.x, m8.x, m9.x, m10.x, m11.x);
	winograd_f4x3_v3(v4.y, v5.y, v6.y, v7.y, m7.y, m8.y, m9.y, m10.y, m11.y);
	winograd_f4x3_v3(v4.z, v5.z, v6.z, v7.z, m7.z, m8.z, m9.z, m10.z, m11.z);
	winograd_f4x3_v3(v4.w, v5.w, v6.w, v7.w, m7.w, m8.w, m9.w, m10.w, m11.w);

	float4 v9, v10;
	winograd_f4x3_v3(v8.x, v9.x, v10.x, v11.x, m13.x, m14.x, m15.x, m16.x);
	winograd_f4x3_v3(v8.y, v9.y, v10.y, v11.y, m13.y, m14.y, m15.y, m16.y);
	winograd_f4x3_v3(v8.z, v9.z, v10.z, v11.z, m13.z, m14.z, m15.z, m16.z);
	winograd_f4x3_v3(v8.w, v9.w, v10.w, v11.w, m13.w, m14.w, m15.w, m16.w);

	float4 v13, v14;
	winograd_f4x3_v3(v12.x, v13.x, v14.x, v15.x, m19.x, m20.x, m21.x, m22.x);
	winograd_f4x3_v3(v12.y, v13.y, v14.y, v15.y, m19.y, m20.y, m21.y, m22.y);
	winograd_f4x3_v3(v12.z, v13.z, v14.z, v15.z, m19.z, m20.z, m21.z, m22.z);
	winograd_f4x3_v3(v12.w, v13.w, v14.w, v15.w, m19.w, m20.w, m21.w, m22.w);

	const int  j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int  j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	const int  j8 = j7 + OC, j9 = j8 + OC, j10 = j9 + OC, j11 = j10 + OC;
	const int j12 = j11 + OC, j13 = j12 + OC, j14 = j13 + OC, j15 = j14 + OC;

	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;

	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;

	*(float4*)(Y + j8) = v8; *(float4*)(Y + j9) = v9;
	*(float4*)(Y + j10) = v10; *(float4*)(Y + j11) = v11;

	*(float4*)(Y + j12) = v12; *(float4*)(Y + j13) = v13;
	*(float4*)(Y + j14) = v14; *(float4*)(Y + j15) = v15;
}

#endif


//double buffer: Split D for m, v
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_B7
#define XENO1_WINOGRAD_F4X3_B7

#define xeno_winograd_f4x3_b7(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_b7<LB, (1<<LB>>2),(2<<LB>>2),(3<<LB>>2), (1<<LB>>2)-1,\
			IC, IC*2, IC*3, IC*4, IC*5>\
		<<< dim3(GN>>LB>>2, GM>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 1.92512 msec, Performace = 10039.6 GFlop/s
template<int LB, int STEP, int STEP2, int STEP3, int STEP_m1,
	int IC, int IC2, int IC3, int IC4, int IC5>
	__global__ void xeno_winograd_f4x3_kernel_b7(
		const float* __restrict__ X, int IH, int IW,
		const float* __restrict__ CW,//[FH, FW, IC, OC]
		float* __restrict__ Y, int OH_OW, int OW, int OC,
		int ph, int pw,
		int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Gsm[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Gsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dsm[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Dsv[2][1 << LB][(1 << LB) + 1];

	float4  v0 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v15 = F32_4_0;

	float4  m1 = F32_4_0, m2 = F32_4_0, m3 = F32_4_0, m4 = F32_4_0;
	float4  m7 = F32_4_0, m8 = F32_4_0, m9 = F32_4_0, m10 = F32_4_0;
	float4 m13 = F32_4_0, m14 = F32_4_0, m15 = F32_4_0, m16 = F32_4_0;
	float4 m19 = F32_4_0, m20 = F32_4_0, m21 = F32_4_0, m22 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + (ty / STEP);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 4) + j_index;
	int tj0 = j0 + ((tx / STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool lx0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool lx1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool lx2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool lx3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool lx4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool lx5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		//load 2 group from X
		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
		float x0 = (lx0 ? X[xoffset] : 0);//0
		float x1 = (lx1 ? X[xoffset + IC] : 0);//1
		float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
		float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
		float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
		float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
		float d0, d1, d2, d3, d4, d5;
		winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
		Dsm[buf][tx][ty] = float4{ d1, d2, d3, d4 };
		Dsv[buf][tx][ty] = float2{ d0, d5 };

		//load 2 group from W
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
		float w0 = CW[woffset];
		float w1 = CW[woffset + Wstride];
		float w2 = CW[woffset + (Wstride << 1)];
		float g0, g1, g2, g3, g4, g5;
		winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
		Gsm[buf][ty][tx] = float4{ g1, g2, g3, g4 };
		Gsv[buf][ty][tx] = float2{ g0, g5 };
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 gm0 = Gsm[buf][ik][tx]; float4 gm1 = Gsm[buf][ik + STEP][tx];
				float4 gm2 = Gsm[buf][ik + STEP2][tx]; float4 gm3 = Gsm[buf][ik + STEP3][tx];
				float4 dm2 = Dsm[buf][ik + STEP2][ty]; float4 dm3 = Dsm[buf][ik + STEP3][ty];
				float4 dm0 = Dsm[buf][ik][ty]; float4 dm1 = Dsm[buf][ik + STEP][ty];

				m1.x += gm0.x * dm0.x; m1.y += gm1.x * dm0.x; m1.z += gm2.x * dm0.x; m1.w += gm3.x * dm0.x;
				m2.x += gm0.y * dm0.y; m2.y += gm1.y * dm0.y; m2.z += gm2.y * dm0.y; m2.w += gm3.y * dm0.y;
				m3.x += gm0.z * dm0.z; m3.y += gm1.z * dm0.z; m3.z += gm2.z * dm0.z; m3.w += gm3.z * dm0.z;
				m4.x += gm0.w * dm0.w; m4.y += gm1.w * dm0.w; m4.z += gm2.w * dm0.w; m4.w += gm3.w * dm0.w;
				m7.x += gm0.x * dm1.x; m7.y += gm1.x * dm1.x; m7.z += gm2.x * dm1.x; m7.w += gm3.x * dm1.x;
				m8.x += gm0.y * dm1.y; m8.y += gm1.y * dm1.y; m8.z += gm2.y * dm1.y; m8.w += gm3.y * dm1.y;
				m9.x += gm0.z * dm1.z; m9.y += gm1.z * dm1.z; m9.z += gm2.z * dm1.z; m9.w += gm3.z * dm1.z;
				m10.x += gm0.w * dm1.w; m10.y += gm1.w * dm1.w; m10.z += gm2.w * dm1.w; m10.w += gm3.w * dm1.w;
				m13.x += gm0.x * dm2.x; m13.y += gm1.x * dm2.x; m13.z += gm2.x * dm2.x; m13.w += gm3.x * dm2.x;
				m14.x += gm0.y * dm2.y; m14.y += gm1.y * dm2.y; m14.z += gm2.y * dm2.y; m14.w += gm3.y * dm2.y;
				m15.x += gm0.z * dm2.z; m15.y += gm1.z * dm2.z; m15.z += gm2.z * dm2.z; m15.w += gm3.z * dm2.z;
				m16.x += gm0.w * dm2.w; m16.y += gm1.w * dm2.w; m16.z += gm2.w * dm2.w; m16.w += gm3.w * dm2.w;
				m19.x += gm0.x * dm3.x; m19.y += gm1.x * dm3.x; m19.z += gm2.x * dm3.x; m19.w += gm3.x * dm3.x;
				m20.x += gm0.y * dm3.y; m20.y += gm1.y * dm3.y; m20.z += gm2.y * dm3.y; m20.w += gm3.y * dm3.y;
				m21.x += gm0.z * dm3.z; m21.y += gm1.z * dm3.z; m21.z += gm2.z * dm3.z; m21.w += gm3.z * dm3.z;
				m22.x += gm0.w * dm3.w; m22.y += gm1.w * dm3.w; m22.z += gm2.w * dm3.w; m22.w += gm3.w * dm3.w;

				float2 gv0 = Gsv[buf][ik][tx]; float2 gv1 = Gsv[buf][ik + STEP][tx];
				float2 gv2 = Gsv[buf][ik + STEP2][tx]; float2 gv3 = Gsv[buf][ik + STEP3][tx];
				float2 dv2 = Dsv[buf][ik + STEP2][ty]; float2 dv3 = Dsv[buf][ik + STEP3][ty];
				float2 dv0 = Dsv[buf][ik][ty]; float2 dv1 = Dsv[buf][ik + STEP][ty];

				v0.x += gv0.x * dv0.x; v0.y += gv1.x * dv0.x; v0.z += gv2.x * dv0.x; v0.w += gv3.x * dv0.x;
				v3.x += gv0.y * dv0.y; v3.y += gv1.y * dv0.y; v3.z += gv2.y * dv0.y; v3.w += gv3.y * dv0.y;
				v4.x += gv0.x * dv1.x; v4.y += gv1.x * dv1.x; v4.z += gv2.x * dv1.x; v4.w += gv3.x * dv1.x;
				v7.x += gv0.y * dv1.y; v7.y += gv1.y * dv1.y; v7.z += gv2.y * dv1.y; v7.w += gv3.y * dv1.y;
				v8.x += gv0.x * dv2.x; v8.y += gv1.x * dv2.x; v8.z += gv2.x * dv2.x; v8.w += gv3.x * dv2.x;
				v11.x += gv0.y * dv2.y; v11.y += gv1.y * dv2.y; v11.z += gv2.y * dv2.y; v11.w += gv3.y * dv2.y;
				v12.x += gv0.x * dv3.x; v12.y += gv1.x * dv3.x; v12.z += gv2.x * dv3.x; v12.w += gv3.x * dv3.x;
				v15.x += gv0.y * dv3.y; v15.y += gv1.y * dv3.y; v15.z += gv2.y * dv3.y; v15.w += gv3.y * dv3.y;
			}
			buf ^= 1;

			//load 2 group from X
			const int xic = oic + (tx & STEP_m1);//with the same ty
			const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);//0
			float x1 = (lx1 ? X[xoffset + IC] : 0);//1
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
			Dsm[buf][tx][ty] = float4{ d1, d2, d3, d4 };
			Dsv[buf][tx][ty] = float2{ d0, d5 };

			//load 2 group from W
			const int wic = oic + (ty & STEP_m1);//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float w0 = CW[woffset];
			float w1 = CW[woffset + Wstride];
			float w2 = CW[woffset + (Wstride << 1)];
			float g0, g1, g2, g3, g4, g5;
			winograd_f4x3_g(g0, g1, g2, g3, g4, g5, w0, w1, w2);
			Gsm[buf][ty][tx] = float4{ g1, g2, g3, g4 };
			Gsv[buf][ty][tx] = float2{ g0, g5 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 gm0 = Gsm[buf][ik][tx]; float4 gm1 = Gsm[buf][ik + STEP][tx];
			float4 gm2 = Gsm[buf][ik + STEP2][tx]; float4 gm3 = Gsm[buf][ik + STEP3][tx];
			float4 dm2 = Dsm[buf][ik + STEP2][ty]; float4 dm3 = Dsm[buf][ik + STEP3][ty];
			float4 dm0 = Dsm[buf][ik][ty]; float4 dm1 = Dsm[buf][ik + STEP][ty];

			m1.x += gm0.x * dm0.x; m1.y += gm1.x * dm0.x; m1.z += gm2.x * dm0.x; m1.w += gm3.x * dm0.x;
			m2.x += gm0.y * dm0.y; m2.y += gm1.y * dm0.y; m2.z += gm2.y * dm0.y; m2.w += gm3.y * dm0.y;
			m3.x += gm0.z * dm0.z; m3.y += gm1.z * dm0.z; m3.z += gm2.z * dm0.z; m3.w += gm3.z * dm0.z;
			m4.x += gm0.w * dm0.w; m4.y += gm1.w * dm0.w; m4.z += gm2.w * dm0.w; m4.w += gm3.w * dm0.w;
			m7.x += gm0.x * dm1.x; m7.y += gm1.x * dm1.x; m7.z += gm2.x * dm1.x; m7.w += gm3.x * dm1.x;
			m8.x += gm0.y * dm1.y; m8.y += gm1.y * dm1.y; m8.z += gm2.y * dm1.y; m8.w += gm3.y * dm1.y;
			m9.x += gm0.z * dm1.z; m9.y += gm1.z * dm1.z; m9.z += gm2.z * dm1.z; m9.w += gm3.z * dm1.z;
			m10.x += gm0.w * dm1.w; m10.y += gm1.w * dm1.w; m10.z += gm2.w * dm1.w; m10.w += gm3.w * dm1.w;
			m13.x += gm0.x * dm2.x; m13.y += gm1.x * dm2.x; m13.z += gm2.x * dm2.x; m13.w += gm3.x * dm2.x;
			m14.x += gm0.y * dm2.y; m14.y += gm1.y * dm2.y; m14.z += gm2.y * dm2.y; m14.w += gm3.y * dm2.y;
			m15.x += gm0.z * dm2.z; m15.y += gm1.z * dm2.z; m15.z += gm2.z * dm2.z; m15.w += gm3.z * dm2.z;
			m16.x += gm0.w * dm2.w; m16.y += gm1.w * dm2.w; m16.z += gm2.w * dm2.w; m16.w += gm3.w * dm2.w;
			m19.x += gm0.x * dm3.x; m19.y += gm1.x * dm3.x; m19.z += gm2.x * dm3.x; m19.w += gm3.x * dm3.x;
			m20.x += gm0.y * dm3.y; m20.y += gm1.y * dm3.y; m20.z += gm2.y * dm3.y; m20.w += gm3.y * dm3.y;
			m21.x += gm0.z * dm3.z; m21.y += gm1.z * dm3.z; m21.z += gm2.z * dm3.z; m21.w += gm3.z * dm3.z;
			m22.x += gm0.w * dm3.w; m22.y += gm1.w * dm3.w; m22.z += gm2.w * dm3.w; m22.w += gm3.w * dm3.w;

			float2 gv0 = Gsv[buf][ik][tx]; float2 gv1 = Gsv[buf][ik + STEP][tx];
			float2 gv2 = Gsv[buf][ik + STEP2][tx]; float2 gv3 = Gsv[buf][ik + STEP3][tx];
			float2 dv2 = Dsv[buf][ik + STEP2][ty]; float2 dv3 = Dsv[buf][ik + STEP3][ty];
			float2 dv0 = Dsv[buf][ik][ty]; float2 dv1 = Dsv[buf][ik + STEP][ty];

			v0.x += gv0.x * dv0.x; v0.y += gv1.x * dv0.x; v0.z += gv2.x * dv0.x; v0.w += gv3.x * dv0.x;
			v3.x += gv0.y * dv0.y; v3.y += gv1.y * dv0.y; v3.z += gv2.y * dv0.y; v3.w += gv3.y * dv0.y;
			v4.x += gv0.x * dv1.x; v4.y += gv1.x * dv1.x; v4.z += gv2.x * dv1.x; v4.w += gv3.x * dv1.x;
			v7.x += gv0.y * dv1.y; v7.y += gv1.y * dv1.y; v7.z += gv2.y * dv1.y; v7.w += gv3.y * dv1.y;
			v8.x += gv0.x * dv2.x; v8.y += gv1.x * dv2.x; v8.z += gv2.x * dv2.x; v8.w += gv3.x * dv2.x;
			v11.x += gv0.y * dv2.y; v11.y += gv1.y * dv2.y; v11.z += gv2.y * dv2.y; v11.w += gv3.y * dv2.y;
			v12.x += gv0.x * dv3.x; v12.y += gv1.x * dv3.x; v12.z += gv2.x * dv3.x; v12.w += gv3.x * dv3.x;
			v15.x += gv0.y * dv3.y; v15.y += gv1.y * dv3.y; v15.z += gv2.y * dv3.y; v15.w += gv3.y * dv3.y;
		}
		buf ^= 1;
	}

	float4 v1, v2;
	winograd_f4x3_v3(v0.x, v1.x, v2.x, v3.x, m1.x, m2.x, m3.x, m4.x);
	winograd_f4x3_v3(v0.y, v1.y, v2.y, v3.y, m1.y, m2.y, m3.y, m4.y);
	winograd_f4x3_v3(v0.z, v1.z, v2.z, v3.z, m1.z, m2.z, m3.z, m4.z);
	winograd_f4x3_v3(v0.w, v1.w, v2.w, v3.w, m1.w, m2.w, m3.w, m4.w);

	float4 v5, v6;
	winograd_f4x3_v3(v4.x, v5.x, v6.x, v7.x, m7.x, m8.x, m9.x, m10.x, m11.x);
	winograd_f4x3_v3(v4.y, v5.y, v6.y, v7.y, m7.y, m8.y, m9.y, m10.y, m11.y);
	winograd_f4x3_v3(v4.z, v5.z, v6.z, v7.z, m7.z, m8.z, m9.z, m10.z, m11.z);
	winograd_f4x3_v3(v4.w, v5.w, v6.w, v7.w, m7.w, m8.w, m9.w, m10.w, m11.w);

	float4 v9, v10;
	winograd_f4x3_v3(v8.x, v9.x, v10.x, v11.x, m13.x, m14.x, m15.x, m16.x);
	winograd_f4x3_v3(v8.y, v9.y, v10.y, v11.y, m13.y, m14.y, m15.y, m16.y);
	winograd_f4x3_v3(v8.z, v9.z, v10.z, v11.z, m13.z, m14.z, m15.z, m16.z);
	winograd_f4x3_v3(v8.w, v9.w, v10.w, v11.w, m13.w, m14.w, m15.w, m16.w);

	float4 v13, v14;
	winograd_f4x3_v3(v12.x, v13.x, v14.x, v15.x, m19.x, m20.x, m21.x, m22.x);
	winograd_f4x3_v3(v12.y, v13.y, v14.y, v15.y, m19.y, m20.y, m21.y, m22.y);
	winograd_f4x3_v3(v12.z, v13.z, v14.z, v15.z, m19.z, m20.z, m21.z, m22.z);
	winograd_f4x3_v3(v12.w, v13.w, v14.w, v15.w, m19.w, m20.w, m21.w, m22.w);

	const int  j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int  j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	const int  j8 = j7 + OC, j9 = j8 + OC, j10 = j9 + OC, j11 = j10 + OC;
	const int j12 = j11 + OC, j13 = j12 + OC, j14 = j13 + OC, j15 = j14 + OC;

	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;

	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;

	*(float4*)(Y + j8) = v8; *(float4*)(Y + j9) = v9;
	*(float4*)(Y + j10) = v10; *(float4*)(Y + j11) = v11;

	*(float4*)(Y + j12) = v12; *(float4*)(Y + j13) = v13;
	*(float4*)(Y + j14) = v14; *(float4*)(Y + j15) = v15;
}

#endif

///ssssssssss
//half shared memory(Try to realse)
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_B8
#define XENO1_WINOGRAD_F4X3_B8

#define xeno_winograd_f4x3_b8(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_b8<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, IC, IC*2, IC*3, IC*4, IC*5, IC*6>\
		<<< dim3(GN>>LB>>2, GM>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 1.94215 msec, Performace = 9951.52 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5, int IC6>
__global__ void xeno_winograd_f4x3_kernel_b8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	__shared__ float4 Gsm[1 << LB][(2 << LB) + 1];
	__shared__ float2 Gsv[1 << LB][(2 << LB) + 1];
	__shared__ float4 Dsm[1 << LB][(2 << LB) + 1];
	__shared__ float2 Dsv[1 << LB][(2 << LB) + 1];

	float4  v0 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v15 = F32_4_0;

	float4  m1 = F32_4_0, m2 = F32_4_0, m3 = F32_4_0, m4 = F32_4_0;
	float4  m7 = F32_4_0, m8 = F32_4_0, m9 = F32_4_0, m10 = F32_4_0;
	float4 m13 = F32_4_0, m14 = F32_4_0, m15 = F32_4_0, m16 = F32_4_0;
	float4 m19 = F32_4_0, m20 = F32_4_0, m21 = F32_4_0, m22 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 4) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 3);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool lx0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool lx1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool lx2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool lx3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool lx4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool lx5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		bool lx6 = lh0 && (tiw0 >= -6) && (tiw0 + 6 < IW);
		bool lx7 = lh0 && (tiw0 >= -7) && (tiw0 + 7 < IW);
		bool lx8 = lh0 && (tiw0 >= -8) && (tiw0 + 8 < IW);
		bool lx9 = lh0 && (tiw0 >= -9) && (tiw0 + 9 < IW);

		for (int oic = 0; oic < IC; oic += STEP)
		{
			//load 2 group from W
			const int wic = oic + (ty & STEP_m1);//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float2 g0, g1, g2, g3, g4, g5;
			winograd_f4x3_g(g0.x, g1.x, g2.x, g3.x, g4.x, g5.x, w0.x, w1.x, w2.x);
			Gsm[ty][tx] = float4{ g1.x, g2.x, g3.x, g4.x };
			Gsv[ty][tx] = float2{ g0.x, g5.x };
			winograd_f4x3_g(g0.y, g1.y, g2.y, g3.y, g4.y, g5.y, w0.y, w1.y, w2.y);
			Gsm[ty][tx + STEP2] = float4{ g1.y, g2.y, g3.y, g4.y };
			Gsv[ty][tx + STEP2] = float2{ g0.y, g5.y };

			//load 2 group from X
			const int xic = oic + (tx & STEP_m1);//with the same ty
			const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);//0
			float x1 = (lx1 ? X[xoffset + IC] : 0);//1
			float x2 = (lx2 ? X[xoffset + IC2] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC3] : 0);//3
			float x4 = (lx4 ? X[xoffset + IC4] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC5] : 0);//5
			float x6 = (lx6 ? X[xoffset + IC6] : 0);//6
			float x7 = (lx7 ? X[xoffset + IC * 7] : 0);//7
			float x8 = (lx8 ? X[xoffset + (IC << 3)] : 0);//8
			float x9 = (lx9 ? X[xoffset + IC * 9] : 0);
			float2 d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0.x, d1.x, d2.x, d3.x, d4.x, d5.x, x0, x1, x2, x3, x4, x5);
			Dsm[tx][ty] = float4{ d1.x, d2.x, d3.x, d4.x };
			Dsv[tx][ty] = float2{ d0.x, d5.x };
			winograd_f4x3_d(d0.y, d1.y, d2.y, d3.y, d4.y, d5.y, x4, x5, x6, x7, x8, x9);
			Dsm[tx][ty + STEP2] = float4{ d1.y, d2.y, d3.y, d4.y };
			Dsv[tx][ty + STEP2] = float2{ d0.y, d5.y };

			
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float2 gv0 = Gsv[ik][tx];
				float2 gv1 = Gsv[ik][tx + STEP2];
				float2 gv2 = Gsv[ik + STEP][tx];
				float2 gv3 = Gsv[ik + STEP][tx + STEP2];

				float2 dv0 = Dsv[ik][ty];
				float2 dv1 = Dsv[ik][ty + STEP2];
				float2 dv2 = Dsv[ik + STEP][ty];
				float2 dv3 = Dsv[ik + STEP][ty + STEP2];

				v0.x += gv0.x * dv0.x; v0.y += gv1.x * dv0.x; v0.z += gv2.x * dv0.x; v0.w += gv3.x * dv0.x;
				v3.x += gv0.y * dv0.y; v3.y += gv1.y * dv0.y; v3.z += gv2.y * dv0.y; v3.w += gv3.y * dv0.y;
				v4.x += gv0.x * dv1.x; v4.y += gv1.x * dv1.x; v4.z += gv2.x * dv1.x; v4.w += gv3.x * dv1.x;
				v7.x += gv0.y * dv1.y; v7.y += gv1.y * dv1.y; v7.z += gv2.y * dv1.y; v7.w += gv3.y * dv1.y;
				v8.x += gv0.x * dv2.x; v8.y += gv1.x * dv2.x; v8.z += gv2.x * dv2.x; v8.w += gv3.x * dv2.x;
				v11.x += gv0.y * dv2.y; v11.y += gv1.y * dv2.y; v11.z += gv2.y * dv2.y; v11.w += gv3.y * dv2.y;
				v12.x += gv0.x * dv3.x; v12.y += gv1.x * dv3.x; v12.z += gv2.x * dv3.x; v12.w += gv3.x * dv3.x;
				v15.x += gv0.y * dv3.y; v15.y += gv1.y * dv3.y; v15.z += gv2.y * dv3.y; v15.w += gv3.y * dv3.y;
			}

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 gm0 = Gsm[ik][tx];
				float4 gm1 = Gsm[ik][tx + STEP2];
				float4 gm2 = Gsm[ik + STEP][tx];
				float4 gm3 = Gsm[ik + STEP][tx + STEP2];

				float4 dm0 = Dsm[ik][ty];
				float4 dm1 = Dsm[ik][ty + STEP2];
				float4 dm2 = Dsm[ik + STEP][ty];
				float4 dm3 = Dsm[ik + STEP][ty + STEP2];

				m1.x += gm0.x * dm0.x; m1.y += gm1.x * dm0.x; m1.z += gm2.x * dm0.x; m1.w += gm3.x * dm0.x;
				m2.x += gm0.y * dm0.y; m2.y += gm1.y * dm0.y; m2.z += gm2.y * dm0.y; m2.w += gm3.y * dm0.y;
				m3.x += gm0.z * dm0.z; m3.y += gm1.z * dm0.z; m3.z += gm2.z * dm0.z; m3.w += gm3.z * dm0.z;
				m4.x += gm0.w * dm0.w; m4.y += gm1.w * dm0.w; m4.z += gm2.w * dm0.w; m4.w += gm3.w * dm0.w;
				m7.x += gm0.x * dm1.x; m7.y += gm1.x * dm1.x; m7.z += gm2.x * dm1.x; m7.w += gm3.x * dm1.x;
				m8.x += gm0.y * dm1.y; m8.y += gm1.y * dm1.y; m8.z += gm2.y * dm1.y; m8.w += gm3.y * dm1.y;
				m9.x += gm0.z * dm1.z; m9.y += gm1.z * dm1.z; m9.z += gm2.z * dm1.z; m9.w += gm3.z * dm1.z;
				m10.x += gm0.w * dm1.w; m10.y += gm1.w * dm1.w; m10.z += gm2.w * dm1.w; m10.w += gm3.w * dm1.w;
				m13.x += gm0.x * dm2.x; m13.y += gm1.x * dm2.x; m13.z += gm2.x * dm2.x; m13.w += gm3.x * dm2.x;
				m14.x += gm0.y * dm2.y; m14.y += gm1.y * dm2.y; m14.z += gm2.y * dm2.y; m14.w += gm3.y * dm2.y;
				m15.x += gm0.z * dm2.z; m15.y += gm1.z * dm2.z; m15.z += gm2.z * dm2.z; m15.w += gm3.z * dm2.z;
				m16.x += gm0.w * dm2.w; m16.y += gm1.w * dm2.w; m16.z += gm2.w * dm2.w; m16.w += gm3.w * dm2.w;
				m19.x += gm0.x * dm3.x; m19.y += gm1.x * dm3.x; m19.z += gm2.x * dm3.x; m19.w += gm3.x * dm3.x;
				m20.x += gm0.y * dm3.y; m20.y += gm1.y * dm3.y; m20.z += gm2.y * dm3.y; m20.w += gm3.y * dm3.y;
				m21.x += gm0.z * dm3.z; m21.y += gm1.z * dm3.z; m21.z += gm2.z * dm3.z; m21.w += gm3.z * dm3.z;
				m22.x += gm0.w * dm3.w; m22.y += gm1.w * dm3.w; m22.z += gm2.w * dm3.w; m22.w += gm3.w * dm3.w;
			}

			__syncthreads();
		}
	}

	float4 v1, v2;
	winograd_f4x3_v3(v0.x, v1.x, v2.x, v3.x, m1.x, m2.x, m3.x, m4.x);
	winograd_f4x3_v3(v0.y, v1.y, v2.y, v3.y, m1.y, m2.y, m3.y, m4.y);
	winograd_f4x3_v3(v0.z, v1.z, v2.z, v3.z, m1.z, m2.z, m3.z, m4.z);
	winograd_f4x3_v3(v0.w, v1.w, v2.w, v3.w, m1.w, m2.w, m3.w, m4.w);

	float4 v5, v6;
	winograd_f4x3_v3(v4.x, v5.x, v6.x, v7.x, m7.x, m8.x, m9.x, m10.x, m11.x);
	winograd_f4x3_v3(v4.y, v5.y, v6.y, v7.y, m7.y, m8.y, m9.y, m10.y, m11.y);
	winograd_f4x3_v3(v4.z, v5.z, v6.z, v7.z, m7.z, m8.z, m9.z, m10.z, m11.z);
	winograd_f4x3_v3(v4.w, v5.w, v6.w, v7.w, m7.w, m8.w, m9.w, m10.w, m11.w);

	float4 v9, v10;
	winograd_f4x3_v3(v8.x, v9.x, v10.x, v11.x, m13.x, m14.x, m15.x, m16.x);
	winograd_f4x3_v3(v8.y, v9.y, v10.y, v11.y, m13.y, m14.y, m15.y, m16.y);
	winograd_f4x3_v3(v8.z, v9.z, v10.z, v11.z, m13.z, m14.z, m15.z, m16.z);
	winograd_f4x3_v3(v8.w, v9.w, v10.w, v11.w, m13.w, m14.w, m15.w, m16.w);

	float4 v13, v14;
	winograd_f4x3_v3(v12.x, v13.x, v14.x, v15.x, m19.x, m20.x, m21.x, m22.x);
	winograd_f4x3_v3(v12.y, v13.y, v14.y, v15.y, m19.y, m20.y, m21.y, m22.y);
	winograd_f4x3_v3(v12.z, v13.z, v14.z, v15.z, m19.z, m20.z, m21.z, m22.z);
	winograd_f4x3_v3(v12.w, v13.w, v14.w, v15.w, m19.w, m20.w, m21.w, m22.w);

	const int  j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int  j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	const int  j8 = j7 + OC, j9 = j8 + OC, j10 = j9 + OC, j11 = j10 + OC;
	const int j12 = j11 + OC, j13 = j12 + OC, j14 = j13 + OC, j15 = j14 + OC;

	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;

	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;

	*(float4*)(Y + j8) = v8; *(float4*)(Y + j9) = v9;
	*(float4*)(Y + j10) = v10; *(float4*)(Y + j11) = v11;

	*(float4*)(Y + j12) = v12; *(float4*)(Y + j13) = v13;
	*(float4*)(Y + j14) = v14; *(float4*)(Y + j15) = v15;
}

#endif


