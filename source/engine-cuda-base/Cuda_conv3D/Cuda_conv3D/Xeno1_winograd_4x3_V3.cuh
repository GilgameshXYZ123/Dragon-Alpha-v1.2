

//half shared memory
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_C1
#define XENO1_WINOGRAD_F4X3_C1

#define xeno_winograd_f4x3_c1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_C1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 2.34835 msec, Performace = 8230.18 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_C1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	__shared__ float4 Gs0[1 << LB][(2 << LB) + 1];
	__shared__ float2 Gs1[1 << LB][(2 << LB) + 1];
	__shared__ float4 Ds0[1 << LB][(1 << LB) + 1];
	__shared__ float2 Ds1[1 << LB][(1 << LB) + 1];

	float4 m0 = F32_4_0, m1 = F32_4_0, m2 = F32_4_0;//0-3
	float4 m3 = F32_4_0, m4 = F32_4_0, m5 = F32_4_0;
	float4 m6 = F32_4_0, m7 = F32_4_0, m8 = F32_4_0;//4-8
	float4 m9 = F32_4_0, ma = F32_4_0, mb = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
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
			float x2 = (lx2 ? X[xoffset + IC * 2] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + IC * 4] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
			Ds0[tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[tx][ty] = float2{ d4, d5 };

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
				float4 d2 = Ds0[ik + STEP][ty]; float2 d3 = Ds1[ik + STEP][ty];

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
				ma.x += g1.x * d3.x; ma.y += g3.x * d3.x; ma.z += g5.x * d3.x; ma.w += g7.x * d3.x;
				mb.x += g1.y * d3.y; mb.y += g3.y * d3.y; mb.z += g5.y * d3.y; mb.w += g7.y * d3.y;
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
	winograd_f4x3_v(v4.x, v5.x, v6.x, v7.x, m6.x, m7.x, m8.x, m9.x, ma.x, mb.x);
	winograd_f4x3_v(v4.y, v5.y, v6.y, v7.y, m6.y, m7.y, m8.y, m9.y, ma.y, mb.y);
	winograd_f4x3_v(v4.z, v5.z, v6.z, v7.z, m6.z, m7.z, m8.z, m9.z, ma.z, mb.z);
	winograd_f4x3_v(v4.w, v5.w, v6.w, v7.w, m6.w, m7.w, m8.w, m9.w, ma.w, mb.w);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4;
	*(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6;
	*(float4*)(Y + j7) = v7;
}

#endif


//half shared memory
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_C2
#define XENO1_WINOGRAD_F4X3_C2

#define xeno_winograd_f4x3_c2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_C2<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 2.32119 msec, Performace = 8326.48 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_C2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	bool buf = 0;

	__shared__ float4 Gs0[2][1 << LB][(2 << LB) + 1];
	__shared__ float2 Gs1[2][1 << LB][(2 << LB) + 1];

	__shared__ float4 Ds0[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ds1[2][1 << LB][(1 << LB) + 1];

	float4 m0 = F32_4_0, m1 = F32_4_0, m2 = F32_4_0;//0-3
	float4 m3 = F32_4_0, m4 = F32_4_0, m5 = F32_4_0;
	float4 m6 = F32_4_0, m7 = F32_4_0, m8 = F32_4_0;//4-8
	float4 m9 = F32_4_0, ma = F32_4_0, mb = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
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
		float x2 = (lx2 ? X[xoffset + IC * 2] : 0);//2
		float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
		float x4 = (lx4 ? X[xoffset + IC * 4] : 0);//4
		float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
		float d0, d1, d2, d3, d4, d5;
		winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
		Ds0[buf][tx][ty] = float4{ d0, d1, d2, d3 };
		Ds1[buf][tx][ty] = float2{ d4, d5 };

		//load 2 group from W
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		float2 g0, g1, g2, g3, g4, g5;
		winograd_f4x3_g(g0.x, g1.x, g2.x, g3.x, g4.x, g5.x, w0.x, w1.x, w2.x);
		Gs0[buf][ty][tx] = float4{ g0.x, g1.x, g2.x, g3.x };
		Gs1[buf][ty][tx] = float2{ g4.x, g5.x };
		winograd_f4x3_g(g0.y, g1.y, g2.y, g3.y, g4.y, g5.y, w0.y, w1.y, w2.y);
		Gs0[buf][ty][tx + STEP2] = float4{ g0.y, g1.y, g2.y, g3.y };
		Gs1[buf][ty][tx + STEP2] = float2{ g4.y, g5.y };
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds0[buf][ik][ty]; float2 d1 = Ds1[buf][ik][ty];
				float4 d2 = Ds0[buf][ik + STEP][ty]; float2 d3 = Ds1[buf][ik + STEP][ty];

				float4 g0 = Gs0[buf][ik][tx]; float2 g1 = Gs1[buf][ik][tx];
				float4 g2 = Gs0[buf][ik][tx + STEP2]; float2 g3 = Gs1[buf][ik][tx + STEP2];
				float4 g4 = Gs0[buf][ik + STEP][tx]; float2 g5 = Gs1[buf][ik + STEP][tx];
				float4 g6 = Gs0[buf][ik + STEP][tx + STEP2]; float2 g7 = Gs1[buf][ik + STEP][tx + STEP2];

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
				ma.x += g1.x * d3.x; ma.y += g3.x * d3.x; ma.z += g5.x * d3.x; ma.w += g7.x * d3.x;
				mb.x += g1.y * d3.y; mb.y += g3.y * d3.y; mb.z += g5.y * d3.y; mb.w += g7.y * d3.y;
			}
			buf ^= 1;

			//load 2 group from X
			const int xic = oic + (tx & STEP_m1);//with the same ty
			const int xoffset = ((tn0*IH + (tih0 + fh))*IW + tiw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);//0
			float x1 = (lx1 ? X[xoffset + IC] : 0);//1
			float x2 = (lx2 ? X[xoffset + IC * 2] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + IC * 4] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
			Ds0[buf][tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[buf][tx][ty] = float2{ d4, d5 };

			//load 2 group from W
			const int wic = oic + (ty & STEP_m1);//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float2 g0, g1, g2, g3, g4, g5;
			winograd_f4x3_g(g0.x, g1.x, g2.x, g3.x, g4.x, g5.x, w0.x, w1.x, w2.x);
			Gs0[buf][ty][tx] = float4{ g0.x, g1.x, g2.x, g3.x };
			Gs1[buf][ty][tx] = float2{ g4.x, g5.x };
			winograd_f4x3_g(g0.y, g1.y, g2.y, g3.y, g4.y, g5.y, w0.y, w1.y, w2.y);
			Gs0[buf][ty][tx + STEP2] = float4{ g0.y, g1.y, g2.y, g3.y };
			Gs1[buf][ty][tx + STEP2] = float2{ g4.y, g5.y };
			__syncthreads();
		}
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 d0 = Ds0[buf][ik][ty]; float2 d1 = Ds1[buf][ik][ty];
			float4 d2 = Ds0[buf][ik + STEP][ty]; float2 d3 = Ds1[buf][ik + STEP][ty];

			float4 g0 = Gs0[buf][ik][tx]; float2 g1 = Gs1[buf][ik][tx];
			float4 g2 = Gs0[buf][ik][tx + STEP2]; float2 g3 = Gs1[buf][ik][tx + STEP2];
			float4 g4 = Gs0[buf][ik + STEP][tx]; float2 g5 = Gs1[buf][ik + STEP][tx];
			float4 g6 = Gs0[buf][ik + STEP][tx + STEP2]; float2 g7 = Gs1[buf][ik + STEP][tx + STEP2];

			m1.x += g0.y * d0.y; m1.y += g2.y * d0.y; m1.z += g4.y * d0.y; m1.w += g6.y * d0.y;
			m2.x += g0.z * d0.z; m2.y += g2.z * d0.z; m2.z += g4.z * d0.z; m2.w += g6.z * d0.z;
			m3.x += g0.w * d0.w; m3.y += g2.w * d0.w; m3.z += g4.w * d0.w; m3.w += g6.w * d0.w;
			m4.x += g1.x * d1.x; m4.y += g3.x * d1.x; m4.z += g5.x * d1.x; m4.w += g7.x * d1.x;
			m5.x += g1.y * d1.y; m5.y += g3.y * d1.y; m5.z += g5.y * d1.y; m5.w += g7.y * d1.y;

			m6.x += g0.x * d2.x; m6.y += g2.x * d2.x; m6.z += g4.x * d2.x; m6.w += g6.x * d2.x;
			m7.x += g0.y * d2.y; m7.y += g2.y * d2.y; m7.z += g4.y * d2.y; m7.w += g6.y * d2.y;
			m8.x += g0.z * d2.z; m8.y += g2.z * d2.z; m8.z += g4.z * d2.z; m8.w += g6.z * d2.z;
			m9.x += g0.w * d2.w; m9.y += g2.w * d2.w; m9.z += g4.w * d2.w; m9.w += g6.w * d2.w;
			ma.x += g1.x * d3.x; ma.y += g3.x * d3.x; ma.z += g5.x * d3.x; ma.w += g7.x * d3.x;
			mb.x += g1.y * d3.y; mb.y += g3.y * d3.y; mb.z += g5.y * d3.y; mb.w += g7.y * d3.y;
		}
		buf ^= 1;
	}

	float4 v0, v1, v2, v3;
	winograd_f4x3_v(v0.x, v1.x, v2.x, v3.x, m0.x, m1.x, m2.x, m3.x, m4.x, m5.x);
	winograd_f4x3_v(v0.y, v1.y, v2.y, v3.y, m0.y, m1.y, m2.y, m3.y, m4.y, m5.y);
	winograd_f4x3_v(v0.z, v1.z, v2.z, v3.z, m0.z, m1.z, m2.z, m3.z, m4.z, m5.z);
	winograd_f4x3_v(v0.w, v1.w, v2.w, v3.w, m0.w, m1.w, m2.w, m3.w, m4.w, m5.w);

	float4 v4, v5, v6, v7;
	winograd_f4x3_v(v4.x, v5.x, v6.x, v7.x, m6.x, m7.x, m8.x, m9.x, ma.x, mb.x);
	winograd_f4x3_v(v4.y, v5.y, v6.y, v7.y, m6.y, m7.y, m8.y, m9.y, ma.y, mb.y);
	winograd_f4x3_v(v4.z, v5.z, v6.z, v7.z, m6.z, m7.z, m8.z, m9.z, ma.z, mb.z);
	winograd_f4x3_v(v4.w, v5.w, v6.w, v7.w, m6.w, m7.w, m8.w, m9.w, ma.w, mb.w);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4;
	*(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6;
	*(float4*)(Y + j7) = v7;
}

#endif