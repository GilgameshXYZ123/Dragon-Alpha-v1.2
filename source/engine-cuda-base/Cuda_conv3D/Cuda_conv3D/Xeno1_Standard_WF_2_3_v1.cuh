#pragma once


#ifndef AWINOGRAD_2X3_KERNEL1
#define AWINOGRAD_2X3_KERNEL1

#define AW23_K1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)


template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void AW23_kernel1(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__  Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Gst[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dst[2][1 << LB][(1 << LB) + 1];

	//compute 4*8 results: 8*8 accumulators
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << LB << 2) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for Y[N, OH, OW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 2);//oc = f(by)
	const int yj0 = bj0 + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)

#pragma once //compute area------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

		//load 2 group from CW
		const int woffset = fh * IC3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
		float4 d1 = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, w2.x, w0.y, w2.y };
		float4 g1 = float4{ gst0, gst1, gst2, gst3 };

		//write to shread memory
		Dsv[buf][tx][ty] = d0; Dst[buf][tx][ty] = d1;
		Gsv[buf][ty][tx] = g0; Gst[buf][ty][tx] = g1;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gsv[buf][ik][ux], g2 = Gsv[buf][ik + STEP][ux];//{x, w} for v;
				float4 d0 = Dsv[buf][ik][uy], d2 = Dsv[buf][ik + STEP][uy];//{x, w} for v;

				winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
				winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

				float4 g1 = Gst[buf][ik][ux], g3 = Gst[buf][ik + STEP][ux];//{y, z} for t
				float4 d1 = Dst[buf][ik][uy], d3 = Dst[buf][ik + STEP][uy];//{y, z} for t

				winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
				winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
			}
			buf ^= 1;

			//load 2 group from X
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

			//load 2 group from CW
			const int woffset = (fh*IC3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//Winograd transform: W(3) -> G(4); X[4] -> D[4]
			float4 d0 = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
			float4 d1 = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, w2.x, w0.y, w2.y };
			float4 g1 = float4{ gst0, gst1, gst2, gst3 };

			//write to shread memory
			Dsv[buf][tx][ty] = d0; Dst[buf][tx][ty] = d1;
			Gsv[buf][ty][tx] = g0; Gst[buf][ty][tx] = g1;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gsv[buf][ik][ux], g2 = Gsv[buf][ik + STEP][ux];//{x, w} for v;
			float4 d0 = Dsv[buf][ik][uy], d2 = Dsv[buf][ik + STEP][uy];//{x, w} for v;

			winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
			winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

			float4 g1 = Gst[buf][ik][ux], g3 = Gst[buf][ik + STEP][ux];//{y, z} for t
			float4 d1 = Dst[buf][ik][uy], d3 = Dst[buf][ik + STEP][uy];//{y, z} for t

			winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
			winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2; *(float4*)(Y + Y3) = v3;
	*(float4*)(Y + Y4) = v4; *(float4*)(Y + Y5) = v5;
	*(float4*)(Y + Y6) = v6; *(float4*)(Y + Y7) = v7;
}

#endif


#define AWINOGRAD_2X3_KERNEL2
#ifndef AWINOGRAD_2X3_KERNEL2
#define AWINOGRAD_2X3_KERNEL2

//64 * 128
//[OW % 4] == 0
#define AW23_K2(stream, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel2\
		<<< dim3(GN>>6, GM>>7), dim3(32, 8), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 18, Time = 3.16233 msec, Performace = 12223.5 GFlop/s
__global__ void AW23_kernel2(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__  Y, int OH_OW, int OW, 
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4][64 + 4];
	__shared__ float Ds[2][8][4][64 + 4];

	//compute 8*8 accumulators
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 1);//2 * 4 = 8 elements
	CW += toc0;//Cw[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (tx << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//prepare for thread_offset: 4 * (64 / 4) = 4 * 16
	const int ux = (ty & 3);//0 -> 4
	const int uy = (tx << 1) + (ty > 4);//0 -> 64
	const int XIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7
	const int GIdx = ((uy & 15) >> 1)              << 3;//0 -> 7

	//prepare for Y[N, OH, OW, OC]
	//uy: oc = GIdx,         j = XIdx;
	//ux: oc = (ux & 1) * 4, j = (ux >> 1)

	//======[compute area1: local]======================================================
	const int Wstride = IC * OC;

#pragma once
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC * 2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC * 3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC * 4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC * 5, -1));

		//load 2 group from CW
		const int woffset = fh * IC * 3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
		float4 d1 = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
		float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

		//write to shread memory
		Gs[buf][ty][0][(tx << 1)] = g0.x; Gs[buf][ty][0][(tx << 1) + 1] = g1.x;
		Gs[buf][ty][1][(tx << 1)] = g0.y; Gs[buf][ty][1][(tx << 1) + 1] = g1.y;
		Gs[buf][ty][2][(tx << 1)] = g0.z; Gs[buf][ty][2][(tx << 1) + 1] = g1.z;
		Gs[buf][ty][3][(tx << 1)] = g0.w; Gs[buf][ty][3][(tx << 1) + 1] = g1.w;

		Ds[buf][ty][0][(tx << 1)] = d0.x; Ds[buf][ty][0][(tx << 1) + 1] = d1.x;
		Ds[buf][ty][1][(tx << 1)] = d0.y; Ds[buf][ty][1][(tx << 1) + 1] = d1.y;
		Ds[buf][ty][2][(tx << 1)] = d0.z; Ds[buf][ty][2][(tx << 1) + 1] = d1.z;
		Ds[buf][ty][3][(tx << 1)] = d0.w; Ds[buf][ty][3][(tx << 1) + 1] = d1.w;
		__syncthreads();

		for (int oic = 8; oic < IC; oic += 8) {
#pragma unroll
			for (int ik = 0; ik < 8; ik++)
			{
				float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[ik][ux][XIdx]), b1 = *(float4*)(&Ds[ik][ux][XIdx + 4]);

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

			//load 2 group from X
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset         , -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC    , -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC * 2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC * 3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC * 4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC * 5, -1));

			//load 2 group from CW
			const int woffset = (fh*IC*3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
			float4 d1 = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
			float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

			//write to shread memory
			Gs[buf][ty][0][(tx << 1)] = g0.x; Gs[buf][ty][0][(tx << 1) + 1] = g1.x;
			Gs[buf][ty][1][(tx << 1)] = g0.y; Gs[buf][ty][1][(tx << 1) + 1] = g1.y;
			Gs[buf][ty][2][(tx << 1)] = g0.z; Gs[buf][ty][2][(tx << 1) + 1] = g1.z;
			Gs[buf][ty][3][(tx << 1)] = g0.w; Gs[buf][ty][3][(tx << 1) + 1] = g1.w;

			Ds[buf][ty][0][(tx << 1)] = d0.x; Ds[buf][ty][0][(tx << 1) + 1] = d1.x;
			Ds[buf][ty][1][(tx << 1)] = d0.y; Ds[buf][ty][1][(tx << 1) + 1] = d1.y;
			Ds[buf][ty][2][(tx << 1)] = d0.z; Ds[buf][ty][2][(tx << 1) + 1] = d1.z;
			Ds[buf][ty][3][(tx << 1)] = d0.w; Ds[buf][ty][3][(tx << 1) + 1] = d1.w;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < 8; ik++)
		{
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][XIdx]), b1 = *(float4*)(&Ds[ik][ux][XIdx + 4]);

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
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
	

	/*const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2; *(float4*)(Y + Y3) = v3;
	*(float4*)(Y + Y4) = v4; *(float4*)(Y + Y5) = v5;
	*(float4*)(Y + Y6) = v6; *(float4*)(Y + Y7) = v7;*/
}

#endif


#ifndef AWINOGRAD_2X3_KERNEL3
#define AWINOGRAD_2X3_KERNEL3

#define AW23_K3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel3<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void AW23_kernel3(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__  Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	//2 * (2 * 8 * 4 * 64) = 32 * 4 * 64
	//combine:
	//[ux = 4][uy = 64][elem = 32]
	//so: write 32 elem to shared memory per turn: split by oc channel

	bool buf = 0;
	__shared__ float Gs[2][8][4][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << LB << 2) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << LB) + tx;
	const int ux = (idx / 64);//0 ->  3
	const int uy = (idx & 63);//0 -> 63
	const int XIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7
	const int GIdx = ((uy & 15) >> 1) << 3;//0 -> 7

	//======[compute area1: local]======================================================
	const int Wstride = IC * OC;
#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

		//load 2 group from CW[FH, 3, IC, OC]
		const int woffset = fh * IC3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset                 );//fw = 0
		float2 w1 = *(float2*)(CW + woffset + Wstride       );//fw = 1
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));//fw = 2

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
		float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, gst0, gst1, w1.x };
		float4 g1 = float4{ w0.y, gst2, gst3, w1.y };
		
		//tj0  = bj0  + (ty << 3) + ((tx >= STEP) << 2);
		//toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
		//CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]
		//const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

		//const int yoc = boc + GIdx + ((ux & 1) << 2);
		//const int yj  = bj  + XIdx + (ux >> 1);

		//we know:
		//in read shared memory: 
		//ty -> oc -> XIdx
		//tx ->  j -> GIdx

		Ds[buf][tx & STEP_m1][0][(ty << 1)] = d0.x; Ds[buf][tx & STEP_m1][0][(ty << 1) + 1] = d1.x;//k = tx & STEP_m1
		Ds[buf][tx & STEP_m1][1][(ty << 1)] = d0.y; Ds[buf][tx & STEP_m1][1][(ty << 1) + 1] = d1.y;
		Ds[buf][tx & STEP_m1][2][(ty << 1)] = d0.z; Ds[buf][tx & STEP_m1][2][(ty << 1) + 1] = d1.z;
		Ds[buf][tx & STEP_m1][3][(ty << 1)] = d0.w; Ds[buf][tx & STEP_m1][3][(ty << 1) + 1] = d1.w;

		Gs[buf][ty & STEP_m1][0][(tx << 1)] = g0.x; Gs[buf][ty & STEP_m1][0][(tx << 1) + 1] = g1.x;//k = ty & STEP_m1
		Gs[buf][ty & STEP_m1][1][(tx << 1)] = g0.y; Gs[buf][ty & STEP_m1][1][(tx << 1) + 1] = g1.y;
		Gs[buf][ty & STEP_m1][2][(tx << 1)] = g0.z; Gs[buf][ty & STEP_m1][2][(tx << 1) + 1] = g1.z;
		Gs[buf][ty & STEP_m1][3][(tx << 1)] = g0.w; Gs[buf][ty & STEP_m1][3][(tx << 1) + 1] = g1.w;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][XIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][XIdx + 4]);

				//oc[0 - 3]             oc[4 - 7]
				simdMM4(v0,  b0.x, a0); simdMM4(v1,  b0.x, a1);//j0
				simdMM4(v2,  b0.y, a0); simdMM4(v3,  b0.y, a1);//j1
				simdMM4(v4,  b0.z, a0); simdMM4(v5,  b0.z, a1);//j2
				simdMM4(v6,  b0.w, a0); simdMM4(v7,  b0.w, a1);//j3
				simdMM4(v8,  b1.x, a0); simdMM4(v9,  b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}
			buf ^= 1;

			//load 2 group from X[N, IH, IW, IC]
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

			//load 2 group from CW[FH, 3, IC, OC]
			const int woffset = (fh*IC3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, gst0, gst1, w1.x };
			float4 g1 = float4{ w0.y, gst2, gst3, w1.y };

			//write to shread memory
			Ds[buf][tx & STEP_m1][0][(ty << 1)] = d0.x; Ds[buf][tx & STEP_m1][0][(ty << 1) + 1] = d1.x;
			Ds[buf][tx & STEP_m1][1][(ty << 1)] = d0.y; Ds[buf][tx & STEP_m1][1][(ty << 1) + 1] = d1.y;
			Ds[buf][tx & STEP_m1][2][(ty << 1)] = d0.z; Ds[buf][tx & STEP_m1][2][(ty << 1) + 1] = d1.z;
			Ds[buf][tx & STEP_m1][3][(ty << 1)] = d0.w; Ds[buf][tx & STEP_m1][3][(ty << 1) + 1] = d1.w;

			Gs[buf][ty & STEP_m1][0][(tx << 1)] = g0.x; Gs[buf][ty & STEP_m1][0][(tx << 1) + 1] = g1.x;
			Gs[buf][ty & STEP_m1][1][(tx << 1)] = g0.y; Gs[buf][ty & STEP_m1][1][(tx << 1) + 1] = g1.y;
			Gs[buf][ty & STEP_m1][2][(tx << 1)] = g0.z; Gs[buf][ty & STEP_m1][2][(tx << 1) + 1] = g1.z;
			Gs[buf][ty & STEP_m1][3][(tx << 1)] = g0.w; Gs[buf][ty & STEP_m1][3][(tx << 1) + 1] = g1.w;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[buf][ik][ux][XIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][XIdx + 4]);

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
	//2 * (2 * 8 * 4 * 64) = 32 * 4 * 64
	//combine:
	//[ux = 4][uy = 64][elem = 32]
	//so: write 32 elem to shared memory per turn: split by oc channel

	//bool buf = 0;
	//__shared__ float Gs[2][8][4][64 + 4];//[buf][ik][elem][64 * oc]
	//__shared__ float Ds[2][8][4][64 + 4];//[buf][ik][elem][64 *  j]

	//4, 64, (32/2) = 16
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
	float a[4], y0[2], y1[2], y2[2], y3[2];
	int turn_j;

	//======[turn01]======================================================
	//write-turn0: oc0 -> oc3
	{
		*(float4*)(&get3d(Ys0, ux, uy, 0, 64, 16)) = v0;//j0 -> j3
		*(float4*)(&get3d(Ys0, ux, uy, 4, 64, 16)) = v2;
		*(float4*)(&get3d(Ys0, ux, uy, 8, 64, 16)) = v4;
		*(float4*)(&get3d(Ys0, ux, uy, 12, 64, 16)) = v6;
	}
	
	//write-turn1: oc4 -> oc7
	{
		*(float4*)(&get3d(Ys1, ux, uy, 0, 64, 16)) = v1;//j0 -> j3
		*(float4*)(&get3d(Ys1, ux, uy, 4, 64, 16)) = v3;
		*(float4*)(&get3d(Ys1, ux, uy, 8, 64, 16)) = v5;
		*(float4*)(&get3d(Ys1, ux, uy, 12, 64, 16)) = v6;
	}
	__syncthreads();

	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + XIdx;
	const int yoffset = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)

	const int Y0 = yoffset + (ux << 1) * OC;
	{
		a[0] = get3d(Ys0, 0, uy, (ux << 2), 64, 16);//oc0
		a[1] = get3d(Ys0, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 1, 64, 16);//oc1
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 2, 64, 16);//oc2
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		//oc = 0 -> 3: += 4
		//j = ux << 1;
		
		*(float4*)(Y + Y0    )  = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y0 + OC) = { y0[1], y1[1], y2[1], y3[1] };
	}
	
	{
		a[0] = get3d(Ys1, 0, uy, (ux << 2), 64, 16);//oc4
		a[1] = get3d(Ys1, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 1, 64, 16);//oc5
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 2, 64, 16);//oc
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y0      + 4) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y0 + OC + 4) = { y0[1], y1[1], y2[1], y3[1] };
	}
	__syncthreads();

	//======[turn23]======================================================
	//write-turn2: j4, j5
	{
		*(float4*)(&get3d(Ys0, ux, uy, 0, 64, 16)) = v8;
		*(float4*)(&get3d(Ys0, ux, uy, 4, 64, 16)) = v9;
		*(float4*)(&get3d(Ys0, ux, uy, 8, 64, 16)) = v10;
		*(float4*)(&get3d(Ys0, ux, uy, 12, 64, 16)) = v11;
	}

	//write-turn3: j6, j7
	{
		*(float4*)(&get3d(Ys1, ux, uy, 0, 64, 16)) = v12;
		*(float4*)(&get3d(Ys1, ux, uy, 4, 64, 16)) = v13;
		*(float4*)(&get3d(Ys1, ux, uy, 8, 64, 16)) = v14;
		*(float4*)(&get3d(Ys1, ux, uy, 12, 64, 16)) = v15;
	}
	__syncthreads();

	const int Y1 = yoffset + ((ux << 1) + 4) * OC;
	{
		a[0] = get3d(Ys0, 0, uy, (ux << 2), 64, 16);//oc0
		a[1] = get3d(Ys0, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 1, 64, 16);//oc1
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 2, 64, 16);//oc2
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y1     ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y1 + OC) = { y0[1], y1[1], y2[1], y3[1] };
	}

	{
		a[0] = get3d(Ys1, 0, uy, (ux << 2), 64, 16);//oc4
		a[1] = get3d(Ys1, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 1, 64, 16);//oc5
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 2, 64, 16);//oc
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y1      + 4) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y1 + OC + 4) = { y0[1], y1[1], y2[1], y3[1] };
	}
}

#endif


#ifndef AWINOGRAD_2X3_KERNEL4
#define AWINOGRAD_2X3_KERNEL4

#define AW23_K4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel4<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void AW23_kernel4(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__  Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << LB) + tx;
	const int ux = (idx / 64);//0 ->  3
	const int uy = (idx & 63);//0 -> 63
	int DIdx = ((uy & 1) + ((uy >> 4) << 1));//0 -> 7, j 8*8 = 64
	int GIdx = ((uy & 15) >> 1)             ;//0 -> 7, oc
	DIdx *= 8;
	GIdx *= 8;

	//======[compute area1: local]======================================================
	const int Wstride = IC * OC;
#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

		//load 2 group from CW[FH, 3, IC, OC]
		const int woffset = fh * IC3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset                 );//fw = 0
		float2 w1 = *(float2*)(CW + woffset + Wstride       );//fw = 1
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));//fw = 2

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
		float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, gst0, gst1, w1.x };
		float4 g1 = float4{ w0.y, gst2, gst3, w1.y };
		
		Ds[buf][tx & STEP_m1][0][(ty << 1)] = d0.x; Ds[buf][tx & STEP_m1][0][(ty << 1) + 1] = d1.x;//k = tx & STEP_m1
		Ds[buf][tx & STEP_m1][1][(ty << 1)] = d0.y; Ds[buf][tx & STEP_m1][1][(ty << 1) + 1] = d1.y;
		Ds[buf][tx & STEP_m1][2][(ty << 1)] = d0.z; Ds[buf][tx & STEP_m1][2][(ty << 1) + 1] = d1.z;
		Ds[buf][tx & STEP_m1][3][(ty << 1)] = d0.w; Ds[buf][tx & STEP_m1][3][(ty << 1) + 1] = d1.w;

		Gs[buf][ty & STEP_m1][0][(tx << 1)] = g0.x; Gs[buf][ty & STEP_m1][0][(tx << 1) + 1] = g1.x;//k = ty & STEP_m1
		Gs[buf][ty & STEP_m1][1][(tx << 1)] = g0.y; Gs[buf][ty & STEP_m1][1][(tx << 1) + 1] = g1.y;
		Gs[buf][ty & STEP_m1][2][(tx << 1)] = g0.z; Gs[buf][ty & STEP_m1][2][(tx << 1) + 1] = g1.z;
		Gs[buf][ty & STEP_m1][3][(tx << 1)] = g0.w; Gs[buf][ty & STEP_m1][3][(tx << 1) + 1] = g1.w;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

				//oc[0 - 3]             oc[4 - 7]
				simdMM4(v0,  b0.x, a0); simdMM4(v1,  b0.x, a1);//j0
				simdMM4(v2,  b0.y, a0); simdMM4(v3,  b0.y, a1);//j1
				simdMM4(v4,  b0.z, a0); simdMM4(v5,  b0.z, a1);//j2
				simdMM4(v6,  b0.w, a0); simdMM4(v7,  b0.w, a1);//j3
				simdMM4(v8,  b1.x, a0); simdMM4(v9,  b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}
			buf ^= 1;

			//load 2 group from X[N, IH, IW, IC]
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

			//load 2 group from CW[FH, 3, IC, OC]
			const int woffset = (fh*IC3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, gst0, gst1, w1.x };
			float4 g1 = float4{ w0.y, gst2, gst3, w1.y };

			//write to shread memory
			Ds[buf][tx & STEP_m1][0][(ty << 1)] = d0.x; Ds[buf][tx & STEP_m1][0][(ty << 1) + 1] = d1.x;
			Ds[buf][tx & STEP_m1][1][(ty << 1)] = d0.y; Ds[buf][tx & STEP_m1][1][(ty << 1) + 1] = d1.y;
			Ds[buf][tx & STEP_m1][2][(ty << 1)] = d0.z; Ds[buf][tx & STEP_m1][2][(ty << 1) + 1] = d1.z;
			Ds[buf][tx & STEP_m1][3][(ty << 1)] = d0.w; Ds[buf][tx & STEP_m1][3][(ty << 1) + 1] = d1.w;

			Gs[buf][ty & STEP_m1][0][(tx << 1)] = g0.x; Gs[buf][ty & STEP_m1][0][(tx << 1) + 1] = g1.x;
			Gs[buf][ty & STEP_m1][1][(tx << 1)] = g0.y; Gs[buf][ty & STEP_m1][1][(tx << 1) + 1] = g1.y;
			Gs[buf][ty & STEP_m1][2][(tx << 1)] = g0.z; Gs[buf][ty & STEP_m1][2][(tx << 1) + 1] = g1.z;
			Gs[buf][ty & STEP_m1][3][(tx << 1)] = g0.w; Gs[buf][ty & STEP_m1][3][(tx << 1) + 1] = g1.w;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
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
	//2 * (2 * 8 * 4 * 64) = 32 * 4 * 64
	//combine:
	//[ux = 4][uy = 64][elem = 32]
	//so: write 32 elem to shared memory per turn: split by oc channel

	//bool buf = 0;
	//__shared__ float Gs[2][8][4][64 + 4];//[buf][ik][elem][64 * oc]
	//__shared__ float Ds[2][8][4][64 + 4];//[buf][ik][elem][64 *  j]

	//4, 64, (32/2) = 16
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
	float a[4], y0[2], y1[2], y2[2], y3[2];

	//======[turn01]======================================================
	//write-turn0: 
	{
		//oc0 -> oc3                                      // oc4 -> oc7
		*(float4*)(&get3d(Ys0, ux, uy,  0, 64, 16)) = v0; *(float4*)(&get3d(Ys1, ux, uy,  0, 64, 16)) = v1;
		*(float4*)(&get3d(Ys0, ux, uy,  4, 64, 16)) = v2; *(float4*)(&get3d(Ys1, ux, uy,  4, 64, 16)) = v3;
		*(float4*)(&get3d(Ys0, ux, uy,  8, 64, 16)) = v4; *(float4*)(&get3d(Ys1, ux, uy,  8, 64, 16)) = v5;
		*(float4*)(&get3d(Ys0, ux, uy, 12, 64, 16)) = v6; *(float4*)(&get3d(Ys1, ux, uy, 12, 64, 16)) = v7;
	}
	__syncthreads();

	//oc0 = 0 -> 63: GIdx = (0 - 7)*8 = 0 -> 56 + 8 -> 64
	//yj0 = (0 -> 7) * 16 = 7 * 16 + 3*2 + 8 + 1
	//

	const int yoc0 = boc0 + GIdx;
	const int yj0  = bj0  + (DIdx << 1);
	const int yoffset = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)

	//[4][64][16]
	const int Y0 = yoffset + (ux << 1) * OC;//[<n, oh, ow>, oc]
	{
		a[0] = get3d(Ys0, 0, uy, (ux << 2), 64, 16);//oc0
		a[1] = get3d(Ys0, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 1, 64, 16);//oc1
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 2, 64, 16);//oc2
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		//oc = 0 -> 3: += 4
		//j = ux << 1;
		
		*(float4*)(Y + Y0     ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y0 + OC) = { y0[1], y1[1], y2[1], y3[1] };
	}

	{
		a[0] = get3d(Ys1, 0, uy, (ux << 2), 64, 16);//oc4
		a[1] = get3d(Ys1, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 1, 64, 16);//oc5
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 2, 64, 16);//oc
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y0      + 4) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y0 + OC + 4) = { y0[1], y1[1], y2[1], y3[1] };
	}
	__syncthreads();

	//======[turn23]======================================================
	//write-turn2: j4, j5
	{
		*(float4*)(&get3d(Ys0, ux, uy,  0, 64, 16)) =  v8;
		*(float4*)(&get3d(Ys0, ux, uy,  4, 64, 16)) = v10;
		*(float4*)(&get3d(Ys0, ux, uy,  8, 64, 16)) = v12;
		*(float4*)(&get3d(Ys0, ux, uy, 12, 64, 16)) = v14;
	}

	//write-turn3: j6, j7
	{
		*(float4*)(&get3d(Ys1, ux, uy,  0, 64, 16)) =  v9;
		*(float4*)(&get3d(Ys1, ux, uy,  4, 64, 16)) = v11;
		*(float4*)(&get3d(Ys1, ux, uy,  8, 64, 16)) = v13;
		*(float4*)(&get3d(Ys1, ux, uy, 12, 64, 16)) = v15;
	}
	__syncthreads();

	const int Y1 = yoffset + ((ux << 1) + 8) * OC;
	{
		a[0] = get3d(Ys0, 0, uy, (ux << 2), 64, 16);//oc0
		a[1] = get3d(Ys0, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 1, 64, 16);//oc1
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 2, 64, 16);//oc2
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y1     ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y1 + OC) = { y0[1], y1[1], y2[1], y3[1] };
	}

	{
		a[0] = get3d(Ys1, 0, uy, (ux << 2), 64, 16);//oc4
		a[1] = get3d(Ys1, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 1, 64, 16);//oc5
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 2, 64, 16);//oc
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y1      + 4) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y1 + OC + 4) = { y0[1], y1[1], y2[1], y3[1] };
	}
}

#endif



#ifndef AWINOGRAD_2X3_KERNEL5
#define AWINOGRAD_2X3_KERNEL5

#define AW23_K5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel5<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//Size = 18, Time = 3.02868 msec, Performace = 12762.9 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void AW23_kernel5(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0  = bj0 + (ty << 3)  + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx / 64);//0 ->  3
	const int uy = (idx % 64);//0 -> 63
	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int GIdx = ((uy & 15) >> 1)              << 3;//0 -> 7, oc

	//======[compute area1: local]======================================================
	const int Wstride = IC * OC;
#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

		//load 2 group from CW[FH, 3, IC, OC]
		const int woffset = fh * IC3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset                 );//fw = 0
		float2 w1 = *(float2*)(CW + woffset + Wstride       );//fw = 1
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));//fw = 2

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
		float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;

		//float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = 0.5f*(w0.x - w1.x + w2.x);
		//float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = 0.5f*(w0.y - w1.y + w2.y);
		float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
		float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

		const int Ds_k = tx & STEP_m1;
		const int Ds_i = (ty << 2) + ((tx >= STEP) << 1);

		const int Gs_k = ty & STEP_m1;
		const int Gs_i = (tx << 2) + ((ty >= STEP) << 1);

		Ds[buf][Ds_k][0][Ds_i] = d0.x; Ds[buf][Ds_k][0][Ds_i + 1] = d1.x;//k = tx & STEP_m1
		Ds[buf][Ds_k][1][Ds_i] = d0.y; Ds[buf][Ds_k][1][Ds_i + 1] = d1.y;
		Ds[buf][Ds_k][2][Ds_i] = d0.z; Ds[buf][Ds_k][2][Ds_i + 1] = d1.z;
		Ds[buf][Ds_k][3][Ds_i] = d0.w; Ds[buf][Ds_k][3][Ds_i + 1] = d1.w;

		Gs[buf][Gs_k][0][Gs_i] = g0.x; Gs[buf][Gs_k][0][Gs_i + 1] = g1.x;//k = ty & STEP_m1
		Gs[buf][Gs_k][1][Gs_i] = g0.y; Gs[buf][Gs_k][1][Gs_i + 1] = g1.y;
		Gs[buf][Gs_k][2][Gs_i] = g0.z; Gs[buf][Gs_k][2][Gs_i + 1] = g1.z;
		Gs[buf][Gs_k][3][Gs_i] = g0.w; Gs[buf][Gs_k][3][Gs_i + 1] = g1.w;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

				//oc[0 - 3]             oc[4 - 7]
				simdMM4(v0,  b0.x, a0); simdMM4(v1,  b0.x, a1);//j0
				simdMM4(v2,  b0.y, a0); simdMM4(v3,  b0.y, a1);//j1
				simdMM4(v4,  b0.z, a0); simdMM4(v5,  b0.z, a1);//j2
				simdMM4(v6,  b0.w, a0); simdMM4(v7,  b0.w, a1);//j3
				simdMM4(v8,  b1.x, a0); simdMM4(v9,  b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}
			buf ^= 1;

			//load 2 group from X[N, IH, IW, IC]
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

			//load 2 group from CW[FH, 3, IC, OC]
			const int woffset = (fh*IC3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
			float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

			//write to shread memory
			Ds[buf][Ds_k][0][Ds_i] = d0.x; Ds[buf][Ds_k][0][Ds_i + 1] = d1.x;//k = tx & STEP_m1
			Ds[buf][Ds_k][1][Ds_i] = d0.y; Ds[buf][Ds_k][1][Ds_i + 1] = d1.y;
			Ds[buf][Ds_k][2][Ds_i] = d0.z; Ds[buf][Ds_k][2][Ds_i + 1] = d1.z;
			Ds[buf][Ds_k][3][Ds_i] = d0.w; Ds[buf][Ds_k][3][Ds_i + 1] = d1.w;

			Gs[buf][Gs_k][0][Gs_i] = g0.x; Gs[buf][Gs_k][0][Gs_i + 1] = g1.x;//k = ty & STEP_m1
			Gs[buf][Gs_k][1][Gs_i] = g0.y; Gs[buf][Gs_k][1][Gs_i + 1] = g1.y;
			Gs[buf][Gs_k][2][Gs_i] = g0.z; Gs[buf][Gs_k][2][Gs_i + 1] = g1.z;
			Gs[buf][Gs_k][3][Gs_i] = g0.w; Gs[buf][Gs_k][3][Gs_i + 1] = g1.w;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
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
	__syncthreads();

	//======[compute area12: block]======================================================
	//2 * (2 * 8 * 4 * 64) = 32 * 4 * 64
	//combine:
	//[ux = 4][uy = 64][elem = 32]
	//so: write 32 elem to shared memory per turn: split by oc channel

	//bool buf = 0;
	//__shared__ float Gs[2][8][4][64 + 4];//[buf][ik][elem][64 * oc]
	//__shared__ float Ds[2][8][4][64 + 4];//[buf][ik][elem][64 *  j]

	//4, 64, (32/2) = 16
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
	float a[4], y0[2], y1[2], y2[2], y3[2];

	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx << 1);
	const int yoffset = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)
	const int Y0 = yoffset + (ux << 1) * OC;//j0 -> j3
	const int Y1 = yoffset + ((ux + 4) << 1) * OC;             //j4 -> j7

	//======[turn01]======================================================
	//write-turn0: j0, j1, j2, j3
	{
		//oc0 -> oc3                                      // oc4 -> oc7
		*(float4*)(&get3d(Ys0, ux, uy,  0, 64, 16)) = v0; *(float4*)(&get3d(Ys1, ux, uy,  0, 64, 16)) = v1;
		*(float4*)(&get3d(Ys0, ux, uy,  4, 64, 16)) = v2; *(float4*)(&get3d(Ys1, ux, uy,  4, 64, 16)) = v3;
		*(float4*)(&get3d(Ys0, ux, uy,  8, 64, 16)) = v4; *(float4*)(&get3d(Ys1, ux, uy,  8, 64, 16)) = v5;
		*(float4*)(&get3d(Ys0, ux, uy, 12, 64, 16)) = v6; *(float4*)(&get3d(Ys1, ux, uy, 12, 64, 16)) = v7;
	}
	__syncthreads();

	{
		a[0] = get3d(Ys0, 0, uy, (ux << 2)    , 64, 16);//oc0
		a[1] = get3d(Ys0, 1, uy, (ux << 2)    , 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2)    , 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2)    , 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 1, 64, 16);//oc1
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 2, 64, 16);//oc2
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		//oc = 0 -> 3: += 4
		//j = ux << 1;
		
		*(float4*)(Y + Y0     ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y0 + OC) = { y0[1], y1[1], y2[1], y3[1] };
	}

	{
		a[0] = get3d(Ys1, 0, uy, (ux << 2), 64, 16);//oc4
		a[1] = get3d(Ys1, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 1, 64, 16);//oc5
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 2, 64, 16);//oc
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y0      + 4) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y0 + OC + 4) = { y0[1], y1[1], y2[1], y3[1] };
	}
	__syncthreads();

	//======[turn23]======================================================
	//write-turn2: j4, j5, j6, j7
	{
		//oc0 -> oc3                                       // oc4 -> oc7
		*(float4*)(&get3d(Ys0, ux, uy,  0, 64, 16)) =  v8; *(float4*)(&get3d(Ys1, ux, uy,  0, 64, 16)) =  v9;
		*(float4*)(&get3d(Ys0, ux, uy,  4, 64, 16)) = v10; *(float4*)(&get3d(Ys1, ux, uy,  4, 64, 16)) = v11;
		*(float4*)(&get3d(Ys0, ux, uy,  8, 64, 16)) = v12; *(float4*)(&get3d(Ys1, ux, uy,  8, 64, 16)) = v13;
		*(float4*)(&get3d(Ys0, ux, uy, 12, 64, 16)) = v14; *(float4*)(&get3d(Ys1, ux, uy, 12, 64, 16)) = v15;
	}
	__syncthreads();

	{
		a[0] = get3d(Ys0, 0, uy, (ux << 2), 64, 16);//oc0
		a[1] = get3d(Ys0, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 1, 64, 16);//oc1
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 2, 64, 16);//oc2
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y1     ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y1 + OC) = { y0[1], y1[1], y2[1], y3[1] };
	}

	{
		a[0] = get3d(Ys1, 0, uy, (ux << 2), 64, 16);//oc4
		a[1] = get3d(Ys1, 1, uy, (ux << 2), 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2), 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2), 64, 16);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 1, 64, 16);//oc5
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 1, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 1, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 1, 64, 16);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 2, 64, 16);//oc
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 2, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 2, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 2, 64, 16);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 3, 64, 16);//oc3
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 3, 64, 16);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 3, 64, 16);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 3, 64, 16);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y1      + 4) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y1 + OC + 4) = { y0[1], y1[1], y2[1], y3[1] };
	}
}

#endif


#ifndef AWINOGRAD_2X3_KERNEL6
#define AWINOGRAD_2X3_KERNEL6

#define AW23_K6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel6<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//Size = 18, Time = 2.97769 msec, Performace = 12981.4 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void AW23_kernel6(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0  = bj0 + (ty << 3)  + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx / 64);//0 ->  3
	const int uy = (idx % 64);//0 -> 63
	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int GIdx = ((uy & 15) >> 1)              << 3;//0 -> 7, oc

	//======[compute area1: local]======================================================
	const int Wstride = IC * OC;
#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

		//load 2 group from CW[FH, 3, IC, OC]
		const int woffset = fh * IC3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset                 );//fw = 0
		float2 w1 = *(float2*)(CW + woffset + Wstride       );//fw = 1
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));//fw = 2

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
		float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
		float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

		const int Ds_k = tx & STEP_m1;
		const int Ds_i = (ty << 2) + ((tx >= STEP) << 1);

		const int Gs_k = ty & STEP_m1;
		const int Gs_i = (tx << 2) + ((ty >= STEP) << 1);

		Gs[buf][Gs_k][0][Gs_i] = g0.x; Gs[buf][Gs_k][0][Gs_i + 1] = g1.x;//k = ty & STEP_m1
		Gs[buf][Gs_k][1][Gs_i] = g0.y; Gs[buf][Gs_k][1][Gs_i + 1] = g1.y;
		Gs[buf][Gs_k][2][Gs_i] = g0.z; Gs[buf][Gs_k][2][Gs_i + 1] = g1.z;
		Gs[buf][Gs_k][3][Gs_i] = g0.w; Gs[buf][Gs_k][3][Gs_i + 1] = g1.w;

		Ds[buf][Ds_k][0][Ds_i] = d0.x; Ds[buf][Ds_k][0][Ds_i + 1] = d1.x;//k = tx & STEP_m1
		Ds[buf][Ds_k][1][Ds_i] = d0.y; Ds[buf][Ds_k][1][Ds_i + 1] = d1.y;
		Ds[buf][Ds_k][2][Ds_i] = d0.z; Ds[buf][Ds_k][2][Ds_i + 1] = d1.z;
		Ds[buf][Ds_k][3][Ds_i] = d0.w; Ds[buf][Ds_k][3][Ds_i + 1] = d1.w;

		
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 a0 = *(float4*)(&Gs[buf][ik][ux][GIdx]), a1 = *(float4*)(&Gs[buf][ik][ux][GIdx + 4]);
				float4 b0 = *(float4*)(&Ds[buf][ik][ux][DIdx]), b1 = *(float4*)(&Ds[buf][ik][ux][DIdx + 4]);

				//oc[0 - 3]             oc[4 - 7]
				simdMM4(v0,  b0.x, a0); simdMM4(v1,  b0.x, a1);//j0
				simdMM4(v2,  b0.y, a0); simdMM4(v3,  b0.y, a1);//j1
				simdMM4(v4,  b0.z, a0); simdMM4(v5,  b0.z, a1);//j2
				simdMM4(v6,  b0.w, a0); simdMM4(v7,  b0.w, a1);//j3
				simdMM4(v8,  b1.x, a0); simdMM4(v9,  b1.x, a1);//j4
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);//j5
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);//j6
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);//j7
			}
			buf ^= 1;

			//load 2 group from X[N, IH, IW, IC]
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

			//load 2 group from CW[FH, 3, IC, OC]
			const int woffset = (fh*IC3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
			float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

			//write to shread memory
			Gs[buf][Gs_k][0][Gs_i] = g0.x; Gs[buf][Gs_k][0][Gs_i + 1] = g1.x;//k = ty & STEP_m1
			Gs[buf][Gs_k][1][Gs_i] = g0.y; Gs[buf][Gs_k][1][Gs_i + 1] = g1.y;
			Gs[buf][Gs_k][2][Gs_i] = g0.z; Gs[buf][Gs_k][2][Gs_i + 1] = g1.z;
			Gs[buf][Gs_k][3][Gs_i] = g0.w; Gs[buf][Gs_k][3][Gs_i + 1] = g1.w;

			Ds[buf][Ds_k][0][Ds_i] = d0.x; Ds[buf][Ds_k][0][Ds_i + 1] = d1.x;//k = tx & STEP_m1
			Ds[buf][Ds_k][1][Ds_i] = d0.y; Ds[buf][Ds_k][1][Ds_i + 1] = d1.y;
			Ds[buf][Ds_k][2][Ds_i] = d0.z; Ds[buf][Ds_k][2][Ds_i + 1] = d1.z;
			Ds[buf][Ds_k][3][Ds_i] = d0.w; Ds[buf][Ds_k][3][Ds_i + 1] = d1.w;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
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
	__syncthreads();

	//======[compute area12: block]======================================================
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
	float a[4], y0[2], y1[2], y2[2], y3[2];

	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx << 1);
	const int yoffset = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)
	const int Y0 = yoffset + (ux << 1) * OC;//j0 -> j3
	const int Y1 = yoffset + ((ux + 4) << 1) * OC;             //j4 -> j7

	//======[turn01]======================================================
	//write-turn0: j0, j1, j2, j3
	{
		//oc0 -> oc3                                      // oc4 -> oc7
		*(float4*)(&get3d(Ys0, ux, uy,  0, 65, 20)) = v0; *(float4*)(&get3d(Ys1, ux, uy,  0, 65, 20)) = v1;
		*(float4*)(&get3d(Ys0, ux, uy,  4, 65, 20)) = v2; *(float4*)(&get3d(Ys1, ux, uy,  4, 65, 20)) = v3;
		*(float4*)(&get3d(Ys0, ux, uy,  8, 65, 20)) = v4; *(float4*)(&get3d(Ys1, ux, uy,  8, 65, 20)) = v5;
		*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v6; *(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v7;
	}
	__syncthreads();

	{
		a[0] = get3d(Ys0, 0, uy, (ux << 2)    , 65, 20);//oc0
		a[1] = get3d(Ys0, 1, uy, (ux << 2)    , 65, 20);
		a[2] = get3d(Ys0, 2, uy, (ux << 2)    , 65, 20);
		a[3] = get3d(Ys0, 3, uy, (ux << 2)    , 65, 20);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20);//oc1
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20);//oc2
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20);//oc3
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);
		winograd_f2x3_y(y3, a);

		//oc = 0 -> 3: += 4
		//j = ux << 1;
		
		*(float4*)(Y + Y0     ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y0 + OC) = { y0[1], y1[1], y2[1], y3[1] };
	}

	{
		a[0] = get3d(Ys1, 0, uy, (ux << 2), 65, 20);//oc4
		a[1] = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		a[2] = get3d(Ys1, 2, uy, (ux << 2), 65, 20);
		a[3] = get3d(Ys1, 3, uy, (ux << 2), 65, 20);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20);//oc5
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20);//oc
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20);//oc3
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y0      + 4) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y0 + OC + 4) = { y0[1], y1[1], y2[1], y3[1] };
	}
	__syncthreads();

	//======[turn23]======================================================
	//write-turn2: j4, j5, j6, j7
	{
		//oc0 -> oc3                                       // oc4 -> oc7
		*(float4*)(&get3d(Ys0, ux, uy,  0, 65, 20)) =  v8; *(float4*)(&get3d(Ys1, ux, uy,  0, 65, 20)) =  v9;
		*(float4*)(&get3d(Ys0, ux, uy,  4, 65, 20)) = v10; *(float4*)(&get3d(Ys1, ux, uy,  4, 65, 20)) = v11;
		*(float4*)(&get3d(Ys0, ux, uy,  8, 65, 20)) = v12; *(float4*)(&get3d(Ys1, ux, uy,  8, 65, 20)) = v13;
		*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v14; *(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v15;
	}
	__syncthreads();

	{
		//oc0
		a[0] = get3d(Ys0, 0, uy, (ux << 2), 65, 20);
		a[1] = get3d(Ys0, 1, uy, (ux << 2), 65, 20);
		a[2] = get3d(Ys0, 2, uy, (ux << 2), 65, 20);
		a[3] = get3d(Ys0, 3, uy, (ux << 2), 65, 20);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20);//oc1
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20);//oc2
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20);//oc3
		a[1] = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);
		a[2] = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20);
		a[3] = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y1     ) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y1 + OC) = { y0[1], y1[1], y2[1], y3[1] };
	}

	{
		a[0] = get3d(Ys1, 0, uy, (ux << 2), 65, 20);//oc4
		a[1] = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		a[2] = get3d(Ys1, 2, uy, (ux << 2), 65, 20);
		a[3] = get3d(Ys1, 3, uy, (ux << 2), 65, 20);
		winograd_f2x3_y(y0, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20);//oc5
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);
		winograd_f2x3_y(y1, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20);//oc
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);
		winograd_f2x3_y(y2, a);

		a[0] = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20);//oc3
		a[1] = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		a[2] = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20);
		a[3] = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);
		winograd_f2x3_y(y3, a);

		*(float4*)(Y + Y1      + 4) = { y0[0], y1[0], y2[0], y3[0] };
		*(float4*)(Y + Y1 + OC + 4) = { y0[1], y1[1], y2[1], y3[1] };
	}
}

#endif


#ifndef AWINOGRAD_2X3_KERNEL7
#define AWINOGRAD_2X3_KERNEL7

#define AW23_K7(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel7<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//Size = 18, Time = 2.97769 msec, Performace = 12981.4 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void AW23_kernel7(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx / 64);//0 ->  3
	const int uy = (idx % 64);//0 -> 63
	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int GIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//======[compute area1: local]======================================================
	const int Wstride = IC * OC;
#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

		//load 2 group from CW[FH, 3, IC, OC]
		const int woffset = fh * IC3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset);//fw = 0
		float2 w1 = *(float2*)(CW + woffset + Wstride);//fw = 1
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));//fw = 2

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
		float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
		float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

		const int Ds_k = tx & STEP_m1;
		const int Ds_i = (ty << 2) + ((tx >= STEP) << 1);

		const int Gs_k = ty & STEP_m1;
		const int Gs_i = (tx << 2) + ((ty >= STEP) << 1);

		Gs[buf][Gs_k][0][Gs_i] = g0.x; Gs[buf][Gs_k][0][Gs_i + 1] = g1.x;//k = ty & STEP_m1
		Gs[buf][Gs_k][1][Gs_i] = g0.y; Gs[buf][Gs_k][1][Gs_i + 1] = g1.y;
		Gs[buf][Gs_k][2][Gs_i] = g0.z; Gs[buf][Gs_k][2][Gs_i + 1] = g1.z;
		Gs[buf][Gs_k][3][Gs_i] = g0.w; Gs[buf][Gs_k][3][Gs_i + 1] = g1.w;

		Ds[buf][Ds_k][0][Ds_i] = d0.x; Ds[buf][Ds_k][0][Ds_i + 1] = d1.x;//k = tx & STEP_m1
		Ds[buf][Ds_k][1][Ds_i] = d0.y; Ds[buf][Ds_k][1][Ds_i + 1] = d1.y;
		Ds[buf][Ds_k][2][Ds_i] = d0.z; Ds[buf][Ds_k][2][Ds_i + 1] = d1.z;
		Ds[buf][Ds_k][3][Ds_i] = d0.w; Ds[buf][Ds_k][3][Ds_i + 1] = d1.w;


		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
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
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

			//load 2 group from CW[FH, 3, IC, OC]
			const int woffset = (fh*IC3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
			float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

			//write to shread memory
			Gs[buf][Gs_k][0][Gs_i] = g0.x; Gs[buf][Gs_k][0][Gs_i + 1] = g1.x;//k = ty & STEP_m1
			Gs[buf][Gs_k][1][Gs_i] = g0.y; Gs[buf][Gs_k][1][Gs_i + 1] = g1.y;
			Gs[buf][Gs_k][2][Gs_i] = g0.z; Gs[buf][Gs_k][2][Gs_i + 1] = g1.z;
			Gs[buf][Gs_k][3][Gs_i] = g0.w; Gs[buf][Gs_k][3][Gs_i + 1] = g1.w;

			Ds[buf][Ds_k][0][Ds_i] = d0.x; Ds[buf][Ds_k][0][Ds_i + 1] = d1.x;//k = tx & STEP_m1
			Ds[buf][Ds_k][1][Ds_i] = d0.y; Ds[buf][Ds_k][1][Ds_i + 1] = d1.y;
			Ds[buf][Ds_k][2][Ds_i] = d0.z; Ds[buf][Ds_k][2][Ds_i + 1] = d1.z;
			Ds[buf][Ds_k][3][Ds_i] = d0.w; Ds[buf][Ds_k][3][Ds_i + 1] = d1.w;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
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
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
	float2 y0, y1, y2, y3;

	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx << 1);
	const int yoffset = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)
	const int Y0 = yoffset + (ux << 1) * OC;//j0 -> j3
	const int Y1 = yoffset + ((ux + 4) << 1) * OC;             //j4 -> j7

	//======[turn01]======================================================
	//write-turn0: j0, j1, j2, j3
	//oc0 -> oc3                                      // oc4 -> oc7
	*(float4*)(&get3d(Ys0, ux, uy, 0, 65, 20)) = v0;
	*(float4*)(&get3d(Ys0, ux, uy, 4, 65, 20)) = v2;
	*(float4*)(&get3d(Ys0, ux, uy, 8, 65, 20)) = v4;
	*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v6;

	*(float4*)(&get3d(Ys1, ux, uy, 0, 65, 20)) = v1;
	*(float4*)(&get3d(Ys1, ux, uy, 4, 65, 20)) = v3;
	*(float4*)(&get3d(Ys1, ux, uy, 8, 65, 20)) = v5;
	*(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v7;
	__syncthreads();

	{
		v0.x = get3d(Ys0, 0, uy, (ux << 2), 65, 20); v0.y = get3d(Ys0, 1, uy, (ux << 2), 65, 20);
		v0.z = get3d(Ys0, 2, uy, (ux << 2), 65, 20); v0.w = get3d(Ys0, 3, uy, (ux << 2), 65, 20);

		v1.x = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20); v1.y = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);
		v1.z = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20); v1.w = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);

		v2.x = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20); v2.y = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);
		v2.z = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20); v2.w = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);

		v3.x = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20); v3.y = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);
		v3.z = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20); v3.w = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, v0);
		winograd_f2x3_y_f32_64(y2, v2);
		winograd_f2x3_y_f32_64(y1, v1);
		winograd_f2x3_y_f32_64(y3, v3);
		*(float4*)(Y + Y0) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y0 + OC) = float4{ y0.y, y1.y, y2.y, y3.y };
	}

	{
		v0.x = get3d(Ys1, 0, uy, (ux << 2), 65, 20); v0.y = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		v0.z = get3d(Ys1, 2, uy, (ux << 2), 65, 20); v0.w = get3d(Ys1, 3, uy, (ux << 2), 65, 20);

		v1.x = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20); v1.y = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		v1.z = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20); v1.w = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);

		v2.x = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20); v2.y = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		v2.z = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20); v2.w = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);

		v3.x = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20); v3.y = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		v3.z = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20); v3.w = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, v0);
		winograd_f2x3_y_f32_64(y1, v1);
		winograd_f2x3_y_f32_64(y2, v2);
		winograd_f2x3_y_f32_64(y3, v3);
		*(float4*)(Y + Y0 + 4) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y0 + OC + 4) = float4{ y0.y, y1.y, y2.y, y3.y };
	}
	__syncthreads();

	//======[turn23]======================================================
	//write-turn2: j4, j5, j6, j7
	//oc0 -> oc3                                      
	*(float4*)(&get3d(Ys0, ux, uy, 0, 65, 20)) = v8;
	*(float4*)(&get3d(Ys0, ux, uy, 4, 65, 20)) = v10;
	*(float4*)(&get3d(Ys0, ux, uy, 8, 65, 20)) = v12;
	*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v14;

	// oc4 -> oc7
	*(float4*)(&get3d(Ys1, ux, uy, 0, 65, 20)) = v9;
	*(float4*)(&get3d(Ys1, ux, uy, 4, 65, 20)) = v11;
	*(float4*)(&get3d(Ys1, ux, uy, 8, 65, 20)) = v13;
	*(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v15;
	__syncthreads();

	{
		v0.x = get3d(Ys0, 0, uy, (ux << 2), 65, 20); v0.y = get3d(Ys0, 1, uy, (ux << 2), 65, 20);//oc0
		v0.z = get3d(Ys0, 2, uy, (ux << 2), 65, 20); v0.w = get3d(Ys0, 3, uy, (ux << 2), 65, 20);

		v1.x = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20); v1.y = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);//oc1
		v1.z = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20); v1.w = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);

		v2.x = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20); v2.y = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);//oc2
		v2.z = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20); v2.w = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);

		v3.x = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20); v3.y = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);//oc3
		v3.z = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20); v3.w = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, v0);
		winograd_f2x3_y_f32_64(y1, v1);
		winograd_f2x3_y_f32_64(y2, v2);
		winograd_f2x3_y_f32_64(y3, v3);
		*(float4*)(Y + Y1) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y1 + OC) = float4{ y0.y, y1.y, y2.y, y3.y };

	}

	{
		v0.x = get3d(Ys1, 0, uy, (ux << 2), 65, 20); v0.y = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		v0.z = get3d(Ys1, 2, uy, (ux << 2), 65, 20); v0.w = get3d(Ys1, 3, uy, (ux << 2), 65, 20);

		v1.x = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20); v1.y = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		v1.z = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20); v1.w = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);

		v2.x = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20); v2.y = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		v2.z = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20); v2.w = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);

		v3.x = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20); v3.y = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		v3.z = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20); v3.w = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, v0);
		winograd_f2x3_y_f32_64(y1, v1);
		winograd_f2x3_y_f32_64(y2, v2);
		winograd_f2x3_y_f32_64(y3, v3);
		*(float4*)(Y + Y1 + 4) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y1 + OC + 4) = float4{ y0.y, y1.y, y2.y, y3.y };
	}
}

#endif


#ifndef AWINOGRAD_2X3_KERNEL8
#define AWINOGRAD_2X3_KERNEL8

#define AW23_K8(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel8<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 18, Time = 2.97769 msec, Performace = 12981.4 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void AW23_kernel8(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx / 64);//0 ->  3
	const int uy = (idx % 64);//0 -> 63
	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int GIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//======[compute area1: local]======================================================
	const int Wstride = IC * OC;
	const int Ds_k = tx & STEP_m1, Ds_i = (ty << 2) + ((tx >= STEP) << 1);
	const int Gs_k = ty & STEP_m1, Gs_i = (tx << 2) + ((ty >= STEP) << 1);

#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC*2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC*3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC*4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC*5, -1));

		//load 2 group from CW[FH, 3, IC, OC]
		const int woffset = fh * IC * 3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset);//fw = 0
		float2 w1 = *(float2*)(CW + woffset + Wstride);//fw = 1
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));//fw = 2

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
		float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
		float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

		//write to shread memory
		Gs[buf][Gs_k][0][Gs_i] = g0.x; Gs[buf][Gs_k][0][Gs_i + 1] = g1.x;//k = ty & STEP_m1
		Gs[buf][Gs_k][1][Gs_i] = g0.y; Gs[buf][Gs_k][1][Gs_i + 1] = g1.y;
		Gs[buf][Gs_k][2][Gs_i] = g0.z; Gs[buf][Gs_k][2][Gs_i + 1] = g1.z;
		Gs[buf][Gs_k][3][Gs_i] = g0.w; Gs[buf][Gs_k][3][Gs_i + 1] = g1.w;

		Ds[buf][Ds_k][0][Ds_i] = d0.x; Ds[buf][Ds_k][0][Ds_i + 1] = d1.x;//k = tx & STEP_m1
		Ds[buf][Ds_k][1][Ds_i] = d0.y; Ds[buf][Ds_k][1][Ds_i + 1] = d1.y;
		Ds[buf][Ds_k][2][Ds_i] = d0.z; Ds[buf][Ds_k][2][Ds_i + 1] = d1.z;
		Ds[buf][Ds_k][3][Ds_i] = d0.w; Ds[buf][Ds_k][3][Ds_i + 1] = d1.w;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
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
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC*2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC*3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC*4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC*5, -1));

			//load 2 group from CW[FH, 3, IC, OC]
			const int woffset = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
			float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

			//write to shread memory
			Gs[buf][Gs_k][0][Gs_i] = g0.x; Gs[buf][Gs_k][0][Gs_i + 1] = g1.x;//k = ty & STEP_m1
			Gs[buf][Gs_k][1][Gs_i] = g0.y; Gs[buf][Gs_k][1][Gs_i + 1] = g1.y;
			Gs[buf][Gs_k][2][Gs_i] = g0.z; Gs[buf][Gs_k][2][Gs_i + 1] = g1.z;
			Gs[buf][Gs_k][3][Gs_i] = g0.w; Gs[buf][Gs_k][3][Gs_i + 1] = g1.w;

			Ds[buf][Ds_k][0][Ds_i] = d0.x; Ds[buf][Ds_k][0][Ds_i + 1] = d1.x;//k = tx & STEP_m1
			Ds[buf][Ds_k][1][Ds_i] = d0.y; Ds[buf][Ds_k][1][Ds_i + 1] = d1.y;
			Ds[buf][Ds_k][2][Ds_i] = d0.z; Ds[buf][Ds_k][2][Ds_i + 1] = d1.z;
			Ds[buf][Ds_k][3][Ds_i] = d0.w; Ds[buf][Ds_k][3][Ds_i + 1] = d1.w;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
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
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
	float2 y0, y1, y2, y3;

	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx << 1);
	const int yoffset = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)
	const int Y0 = yoffset + (ux << 1) * OC;//j0 -> j3
	const int Y1 = yoffset + ((ux + 4) << 1) * OC;             //j4 -> j7

	//======[turn01]======================================================
	//write-turn0: j0, j1, j2, j3
	//oc0 -> oc3                                      // oc4 -> oc7
	*(float4*)(&get3d(Ys0, ux, uy, 0, 65, 20)) = v0;
	*(float4*)(&get3d(Ys0, ux, uy, 4, 65, 20)) = v2;
	*(float4*)(&get3d(Ys0, ux, uy, 8, 65, 20)) = v4;
	*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v6;

	*(float4*)(&get3d(Ys1, ux, uy, 0, 65, 20)) = v1;
	*(float4*)(&get3d(Ys1, ux, uy, 4, 65, 20)) = v3;
	*(float4*)(&get3d(Ys1, ux, uy, 8, 65, 20)) = v5;
	*(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v7;
	__syncthreads();

	{
		v0.x = get3d(Ys0, 0, uy, (ux << 2), 65, 20); v0.y = get3d(Ys0, 1, uy, (ux << 2), 65, 20);
		v0.z = get3d(Ys0, 2, uy, (ux << 2), 65, 20); v0.w = get3d(Ys0, 3, uy, (ux << 2), 65, 20);

		v1.x = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20); v1.y = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);
		v1.z = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20); v1.w = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);

		v2.x = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20); v2.y = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);
		v2.z = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20); v2.w = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);

		v3.x = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20); v3.y = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);
		v3.z = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20); v3.w = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, v0);
		winograd_f2x3_y_f32_64(y2, v2);
		winograd_f2x3_y_f32_64(y1, v1);
		winograd_f2x3_y_f32_64(y3, v3);
		*(float4*)(Y + Y0) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y0 + OC) = float4{ y0.y, y1.y, y2.y, y3.y };
	}

	{
		v0.x = get3d(Ys1, 0, uy, (ux << 2), 65, 20); v0.y = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		v0.z = get3d(Ys1, 2, uy, (ux << 2), 65, 20); v0.w = get3d(Ys1, 3, uy, (ux << 2), 65, 20);

		v1.x = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20); v1.y = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		v1.z = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20); v1.w = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);

		v2.x = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20); v2.y = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		v2.z = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20); v2.w = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);

		v3.x = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20); v3.y = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		v3.z = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20); v3.w = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, v0);
		winograd_f2x3_y_f32_64(y1, v1);
		winograd_f2x3_y_f32_64(y2, v2);
		winograd_f2x3_y_f32_64(y3, v3);
		*(float4*)(Y + Y0 + 4) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y0 + OC + 4) = float4{ y0.y, y1.y, y2.y, y3.y };
	}
	__syncthreads();

	//======[turn23]======================================================
	//write-turn2: j4, j5, j6, j7
	//oc0 -> oc3                                      
	*(float4*)(&get3d(Ys0, ux, uy, 0, 65, 20)) = v8;
	*(float4*)(&get3d(Ys0, ux, uy, 4, 65, 20)) = v10;
	*(float4*)(&get3d(Ys0, ux, uy, 8, 65, 20)) = v12;
	*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v14;

	// oc4 -> oc7
	*(float4*)(&get3d(Ys1, ux, uy, 0, 65, 20)) = v9;
	*(float4*)(&get3d(Ys1, ux, uy, 4, 65, 20)) = v11;
	*(float4*)(&get3d(Ys1, ux, uy, 8, 65, 20)) = v13;
	*(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v15;
	__syncthreads();

	{
		v0.x = get3d(Ys0, 0, uy, (ux << 2), 65, 20); v0.y = get3d(Ys0, 1, uy, (ux << 2), 65, 20);//oc0
		v0.z = get3d(Ys0, 2, uy, (ux << 2), 65, 20); v0.w = get3d(Ys0, 3, uy, (ux << 2), 65, 20);

		v1.x = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20); v1.y = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);//oc1
		v1.z = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20); v1.w = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);

		v2.x = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20); v2.y = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);//oc2
		v2.z = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20); v2.w = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);

		v3.x = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20); v3.y = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);//oc3
		v3.z = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20); v3.w = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, v0);
		winograd_f2x3_y_f32_64(y1, v1);
		winograd_f2x3_y_f32_64(y2, v2);
		winograd_f2x3_y_f32_64(y3, v3);
		*(float4*)(Y + Y1) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y1 + OC) = float4{ y0.y, y1.y, y2.y, y3.y };

	}

	{
		v0.x = get3d(Ys1, 0, uy, (ux << 2), 65, 20); v0.y = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		v0.z = get3d(Ys1, 2, uy, (ux << 2), 65, 20); v0.w = get3d(Ys1, 3, uy, (ux << 2), 65, 20);

		v1.x = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20); v1.y = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		v1.z = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20); v1.w = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);

		v2.x = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20); v2.y = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		v2.z = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20); v2.w = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);

		v3.x = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20); v3.y = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		v3.z = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20); v3.w = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, v0);
		winograd_f2x3_y_f32_64(y1, v1);
		winograd_f2x3_y_f32_64(y2, v2);
		winograd_f2x3_y_f32_64(y3, v3);
		*(float4*)(Y + Y1 + 4) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y1 + OC + 4) = float4{ y0.y, y1.y, y2.y, y3.y };
	}
}

#endif


//standard: Size = 72, Time = 11.0702 msec, Performace = 13967.1 GFlop/s
#ifndef AWINOGRAD_2X3_KERNEL9
#define AWINOGRAD_2X3_KERNEL9

#define AW23_K9(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel9<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 18, Time = 2.97769 msec, Performace = 12981.4 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void AW23_kernel9(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx / 64), uy = (idx & 63);
	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int GIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//======[compute area1: local]======================================================
	const int Wstride = IC * OC;
	const int Ds_k = tx & STEP_m1, Ds_i = (ty << 2) + ((tx >= STEP) << 1);
	const int Gs_k = ty & STEP_m1, Gs_i = (tx << 2) + ((ty >= STEP) << 1);

#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC * 2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC * 3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC * 4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC * 5, -1));

		//Size = 72, Time = 10.9697 msec, Performace = 14095.1 GFlop/s

		//load 2 group from CW[FH, 3, IC, OC]
		const int woffset = fh * IC * 3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset);//fw = 0
		float2 w1 = *(float2*)(CW + woffset + Wstride);//fw = 1
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));//fw = 2

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
		float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
		float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

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

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
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
			const int woffset = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//load 2 group from X[N, IH, IW, IC]
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC * 2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC * 3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC * 4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC * 5, -1));

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
			float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

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
		for (int ik = 0; ik < STEP; ik++)
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
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
	float4 k0, k1, k2, k3;
	float2 y0, y1, y2, y3;

	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx << 1);
	const int yoffset = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)
	const int Y0 = yoffset + (ux << 1) * OC;//j0 -> j3
	const int Y1 = yoffset + ((ux + 4) << 1) * OC;             //j4 -> j7

	//======[turn01]======================================================
	//write-turn0: j0, j1, j2, j3
	//oc0 -> oc3                                      // oc4 -> oc7
	*(float4*)(&get3d(Ys0, ux, uy, 0, 65, 20)) = v0;
	*(float4*)(&get3d(Ys0, ux, uy, 4, 65, 20)) = v2;
	*(float4*)(&get3d(Ys0, ux, uy, 8, 65, 20)) = v4;
	*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v6;

	*(float4*)(&get3d(Ys1, ux, uy, 0, 65, 20)) = v1;
	*(float4*)(&get3d(Ys1, ux, uy, 4, 65, 20)) = v3;
	*(float4*)(&get3d(Ys1, ux, uy, 8, 65, 20)) = v5;
	*(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v7;
	__syncthreads();

	{
		k0.x = get3d(Ys0, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys0, 1, uy, (ux << 2), 65, 20);
		k0.z = get3d(Ys0, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys0, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);
		k1.z = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);
		k2.z = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);
		k3.z = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y0) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y0 + OC) = float4{ y0.y, y1.y, y2.y, y3.y };
	}

	{
		k0.x = get3d(Ys1, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		k0.z = get3d(Ys1, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys1, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		k1.z = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		k2.z = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		k3.z = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y0 + 4) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y0 + OC + 4) = float4{ y0.y, y1.y, y2.y, y3.y };
	}
	__syncthreads();

	//======[turn23]======================================================
	//write-turn2: j4, j5, j6, j7
	//oc0 -> oc3                                      
	*(float4*)(&get3d(Ys0, ux, uy, 0, 65, 20)) = v8;
	*(float4*)(&get3d(Ys0, ux, uy, 4, 65, 20)) = v10;
	*(float4*)(&get3d(Ys0, ux, uy, 8, 65, 20)) = v12;
	*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v14;

	// oc4 -> oc7
	*(float4*)(&get3d(Ys1, ux, uy, 0, 65, 20)) = v9;
	*(float4*)(&get3d(Ys1, ux, uy, 4, 65, 20)) = v11;
	*(float4*)(&get3d(Ys1, ux, uy, 8, 65, 20)) = v13;
	*(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v15;
	__syncthreads();

	{
		k0.x = get3d(Ys0, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys0, 1, uy, (ux << 2), 65, 20);//oc0
		k0.z = get3d(Ys0, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys0, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);//oc1
		k1.z = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);//oc2
		k2.z = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);//oc3
		k3.z = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y1) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y1 + OC) = float4{ y0.y, y1.y, y2.y, y3.y };
	}

	{
		k0.x = get3d(Ys1, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		k0.z = get3d(Ys1, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys1, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		k1.z = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		k2.z = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		k3.z = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y1 + 4) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y1 + OC + 4) = float4{ y0.y, y1.y, y2.y, y3.y };
	}
}

#endif


#ifndef AWINOGRAD_2X3_KERNEL10
#define AWINOGRAD_2X3_KERNEL10

#define AW23_K10(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel10<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 18, Time = 2.81464 msec, Performace = 13733.5 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void AW23_kernel10(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx / 64), uy = (idx & 63);
	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int GIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//======[compute area1: local]======================================================
	const int Wstride = IC * OC;
	const int Ds_k = tx & STEP_m1, Ds_i = (ty << 2) + ((tx >= STEP) << 1);
	const int Gs_k = ty & STEP_m1, Gs_i = (tx << 2) + ((ty >= STEP) << 1);

#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from CW[FH, 3, IC, OC]
		const int woffset = fh * IC * 3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset);//fw = 0
		float2 w1 = *(float2*)(CW + woffset + Wstride);//fw = 1
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));//fw = 2

		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC * 2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC * 3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC * 4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC * 5, -1));

		float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
		float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
		float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

		*(float2*)(&Ds[buf][Ds_k][0][Ds_i]) = float2{ d0.x, d1.x };
		*(float2*)(&Ds[buf][Ds_k][1][Ds_i]) = float2{ d0.y, d1.y };
		*(float2*)(&Ds[buf][Ds_k][2][Ds_i]) = float2{ d0.z, d1.z };
		*(float2*)(&Ds[buf][Ds_k][3][Ds_i]) = float2{ d0.w, d1.w };

		*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = float2{ g0.x, g1.x };
		*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = float2{ g0.y, g1.y };
		*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = float2{ g0.z, g1.z };
		*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = float2{ g0.w, g1.w };
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
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
			const int woffset = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//load 2 group from X[N, IH, IW, IC]
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC * 2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC * 3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC * 4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC * 5, -1));

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, gst0, gst1, w2.x };
			float4 g1 = float4{ w0.y, gst2, gst3, w2.y };

			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

			*(float2*)(&Ds[buf][Ds_k][0][Ds_i]) = float2{ d0.x, d1.x };
			*(float2*)(&Ds[buf][Ds_k][1][Ds_i]) = float2{ d0.y, d1.y };
			*(float2*)(&Ds[buf][Ds_k][2][Ds_i]) = float2{ d0.z, d1.z };
			*(float2*)(&Ds[buf][Ds_k][3][Ds_i]) = float2{ d0.w, d1.w };

			*(float2*)(&Gs[buf][Gs_k][0][Gs_i]) = float2{ g0.x, g1.x };
			*(float2*)(&Gs[buf][Gs_k][1][Gs_i]) = float2{ g0.y, g1.y };
			*(float2*)(&Gs[buf][Gs_k][2][Gs_i]) = float2{ g0.z, g1.z };
			*(float2*)(&Gs[buf][Gs_k][3][Gs_i]) = float2{ g0.w, g1.w };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
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
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
	float4 k0, k1, k2, k3;
	float2 y0, y1, y2, y3;

	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx << 1);
	const int yoffset = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)
	const int Y0 = yoffset + (ux << 1) * OC;//j0 -> j3
	const int Y1 = yoffset + ((ux + 4) << 1) * OC;             //j4 -> j7

	//======[turn01]======================================================
	//write-turn0: j0, j1, j2, j3
	//oc0 -> oc3                                      // oc4 -> oc7
	*(float4*)(&get3d(Ys0, ux, uy, 0, 65, 20)) = v0;
	*(float4*)(&get3d(Ys0, ux, uy, 4, 65, 20)) = v2;
	*(float4*)(&get3d(Ys0, ux, uy, 8, 65, 20)) = v4;
	*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v6;

	*(float4*)(&get3d(Ys1, ux, uy, 0, 65, 20)) = v1;
	*(float4*)(&get3d(Ys1, ux, uy, 4, 65, 20)) = v3;
	*(float4*)(&get3d(Ys1, ux, uy, 8, 65, 20)) = v5;
	*(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v7;
	__syncthreads();

	{
		k0.x = get3d(Ys0, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys0, 1, uy, (ux << 2), 65, 20);
		k0.z = get3d(Ys0, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys0, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);
		k1.z = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);
		k2.z = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);
		k3.z = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y0) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y0 + OC) = float4{ y0.y, y1.y, y2.y, y3.y };
	}

	{
		k0.x = get3d(Ys1, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		k0.z = get3d(Ys1, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys1, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		k1.z = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		k2.z = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		k3.z = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y0 + 4) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y0 + OC + 4) = float4{ y0.y, y1.y, y2.y, y3.y };
	}
	__syncthreads();

	//======[turn23]======================================================
	//write-turn2: j4, j5, j6, j7
	//oc0 -> oc3                                      
	*(float4*)(&get3d(Ys0, ux, uy, 0, 65, 20)) = v8;
	*(float4*)(&get3d(Ys0, ux, uy, 4, 65, 20)) = v10;
	*(float4*)(&get3d(Ys0, ux, uy, 8, 65, 20)) = v12;
	*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v14;

	// oc4 -> oc7
	*(float4*)(&get3d(Ys1, ux, uy, 0, 65, 20)) = v9;
	*(float4*)(&get3d(Ys1, ux, uy, 4, 65, 20)) = v11;
	*(float4*)(&get3d(Ys1, ux, uy, 8, 65, 20)) = v13;
	*(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v15;
	__syncthreads();

	{
		k0.x = get3d(Ys0, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys0, 1, uy, (ux << 2), 65, 20);//oc0
		k0.z = get3d(Ys0, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys0, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);//oc1
		k1.z = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);//oc2
		k2.z = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);//oc3
		k3.z = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y1) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y1 + OC) = float4{ y0.y, y1.y, y2.y, y3.y };
	}

	{
		k0.x = get3d(Ys1, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		k0.z = get3d(Ys1, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys1, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		k1.z = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		k2.z = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		k3.z = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y1 + 4) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y1 + OC + 4) = float4{ y0.y, y1.y, y2.y, y3.y };
	}
}

#endif


//standard: Size = 18, Time = 2.76092 msec, Performace = 14000.7 GFlop/s
#ifndef AWINOGRAD_2X3_KERNEL11
#define AWINOGRAD_2X3_KERNEL11

#define AW23_K11(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel11<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 18, Time = 2.76092 msec, Performace = 14000.7 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void AW23_kernel11(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx / 64), uy = (idx & 63);
	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int GIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//======[compute area1: local]======================================================
	float w[6], x[6];
	const int Wstride = IC * OC;
	const int Ds_k = tx & STEP_m1, Ds_i = (ty << 2) + ((tx >= STEP) << 1);
	const int Gs_k = ty & STEP_m1, Gs_i = (tx << 2) + ((ty >= STEP) << 1);

#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2), tX5 = tX0 + IC * 5;
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
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + Wstride, W2 = W0 + (Wstride << 1);
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

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
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
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2), tX5 = tX0 + IC * 5;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
			float x0 = tex1Dfetch<float>(X, tX0);
			float x1 = tex1Dfetch<float>(X, tX1);
			float x2 = tex1Dfetch<float>(X, tX2);
			float x3 = tex1Dfetch<float>(X, tX3);
			float x4 = tex1Dfetch<float>(X, tX4);
			float x5 = tex1Dfetch<float>(X, tX5);

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + Wstride, W2 = W0 + (Wstride << 1);
			*(float2*)(w    ) = *(float2*)(CW + W0);//fw = 0
			*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1
			*(float2*)(w + 4) = *(float2*)(CW + W2);//fw = 2

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

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
		for (int ik = 0; ik < STEP; ik++)
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
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
	float4 k0, k1, k2, k3;
	float2 y0, y1, y2, y3;

	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx << 1);
	const int yoffset = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)
	const int Y0 = yoffset + (ux << 1) * OC;//j0 -> j3
	const int Y1 = yoffset + ((ux + 4) << 1) * OC;             //j4 -> j7

	//======[turn01]======================================================
	//write-turn0: j0, j1, j2, j3
	//oc0 -> oc3                                      // oc4 -> oc7
	*(float4*)(&get3d(Ys0, ux, uy, 0, 65, 20)) = v0;
	*(float4*)(&get3d(Ys0, ux, uy, 4, 65, 20)) = v2;
	*(float4*)(&get3d(Ys0, ux, uy, 8, 65, 20)) = v4;
	*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v6;

	*(float4*)(&get3d(Ys1, ux, uy, 0, 65, 20)) = v1;
	*(float4*)(&get3d(Ys1, ux, uy, 4, 65, 20)) = v3;
	*(float4*)(&get3d(Ys1, ux, uy, 8, 65, 20)) = v5;
	*(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v7;
	__syncthreads();

	{
		k0.x = get3d(Ys0, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys0, 1, uy, (ux << 2), 65, 20);
		k0.z = get3d(Ys0, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys0, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);
		k1.z = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);
		k2.z = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);
		k3.z = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y0) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y0 + OC) = float4{ y0.y, y1.y, y2.y, y3.y };
	}

	{
		k0.x = get3d(Ys1, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		k0.z = get3d(Ys1, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys1, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		k1.z = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		k2.z = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		k3.z = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y0 + 4) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y0 + OC + 4) = float4{ y0.y, y1.y, y2.y, y3.y };
	}
	__syncthreads();

	//======[turn23]======================================================
	//write-turn2: j4, j5, j6, j7
	//oc0 -> oc3                                      
	*(float4*)(&get3d(Ys0, ux, uy, 0, 65, 20)) = v8;
	*(float4*)(&get3d(Ys0, ux, uy, 4, 65, 20)) = v10;
	*(float4*)(&get3d(Ys0, ux, uy, 8, 65, 20)) = v12;
	*(float4*)(&get3d(Ys0, ux, uy, 12, 65, 20)) = v14;

	// oc4 -> oc7
	*(float4*)(&get3d(Ys1, ux, uy, 0, 65, 20)) = v9;
	*(float4*)(&get3d(Ys1, ux, uy, 4, 65, 20)) = v11;
	*(float4*)(&get3d(Ys1, ux, uy, 8, 65, 20)) = v13;
	*(float4*)(&get3d(Ys1, ux, uy, 12, 65, 20)) = v15;
	__syncthreads();

	{
		k0.x = get3d(Ys0, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys0, 1, uy, (ux << 2), 65, 20);//oc0
		k0.z = get3d(Ys0, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys0, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys0, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys0, 1, uy, (ux << 2) + 1, 65, 20);//oc1
		k1.z = get3d(Ys0, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys0, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys0, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys0, 1, uy, (ux << 2) + 2, 65, 20);//oc2
		k2.z = get3d(Ys0, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys0, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys0, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys0, 1, uy, (ux << 2) + 3, 65, 20);//oc3
		k3.z = get3d(Ys0, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys0, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y1) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y1 + OC) = float4{ y0.y, y1.y, y2.y, y3.y };
	}

	{
		k0.x = get3d(Ys1, 0, uy, (ux << 2), 65, 20); k0.y = get3d(Ys1, 1, uy, (ux << 2), 65, 20);
		k0.z = get3d(Ys1, 2, uy, (ux << 2), 65, 20); k0.w = get3d(Ys1, 3, uy, (ux << 2), 65, 20);

		k1.x = get3d(Ys1, 0, uy, (ux << 2) + 1, 65, 20); k1.y = get3d(Ys1, 1, uy, (ux << 2) + 1, 65, 20);
		k1.z = get3d(Ys1, 2, uy, (ux << 2) + 1, 65, 20); k1.w = get3d(Ys1, 3, uy, (ux << 2) + 1, 65, 20);

		k2.x = get3d(Ys1, 0, uy, (ux << 2) + 2, 65, 20); k2.y = get3d(Ys1, 1, uy, (ux << 2) + 2, 65, 20);
		k2.z = get3d(Ys1, 2, uy, (ux << 2) + 2, 65, 20); k2.w = get3d(Ys1, 3, uy, (ux << 2) + 2, 65, 20);

		k3.x = get3d(Ys1, 0, uy, (ux << 2) + 3, 65, 20); k3.y = get3d(Ys1, 1, uy, (ux << 2) + 3, 65, 20);
		k3.z = get3d(Ys1, 2, uy, (ux << 2) + 3, 65, 20); k3.w = get3d(Ys1, 3, uy, (ux << 2) + 3, 65, 20);

		winograd_f2x3_y_f32_64(y0, k0);
		winograd_f2x3_y_f32_64(y1, k1);
		winograd_f2x3_y_f32_64(y2, k2);
		winograd_f2x3_y_f32_64(y3, k3);
		*(float4*)(Y + Y1 + 4) = float4{ y0.x, y1.x, y2.x, y3.x };
		*(float4*)(Y + Y1 + OC + 4) = float4{ y0.y, y1.y, y2.y, y3.y };
	}
}

#endif


//standard: Size = 18, Time = 2.76092 msec, Performace = 14000.7 GFlop/s
#ifndef AWINOGRAD_2X3_KERNEL12
#define AWINOGRAD_2X3_KERNEL12

#define AW23_K12(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel12<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 18, Time = 2.76092 msec, Performace = 14000.7 GFlop/s
//Size = 18, Time = 2.56932 msec, Performace = 15044.7 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void AW23_kernel12(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx / 64), uy = (idx & 63);
	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int GIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//prepare for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + (DIdx << 1);
	const int yoffset = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)
	const int Y00 = yoffset + (ux << 1) * OC;//j0 -> j3

	//======[compute area1: local]======================================================
	float w[6], x[6];
	const int Wstride = IC * OC;
	const int Ds_k = tx & STEP_m1, Ds_i = (ty << 2) + ((tx >= STEP) << 1);
	const int Gs_k = ty & STEP_m1, Gs_i = (tx << 2) + ((ty >= STEP) << 1);

#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2), tX5 = tX0 + IC * 5;
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
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + Wstride, W2 = W0 + (Wstride << 1);
		*(float2*)(w) = *(float2*)(CW + W0);//fw = 0
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

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
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
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2), tX5 = tX0 + IC * 5;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
			float x0 = tex1Dfetch<float>(X, tX0);
			float x1 = tex1Dfetch<float>(X, tX1);
			float x2 = tex1Dfetch<float>(X, tX2);
			float x3 = tex1Dfetch<float>(X, tX3);
			float x4 = tex1Dfetch<float>(X, tX4);
			float x5 = tex1Dfetch<float>(X, tX5);

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + Wstride, W2 = W0 + (Wstride << 1);
			*(float2*)(w) = *(float2*)(CW + W0);//fw = 0
			*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1
			*(float2*)(w + 4) = *(float2*)(CW + W2);//fw = 2

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

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
		for (int ik = 0; ik < STEP; ik++)
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
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
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


//standard: Size = 18, Time = 2.76092 msec, Performace = 14000.7 GFlop/s
#ifndef AWINOGRAD_2X3_KERNEL13
#define AWINOGRAD_2X3_KERNEL13

#define AW23_K13(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel13<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 18, Time = 2.76092 msec, Performace = 14000.7 GFlop/s
//Size = 18, Time = 2.56932 msec, Performace = 15044.7 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void AW23_kernel13(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 * oc]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik][elem][64 *  j]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
	const int DIdx = ((uy & 1) + ((uy >> 4) << 1)) << 3;//0 -> 7, j 8*8 = 64
	const int GIdx = ((uy & 15) >> 1) << 3;//0 -> 7, oc

	//prepare for Y[N, OH, OW, OC]
	const int yoc0 = boc0 + GIdx;
	const int yj0 = bj0 + ((DIdx + ux) << 1);
	const int Y00 = yj0 * OC + yoc0;//ux: j0 -> j3

	//======[compute area1: local]======================================================
	float w[6], x[6];
	const int Wstride = IC * OC;
	const int Ds_k = tx & STEP_m1, Ds_i = (ty << 2) + ((tx >= STEP) << 1);
	const int Gs_k = ty & STEP_m1, Gs_i = (tx << 2) + ((ty >= STEP) << 1);

#pragma once 
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2), tX5 = tX0 + IC * 5;
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
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + Wstride, W2 = W0 + (Wstride << 1);
		*(float2*)(w) = *(float2*)(CW + W0);//fw = 0
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

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
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
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2), tX5 = tX0 + IC * 5;
			tX0 = IF_int(ly0, tX0, -1); tX1 = IF_int(ly1, tX1, -1);
			tX2 = IF_int(ly2, tX2, -1); tX3 = IF_int(ly3, tX3, -1);
			tX4 = IF_int(ly4, tX4, -1); tX5 = IF_int(ly5, tX5, -1);
			float x0 = tex1Dfetch<float>(X, tX0);
			float x1 = tex1Dfetch<float>(X, tX1);
			float x2 = tex1Dfetch<float>(X, tX2);
			float x3 = tex1Dfetch<float>(X, tX3);
			float x4 = tex1Dfetch<float>(X, tX4);
			float x5 = tex1Dfetch<float>(X, tX5);

			//load 2 group from CW[FH, 3, IC, OC]
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + Wstride, W2 = W0 + (Wstride << 1);
			*(float2*)(w) = *(float2*)(CW + W0);//fw = 0
			*(float2*)(w + 2) = *(float2*)(CW + W1);//fw = 1
			*(float2*)(w + 4) = *(float2*)(CW + W2);//fw = 2

			//Winograd transform: W(3) -> G(4); X(4) -> D(4)
			float4 d0 = float4{ x0 - x2, x1 + x2, x2 - x1, x1 - x3 };
			float4 d1 = float4{ x2 - x4, x3 + x4, x4 - x3, x3 - x5 };

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
		for (int ik = 0; ik < STEP; ik++)
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
	float *Ys0 = &Gs[0][0][0][0], *Ys1 = &Ds[0][0][0][0];
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


//SELECT: Size = 72, Time = 10.6781 msec, Performace = 14480 GFlop/s
//        Size = 18, Time = 2.74469 msec, Performace = 14083.5 GFlop/s
#ifndef AWINOGRAD_2X3_KERNEL14
#define AWINOGRAD_2X3_KERNEL14

#define AW23_K14(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel14\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 72, Time = 9.72576 msec, Performace = 15897.9 GFlop/s
//Size = 72, Time = 9.67849 msec, Performace = 15975.5 GFlop/s
//Size = 72, Time = 9.83459 msec, Performace = 15721.9 GFlop/s

//Size = 18, Time = 2.7521 msec, Performace = 14045.5 GFlop/s
//Size = 18, Time = 2.56932 msec, Performace = 15044.7 GFlop/s
__global__ void AW23_kernel14(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][g(oc): 64]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][d( j): 64]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty > 7) << 1);
	CW += (ty & 7)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx > 7) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
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
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X[N, IH, IW, IC]
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2), tX5 = tX0 + IC * 5;
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
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + IC_OC, W2 = W0 + (IC_OC << 1);
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
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2), tX5 = tX0 + IC * 5;
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
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC, W2 = W0 + (IC_OC << 1);
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


#ifndef AWINOGRAD_2X3_KERNEL15
#define AWINOGRAD_2X3_KERNEL15

#define AW23_K15(stream, oc_index, j_index, X, IH, IW, CW, FH, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	AW23_kernel15<FH>\
		<<< dim3(GN>>6, GM>>7), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 72, Time = 9.72576 msec, Performace = 15897.9 GFlop/s
//Size = 72, Time = 9.67849 msec, Performace = 15975.5 GFlop/s
//Size = 72, Time = 9.83459 msec, Performace = 15721.9 GFlop/s

//Size = 18, Time = 2.73548 msec, Performace = 14130.9 GFlop/s
//Size = 18, Time = 2.56932 msec, Performace = 15044.7 GFlop/s
template<int FH>
__global__ void AW23_kernel15(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][g(oc): 64]
	__shared__ float Ds[2][8][4 + 1][64 + 4];//[buf][ik: STEP][elem: 4][d( j): 64]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << 6) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty > 7) << 1);
	CW += (ty & 7)*OC + toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << 7) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx > 7) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & 7);

	//prepare for threadIdx
	const int idx = (ty << 4) + tx;
	const int ux = (idx >> 6), uy = (idx & 63);
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
		int tX0 = X0 + fh * IW * IC;
		int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
		int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2), tX5 = tX0 + IC * 5;
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
		const int W0 = fh * IC * 3 * OC;//with the same tx
		const int W1 = W0 + IC_OC, W2 = W0 + (IC_OC << 1);
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
			int tX0 = X0 + fh * IW * IC + oic;
			int tX1 = tX0 + IC, tX2 = tX0 + (IC << 1);
			int tX3 = tX0 + IC * 3, tX4 = tX0 + (IC << 2), tX5 = tX0 + IC * 5;
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
			const int W0 = (fh*IC * 3 + oic) * OC;//fh, ic, fw, oc
			const int W1 = W0 + IC_OC, W2 = W0 + (IC_OC << 1);
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
