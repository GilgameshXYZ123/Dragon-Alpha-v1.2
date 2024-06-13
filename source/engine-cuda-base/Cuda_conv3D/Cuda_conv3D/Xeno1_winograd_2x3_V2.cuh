

//FH = FW = 3
//sh = sw = 1
//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V1
#define CONV_3D_WINOGRAD_KERNEL_W3_V1

#define wingrad_v1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, 3, 3, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	int ow1 = ow0 + 1;

	//prepare for GK = FH * FW
	int GK = FH * FW;

	//compute area---------------------------------------
	float v0 = 0, v1 = 0;
	for (int k = 0; k < GK; k += 3)
	{
		int fh = k / FW, fw0 = k % FW;
		int fw1 = fw0 + 1;
		int fw2 = fw0 + 2;

		int ih = oh0 - ph + fh;
		int iw0 = ow0 - pw + fw0;
		int iw1 = iw0 + 1;
		int iw2 = iw0 + 2;
		int iw3 = iw0 + 3;

		for (int ic = 0; ic < IC; ic++)
		{
			bool ly = (ih >= 0) && (ih < IH);
			bool ly0 = ly && (iw0 >= 0) && (iw0 < IW);
			bool ly1 = ly && (iw1 >= 0) && (iw1 < IW);
			bool ly2 = ly && (iw2 >= 0) && (iw2 < IW);
			bool ly3 = ly && (iw3 >= 0) && (iw3 < IW);

			//F(3*3, 2*2)
			//

			float d0 = (ly0 ? get4d(X, n0, ih, iw0, ic, IH, IW, IC) : 0);
			float d1 = (ly1 ? get4d(X, n0, ih, iw1, ic, IH, IW, IC) : 0);
			float d2 = (ly2 ? get4d(X, n0, ih, iw2, ic, IH, IW, IC) : 0);
			float d3 = (ly3 ? get4d(X, n0, ih, iw3, ic, IH, IW, IC) : 0);

			float g0 = get4d(W, oc0, fh, fw0, ic, FH, FW, IC);
			float g1 = get4d(W, oc0, fh, fw1, ic, FH, FW, IC);
			float g2 = get4d(W, oc0, fh, fw2, ic, FH, FW, IC);

			float m1 = g0 * (d0 - d2);
			float m2 = 0.5f * (d1 + d2) * (g0 + g1 + g2);
			float m3 = 0.5f * (d2 - d1) * (g0 - g1 + g2);
			float m4 = g2 * (d1 - d3);

			v0 += (m1 + m2 + m3);
			v1 += (m2 - m3 - m4);
		}
	}

	get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC) = v0;
	get4d(Y, n0, oh0, ow1, oc0, OH, OW, OC) = v1;
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_B2
#define CONV_3D_WINOGRAD_KERNEL_W3_B2

#define wingrad_b2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_b2<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 1.125, Time = 4.0287 msec, Performace = 599.677 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dWinoGrad_kernel_b2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(1 << LB) + 1];//{tgroup0}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index, j2 = j0 + 2;
	int tj0 = j0 + ((tx >= STEP) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);

	//compute area---------------------------------------
	float4 v0 = float4{ 0, 0, 0, 0 };
	float4 v1 = float4{ 0, 0, 0, 0 };
	float4 v2 = float4{ 0, 0, 0, 0 };
	float4 v3 = float4{ 0, 0, 0, 0 };

	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from CW
		int wic = ty - ((ty >= STEP) << LB >> 1);
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 g0 = *(float2*)(CW + W0);
		float2 g1 = *(float2*)(CW + W1);
		float2 g2 = *(float2*)(CW + W2);
		Gs[buf][ty][tx] = winograd_f2x3_g(g0.x, g1.x, g2.x);
		Gs[buf][ty][tx + STEP2] = winograd_f2x3_g(g0.y, g1.y, g2.y);

		//load 1 group from X
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);

		int xic = tx - ((tx >= STEP) << LB >> 1);
		int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
		float d0 = (ly0 ? X[X0] : 0);
		float d1 = (ly1 ? X[X1] : 0);
		float d2 = (ly2 ? X[X2] : 0);
		float d3 = (ly3 ? X[X3] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		__syncthreads();

		for (int oic = 1, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2];
				float4 g2 = Gs[buf][ik + STEP][tx];
				float4 g3 = Gs[buf][ik + STEP][tx + STEP2];

				float4 d0 = Ds[buf][ik][ty];
				float4 d1 = Ds[buf][ik + STEP][ty];

				wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 1 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);

			//load 2 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[buf][ty][tx] = winograd_f2x3_g(g0.x, g1.x, g2.x);
			Gs[buf][ty][tx + STEP2] = winograd_f2x3_g(g0.y, g1.y, g2.y);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 d0 = Ds[buf][ik][ty];
			float4 d1 = Ds[buf][ik + STEP][ty];

			float4 g0 = Gs[buf][ik][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2];
			float4 g2 = Gs[buf][ik + STEP][tx];
			float4 g3 = Gs[buf][ik + STEP][tx + STEP2];

			wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC;
	const int Y2 = Y1 + OC;
	const int Y3 = Y2 + OC;

	*(float4*)(Y + Y0) = v0;
	*(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2;
	*(float4*)(Y + Y3) = v3;
}

#endif

//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_B0
#define XENO1_WINOGRAD_F2x3_B0

#define xeno_winograd_2x3_b0(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_B0<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, IC>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 9, Time = 2.12893 msec, Performace = 9078.43 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC>
__global__ void xeno_winograd_f2x3_kernel_B0(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from CW
		int wic = (ty & STEP_m1);//with the same tx
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 g0 = *(float2*)(CW + W0);
		float2 g1 = *(float2*)(CW + W1);
		float2 g2 = *(float2*)(CW + W2);
		Gs[buf][ty][tx] = winograd_f2x3_g(g0.x, g1.x, g2.x);
		Gs[buf][ty][tx + STEP2] = winograd_f2x3_g(g0.y, g1.y, g2.y);

		//load 1 group from X
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		int xic = (tx & STEP_m1);//with the same ty
		int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		int X1 = X0 + IC, X2 = X1 + IC;
		int X3 = X2 + IC, X4 = X3 + IC, X5 = X4 + IC;
		float d0 = (ly0 ? X[X0] : 0);
		float d1 = (ly1 ? X[X1] : 0);
		float d2 = (ly2 ? X[X2] : 0);
		float d3 = (ly3 ? X[X3] : 0);
		float d4 = (ly4 ? X[X4] : 0);
		float d5 = (ly5 ? X[X5] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
				float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

				wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}

				wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 2 group from CW
			int wic = oic + (ty & STEP_m1);
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[buf][ty][tx] = winograd_f2x3_g(g0.x, g1.x, g2.x);
			Gs[buf][ty][tx + STEP2] = winograd_f2x3_g(g0.y, g1.y, g2.y);

			//load 1 group from X
			int xic = oic + (tx & STEP_m1);
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC;
			int X3 = X2 + IC, X4 = X3 + IC, X5 = X4 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			float d4 = (ly4 ? X[X4] : 0);
			float d5 = (ly5 ? X[X5] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
			float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
		buf ^= 1;
	}

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_B1
#define XENO1_WINOGRAD_F2x3_KERNEL_B1

#define xeno_winograd_2x3_b1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel6_B1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5), OC>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 2.12893 msec, Performace = 9078.43 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5, int OC>
__global__ void xeno_winograd_f2x3_kernel6_B1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from CW
		int wic = (ty & STEP_m1);//with the same tx
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 g0 = *(float2*)(CW + W0);
		float2 g1 = *(float2*)(CW + W1);
		float2 g2 = *(float2*)(CW + W2);
		Gs[buf][ty][tx] = winograd_f2x3_g(g0.x, g1.x, g2.x);
		Gs[buf][ty][tx + STEP2] = winograd_f2x3_g(g0.y, g1.y, g2.y);

		//load 1 group from X
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		
		const int xic = (tx & STEP_m1);//with the same ty
		const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		float d0 = (ly0 ? X[X0      ] : 0);
		float d1 = (ly1 ? X[X0 + IC ] : 0);
		float d2 = (ly2 ? X[X0 + IC2] : 0);
		float d3 = (ly3 ? X[X0 + IC3] : 0);
		float d4 = (ly4 ? X[X0 + IC4] : 0);
		float d5 = (ly5 ? X[X0 + IC5] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

				wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[buf][ty][tx] = winograd_f2x3_g(g0.x, g1.x, g2.x);
			Gs[buf][ty][tx + STEP2] = winograd_f2x3_g(g0.y, g1.y, g2.y);

			//load 1 group from X
			int xic = oic + (tx & STEP_m1);
			const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			float d0 = (ly0 ? X[X0      ] : 0);
			float d1 = (ly1 ? X[X0 + IC ] : 0);
			float d2 = (ly2 ? X[X0 + IC2] : 0);
			float d3 = (ly3 ? X[X0 + IC3] : 0);
			float d4 = (ly4 ? X[X0 + IC4] : 0);
			float d5 = (ly5 ? X[X0 + IC5] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
		buf ^= 1;
	}

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif



//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_B2
#define XENO1_WINOGRAD_F2x3_KERNEL_B2

#define xeno_winograd_2x3_b2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel6_B2<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 2.18588 msec, Performace = 8841.91 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel6_B2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		float d0 = (ly0 ? X[X0] : 0);
		float d1 = (ly1 ? X[X0 + IC] : 0);
		float d2 = (ly2 ? X[X0 + IC2] : 0);
		float d3 = (ly3 ? X[X0 + IC3] : 0);
		float d4 = (ly4 ? X[X0 + IC4] : 0);
		float d5 = (ly5 ? X[X0 + IC5] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

		//load 2 group from CW
		int wic = (ty & STEP_m1);//with the same tx
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 g0 = *(float2*)(CW + W0);
		float2 g1 = *(float2*)(CW + W1);
		float2 g2 = *(float2*)(CW + W2);
		Gs[buf][ty][tx] = winograd_f2x3_g(g0.x, g1.x, g2.x);
		Gs[buf][ty][tx + STEP2] = winograd_f2x3_g(g0.y, g1.y, g2.y);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

				G_winograd4_W(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				G_winograd4_W(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				G_winograd4_W(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				G_winograd4_W(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 1 group from X
			int xic = oic + (tx & STEP_m1);
			const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X0 + IC] : 0);
			float d2 = (ly2 ? X[X0 + IC2] : 0);
			float d3 = (ly3 ? X[X0 + IC3] : 0);
			float d4 = (ly4 ? X[X0 + IC4] : 0);
			float d5 = (ly5 ? X[X0 + IC5] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[buf][ty][tx] = winograd_f2x3_g(g0.x, g1.x, g2.x);
			Gs[buf][ty][tx + STEP2] = winograd_f2x3_g(g0.y, g1.y, g2.y);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			G_winograd4_W(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			G_winograd4_W(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			G_winograd4_W(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			G_winograd4_W(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
		buf ^= 1;
	}

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_B3
#define XENO1_WINOGRAD_F2x3_KERNEL_B3

#define xeno_winograd_2x3_b3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel6_B3<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 2.0847 msec, Performace = 9271.05 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel6_B3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		float d0 = (ly0 ? X[X0] : 0);
		float d1 = (ly1 ? X[X0 + IC] : 0);
		float d2 = (ly2 ? X[X0 + IC2] : 0);
		float d3 = (ly3 ? X[X0 + IC3] : 0);
		float d4 = (ly4 ? X[X0 + IC4] : 0);
		float d5 = (ly5 ? X[X0 + IC5] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

		//load 2 group from CW
		int wic = (ty & STEP_m1);//with the same tx
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
		float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
		Gs[buf][ty][tx] = g0;
		Gs[buf][ty][tx + STEP2] = g1;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
				
				G_winograd4_W(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				G_winograd4_W(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				G_winograd4_W(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				G_winograd4_W(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X0 + IC] : 0);
			float d2 = (ly2 ? X[X0 + IC2] : 0);
			float d3 = (ly3 ? X[X0 + IC3] : 0);
			float d4 = (ly4 ? X[X0 + IC4] : 0);
			float d5 = (ly5 ? X[X0 + IC5] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int W0 = ((fh * 3)*IC + wic)*OC;
			const int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);
			float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
			float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
			Gs[buf][ty][tx] = g0;
			Gs[buf][ty][tx + STEP2] = g1;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			G_winograd4_W(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			G_winograd4_W(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			G_winograd4_W(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			G_winograd4_W(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
		buf ^= 1;
	}

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_B4
#define XENO1_WINOGRAD_F2x3_KERNEL_B4

#define xeno_winograd_2x3_b4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel6_B4<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 2.0847 msec, Performace = 9271.05 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel6_B4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		float d0 = (ly0 ? X[X0] : 0);
		float d1 = (ly1 ? X[X0 + IC] : 0);
		float d2 = (ly2 ? X[X0 + IC2] : 0);
		float d3 = (ly3 ? X[X0 + IC3] : 0);
		float d4 = (ly4 ? X[X0 + IC4] : 0);
		float d5 = (ly5 ? X[X0 + IC5] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

		//load 2 group from CW
		int wic = (ty & STEP_m1);//with the same tx
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
		float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
		Gs[buf][ty][tx] = g0;
		Gs[buf][ty][tx + STEP2] = g1;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

				G_winograd4_W_V2(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				G_winograd4_W_V2(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				G_winograd4_W_V2(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				G_winograd4_W_V2(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X0 + IC] : 0);
			float d2 = (ly2 ? X[X0 + IC2] : 0);
			float d3 = (ly3 ? X[X0 + IC3] : 0);
			float d4 = (ly4 ? X[X0 + IC4] : 0);
			float d5 = (ly5 ? X[X0 + IC5] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int W0 = ((fh * 3)*IC + wic)*OC;
			const int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);
			float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
			float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
			Gs[buf][ty][tx] = g0;
			Gs[buf][ty][tx + STEP2] = g1;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			G_winograd4_W_V2(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			G_winograd4_W_V2(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			G_winograd4_W_V2(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			G_winograd4_W_V2(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
		buf ^= 1;
	}

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)//BEST
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_B5
#define XENO1_WINOGRAD_F2x3_KERNEL_B5

#define xeno_winograd_2x3_b5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel6_B5<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.90117 msec, Performace = 10166 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel6_B5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int W0 = ((fh * 3)*IC + wic)*OC;
		const int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
		float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
		Gs[buf][ty][tx] = g0;
		Gs[buf][ty][tx + STEP2] = g1;

		//load 1 group from X
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		float d0 = (ly0 ? X[X0] : 0);
		float d1 = (ly1 ? X[X0 + IC] : 0);
		float d2 = (ly2 ? X[X0 + IC2] : 0);
		float d3 = (ly3 ? X[X0 + IC3] : 0);
		float d4 = (ly4 ? X[X0 + IC4] : 0);
		float d5 = (ly5 ? X[X0 + IC5] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];

				G_winograd4_W_V3(v0, v1, t0, t1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v2, v3, t2, t3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v4, v5, t4, t5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v6, v7, t6, t7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int W0 = ((fh * 3)*IC + wic)*OC;
			const int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);
			float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
			float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
			Gs[buf][ty][tx] = g0;
			Gs[buf][ty][tx + STEP2] = g1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X0 + IC] : 0);
			float d2 = (ly2 ? X[X0 + IC2] : 0);
			float d3 = (ly3 ? X[X0 + IC3] : 0);
			float d4 = (ly4 ? X[X0 + IC4] : 0);
			float d5 = (ly5 ? X[X0 + IC5] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			G_winograd4_W_V3(v0, v1, t0, t1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v2, v3, t2, t3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v4, v5, t4, t5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v6, v7, t6, t7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_B6
#define XENO1_WINOGRAD_F2x3_KERNEL_B6

#define xeno_winograd_2x3_b6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_B6<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.90117 msec, Performace = 10166 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel_B6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += (ty & STEP_m1)*OC + oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	X += ((tn0 *IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);//X[tn0, 0, tiw0, 0]
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from CW
		const int W0 = fh * 3 * IC*OC;//with the same tx
		const int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
		float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
		Gs[buf][ty][tx] = g0;
		Gs[buf][ty][tx + STEP2] = g1;

		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 < IH - fh);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		
		const int X0 = fh * IW*IC;//with the same ty
		float d0 = (ly0 ? X[X0] : 0);
		float d1 = (ly1 ? X[X0 + IC] : 0);
		float d2 = (ly2 ? X[X0 + IC2] : 0);
		float d3 = (ly3 ? X[X0 + IC3] : 0);
		float d4 = (ly4 ? X[X0 + IC4] : 0);
		float d5 = (ly5 ? X[X0 + IC5] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];

				G_winograd4_W_V3(v0, v1, t0, t1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v2, v3, t2, t3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v4, v5, t4, t5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v6, v7, t6, t7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 2 group from CW
			const int W0 = (fh * 3 * IC + oic)*OC;
			const int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);
			float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
			float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
			Gs[buf][ty][tx] = g0;
			Gs[buf][ty][tx + STEP2] = g1;

			//load 1 group from X
			const int X0 = fh * IW*IC + oic;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X0 + IC] : 0);
			float d2 = (ly2 ? X[X0 + IC2] : 0);
			float d3 = (ly3 ? X[X0 + IC3] : 0);
			float d4 = (ly4 ? X[X0 + IC4] : 0);
			float d5 = (ly5 ? X[X0 + IC5] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			G_winograd4_W_V3(v0, v1, t0, t1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v2, v3, t2, t3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v4, v5, t4, t5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v6, v7, t6, t7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
		buf ^= 1;
	}

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	winograd_f2x3_VT4(v0, v1, t0, t1); *(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	winograd_f2x3_VT4(v2, v3, t2, t3); *(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	winograd_f2x3_VT4(v4, v5, t4, t5); *(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	winograd_f2x3_VT4(v6, v7, t6, t7); *(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)//BEST
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_B7
#define XENO1_WINOGRAD_F2x3_KERNEL_B7

#define xeno_winograd_2x3_b7(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel6_B7<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.87129 msec, Performace = 10328.3 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel6_B7(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int X0 = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float d0 = tex1Dfetch<float>(X, IF_int(ly0, X0      , -1));
		float d1 = tex1Dfetch<float>(X, IF_int(ly1, X0 + IC , -1));
		float d2 = tex1Dfetch<float>(X, IF_int(ly2, X0 + IC2, -1));
		float d3 = tex1Dfetch<float>(X, IF_int(ly3, X0 + IC3, -1));
		float d4 = tex1Dfetch<float>(X, IF_int(ly4, X0 + IC4, -1));
		float d5 = tex1Dfetch<float>(X, IF_int(ly5, X0 + IC5, -1));
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int W0 = ((fh * 3)*IC + wic)*OC;
		const int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 w0 = *(float2*)(CW + W0);
		float2 w1 = *(float2*)(CW + W1);
		float2 w2 = *(float2*)(CW + W2);
		float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
		float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
		Gs[buf][ty][tx] = g0;
		Gs[buf][ty][tx + STEP2] = g1;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

				G_winograd4_W_V3(v0, v1, t0, t1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v2, v3, t2, t3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v4, v5, t4, t5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v6, v7, t6, t7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int X0 = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float d0 = tex1Dfetch<float>(X, IF_int(ly0, X0      , -1));
			float d1 = tex1Dfetch<float>(X, IF_int(ly1, X0 + IC , -1));
			float d2 = tex1Dfetch<float>(X, IF_int(ly2, X0 + IC2, -1));
			float d3 = tex1Dfetch<float>(X, IF_int(ly3, X0 + IC3, -1));
			float d4 = tex1Dfetch<float>(X, IF_int(ly4, X0 + IC4, -1));
			float d5 = tex1Dfetch<float>(X, IF_int(ly5, X0 + IC5, -1));
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int W0 = ((fh * 3)*IC + wic)*OC;
			const int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 w0 = *(float2*)(CW + W0);
			float2 w1 = *(float2*)(CW + W1);
			float2 w2 = *(float2*)(CW + W2);
			float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
			float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
			Gs[buf][ty][tx] = g0;
			Gs[buf][ty][tx + STEP2] = g1;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			G_winograd4_W_V3(v0, v1, t0, t1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v2, v3, t2, t3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v4, v5, t4, t5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v6, v7, t6, t7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


///BEST STANDARD
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_B8
#define XENO1_WINOGRAD_F2x3_KERNEL_B8

#define xeno_winograd_2x3_b8(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel6_B9<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.87129 msec, Performace = 10328.3 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel6_B8(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float d0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float d1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float d2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float d3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float d4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float d5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh*IC3 + wic)*OC;
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		WinoGrad_produce_G(Gs[buf][ty][tx], w0.x, w1.x, w2.x);
		WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], w0.y, w1.y, w2.y);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];

				G_winograd4_W_V3(v0, v1, t0, t1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v2, v3, t2, t3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v4, v5, t4, t5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v6, v7, t6, t7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float d0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float d1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float d2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float d3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float d4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float d5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = (fh*IC3 + wic)*OC;
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			WinoGrad_produce_G(Gs[buf][ty][tx], w0.x, w1.x, w2.x);
			WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], w0.y, w1.y, w2.y);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];

			G_winograd4_W_V3(v0, v1, t0, t1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v2, v3, t2, t3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v4, v5, t4, t5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v6, v7, t6, t7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//FOR TEST: NOT CORRECT
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_B9
#define XENO1_WINOGRAD_F2x3_KERNEL_B9

#define xeno_winograd_2x3_b9(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel6_B9<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.87129 msec, Performace = 10328.3 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel6_B9(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 8*4 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float d0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float d1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float d2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float d3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float d4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float d5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh*IC3 + wic)*OC;
		float2 w0 = *(float2*)(CW + woffset);//[oc, fh, fw, wic]
		float2 w1 = *(float2*)(CW + woffset + Wstride);//[oc, fh, fw, wic]
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		WinoGrad_produce_G(Gs[buf][ty][tx], w0.x, w1.x, w2.x);
		WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], w0.y, w1.y, w2.y);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
				G_winograd4_W_V3(v0, v1, v4, v5, g0, g0, g2, g2, d0);//d0 * {g0, g1, g2, g3}
				G_winograd4_W_V3(v2, v3, v6, v7, g0, g0, g2, g2, d2);//d1 * {g0, g1, g2, g3}
			}
			float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
			float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

				G_winograd4_W_V3(t0, t1, t4, t5, g1, g1, g3, g3, d1);//d2 * {g0, g1, g2, g3}
				G_winograd4_W_V3(t2, t3, t6, t7, g1, g1, g3, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			winograd_f2x3_VT4(v0, v1, t0, t1);
			winograd_f2x3_VT4(v2, v3, t2, t3);
			winograd_f2x3_VT4(v4, v5, t4, t5);
			winograd_f2x3_VT4(v6, v7, t6, t7);

			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float d0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float d1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float d2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float d3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float d4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float d5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = (fh*IC3 + wic)*OC;
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			WinoGrad_produce_G(Gs[buf][ty][tx], w0.x, w1.x, w2.x);
			WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], w0.y, w1.y, w2.y);
			__syncthreads();
		}

#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];
			G_winograd4_W_V3(v0, v1, v4, v5, g0, g0, g2, g2, d0);//d0 * {g0, g1, g2, g3}
			G_winograd4_W_V3(v2, v3, v6, v7, g0, g0, g2, g2, d2);//d1 * {g0, g1, g2, g3}
		}
		float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
		float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			G_winograd4_W_V3(t0, t1, t4, t5, g1, g1, g3, g3, d1);//d2 * {g0, g1, g2, g3}
			G_winograd4_W_V3(t2, t3, t6, t7, g1, g1, g3, g3, d3);//d3 * {g0, g1, g2, g3}
		}
		winograd_f2x3_VT4(v0, v1, t0, t1);
		winograd_f2x3_VT4(v2, v3, t2, t3);
		winograd_f2x3_VT4(v4, v5, t4, t5);
		winograd_f2x3_VT4(v6, v7, t6, t7);
	}

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif

