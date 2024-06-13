

//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL1
#define XENO1_WINOGRAD_F2x3_KERNEL1

#define xeno_winograd_2x3_1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel1<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 1.125, Time = 4.0287 msec, Performace = 599.677 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void xeno_winograd_f2x3_kernel1(
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

#pragma once
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


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL2
#define XENO1_WINOGRAD_F2x3_KERNEL2

#define xeno_winograd_2x3_2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel2<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 9, Time = 2.53436 msec, Performace = 7626.14 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6, int STEP_m1>
__global__ void xeno_winograd_f2x3_kernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(4 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(1 << LB) + 1];//{tgroup0}

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index, j2 = j0 + 2;
	int tj0 = j0 + ((tx >= STEP) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);

	//compute area---------------------------------------
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

#pragma once
	for (int fh = 0; fh < 3; fh++)
	{
		//load 4 group from CW
		const int wic = ty - ((ty >= STEP) << LB >> 1);
		const int woffset0 = ((fh * 3)*IC + wic)*OC;
		const int woffset1 = woffset0 + Wstride;
		const int woffset2 = woffset1 + Wstride;
		float4 g0 = *(float4*)(CW + woffset0);
		float4 g1 = *(float4*)(CW + woffset1);
		float4 g2 = *(float4*)(CW + woffset2);
		WinoGrad_produce_G(Gs[buf][ty][tx        ], g0.x, g1.x, g2.x);
		WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], g0.y, g1.y, g2.y);
		WinoGrad_produce_G(Gs[buf][ty][tx + STEP4], g0.z, g1.z, g2.z);
		WinoGrad_produce_G(Gs[buf][ty][tx + STEP6], g0.w, g1.w, g2.w);

		//load 1 group from X
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);

		const int xic = tx - ((tx >= STEP) << LB >> 1);
		const int xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		const int xoffset1 = xoffset0 + IC;
		const int xoffset2 = xoffset1 + IC;
		const int xoffset3 = xoffset2 + IC;
		float d0 = (ly0 ? X[xoffset0] : 0);
		float d1 = (ly1 ? X[xoffset1] : 0);
		float d2 = (ly2 ? X[xoffset2] : 0);
		float d3 = (ly3 ? X[xoffset3] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		__syncthreads();

		for (int oic = 1, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx        ], g4 = Gs[buf][ik + STEP][tx        ];
				float4 g1 = Gs[buf][ik][tx + STEP2], g5 = Gs[buf][ik + STEP][tx + STEP2];
				float4 g2 = Gs[buf][ik][tx + STEP4], g6 = Gs[buf][ik + STEP][tx + STEP4];
				float4 g3 = Gs[buf][ik][tx + STEP6], g7 = Gs[buf][ik + STEP][tx + STEP6];
				float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0); wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1); wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
			}
			buf ^= 1;

			//load 1 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			const int xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			const int xoffset1 = xoffset0 + IC;
			const int xoffset2 = xoffset1 + IC;
			const int xoffset3 = xoffset2 + IC;
			float d0 = (ly0 ? X[xoffset0] : 0);
			float d1 = (ly1 ? X[xoffset1] : 0);
			float d2 = (ly2 ? X[xoffset2] : 0);
			float d3 = (ly3 ? X[xoffset3] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);

			//load 2 group from CW
			const int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			const int woffset0 = ((fh * 3)*IC + wic)*OC;
			const int woffset1 = woffset0 + Wstride;
			const int woffset2 = woffset1 + Wstride;
			float4 g0 = *(float4*)(CW + woffset0);
			float4 g1 = *(float4*)(CW + woffset1);
			float4 g2 = *(float4*)(CW + woffset2);
			WinoGrad_produce_G(Gs[buf][ty][tx], g0.x, g1.x, g2.x);
			WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], g0.y, g1.y, g2.y);
			WinoGrad_produce_G(Gs[buf][ty][tx + STEP4], g0.z, g1.z, g2.z);
			WinoGrad_produce_G(Gs[buf][ty][tx + STEP6], g0.w, g1.w, g2.w);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2];
			float4 g2 = Gs[buf][ik][tx + STEP4];
			float4 g3 = Gs[buf][ik][tx + STEP6];
			float4 g4 = Gs[buf][ik + STEP][tx];
			float4 g5 = Gs[buf][ik + STEP][tx + STEP2];
			float4 g6 = Gs[buf][ik + STEP][tx + STEP4];
			float4 g7 = Gs[buf][ik + STEP][tx + STEP6];

			float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];

			wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0); wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
			wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1); wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2; *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4; *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6; *(float4*)(Y + Y3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL3
#define XENO1_WINOGRAD_F2x3_KERNEL3

#define xeno_winograd_2x3_3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel3<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 9, Time = 2.50928 msec, Performace = 7702.35 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6, int STEP_m1>
__global__ void xeno_winograd_f2x3_kernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(4 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(1 << LB) + 1];//{tgroup0}

	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index, j2 = j0 + 2;
	int tj0 = j0 + ((tx >= STEP) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);

#pragma once//compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 4 group from CW
		const int wic = (ty & STEP_m1);
		const int woffset = ((fh * 3)*IC + wic)*OC;
		float4 g0 = *(float4*)(CW + woffset);
		float4 g1 = *(float4*)(CW + woffset + Wstride);
		float4 g2 = *(float4*)(CW + woffset + (Wstride << 1));
		WinoGrad_produce_G(Gs[buf][ty][tx], g0.x, g1.x, g2.x);
		WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], g0.y, g1.y, g2.y);
		WinoGrad_produce_G(Gs[buf][ty][tx + STEP4], g0.z, g1.z, g2.z);
		WinoGrad_produce_G(Gs[buf][ty][tx + STEP6], g0.w, g1.w, g2.w);

		//load 1 group from X
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);

		const int xic = (tx & STEP_m1);
		const int xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		const int xoffset1 = xoffset0 + IC;
		const int xoffset2 = xoffset1 + IC;
		const int xoffset3 = xoffset2 + IC;
		float d0 = (ly0 ? X[xoffset0] : 0);
		float d1 = (ly1 ? X[xoffset1] : 0);
		float d2 = (ly2 ? X[xoffset2] : 0);
		float d3 = (ly3 ? X[xoffset3] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g4 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g5 = Gs[buf][ik + STEP][tx + STEP2];
				float4 g2 = Gs[buf][ik][tx + STEP4], g6 = Gs[buf][ik + STEP][tx + STEP4];
				float4 g3 = Gs[buf][ik][tx + STEP6], g7 = Gs[buf][ik + STEP][tx + STEP6];
				float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0); wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1); wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			const int xoffset1 = xoffset0 + IC;
			const int xoffset2 = xoffset1 + IC;
			const int xoffset3 = xoffset2 + IC;
			float d0 = (ly0 ? X[xoffset0] : 0);
			float d1 = (ly1 ? X[xoffset1] : 0);
			float d2 = (ly2 ? X[xoffset2] : 0);
			float d3 = (ly3 ? X[xoffset3] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = ((fh * 3)*IC + wic)*OC;
			float4 g0 = *(float4*)(CW + woffset);
			float4 g1 = *(float4*)(CW + woffset + Wstride);
			float4 g2 = *(float4*)(CW + woffset + (Wstride << 1));
			WinoGrad_produce_G(Gs[buf][ty][tx], g0.x, g1.x, g2.x);
			WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], g0.y, g1.y, g2.y);
			WinoGrad_produce_G(Gs[buf][ty][tx + STEP4], g0.z, g1.z, g2.z);
			WinoGrad_produce_G(Gs[buf][ty][tx + STEP6], g0.w, g1.w, g2.w);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g4 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g5 = Gs[buf][ik + STEP][tx + STEP2];
			float4 g2 = Gs[buf][ik][tx + STEP4], g6 = Gs[buf][ik + STEP][tx + STEP4];
			float4 g3 = Gs[buf][ik][tx + STEP6], g7 = Gs[buf][ik + STEP][tx + STEP6];
			float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];

			wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0); wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
			wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1); wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2; *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4; *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6; *(float4*)(Y + Y3 + 4) = v7;
}

#endif



//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL4
#define XENO1_WINOGRAD_F2x3_KERNEL4

#define xeno_winograd_2x3_4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel4<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 9, Time = 2.29791 msec, Performace = 8410.83 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void xeno_winograd_f2x3_kernel4(
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
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);

#pragma once//compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from CW
		int wic = ty - ((ty >= STEP) << LB >> 1);//with the same tx
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
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		int xic = tx - ((tx >= STEP) << LB >> 1);//with the same ty
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

		for (int oic = 1, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx        ], g2 = Gs[buf][ik + STEP][tx        ];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d0 = Ds[buf][ik][ty        ], d2 = Ds[buf][ik + STEP][ty        ];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

				wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 1 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
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
			float4 g0 = Gs[buf][ik][tx        ], g2 = Gs[buf][ik + STEP][tx        ];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d0 = Ds[buf][ik][ty        ], d2 = Ds[buf][ik + STEP][ty        ];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0;
	*(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2;
	*(float4*)(Y + Y3) = v3;
	*(float4*)(Y + Y4) = v4;
	*(float4*)(Y + Y5) = v5;
	*(float4*)(Y + Y6) = v6;
	*(float4*)(Y + Y7) = v7;
}

#endif


//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL5
#define XENO1_WINOGRAD_F2x3_KERNEL5

#define xeno_winograd_2x3_5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel5<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 9, Time = 2.22246 msec, Performace = 8696.4 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void xeno_winograd_f2x3_kernel5(
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
#ifndef XENO1_WINOGRAD_F2x3_KERNEL6
#define XENO1_WINOGRAD_F2x3_KERNEL6

#define xeno_winograd_2x3_6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel6<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, IC>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 9, Time = 2.12893 msec, Performace = 9078.43 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC>
__global__ void xeno_winograd_f2x3_kernel6(
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
			Ds[buf][tx][ty        ] = winograd_f2x3_d(d0, d1, d2, d3);
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
#ifndef XENO1_WINOGRAD_F2x3_KERNEL6A
#define XENO1_WINOGRAD_F2x3_KERNEL6A

#define xeno_winograd_2x3_6A(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel6A<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, IC>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 9, Time = 2.36388 msec, Performace = 8176.12 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC>
__global__ void xeno_winograd_f2x3_kernel6A(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float3 Gs[2][1 << LB][(4 << LB) + 1];//{toc0, toc1}
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
		Gs[buf][ty][tx        ] = winograd_g3(g0.x, g1.x, g2.x);
		Gs[buf][ty][tx + STEP2] = winograd_g3(g0.y, g1.y, g2.y);

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
		Ds[buf][tx][ty        ] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float3 rg0 = Gs[buf][ik][tx        ], rg2 = Gs[buf][ik + STEP][tx];
				float3 rg1 = Gs[buf][ik][tx + STEP2], rg3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
				float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

				float4 g0 = winograd_g3_to_g4(rg0);
				float4 g1 = winograd_g3_to_g4(rg1);
				float4 g2 = winograd_g3_to_g4(rg2);
				float4 g3 = winograd_g3_to_g4(rg3);

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
			Gs[buf][ty][tx        ] = winograd_g3(g0.x, g1.x, g2.x);
			Gs[buf][ty][tx + STEP2] = winograd_g3(g0.y, g1.y, g2.y);

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
			float3 rg0 = Gs[buf][ik][tx], rg2 = Gs[buf][ik + STEP][tx];
			float3 rg1 = Gs[buf][ik][tx + STEP2], rg3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
			float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			float4 g0 = winograd_g3_to_g4(rg0);
			float4 g1 = winograd_g3_to_g4(rg1);
			float4 g2 = winograd_g3_to_g4(rg2);
			float4 g3 = winograd_g3_to_g4(rg3);

			wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
	}

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif



//==============================================================

//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL7
#define XENO1_WINOGRAD_F2x3_KERNEL7

#define xeno_winograd_2x3_7(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel7<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, G, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 9, Time = 2.42484 msec, Performace = 7970.56 GFlop/
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void xeno_winograd_f2x3_kernel7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, FH: 3, IC, GW: 4]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
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
	G += (oc0 + ((ty >= STEP) << 1)) * 12 * IC;//G[oc0, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	j0 = j0 * OC + oc0;//j0 = Y0

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

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int goffset0 = (fh*IC + wic) << 2;//G[toc0, fh, ic, gw: 0-3]
		const int goffset1 = goffset0 + 12 * IC;//G[toc0, fh, ic, gw: 0-3]
		Gs[buf][ty][tx] = *(float4*)(G + goffset0);
		Gs[buf][ty][tx + STEP2] = *(float4*)(G + goffset1);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
				float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];

				wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

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

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int goffset0 = (fh*IC + wic) << 2;//G[toc0, fh, ic, gw: 0-3]
			const int goffset1 = goffset0 + 12 * IC;//G[toc0, fh, ic, gw: 0-3]
			Gs[buf][ty][tx] = *(float4*)(G + goffset0);
			Gs[buf][ty][tx + STEP2] = *(float4*)(G + goffset1);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
			float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];

			wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
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
#ifndef XENO1_WINOGRAD_F2x3_KERNEL8
#define XENO1_WINOGRAD_F2x3_KERNEL8

#define xeno_winograd_2x3_8(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel8<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, IC>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, G, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4:Size = 9, Time = 2.39855 msec, Performace = 8057.92 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC>
__global__ void xeno_winograd_f2x3_kernel8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, FH: 3, IC, GW: 4]
	float* __restrict__ Y, int OH_OW, int OW,
	int OC,
	int ph, int pw,//sh = sw = 1
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
	G += (oc0 + ((ty >= STEP) << 1)) * 12 * IC;//G[oc0, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	j0 = j0 * OC + oc0;//j0 = Y0

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
		int xic = (tx & STEP_m1);//with the same ty
		int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		float d0 = (ly0 ? X[X0     ] : 0);
		float d1 = (ly1 ? X[X0 + IC] : 0);
		float d2 = (ly2 ? X[X0 + (IC << 1)] : 0);
		float d3 = (ly3 ? X[X0 + (IC * 3)] : 0);
		float d4 = (ly4 ? X[X0 + (IC << 2)] : 0);
		float d5 = (ly5 ? X[X0 + (IC * 5)] : 0);
		Ds[buf][tx][ty       ] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int goffset0 = (fh*IC + wic) << 2;//G[toc0, fh, ic, gw: 0-3]
		const int goffset1 = goffset0 + 12 * IC;//G[toc0, fh, ic, gw: 0-3]
		Gs[buf][ty][tx] = *(float4*)(G + goffset0);
		Gs[buf][ty][tx + STEP2] = *(float4*)(G + goffset1);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
				float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];

				wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 1 group from X
			int xic = oic + (tx & STEP_m1);
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC;
			int X3 = X2 + IC, X4 = X3 + IC, X5 = X4 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X0 + IC] : 0);
			float d2 = (ly2 ? X[X0 + (IC << 1)] : 0);
			float d3 = (ly3 ? X[X0 + (IC * 3)] : 0);
			float d4 = (ly4 ? X[X0 + (IC << 2)] : 0);
			float d5 = (ly5 ? X[X0 + (IC * 5)] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int goffset0 = (fh*IC + wic) << 2;//G[toc0, fh, ic, gw: 0-3]
			const int goffset1 = goffset0 + 12 * IC;//G[toc0, fh, ic, gw: 0-3]
			Gs[buf][ty][tx] = *(float4*)(G + goffset0);
			Gs[buf][ty][tx + STEP2] = *(float4*)(G + goffset1);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
			float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];

			wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
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
#ifndef XENO1_WINOGRAD_F2x3_KERNEL9
#define XENO1_WINOGRAD_F2x3_KERNEL9

#define xeno_winograd_2x3_9(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel9<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, IC>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, G, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 9, Time = 2.39855 msec, Performace = 8057.92 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC>
__global__ void xeno_winograd_f2x3_kernel9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, FH: 3, IC, GW: 4]
	float* __restrict__ Y, int OH_OW, int OW,
	int OC,
	int ph, int pw,//sh = sw = 1
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
	G += (oc0 + ((ty >= STEP) << 1)) * 12 * IC;//G[oc0, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0); 
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	X += ((tn0*IH + tih0)*IW + tiw0)*IC;
	j0 = j0 * OC + oc0;//j0 = Y0

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 < IH - fh);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		int xic = (tx & STEP_m1);//with the same ty
		int X0 = fh * IW*IC + xic;
		float d0 = (ly0 ? X[X0] : 0);
		float d1 = (ly1 ? X[X0 + IC] : 0);
		float d2 = (ly2 ? X[X0 + (IC << 1)] : 0);
		float d3 = (ly3 ? X[X0 + (IC * 3)] : 0);
		float d4 = (ly4 ? X[X0 + (IC << 2)] : 0);
		float d5 = (ly5 ? X[X0 + (IC * 5)] : 0);
		Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int goffset0 = (fh*IC + wic) << 2;//G[toc0, fh, ic, gw: 0-3]
		const int goffset1 = goffset0 + 12 * IC;//G[toc0, fh, ic, gw: 0-3]
		Gs[buf][ty][tx] = *(float4*)(G + goffset0);
		Gs[buf][ty][tx + STEP2] = *(float4*)(G + goffset1);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) 
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
				float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
				
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];

				wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
				wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
				wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 1 group from X
			int xic = oic + (tx & STEP_m1);
			int X0 = fh * IW*IC + xic;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X0 + IC] : 0);
			float d2 = (ly2 ? X[X0 + (IC << 1)] : 0);
			float d3 = (ly3 ? X[X0 + (IC * 3)] : 0);
			float d4 = (ly4 ? X[X0 + (IC << 2)] : 0);
			float d5 = (ly5 ? X[X0 + (IC * 5)] : 0);
			Ds[buf][tx][ty] = winograd_f2x3_d(d0, d1, d2, d3);
			Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int goffset0 = (fh*IC + wic) << 2;//G[toc0, fh, ic, gw: 0-3]
			const int goffset1 = goffset0 + 12 * IC;//G[toc0, fh, ic, gw: 0-3]
			Gs[buf][ty][tx] = *(float4*)(G + goffset0);
			Gs[buf][ty][tx + STEP2] = *(float4*)(G + goffset1);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
			float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];

			wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			wino_grad4_GxW(v4, v5, g0, g1, g2, g3, d2);//d2 * {g0, g1, g2, g3}
			wino_grad4_GxW(v6, v7, g0, g1, g2, g3, d3);//d3 * {g0, g1, g2, g3}
		}
	}

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif

//===============================================================


//(OH, OW) % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL10
#define XENO1_WINOGRAD_F2x3_KERNEL10

#define xeno_winograd_2x3_10(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel10<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB), (1<<LB>>1)-1, IC>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 9, Time = 2.36388 msec, Performace = 8176.12 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6, int STEP_m1, int IC>
__global__ void xeno_winograd_f2x3_kernel10(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float3 Gs[2][1 << LB][(4 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 8*8 results
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from CW
		int wic = (ty & STEP_m1);//with the same tx
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float4 g0 = *(float4*)(CW + W0);//fw0
		float4 g1 = *(float4*)(CW + W1);//fw1
		float4 g2 = *(float4*)(CW + W2);//fw2
		Gs[buf][ty][tx        ] = winograd_g3(g0.x, g1.x, g2.x);
		Gs[buf][ty][tx + STEP2] = winograd_g3(g0.y, g1.y, g2.y);
		Gs[buf][ty][tx + STEP4] = winograd_g3(g0.z, g1.z, g2.z);
		Gs[buf][ty][tx + STEP6] = winograd_g3(g0.w, g1.w, g2.w);

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
		Ds[buf][tx][ty        ] = winograd_f2x3_d(d0, d1, d2, d3);
		Ds[buf][tx][ty + STEP2] = winograd_f2x3_d(d2, d3, d4, d5);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float3 rg0 = Gs[buf][ik][tx];
				float3 rg1 = Gs[buf][ik][tx + STEP2];
				float3 rg2 = Gs[buf][ik][tx + STEP4];
				float3 rg3 = Gs[buf][ik][tx + STEP6];
				float3 rg4 = Gs[buf][ik + STEP][tx];
				float3 rg5 = Gs[buf][ik + STEP][tx + STEP2];
				float3 rg6 = Gs[buf][ik + STEP][tx + STEP4];
				float3 rg7 = Gs[buf][ik + STEP][tx + STEP6];

				float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
				float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

				float4 g0 = winograd_g3_to_g4(rg0);
				float4 g1 = winograd_g3_to_g4(rg1);
				float4 g2 = winograd_g3_to_g4(rg2);
				float4 g3 = winograd_g3_to_g4(rg3);
				float4 g4 = winograd_g3_to_g4(rg4);
				float4 g5 = winograd_g3_to_g4(rg5);
				float4 g6 = winograd_g3_to_g4(rg6);
				float4 g7 = winograd_g3_to_g4(rg7);

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			buf ^= 1;

			//load 2 group from CW
			int wic = oic + (ty & STEP_m1);
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);
			Gs[buf][ty][tx        ] = winograd_g3(g0.x, g1.x, g2.x);
			Gs[buf][ty][tx + STEP2] = winograd_g3(g0.y, g1.y, g2.y);
			Gs[buf][ty][tx + STEP4] = winograd_g3(g0.z, g1.z, g2.z);
			Gs[buf][ty][tx + STEP6] = winograd_g3(g0.w, g1.w, g2.w);

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
			float3 rg0 = Gs[buf][ik][tx];
			float3 rg1 = Gs[buf][ik][tx + STEP2];
			float3 rg2 = Gs[buf][ik][tx + STEP4];
			float3 rg3 = Gs[buf][ik][tx + STEP6];
			float3 rg4 = Gs[buf][ik + STEP][tx];
			float3 rg5 = Gs[buf][ik + STEP][tx + STEP2];
			float3 rg6 = Gs[buf][ik + STEP][tx + STEP4];
			float3 rg7 = Gs[buf][ik + STEP][tx + STEP6];

			float4 d0 = Ds[buf][ik][ty], d1 = Ds[buf][ik + STEP][ty];
			float4 d2 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];

			float4 g0 = winograd_g3_to_g4(rg0);
			float4 g1 = winograd_g3_to_g4(rg1);
			float4 g2 = winograd_g3_to_g4(rg2);
			float4 g3 = winograd_g3_to_g4(rg3);
			float4 g4 = winograd_g3_to_g4(rg4);
			float4 g5 = winograd_g3_to_g4(rg5);
			float4 g6 = winograd_g3_to_g4(rg6);
			float4 g7 = winograd_g3_to_g4(rg7);

			wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
			wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
			wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
			wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
			wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
			wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
			wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);
			wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
		}
	}

	
	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif