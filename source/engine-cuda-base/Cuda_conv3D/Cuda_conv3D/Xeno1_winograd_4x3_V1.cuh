



#ifndef XENO1_WINOGRAD_F4X3_V1
#define XENO1_WINOGRAD_F4X3_V1

#define xeno_winograd_f4x3_k1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 2.25, Time = 10.2949 msec, Performace = 469.343 GFlop/s
template<int LB, int STEP>
__global__ void xeno_winograd_f4x3_kernel_v1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j0, n0, oh0, ow0);
	const int ow1 = ow0 + 1, ow2 = ow0 + 2, ow3 = ow0 + 3;
	const int ih0 = oh0 - ph, iw0 = ow0 - pw;

	float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f;
	float m3 = 0.0f, m4 = 0.0f, m5 = 0.0f;
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (ih0 >= -fh) && (ih0 + fh < IH);
		bool lx0 = lh0 && (iw0 >=  0) && (iw0     < IW);
		bool lx1 = lh0 && (iw0 >= -1) && (iw0 + 1 < IW);
		bool lx2 = lh0 && (iw0 >= -2) && (iw0 + 2 < IW);
		bool lx3 = lh0 && (iw0 >= -3) && (iw0 + 3 < IW);
		bool lx4 = lh0 && (iw0 >= -4) && (iw0 + 4 < IW);
		bool lx5 = lh0 && (iw0 >= -5) && (iw0 + 5 < IW);

		for (int ic = 0; ic < IC; ic++)
		{
			float w0 = get4d(W, oc0, fh, 0, ic, 3, 3, IC);
			float w1 = get4d(W, oc0, fh, 1, ic, 3, 3, IC);
			float w2 = get4d(W, oc0, fh, 2, ic, 3, 3, IC);

			float x0 = (lx0 ? get4d(X, n0, (ih0 + fh), iw0    , ic, IH, IW, IC) : 0);
			float x1 = (lx1 ? get4d(X, n0, (ih0 + fh), iw0 + 1, ic, IH, IW, IC) : 0);
			float x2 = (lx2 ? get4d(X, n0, (ih0 + fh), iw0 + 2, ic, IH, IW, IC) : 0);
			float x3 = (lx3 ? get4d(X, n0, (ih0 + fh), iw0 + 3, ic, IH, IW, IC) : 0);
			float x4 = (lx4 ? get4d(X, n0, (ih0 + fh), iw0 + 4, ic, IH, IW, IC) : 0);
			float x5 = (lx5 ? get4d(X, n0, (ih0 + fh), iw0 + 5, ic, IH, IW, IC) : 0);

			float g0 = w0;
			float g1 = w0 + w1 + w2;
			float g2 = w0 - w1 + w2;
			float g3 = w0 + 2 * w1 + 4 * w2;
			float g4 = w0 - 2 * w1 + 4 * w2;
			float g5 = w2;

			float d0 = 24 * x0 - 30 * x2 + 6 * x4;
			float d1 = 16 * x1 + 16 * x2 - 4 * x3 - 4 * x4;
			float d2 = -16 * x1 + 16 * x2 + 4 * x3 - 4 * x4;
			float d3 = -2 * x1 - x2 + 2 * x3 + x4;
			float d4 = 2 * x1 - x2 - 2 * x3 + x4;
			float d5 = 96 * x1 - 120 * x3 + 24 * x5;

			m0 += g0 * d0;
			m1 += g1 * d1;
			m2 += g2 * d2;
			m3 += g3 * d3;
			m4 += g4 * d4;
			m5 += g5 * d5;
		}
	}

	float v0 = (m0 + m1 + m2 + m3 + m4) / 24.0f;
	float v1 = (m1 - m2 + 2 * m3 - 2 * m4) / 24.0f;
	float v2 = (m1 + m2 + 4 * m3 + 4 * m4) / 24.0f;
	float v3 = (m1 - m2 + 8 * m3 - 8 * m4 + m5) / 24.0f;

	get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC) = v0;
	get4d(Y, n0, oh0, ow1, oc0, OH, OW, OC) = v1;
	get4d(Y, n0, oh0, ow2, oc0, OH, OW, OC) = v2;
	get4d(Y, n0, oh0, ow3, oc0, OH, OW, OC) = v3;

}

#endif



#ifndef XENO1_WINOGRAD_F4X3_V2
#define XENO1_WINOGRAD_F4X3_V2

#define xeno_winograd_f4x3_k2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v2<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 2.25, Time = 10.0122 msec, Performace = 482.597 GFlop/s
template<int LB, int STEP>
__global__ void xeno_winograd_f4x3_kernel_v2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	CW += oc0;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j0, n0, oh0, ow0);
	const int ow1 = ow0 + 1, ow2 = ow0 + 2, ow3 = ow0 + 3;
	const int ih0 = oh0 - ph, iw0 = ow0 - pw;

	//compute area------------------------------------------------
	const int Wstride = IC * OC;

	float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f;
	float m3 = 0.0f, m4 = 0.0f, m5 = 0.0f;
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (ih0 >= -fh) && (ih0 + fh < IH);
		bool lx0 = lh0 && (iw0 >= 0) && (iw0 < IW);
		bool lx1 = lh0 && (iw0 >= -1) && (iw0 + 1 < IW);
		bool lx2 = lh0 && (iw0 >= -2) && (iw0 + 2 < IW);
		bool lx3 = lh0 && (iw0 >= -3) && (iw0 + 3 < IW);
		bool lx4 = lh0 && (iw0 >= -4) && (iw0 + 4 < IW);
		bool lx5 = lh0 && (iw0 >= -5) && (iw0 + 5 < IW);

		for (int ic = 0; ic < IC; ic++)
		{
			//load 1 group from W
			const int woffset = (fh * 3 * IC + ic)*OC;//[fh, fw, ic, oc]
			float w0 = CW[woffset];
			float w1 = CW[woffset + Wstride];
			float w2 = CW[woffset + (Wstride << 1)];
				
			//load 1 group from X
			const int xoffset = ((n0*IH + (ih0 + fh))*IW + iw0)*IC + ic;
			float x0 = (lx0 ? X[xoffset] : 0);
			float x1 = (lx1 ? X[xoffset + IC] : 0);
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);
			float x5 = (lx5 ? X[xoffset + (IC * 5)] : 0);

			float g0 = w0;
			float g1 = w0 + w1 + w2;
			float g2 = w0 - w1 + w2;
			float g3 = w0 + 2 * w1 + 4 * w2;
			float g4 = w0 - 2 * w1 + 4 * w2;
			float g5 = w2;

			float d0 = 24 * x0 - 30 * x2 + 6 * x4;
			float d1 = 16 * x1 + 16 * x2 - 4 * x3 - 4 * x4;
			float d2 = -16 * x1 + 16 * x2 + 4 * x3 - 4 * x4;
			float d3 = -2 * x1 - x2 + 2 * x3 + x4;
			float d4 = 2 * x1 - x2 - 2 * x3 + x4;
			float d5 = 96 * x1 - 120 * x3 + 24 * x5;

			m0 += g0 * d0;
			m1 += g1 * d1;
			m2 += g2 * d2;
			m3 += g3 * d3;
			m4 += g4 * d4;
			m5 += g5 * d5;
		}
	}

	float v0 = (m0 + m1 + m2 + m3 + m4) / 24.0f;
	float v1 = (m1 - m2 + 2 * m3 - 2 * m4) / 24.0f;
	float v2 = (m1 + m2 + 4 * m3 + 4 * m4) / 24.0f;
	float v3 = (m1 - m2 + 8 * m3 - 8 * m4 + m5) / 24.0f;

	get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC) = v0;
	get4d(Y, n0, oh0, ow1, oc0, OH, OW, OC) = v1;
	get4d(Y, n0, oh0, ow2, oc0, OH, OW, OC) = v2;
	get4d(Y, n0, oh0, ow3, oc0, OH, OW, OC) = v3;

}

#endif


//(Y: BLOCK_SIZE*1, X:BLOCK_SIZE*4)
#ifndef XENO1_WINOGRAD_F4X3_V3
#define XENO1_WINOGRAD_F4X3_V3

#define xeno_winograd_f4x3_k3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v3<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 7.3387 msec, Performace = 2633.62 GFlop/s
template<int LB, int STEP>
__global__ void xeno_winograd_f4x3_kernel_v3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Gs0[1 << LB][(1 << LB) + 1];
	__shared__ float2 Gs1[1 << LB][(1 << LB) + 1];
	__shared__ float4 Ds0[1 << LB][(1 << LB) + 1];
	__shared__ float2 Ds1[1 << LB][(1 << LB) + 1];

	float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f;
	float m3 = 0.0f, m4 = 0.0f, m5 = 0.0f;

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	CW += oc0;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j0, n0, oh0, ow0);
	const int ow1 = ow0 + 1, ow2 = ow0 + 2, ow3 = ow0 + 3;
	const int ih0 = oh0 - ph, iw0 = ow0 - pw;

	//compute area------------------------------------------------
	const int Wstride = IC * OC;
#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (ih0 >= -fh) && (ih0 + fh < IH);
		bool lx0 = lh0 && (iw0 >= 0) && (iw0 < IW);
		bool lx1 = lh0 && (iw0 >= -1) && (iw0 + 1 < IW);
		bool lx2 = lh0 && (iw0 >= -2) && (iw0 + 2 < IW);
		bool lx3 = lh0 && (iw0 >= -3) && (iw0 + 3 < IW);
		bool lx4 = lh0 && (iw0 >= -4) && (iw0 + 4 < IW);
		bool lx5 = lh0 && (iw0 >= -5) && (iw0 + 5 < IW);

		for (int oic = 0; oic < IC; oic += STEP)
		{
			const int xic = oic + ty;//with the same tx
			const int xoffset = ((n0*IH + (ih0 + fh))*IW + iw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);
			float x1 = (lx1 ? X[xoffset + IC] : 0);
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);
			float x5 = (lx5 ? X[xoffset + (IC * 5)] : 0);
			float d0 = 24 * x0 - 30 * x2 + 6 * x4;
			float d1 = 16 * x1 + 16 * x2 - 4 * x3 - 4 * x4;
			float d2 = -16 * x1 + 16 * x2 + 4 * x3 - 4 * x4;
			float d3 = -2 * x1 - x2 + 2 * x3 + x4;
			float d4 = 2 * x1 - x2 - 2 * x3 + x4;
			float d5 = 96 * x1 - 120 * x3 + 24 * x5;
			Ds0[ty][tx] = float4{ d0, d1, d2, d3 }; 
			Ds1[ty][tx] = float2{ d4, d5 };
		
			const int wic = oic + tx;//with the same ty
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float w0 = CW[woffset];
			float w1 = CW[woffset + Wstride];
			float w2 = CW[woffset + (Wstride << 1)];
			float g0 = w0;
			float g1 = w0 + w1 + w2;
			float g2 = w0 - w1 + w2;
			float g3 = w0 + 2 * w1 + 4 * w2;
			float g4 = w0 - 2 * w1 + 4 * w2;
			float g5 = w2;
			Gs0[tx][ty] = float4{ g0, g1, g2, g3 };
			Gs1[tx][ty] = float2{ g4, g5 };
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds0[ik][tx]; float2 d1 = Ds1[ik][tx];
				float4 g0 = Gs0[ik][ty]; float2 g1 = Gs1[ik][ty];

				m0 += g0.x * d0.x;
				m1 += g0.y * d0.y;
				m2 += g0.z * d0.z;
				m3 += g0.w * d0.w;
				m4 += g1.x * d1.x;
				m5 += g1.y * d1.y;
			}
			__syncthreads();
		}
	}

	float v0 = (m0 + m1 + m2 + m3 + m4) / 24.0f;
	float v1 = (m1 - m2 + 2 * m3 - 2 * m4) / 24.0f;
	float v2 = (m1 + m2 + 4 * m3 + 4 * m4) / 24.0f;
	float v3 = (m1 - m2 + 8 * m3 - 8 * m4 + m5) / 24.0f;

	get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC) = v0;
	get4d(Y, n0, oh0, ow1, oc0, OH, OW, OC) = v1;
	get4d(Y, n0, oh0, ow2, oc0, OH, OW, OC) = v2;
	get4d(Y, n0, oh0, ow3, oc0, OH, OW, OC) = v3;

}

#endif


//(Y: BLOCK_SIZE*1, X:BLOCK_SIZE*4)
#ifndef XENO1_WINOGRAD_F4X3_V4
#define XENO1_WINOGRAD_F4X3_V4

#define xeno_winograd_f4x3_k4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v4<LB, (1<<LB)>\
		<<< dim3(GN>>LB, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 7.21622 msec, Performace = 2678.32 GFlop/s
template<int LB, int STEP>
__global__ void xeno_winograd_f4x3_kernel_v4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	__shared__ float4 Gs0[1 << LB][(1 << LB) + 1];
	__shared__ float2 Gs1[1 << LB][(1 << LB) + 1];
	__shared__ float4 Ds0[1 << LB][(1 << LB) + 1];
	__shared__ float2 Ds1[1 << LB][(1 << LB) + 1];

	float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f;
	float m3 = 0.0f, m4 = 0.0f, m5 = 0.0f;

	//prepare for GN = OC
	const int oc0 = ((blockIdx.x << LB) + tx) + oc_index;
	CW += oc0;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j0, n0, oh0, ow0);
	const int ow1 = ow0 + 1, ow2 = ow0 + 2, ow3 = ow0 + 3;
	const int ih0 = oh0 - ph, iw0 = ow0 - pw;
	j0 = j0 * OC + oc0;

	//compute area------------------------------------------------
	const int Wstride = IC * OC;
#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (ih0 >= -fh) && (ih0 + fh < IH);
		bool lx0 = lh0 && (iw0 >= 0) && (iw0 < IW);
		bool lx1 = lh0 && (iw0 >= -1) && (iw0 + 1 < IW);
		bool lx2 = lh0 && (iw0 >= -2) && (iw0 + 2 < IW);
		bool lx3 = lh0 && (iw0 >= -3) && (iw0 + 3 < IW);
		bool lx4 = lh0 && (iw0 >= -4) && (iw0 + 4 < IW);
		bool lx5 = lh0 && (iw0 >= -5) && (iw0 + 5 < IW);

		for (int oic = 0; oic < IC; oic += STEP)
		{
			const int xic = oic + tx;//with the same ty
			const int xoffset = ((n0*IH + (ih0 + fh))*IW + iw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);
			float x1 = (lx1 ? X[xoffset + IC] : 0);
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);
			float x5 = (lx5 ? X[xoffset + (IC * 5)] : 0);
			float d0 = 24 * x0 - 30 * x2 + 6 * x4;
			float d1 = 16 * x1 + 16 * x2 - 4 * x3 - 4 * x4;
			float d2 = -16 * x1 + 16 * x2 + 4 * x3 - 4 * x4;
			float d3 = -2 * x1 - x2 + 2 * x3 + x4;
			float d4 = 2 * x1 - x2 - 2 * x3 + x4;
			float d5 = 96 * x1 - 120 * x3 + 24 * x5;
			Ds0[tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[tx][ty] = float2{ d4, d5 };

			const int wic = oic + ty;//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float w0 = CW[woffset];
			float w1 = CW[woffset + Wstride];
			float w2 = CW[woffset + (Wstride << 1)];
			float g0 = w0;
			float g1 = w0 + w1 + w2;
			float g2 = w0 - w1 + w2;
			float g3 = w0 + 2 * w1 + 4 * w2;
			float g4 = w0 - 2 * w1 + 4 * w2;
			float g5 = w2;
			Gs0[ty][tx] = float4{ g0, g1, g2, g3 };
			Gs1[ty][tx] = float2{ g4, g5 };
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds0[ik][ty]; float2 d1 = Ds1[ik][ty];
				float4 g0 = Gs0[ik][tx]; float2 g1 = Gs1[ik][tx];

				m0 += g0.x * d0.x;
				m1 += g0.y * d0.y;
				m2 += g0.z * d0.z;
				m3 += g0.w * d0.w;
				m4 += g1.x * d1.x;
				m5 += g1.y * d1.y;
			}
			__syncthreads();
		}
	}

	float v0 = 0.0416667f * (m0 + m1 + m2 + m3 + m4);
	float v1 = 0.0416667f * (m1 - m2 + 2 * m3 - 2 * m4);
	float v2 = 0.0416667f * (m1 + m2 + 4 * m3 + 4 * m4);
	float v3 = 0.0416667f * (m1 - m2 + 8 * m3 - 8 * m4 + m5);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	Y[j0] = v0;
	Y[j1] = v1;
	Y[j2] = v2;
	Y[j3] = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*4)
#ifndef XENO1_WINOGRAD_F4X3_V5
#define XENO1_WINOGRAD_F4X3_V5

#define xeno_winograd_f4x3_k5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v5<LB, (1<<LB)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 4.8993 msec, Performace = 3944.92 GFlop/s
template<int LB, int STEP>
__global__ void xeno_winograd_f4x3_kernel_v5(
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

	float2 m0 = F32_2_0, m1 = F32_2_0, m2 = F32_2_0;
	float2 m3 = F32_2_0, m4 = F32_2_0, m5 = F32_2_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 1) + oc_index;
	CW += oc0;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j0, n0, oh0, ow0);
	const int ow1 = ow0 + 1, ow2 = ow0 + 2, ow3 = ow0 + 3;
	const int ih0 = oh0 - ph, iw0 = ow0 - pw;
	j0 = j0 * OC + oc0;

	//compute area------------------------------------------------
	const int Wstride = IC * OC;
#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (ih0 >= -fh) && (ih0 + fh < IH);
		bool lx0 = lh0 && (iw0 >= 0) && (iw0 < IW);
		bool lx1 = lh0 && (iw0 >= -1) && (iw0 + 1 < IW);
		bool lx2 = lh0 && (iw0 >= -2) && (iw0 + 2 < IW);
		bool lx3 = lh0 && (iw0 >= -3) && (iw0 + 3 < IW);
		bool lx4 = lh0 && (iw0 >= -4) && (iw0 + 4 < IW);
		bool lx5 = lh0 && (iw0 >= -5) && (iw0 + 5 < IW);

		for (int oic = 0; oic < IC; oic += STEP)
		{
			const int xic = oic + tx;//with the same ty
			const int xoffset = ((n0*IH + (ih0 + fh))*IW + iw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);
			float x1 = (lx1 ? X[xoffset + IC] : 0);
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);
			float x5 = (lx5 ? X[xoffset + (IC * 5)] : 0);
			float d0 = 24 * x0 - 30 * x2 + 6 * x4;
			float d1 = 16 * x1 + 16 * x2 - 4 * x3 - 4 * x4;
			float d2 = -16 * x1 + 16 * x2 + 4 * x3 - 4 * x4;
			float d3 = -2 * x1 - x2 + 2 * x3 + x4;
			float d4 = 2 * x1 - x2 - 2 * x3 + x4;
			float d5 = 96 * x1 - 120 * x3 + 24 * x5;
			Ds0[tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[tx][ty] = float2{ d4, d5 };

			const int wic = oic + ty;//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float2 g0 = { w0.x, w0.y };
			float2 g1 = { w0.x + w1.x + w2.x, w0.y + w1.y + w2.y };
			float2 g2 = { w0.x - w1.x + w2.x, w0.y - w1.y + w2.y };
			float2 g3 = { w0.x + 2 * w1.x + 4 * w2.x, w0.y + 2 * w1.y + 4 * w2.y };
			float2 g4 = { w0.x - 2 * w1.x + 4 * w2.x, w0.y - 2 * w1.y + 4 * w2.y };
			float2 g5 = { w2.x, w2.y };
			Gs0[ty][(tx << 1)    ] = float4{ g0.x, g1.x, g2.x, g3.x };
			Gs0[ty][(tx << 1) + 1] = float4{ g0.y, g1.y, g2.y, g3.y };
			Gs1[ty][(tx << 1)    ] = float2{ g4.x, g5.x };
			Gs1[ty][(tx << 1) + 1] = float2{ g4.y, g5.y };
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds0[ik][ty]; float2 d1 = Ds1[ik][ty];
				float4 g0 = Gs0[ik][(tx << 1)    ]; float2 g1 = Gs1[ik][(tx << 1)    ];
				float4 g2 = Gs0[ik][(tx << 1) + 1]; float2 g3 = Gs1[ik][(tx << 1) + 1];

				m0.x += g0.x * d0.x;
				m1.x += g0.y * d0.y;
				m2.x += g0.z * d0.z;
				m3.x += g0.w * d0.w;
				m4.x += g1.x * d1.x;
				m5.x += g1.y * d1.y;

				m0.y += g2.x * d0.x;
				m1.y += g2.y * d0.y;
				m2.y += g2.z * d0.z;
				m3.y += g2.w * d0.w;
				m4.y += g3.x * d1.x;
				m5.y += g3.y * d1.y;
			}
			__syncthreads();
		}
	}

	float2 v0, v1, v2, v3;
	v0.x = 0.0416667f * (m0.x + m1.x + m2.x + m3.x + m4.x);
	v0.y = 0.0416667f * (m0.y + m1.y + m2.y + m3.y + m4.y);
	v1.x = 0.0416667f * (m1.x - m2.x + 2.0f * m3.x - 2.0f * m4.x);
	v1.y = 0.0416667f * (m1.y - m2.y + 2.0f * m3.y - 2.0f * m4.y);
	v2.x = 0.0416667f * (m1.x + m2.x + 4.0f * m3.x + 4.0f * m4.x);
	v2.y = 0.0416667f * (m1.y + m2.y + 4.0f * m3.y + 4.0f * m4.y);
	v3.x = 0.0416667f * (m1.x - m2.x + 8.0f * m3.x - 8.0f * m4.x + m5.x);
	v3.y = 0.0416667f * (m1.y - m2.y + 8.0f * m3.y - 8.0f * m4.y + m5.y);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
	*(float2*)(Y + j2) = v2;
	*(float2*)(Y + j3) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*4)
#ifndef XENO1_WINOGRAD_F4X3_V6
#define XENO1_WINOGRAD_F4X3_V6

#define xeno_winograd_f4x3_k6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v6<LB, (1<<LB), (2<<LB)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 4.8993 msec, Performace = 3944.92 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void xeno_winograd_f4x3_kernel_v6(
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

	float2 m0 = F32_2_0, m1 = F32_2_0, m2 = F32_2_0;
	float2 m3 = F32_2_0, m4 = F32_2_0, m5 = F32_2_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 1) + oc_index;
	CW += oc0;
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j0, n0, oh0, ow0);
	const int ow1 = ow0 + 1, ow2 = ow0 + 2, ow3 = ow0 + 3;
	const int ih0 = oh0 - ph, iw0 = ow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (ih0 >= -fh) && (ih0 + fh < IH);
		bool lx0 = lh0 && (iw0 >= 0) && (iw0 < IW);
		bool lx1 = lh0 && (iw0 >= -1) && (iw0 + 1 < IW);
		bool lx2 = lh0 && (iw0 >= -2) && (iw0 + 2 < IW);
		bool lx3 = lh0 && (iw0 >= -3) && (iw0 + 3 < IW);
		bool lx4 = lh0 && (iw0 >= -4) && (iw0 + 4 < IW);
		bool lx5 = lh0 && (iw0 >= -5) && (iw0 + 5 < IW);

		for (int oic = 0; oic < IC; oic += STEP)
		{
			const int xic = oic + tx;//with the same ty
			const int xoffset = ((n0*IH + (ih0 + fh))*IW + iw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset] : 0);
			float x1 = (lx1 ? X[xoffset + IC] : 0);
			float x2 = (lx2 ? X[xoffset + (IC << 1)] : 0);
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);
			float x4 = (lx4 ? X[xoffset + (IC << 2)] : 0);
			float x5 = (lx5 ? X[xoffset + (IC * 5)] : 0);
			float d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0, d1, d2, d3, d4, d5, x0, x1, x2, x3, x4, x5);
			Ds0[tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[tx][ty] = float2{ d4, d5 };

			const int wic = oic + ty;//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float2 g0, g1, g2, g3, g4, g5;
			winograd_f4x3_g(g0.x, g1.x, g2.x, g3.x, g4.x, g5.x, w0.x, w1.x, w2.x);
			winograd_f4x3_g(g0.y, g1.y, g2.y, g3.y, g4.y, g5.y, w0.y, w1.y, w2.y);
			Gs0[ty][tx       ] = float4{ g0.x, g1.x, g2.x, g3.x };
			Gs0[ty][tx + STEP] = float4{ g0.y, g1.y, g2.y, g3.y };
			Gs1[ty][tx       ] = float2{ g4.x, g5.x };
			Gs1[ty][tx + STEP] = float2{ g4.y, g5.y };
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds0[ik][ty]; float2 d1 = Ds1[ik][ty];
				float4 g0 = Gs0[ik][tx       ]; float2 g1 = Gs1[ik][tx        ];
				float4 g2 = Gs0[ik][tx + STEP]; float2 g3 = Gs1[ik][tx +  STEP];

				m0.x += g0.x * d0.x; m0.y += g2.x * d0.x;
				m1.x += g0.y * d0.y; m1.y += g2.y * d0.y;
				m2.x += g0.z * d0.z; m2.y += g2.z * d0.z;
				m3.x += g0.w * d0.w; m3.y += g2.w * d0.w;
				m4.x += g1.x * d1.x; m4.y += g3.x * d1.x;
				m5.x += g1.y * d1.y; m5.y += g3.y * d1.y;
			}
			__syncthreads();
		}
	}

	float2 v0, v1, v2, v3;
	winograd_f4x3_v(v0.x, v1.x, v2.x, v3.x, m0.x, m1.x, m2.x, m3.x, m4.x, m5.x);
	winograd_f4x3_v(v0.y, v1.y, v2.y, v3.y, m0.y, m1.y, m2.y, m3.y, m4.y, m5.y);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
	*(float2*)(Y + j2) = v2;
	*(float2*)(Y + j3) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_V7
#define XENO1_WINOGRAD_F4X3_V7

#define xeno_winograd_f4x3_k7(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v7<LB, (1<<LB)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 3.50883 msec, Performace = 5508.21 GFlop/s
template<int LB, int STEP>
__global__ void xeno_winograd_f4x3_kernel_v7(
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
	__shared__ float4 Ds0[1 << LB][(2 << LB) + 1];
	__shared__ float2 Ds1[1 << LB][(2 << LB) + 1];

	float2 m0 = F32_2_0, m1 = F32_2_0, m2 = F32_2_0;//0-3
	float2 m3 = F32_2_0, m4 = F32_2_0, m5 = F32_2_0;
	float2 m6 = F32_2_0, m7 = F32_2_0, m8 = F32_2_0;//4-8
	float2 m9 = F32_2_0, ma = F32_2_0, mb = F32_2_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 1) + oc_index;
	CW += oc0;
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j0, n0, oh0, ow0);
	const int ow1 = ow0 + 1, ow2 = ow0 + 2, ow3 = ow0 + 3;
	const int ow4 = ow0 + 4, ow5 = ow0 + 5, ow6 = ow0 + 6, ow7 = ow0 + 7;
	const int ih0 = oh0 - ph, iw0 = ow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (ih0 >= -fh) && (ih0 + fh < IH);
		bool lx0 = lh0 && (iw0 >= 0) && (iw0 < IW);
		bool lx1 = lh0 && (iw0 >= -1) && (iw0 + 1 < IW);
		bool lx2 = lh0 && (iw0 >= -2) && (iw0 + 2 < IW);
		bool lx3 = lh0 && (iw0 >= -3) && (iw0 + 3 < IW);
		bool lx4 = lh0 && (iw0 >= -4) && (iw0 + 4 < IW);
		bool lx5 = lh0 && (iw0 >= -5) && (iw0 + 5 < IW);
		bool lx6 = lh0 && (iw0 >= -6) && (iw0 + 6 < IW);
		bool lx7 = lh0 && (iw0 >= -7) && (iw0 + 7 < IW);
		bool lx8 = lh0 && (iw0 >= -8) && (iw0 + 8 < IW);
		bool lx9 = lh0 && (iw0 >= -9) && (iw0 + 9 < IW);

		for (int oic = 0; oic < IC; oic += STEP)
		{
			//load 2 group from X
			const int xic = oic + tx;//with the same ty
			const int xoffset = ((n0*IH + (ih0 + fh))*IW + iw0)*IC + xic;
			float x0 = (lx0 ? X[xoffset         ] : 0);//0
			float x1 = (lx1 ? X[xoffset +     IC] : 0);//1
			float x2 = (lx2 ? X[xoffset + IC * 2] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + IC * 4] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5

			float x6 = (lx6 ? X[xoffset + IC * 6] : 0);//6
			float x7 = (lx7 ? X[xoffset + IC * 7] : 0);//7
			float x8 = (lx8 ? X[xoffset + IC * 8] : 0);//8
			float x9 = (lx9 ? X[xoffset + IC * 9] : 0);
			float2 d0, d1, d2, d3, d4, d5;
			winograd_f4x3_d(d0.x, d1.x, d2.x, d3.x, d4.x, d5.x, x0, x1, x2, x3, x4, x5);
			Ds0[tx][ty] = float4{ d0.x, d1.x, d2.x, d3.x };
			Ds1[tx][ty] = float2{ d4.x, d5.x };
			winograd_f4x3_d(d0.y, d1.y, d2.y, d3.y, d4.y, d5.y, x4, x5, x6, x7, x8, x9);
			Ds0[tx][ty + STEP] = float4{ d0.y, d1.y, d2.y, d3.y };
			Ds1[tx][ty + STEP] = float2{ d4.y, d5.y };

			//load 2 group from W
			const int wic = oic + ty;//with the same tx
			const int woffset = (fh * 3 * IC + wic)*OC;//[fh, fw, ic, oc]
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float2 g0, g1, g2, g3, g4, g5;
			winograd_f4x3_g(g0.x, g1.x, g2.x, g3.x, g4.x, g5.x, w0.x, w1.x, w2.x);
			Gs0[ty][tx] = float4{ g0.x, g1.x, g2.x, g3.x };
			Gs1[ty][tx] = float2{ g4.x, g5.x };
			winograd_f4x3_g(g0.y, g1.y, g2.y, g3.y, g4.y, g5.y, w0.y, w1.y, w2.y);
			Gs0[ty][tx + STEP] = float4{ g0.y, g1.y, g2.y, g3.y };
			Gs1[ty][tx + STEP] = float2{ g4.y, g5.y };
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds0[ik][ty]; float2 d1 = Ds1[ik][ty];
				float4 d2 = Ds0[ik][ty + STEP]; float2 d3 = Ds1[ik][ty + STEP];
				
				float4 g0 = Gs0[ik][tx]; float2 g1 = Gs1[ik][tx];
				float4 g2 = Gs0[ik][tx + STEP]; float2 g3 = Gs1[ik][tx + STEP];

				m0.x += g0.x * d0.x; m0.y += g2.x * d0.x;
				m1.x += g0.y * d0.y; m1.y += g2.y * d0.y;
				m2.x += g0.z * d0.z; m2.y += g2.z * d0.z;
				m3.x += g0.w * d0.w; m3.y += g2.w * d0.w;
				m4.x += g1.x * d1.x; m4.y += g3.x * d1.x;
				m5.x += g1.y * d1.y; m5.y += g3.y * d1.y;

				m6.x += g0.x * d2.x; m6.y += g2.x * d2.x;
				m7.x += g0.y * d2.y; m7.y += g2.y * d2.y;
				m8.x += g0.z * d2.z; m8.y += g2.z * d2.z;
				m9.x += g0.w * d2.w; m9.y += g2.w * d2.w;
				ma.x += g1.x * d3.x; ma.y += g3.x * d3.x;
				mb.x += g1.y * d3.y; mb.y += g3.y * d3.y;
			}
			__syncthreads();
		}
	}

	float2 v0, v1, v2, v3;
	winograd_f4x3_v(v0.x, v1.x, v2.x, v3.x, m0.x, m1.x, m2.x, m3.x, m4.x, m5.x);
	winograd_f4x3_v(v0.y, v1.y, v2.y, v3.y, m0.y, m1.y, m2.y, m3.y, m4.y, m5.y);
	
	float2 v4, v5, v6, v7;
	winograd_f4x3_v(v4.x, v5.x, v6.x, v7.x, m6.x, m7.x, m8.x, m9.x, ma.x, mb.x);
	winograd_f4x3_v(v4.y, v5.y, v6.y, v7.y, m6.y, m7.y, m8.y, m9.y, ma.y, mb.y);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
	*(float2*)(Y + j2) = v2;
	*(float2*)(Y + j3) = v3;
	*(float2*)(Y + j4) = v4;
	*(float2*)(Y + j5) = v5;
	*(float2*)(Y + j6) = v6;
	*(float2*)(Y + j7) = v7;
}

#endif


//half shared memory
//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_V8
#define XENO1_WINOGRAD_F4X3_V8

#define xeno_winograd_f4x3_k8(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v8<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>1, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 3.46436 msec, Performace = 5578.91 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_v8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.y, ty = threadIdx.x;

	__shared__ float4 Gs0[1 << LB][(1 << LB) + 1];
	__shared__ float2 Gs1[1 << LB][(1 << LB) + 1];
	__shared__ float4 Ds0[1 << LB][(1 << LB) + 1];
	__shared__ float2 Ds1[1 << LB][(1 << LB) + 1];

	float2 m0 = F32_2_0, m1 = F32_2_0, m2 = F32_2_0;//0-3
	float2 m3 = F32_2_0, m4 = F32_2_0, m5 = F32_2_0;
	float2 m6 = F32_2_0, m7 = F32_2_0, m8 = F32_2_0;//4-8
	float2 m9 = F32_2_0, ma = F32_2_0, mb = F32_2_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 1) + oc_index;
	CW += oc0 + ((ty >= STEP));//CW[0, 0, 0, toc0]
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
		bool lx0 = lh0 && (tiw0 >=  0) && (tiw0 < IW);
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
				float4 d0 = Ds0[ik       ][ty]; float2 d1 = Ds1[ik       ][ty];
				float4 d2 = Ds0[ik + STEP][ty]; float2 d3 = Ds1[ik + STEP][ty];

				float4 g0 = Gs0[ik       ][tx]; float2 g1 = Gs1[ik       ][tx];
				float4 g2 = Gs0[ik + STEP][tx]; float2 g3 = Gs1[ik + STEP][tx];

				m0.x += g0.x * d0.x; m0.y += g2.x * d0.x;
				m1.x += g0.y * d0.y; m1.y += g2.y * d0.y;
				m2.x += g0.z * d0.z; m2.y += g2.z * d0.z;
				m3.x += g0.w * d0.w; m3.y += g2.w * d0.w;
				m4.x += g1.x * d1.x; m4.y += g3.x * d1.x;
				m5.x += g1.y * d1.y; m5.y += g3.y * d1.y;

				m6.x += g0.x * d2.x; m6.y += g2.x * d2.x;
				m7.x += g0.y * d2.y; m7.y += g2.y * d2.y;
				m8.x += g0.z * d2.z; m8.y += g2.z * d2.z;
				m9.x += g0.w * d2.w; m9.y += g2.w * d2.w;
				ma.x += g1.x * d3.x; ma.y += g3.x * d3.x;
				mb.x += g1.y * d3.y; mb.y += g3.y * d3.y;
			}
			__syncthreads();
		}
	}

	float2 v0, v1, v2, v3;
	winograd_f4x3_v(v0.x, v1.x, v2.x, v3.x, m0.x, m1.x, m2.x, m3.x, m4.x, m5.x);
	winograd_f4x3_v(v0.y, v1.y, v2.y, v3.y, m0.y, m1.y, m2.y, m3.y, m4.y, m5.y);

	float2 v4, v5, v6, v7;
	winograd_f4x3_v(v4.x, v5.x, v6.x, v7.x, m6.x, m7.x, m8.x, m9.x, ma.x, mb.x);
	winograd_f4x3_v(v4.y, v5.y, v6.y, v7.y, m6.y, m7.y, m8.y, m9.y, ma.y, mb.y);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
	*(float2*)(Y + j2) = v2;
	*(float2*)(Y + j3) = v3;
	*(float2*)(Y + j4) = v4;
	*(float2*)(Y + j5) = v5;
	*(float2*)(Y + j6) = v6;
	*(float2*)(Y + j7) = v7;
}

#endif


//half shared memory
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_V9
#define XENO1_WINOGRAD_F4X3_V9

#define xeno_winograd_f4x3_k9(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v9<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 2.37849 msec, Performace = 8125.91 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_v9(
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
#ifndef XENO1_WINOGRAD_F4X3_V10
#define XENO1_WINOGRAD_F4X3_V10

#define xeno_winograd_f4x3_k10(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v10<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 1.94215 msec, Performace = 9951.52 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_v10(
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
	__shared__ float4 Ds0[1 << LB][(2 << LB) + 1];
	__shared__ float2 Ds1[1 << LB][(2 << LB) + 1];

	float4  m0 = F32_4_0,  m1 = F32_4_0,  m2 = F32_4_0;//0-3
	float4  m3 = F32_4_0,  m4 = F32_4_0,  m5 = F32_4_0;

	float4  m6 = F32_4_0,  m7 = F32_4_0,  m8 = F32_4_0;//4-8
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
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma unroll//compute area------------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool lx0 = lh0 && (tiw0 >=  0) && (tiw0 < IW);
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
			float x2 = (lx2 ? X[xoffset + IC * 2] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + IC * 4] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float x6 = (lx6 ? X[xoffset + IC * 6] : 0);//6
			float x7 = (lx7 ? X[xoffset + IC * 7] : 0);//7
			float x8 = (lx8 ? X[xoffset + IC * 8] : 0);//8
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

				m0.x  += g0.x * d0.x; m0.y  += g2.x * d0.x; m0.z  += g4.x * d0.x; m0.w  += g6.x * d0.x;
				m1.x  += g0.y * d0.y; m1.y  += g2.y * d0.y; m1.z  += g4.y * d0.y; m1.w  += g6.y * d0.y;
				m2.x  += g0.z * d0.z; m2.y  += g2.z * d0.z; m2.z  += g4.z * d0.z; m2.w  += g6.z * d0.z;
				m3.x  += g0.w * d0.w; m3.y  += g2.w * d0.w; m3.z  += g4.w * d0.w; m3.w  += g6.w * d0.w;
				m4.x  += g1.x * d1.x; m4.y  += g3.x * d1.x; m4.z  += g5.x * d1.x; m4.w  += g7.x * d1.x;
				m5.x  += g1.y * d1.y; m5.y  += g3.y * d1.y; m5.z  += g5.y * d1.y; m5.w  += g7.y * d1.y;

				m6.x  += g0.x * d2.x; m6.y  += g2.x * d2.x; m6.z  += g4.x * d2.x; m6.w  += g6.x * d2.x;
				m7.x  += g0.y * d2.y; m7.y  += g2.y * d2.y; m7.z  += g4.y * d2.y; m7.w  += g6.y * d2.y;
				m8.x  += g0.z * d2.z; m8.y  += g2.z * d2.z; m8.z  += g4.z * d2.z; m8.w  += g6.z * d2.z;
				m9.x  += g0.w * d2.w; m9.y  += g2.w * d2.w; m9.z  += g4.w * d2.w; m9.w  += g6.w * d2.w;
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
	winograd_f4x3_v( v8.x,  v9.x, v10.x, v11.x, m12.x, m13.x, m14.x, m15.x, m16.x, m17.x);
	winograd_f4x3_v( v8.y,  v9.y, v10.y, v11.y, m12.y, m13.y, m14.y, m15.y, m16.y, m17.y);
	winograd_f4x3_v( v8.z,  v9.z, v10.z, v11.z, m12.z, m13.z, m14.z, m15.z, m16.z, m17.z);
	winograd_f4x3_v( v8.w,  v9.w, v10.w, v11.w, m12.w, m13.w, m14.w, m15.w, m16.w, m17.w);

	float4 v12, v13, v14, v15;
	winograd_f4x3_v(v12.x, v13.x, v14.x, v15.x, m18.x, m19.x, m20.x, m21.x, m22.x, m23.x);
	winograd_f4x3_v(v12.y, v13.y, v14.y, v15.y, m18.y, m19.y, m20.y, m21.y, m22.y, m23.y);
	winograd_f4x3_v(v12.z, v13.z, v14.z, v15.z, m18.z, m19.z, m20.z, m21.z, m22.z, m23.z);
	winograd_f4x3_v(v12.w, v13.w, v14.w, v15.w, m18.w, m19.w, m20.w, m21.w, m22.w, m23.w);

	const int  j1 =  j0 + OC,  j2 =  j1 + OC,  j3 =  j2 + OC;
	const int  j4 =  j3 + OC,  j5 =  j4 + OC,  j6 =  j5 + OC,  j7 = j6 + OC;
	const int  j8 =  j7 + OC,  j9 =  j8 + OC,  j10 = j9 + OC, j11 = j10 + OC;
	const int j12 = j11 + OC, j13 = j12 + OC, j14 = j13 + OC, j15 = j14 + OC;

	*(float4*)(Y +  j0) =  v0; *(float4*)(Y +  j1) =  v1;
	*(float4*)(Y +  j2) =  v2; *(float4*)(Y +  j3) =  v3;

	*(float4*)(Y +  j4) =  v4; *(float4*)(Y +  j5) =  v5;
	*(float4*)(Y +  j6) =  v6; *(float4*)(Y +  j7) =  v7;

	*(float4*)(Y +  j8) =  v8; *(float4*)(Y +  j9) =  v9;
	*(float4*)(Y + j10) = v10; *(float4*)(Y + j11) = v11;

	*(float4*)(Y + j12) = v12; *(float4*)(Y + j13) = v13;
	*(float4*)(Y + j14) = v14; *(float4*)(Y + j15) = v15;
}

#endif


//half shared memory
//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8)
#ifndef XENO1_WINOGRAD_F4X3_V11
#define XENO1_WINOGRAD_F4X3_V11

#define xeno_winograd_f4x3_k11(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f4x3_kernel_v11<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>2, GM>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//(OH, OW) % 8 == 0
//Size = 9, Time = 1.94215 msec, Performace = 9951.52 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void xeno_winograd_f4x3_kernel_v11(
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
			float x2 = (lx2 ? X[xoffset + IC * 2] : 0);//2
			float x3 = (lx3 ? X[xoffset + IC * 3] : 0);//3
			float x4 = (lx4 ? X[xoffset + IC * 4] : 0);//4
			float x5 = (lx5 ? X[xoffset + IC * 5] : 0);//5
			float x6 = (lx6 ? X[xoffset + IC * 6] : 0);//6
			float x7 = (lx7 ? X[xoffset + IC * 7] : 0);//7
			float x8 = (lx8 ? X[xoffset + IC * 8] : 0);//8
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


