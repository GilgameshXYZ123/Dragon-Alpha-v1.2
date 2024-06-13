
//winograd9 -> A1
#define C3D_WG2D_KERNEL_A1
#ifndef C3D_WG2D_KERNEL_A1
#define C3D_WG2D_KERNEL_A1

#define c3d_wg2d_a1(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel_A1\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.53425 msec, Performace = 12597.3 GFlop/s
__global__ void c3d_wg2d_kernel_A1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8][16][36];
	__shared__ float Xs[8][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int oc = boc + tx;//[0 - 31]
	G += oc << 4;//G[0, oc, 0, 0]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int xn, ih, iw; {
		xn = xj / THW; int jr = xj - xn * THW;
		int th = jr / TW, tw = jr - th * TW;
		ih = (th << 1) - 1; iw = (tw << 1) - 1;
	}

	//[ic, n, ih, iw]
	const int xoffset = (xn*IH + ih)*IW + iw;
	for (int oic = 0; oic < IC; oic += 8)
	{
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float x[16]; {//load 4 * 4 elements
			const int ly0 = (ih >= 0) && (ih < IH);
			const int ly3 = (ih >= -3) && (ih < IH - 3);
			const int lx0 = (iw >= 0) && (iw < IW);
			const int lx3 = (iw >= -3) && (iw < IW - 3);
			// 0,  1,  2,  3
			// 4, ( 5), ( 6),  7
			// 8, ( 9), (10), 11
			//12, 13, 14, 15
			//no need to check xy: 5,  6,  9, 10
			//no need to check  y: 4,  8,  7, 11
			//no need to check  x: 1,  2, 13, 14

			//[ic, n, ih, iw]
			const int X00 = xoffset + ic * N * IH * IW;
			const int X01 = X00 + 1, X02 = X00 + 2, X03 = X00 + 3;
			const int X10 = X00 + IW, X11 = X10 + 1, X12 = X10 + 2, X13 = X10 + 3;
			const int X20 = X10 + IW, X21 = X20 + 1, X22 = X20 + 2, X23 = X20 + 3;
			const int X30 = X20 + IW, X31 = X30 + 1, X32 = X30 + 3, X33 = X30 + 3;

			x[0] = X[X00];
			x[1] = X[X01];
			x[2] = X[X02];
			x[3] = X[X03];

			x[4] = X[X10];
			x[5] = X[X11];
			x[6] = X[X12];
			x[7] = X[X13];

			x[8] = X[X20];
			x[9] = X[X21];
			x[10] = X[X22];
			x[11] = X[X23];

			x[12] = X[X30];
			x[13] = X[X31];
			x[14] = X[X32];
			x[15] = X[X33];

			Winograd2D_f22x33_transform_X(x);

			//global -> regitser: Xs[ty][][tx], [8][16][32]
			//(1) ty -> ic = 0 - 8
			//(2) 1 group: x0 -> x16, 4 * 4 elements
			//(3) tx -> j: (n, oh, ow)
			Xs[ty][0][tx] = x[0]; Xs[ty][1][tx] = x[1]; Xs[ty][2][tx] = x[2]; Xs[ty][3][tx] = x[3];
			Xs[ty][4][tx] = x[4]; Xs[ty][5][tx] = x[5]; Xs[ty][6][tx] = x[6]; Xs[ty][7][tx] = x[7];
			Xs[ty][8][tx] = x[8]; Xs[ty][9][tx] = x[9]; Xs[ty][10][tx] = x[10]; Xs[ty][11][tx] = x[11];
			Xs[ty][12][tx] = x[12]; Xs[ty][13][tx] = x[13]; Xs[ty][14][tx] = x[14]; Xs[ty][15][tx] = x[15];
		}

		//======[load 1 group from G[OC, IC, 4, 4]]==============================
		float g[16]; {//load 4 * 4 elements
			const int goffset = (ic*OC) << 4;//[gic, goc, 0, 0]

			*(float4*)(g) = *(float4*)(G + goffset);
			*(float4*)(g + 4) = *(float4*)(G + goffset + 4);
			*(float4*)(g + 8) = *(float4*)(G + goffset + 8);
			*(float4*)(g + 12) = *(float4*)(G + goffset + 12);

			//register -> shread: Gs[ty][][tx], [8][16][32]
			//(1) ty -> ic = 0 - 8
			//(2) 1 group: g0 - g16
			//(3) tx -> oc = 0 -31
			Gs[ty][0][tx] = g[0]; Gs[ty][1][tx] = g[1]; Gs[ty][2][tx] = g[2]; Gs[ty][3][tx] = g[3];
			Gs[ty][4][tx] = g[4]; Gs[ty][5][tx] = g[5]; Gs[ty][6][tx] = g[6]; Gs[ty][7][tx] = g[7];
			Gs[ty][8][tx] = g[8]; Gs[ty][9][tx] = g[9]; Gs[ty][10][tx] = g[10]; Gs[ty][11][tx] = g[11];
			Gs[ty][12][tx] = g[12]; Gs[ty][13][tx] = g[13]; Gs[ty][14][tx] = g[14]; Gs[ty][15][tx] = g[15];
		}
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
		__syncthreads();
	}

	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	//=================================================================================
	//turn0: x, [ux, uy, elem]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn1: y
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//(elem, uy, ux)
	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y0);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y2);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y3);
	__syncthreads();

	//======[write to Y]===================================================================
	const int yoc = boc + GIdx + (ux & 1) * 4;
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; {
		yn = yj / THW; int jr = yj - yn * THW;
		int th = jr / TW, tw = jr - th * TW;
		oh = (th << 1); ow = (tw << 1);
	}

	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


#define C3D_WG2D_KERNEL_A2
#ifndef C3D_WG2D_KERNEL_A2
#define C3D_WG2D_KERNEL_A2

#define c3d_wg2d_a2(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel_A2\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 2.06779 msec, Performace = 9346.88 GFlop/s
//Size = 9, Time = 1.46301 msec, Performace = 13210.7 GFlop/s
__global__ void c3d_wg2d_kernel_A2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8][16][36];
	__shared__ float Xs[8][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int oc = boc + tx;//[0 - 31]
	G += oc << 4;//G[0, oc, 0, 0]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int xn, ih, iw; {
		xn = xj / THW; int jr = xj - xn * THW;
		int th = jr / TW, tw = jr - th * TW;
		ih = (th << 1); iw = (tw << 1);
	}

	//2, 4

	const int xoffset = (xn*IH + ih)*IW + iw;//[ic, n, ih, iw]
	for (int oic = 0; oic < IC; oic += 8)
	{
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float x[16]; {//load 4 * 4 elements
			//[ic, n, ih, iw]
			const int X00 = xoffset + ic * N * IH * IW;//[ic, n, ih, iw]
			const int X10 = X00 + IW;
			const int X20 = X10 + IW;
			const int X30 = X20 + IW;

			*(float2*)(x) = *(float2*)(X + X00);
			*(float2*)(x + 2) = *(float2*)(X + X00 + 2);
			*(float2*)(x + 4) = *(float2*)(X + X10);
			*(float2*)(x + 6) = *(float2*)(X + X10 + 2);
			*(float2*)(x + 8) = *(float2*)(X + X20);
			*(float2*)(x + 10) = *(float2*)(X + X20 + 2);
			*(float2*)(x + 12) = *(float2*)(X + X30);
			*(float2*)(x + 14) = *(float2*)(X + X30 + 2);

			Winograd2D_f22x33_transform_X(x);

			//global -> regitser: Xs[ty][][tx], [8][16][32]
			//(1) ty -> ic = 0 - 8
			//(2) 1 group: x0 -> x16, 4 * 4 elements
			//(3) tx -> j: (n, oh, ow)
			Xs[ty][0][tx] = x[0]; Xs[ty][1][tx] = x[1]; Xs[ty][2][tx] = x[2]; Xs[ty][3][tx] = x[3];
			Xs[ty][4][tx] = x[4]; Xs[ty][5][tx] = x[5]; Xs[ty][6][tx] = x[6]; Xs[ty][7][tx] = x[7];
			Xs[ty][8][tx] = x[8]; Xs[ty][9][tx] = x[9]; Xs[ty][10][tx] = x[10]; Xs[ty][11][tx] = x[11];
			Xs[ty][12][tx] = x[12]; Xs[ty][13][tx] = x[13]; Xs[ty][14][tx] = x[14]; Xs[ty][15][tx] = x[15];
		}

		//[ic, <n, ih, iw>]
		//[OC, IC, FH, FW] => [FH, FW, IC, OC]
		//======[load 1 group from G[OC, IC, 4, 4]]==============================
		float g[16]; {//load 4 * 4 elements
			const int goffset = (ic*OC) << 4;//[gic, goc, 0, 0]

			*(float4*)(g) = *(float4*)(G + goffset);
			*(float4*)(g + 4) = *(float4*)(G + goffset + 4);
			*(float4*)(g + 8) = *(float4*)(G + goffset + 8);
			*(float4*)(g + 12) = *(float4*)(G + goffset + 12);

			//register -> shread: Gs[ty][][tx], [8][16][32]
			//(1) ty -> ic = 0 - 8
			//(2) 1 group: g0 - g16
			//(3) tx -> oc = 0 -31
			Gs[ty][0][tx] = g[0]; Gs[ty][1][tx] = g[1]; Gs[ty][2][tx] = g[2]; Gs[ty][3][tx] = g[3];
			Gs[ty][4][tx] = g[4]; Gs[ty][5][tx] = g[5]; Gs[ty][6][tx] = g[6]; Gs[ty][7][tx] = g[7];
			Gs[ty][8][tx] = g[8]; Gs[ty][9][tx] = g[9]; Gs[ty][10][tx] = g[10]; Gs[ty][11][tx] = g[11];
			Gs[ty][12][tx] = g[12]; Gs[ty][13][tx] = g[13]; Gs[ty][14][tx] = g[14]; Gs[ty][15][tx] = g[15];
		}
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
		__syncthreads();
	}

	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	//=================================================================================
	//turn0: x, [ux, uy, elem]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn1: y
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//(elem, uy, ux)
	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y0);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y2);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y3);
	__syncthreads();

	//======[write to Y]===================================================================
	const int yoc = boc + GIdx + (ux & 1) * 4;
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; {
		yn = yj / THW; int jr = yj - yn * THW;
		int th = jr / TW, tw = jr - th * TW;
		oh = (th << 1); ow = (tw << 1);
	}

	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


//select: Size = 9, Time = 1.62279 msec, Performace = 11910 GFlop/s
#define C3D_WG2D_KERNEL_A3
#ifndef C3D_WG2D_KERNEL_A3
#define C3D_WG2D_KERNEL_A3

#define c3d_wg2d_a3(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_A3\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, DH, DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.62279 msec, Performace = 11910 GFlop/s
//Size = 9, Time = 2.06779 msec, Performace = 9346.88 GFlop/s
//Size = 9, Time = 1.46301 msec, Performace = 13210.7 GFlop/s
__global__ void c3d_wg2d_kernel_A3(
	const float* __restrict__ D, int DH, int DW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8][16][36];
	__shared__ float Xs[8][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int oc = boc + tx;//[0 - 31]
	G += oc << 4;//G[0, oc, 0, 0]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int xn, ih, iw; {
		xn = xj / THW; int jr = xj - xn * THW;
		int th = jr / TW, tw = jr - th * TW;
		ih = (th << 1); iw = (tw << 1);
	}

	//2, 4

	const int xoffset = (xn*DH + ih)*DW + iw;//[ic, n, ih, iw]
	for (int oic = 0; oic < IC; oic += 8)
	{
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float x[16]; {//load 4 * 4 elements
			//[ic, n, ih, iw]
			const int X00 = xoffset + ic * N * DH * DW;//[ic, n, ih, iw]
			const int X10 = X00 + DW;
			const int X20 = X10 + DW;
			const int X30 = X20 + DW;

			*(float2*)(x) = *(float2*)(D + X00);
			*(float2*)(x + 2) = *(float2*)(D + X00 + 2);
			*(float2*)(x + 4) = *(float2*)(D + X10);
			*(float2*)(x + 6) = *(float2*)(D + X10 + 2);
			*(float2*)(x + 8) = *(float2*)(D + X20);
			*(float2*)(x + 10) = *(float2*)(D + X20 + 2);
			*(float2*)(x + 12) = *(float2*)(D + X30);
			*(float2*)(x + 14) = *(float2*)(D + X30 + 2);

			Winograd2D_f22x33_transform_X(x);

			//global -> regitser: Xs[ty][][tx], [8][16][32]
			//(1) ty -> ic = 0 - 8
			//(2) 1 group: x0 -> x16, 4 * 4 elements
			//(3) tx -> j: (n, oh, ow)
			Xs[ty][0][tx] = x[0]; Xs[ty][1][tx] = x[1]; Xs[ty][2][tx] = x[2]; Xs[ty][3][tx] = x[3];
			Xs[ty][4][tx] = x[4]; Xs[ty][5][tx] = x[5]; Xs[ty][6][tx] = x[6]; Xs[ty][7][tx] = x[7];
			Xs[ty][8][tx] = x[8]; Xs[ty][9][tx] = x[9]; Xs[ty][10][tx] = x[10]; Xs[ty][11][tx] = x[11];
			Xs[ty][12][tx] = x[12]; Xs[ty][13][tx] = x[13]; Xs[ty][14][tx] = x[14]; Xs[ty][15][tx] = x[15];
		}

		//[ic, <n, ih, iw>]
		//[OC, IC, FH, FW] => [FH, FW, IC, OC]
		//======[load 1 group from G[OC, IC, 4, 4]]==============================
		float g[16]; {//load 4 * 4 elements
			const int goffset = (ic*OC) << 4;//[gic, goc, 0, 0]

			*(float4*)(g) = *(float4*)(G + goffset);
			*(float4*)(g + 4) = *(float4*)(G + goffset + 4);
			*(float4*)(g + 8) = *(float4*)(G + goffset + 8);
			*(float4*)(g + 12) = *(float4*)(G + goffset + 12);

			//register -> shread: Gs[ty][][tx], [8][16][32]
			//(1) ty -> ic = 0 - 8
			//(2) 1 group: g0 - g16
			//(3) tx -> oc = 0 -31
			Gs[ty][0][tx] = g[0]; Gs[ty][1][tx] = g[1]; Gs[ty][2][tx] = g[2]; Gs[ty][3][tx] = g[3];
			Gs[ty][4][tx] = g[4]; Gs[ty][5][tx] = g[5]; Gs[ty][6][tx] = g[6]; Gs[ty][7][tx] = g[7];
			Gs[ty][8][tx] = g[8]; Gs[ty][9][tx] = g[9]; Gs[ty][10][tx] = g[10]; Gs[ty][11][tx] = g[11];
			Gs[ty][12][tx] = g[12]; Gs[ty][13][tx] = g[13]; Gs[ty][14][tx] = g[14]; Gs[ty][15][tx] = g[15];
		}
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
		__syncthreads();
	}

	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	//=================================================================================
	//turn0: x, [ux, uy, elem]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn1: y
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//(elem, uy, ux)
	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y0);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y2);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y3);
	__syncthreads();

	//======[write to Y]===================================================================
	const int yoc = boc + GIdx + (ux & 1) * 4;
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; {
		yn = yj / THW; int jr = yj - yn * THW;
		int th = jr / TW, tw = jr - th * TW;
		oh = (th << 1); ow = (tw << 1);
	}

	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


//select: Size = 9, Time = 1.56808 msec, Performace = 12325.5 GFlop/s
#define C3D_WG2D_KERNEL_A4
#ifndef C3D_WG2D_KERNEL_A4
#define C3D_WG2D_KERNEL_A4

#define c3d_wg2d_a4(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_A4\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, DH, DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.62279 msec, Performace = 11910 GFlop/s
//Size = 9, Time = 2.06779 msec, Performace = 9346.88 GFlop/s
//Size = 9, Time = 1.46301 msec, Performace = 13210.7 GFlop/s
__global__ void c3d_wg2d_kernel_A4(
	const float* __restrict__ D, int DH, int DW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8][16][36];
	__shared__ float Xs[8][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int oc = boc + tx;//[0 - 31]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int xn, ih, iw; {
		xn = xj / THW; int jr = xj - xn * THW;
		int th = jr / TW, tw = jr - th * TW;
		ih = (th << 1); iw = (tw << 1);
	}

	const int xoffset = (xn*DH + ih)*DW + iw;//[ic, n, ih, iw]
	for (int oic = 0; oic < IC; oic += 8)
	{
		const int ic = oic + ty;

		//[ic, n, ih, iw]
		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float x[16]; {//load 4 * 4 elements
			//[ic, n, ih, iw]
			const int X00 = xoffset + ic * N * DH * DW;//[ic, n, ih, iw]
			const int X10 = X00 + DW;
			const int X20 = X10 + DW;
			const int X30 = X20 + DW;

			*(float2*)(x) = *(float2*)(D + X00);
			*(float2*)(x + 2) = *(float2*)(D + X00 + 2);
			*(float2*)(x + 4) = *(float2*)(D + X10);
			*(float2*)(x + 6) = *(float2*)(D + X10 + 2);
			*(float2*)(x + 8) = *(float2*)(D + X20);
			*(float2*)(x + 10) = *(float2*)(D + X20 + 2);
			*(float2*)(x + 12) = *(float2*)(D + X30);
			*(float2*)(x + 14) = *(float2*)(D + X30 + 2);

			Winograd2D_f22x33_transform_X(x);

			//global -> regitser: Xs[ty][][tx], [8][16][32]
			//(1) ty -> ic = 0 - 8
			//(2) 1 group: x0 -> x16, 4 * 4 elements
			//(3) tx -> j: (n, oh, ow)
			Xs[ty][0][tx] = x[0]; Xs[ty][1][tx] = x[1]; Xs[ty][2][tx] = x[2]; Xs[ty][3][tx] = x[3];
			Xs[ty][4][tx] = x[4]; Xs[ty][5][tx] = x[5]; Xs[ty][6][tx] = x[6]; Xs[ty][7][tx] = x[7];
			Xs[ty][8][tx] = x[8]; Xs[ty][9][tx] = x[9]; Xs[ty][10][tx] = x[10]; Xs[ty][11][tx] = x[11];
			Xs[ty][12][tx] = x[12]; Xs[ty][13][tx] = x[13]; Xs[ty][14][tx] = x[14]; Xs[ty][15][tx] = x[15];
		}

		//[ic, 4, 4, oc]
		//======[load 1 group from G[OC, IC, 4, 4]]==============================
		float g[16]; {//load 4 * 4 elements
			//[ic, 4, 4, oc]
			const int goffset = (ic * OC * 16) + oc;//[gic, goc, 0, 0]

			g[0] = G[goffset + OC * 0]; g[1] = G[goffset + OC * 1]; g[2] = G[goffset + OC * 2]; g[3] = G[goffset + OC * 3];
			g[4] = G[goffset + OC * 4]; g[5] = G[goffset + OC * 5]; g[6] = G[goffset + OC * 6]; g[7] = G[goffset + OC * 7];
			g[8] = G[goffset + OC * 8]; g[9] = G[goffset + OC * 9]; g[10] = G[goffset + OC * 10]; g[11] = G[goffset + OC * 11];
			g[12] = G[goffset + OC * 12]; g[13] = G[goffset + OC * 13]; g[14] = G[goffset + OC * 14]; g[15] = G[goffset + OC * 15];

			/**(float4*)(g) = *(float4*)(G + goffset);
			*(float4*)(g + 4) = *(float4*)(G + goffset + 4);
			*(float4*)(g + 8) = *(float4*)(G + goffset + 8);
			*(float4*)(g + 12) = *(float4*)(G + goffset + 12);*/

			//register -> shread: Gs[ty][][tx], [8][16][32]
			//(1) ty -> ic = 0 - 8
			//(2) 1 group: g0 - g16
			//(3) tx -> oc = 0 -31
			Gs[ty][0][tx] = g[0]; Gs[ty][1][tx] = g[1]; Gs[ty][2][tx] = g[2]; Gs[ty][3][tx] = g[3];
			Gs[ty][4][tx] = g[4]; Gs[ty][5][tx] = g[5]; Gs[ty][6][tx] = g[6]; Gs[ty][7][tx] = g[7];
			Gs[ty][8][tx] = g[8]; Gs[ty][9][tx] = g[9]; Gs[ty][10][tx] = g[10]; Gs[ty][11][tx] = g[11];
			Gs[ty][12][tx] = g[12]; Gs[ty][13][tx] = g[13]; Gs[ty][14][tx] = g[14]; Gs[ty][15][tx] = g[15];
		}
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
		__syncthreads();
	}

	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	//=================================================================================
	//turn0: x, [ux, uy, elem]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn1: y
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//(elem, uy, ux)
	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y0);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y2);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y3);
	__syncthreads();

	//======[write to Y]===================================================================
	const int yoc = boc + GIdx + (ux & 1) * 4;
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; {
		yn = yj / THW; int jr = yj - yn * THW;
		int th = jr / TW, tw = jr - th * TW;
		oh = (th << 1); ow = (tw << 1);
	}

	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif



//select: Size = 9, Time = 1.49147 msec, Performace = 12958.6 GFlop/s
#define C3D_WG2D_KERNEL_A6
#ifndef C3D_WG2D_KERNEL_A6
#define C3D_WG2D_KERNEL_A6

#define c3d_wg2d_a6(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_A6\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, DH, DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.49147 msec, Performace = 12958.6 GFlop/s
//Size = 9, Time = 2.06779 msec, Performace = 9346.88 GFlop/s
__global__ void c3d_wg2d_kernel_A6(
	const float* __restrict__ D, int DH, int DW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8][16][36];
	__shared__ float Xs[8][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	G += tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int xn, ih, iw; {
		xn = xj / THW; int jr = xj - xn * THW;
		int th = jr / TW, tw = jr - th * TW;
		ih = (th << 1); iw = (tw << 1);
	}
	D += (xn*DH + ih)*DW + iw;//[ic, n, ih, iw]

	const int GM = N * DH * DW;
	for (int oic = 0; oic < IC; oic += 8) {
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float2 x[8];//[ic, n, ih, iw]
		const int X0 = ic * GM;
		const int X1 = X0 + DW;
		const int X2 = X0 + (DW << 1);
		const int X3 = X0 + DW * 3;

		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3

		Winograd2D_f22x33_transform_X((float*)x);

		Xs[ty][0][tx] = x[0].x; Xs[ty][1][tx] = x[0].y;
		Xs[ty][2][tx] = x[1].x; Xs[ty][3][tx] = x[1].y;
		Xs[ty][4][tx] = x[2].x; Xs[ty][5][tx] = x[2].y;
		Xs[ty][6][tx] = x[3].x; Xs[ty][7][tx] = x[3].y;
		Xs[ty][8][tx] = x[4].x; Xs[ty][9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		//======[load 1 group from G[OC, IC, 4, 4]]==============================
		float4 g[4];//[ic, 4, 4, oc]
		const int G0 = (ic << 4) * OC;
		const int G1 = G0 + OC;
		const int G2 = G0 + (OC << 1);
		const int G3 = G0 + OC * 3;

		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

		*(float4*)(&Gs[ty][tg][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
		__syncthreads();
	}

	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	//=================================================================================
	//turn0: x, [ux, uy, elem]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn1: y
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//(elem, uy, ux)
	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y0);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y2);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y3);

	//======[write to Y]===================================================================
	const int yoc = boc + GIdx + (ux & 1) * 4;
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; {
		yn = yj / THW; int jr = yj - yn * THW;
		int th = jr / TW, tw = jr - th * TW;
		oh = (th << 1); ow = (tw << 1);
	}

	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


//shared memory optimize:
//select: Size = 9, Time = 1.43063 msec, Performace = 13509.7 GFlop/s
#define C3D_WG2D_KERNEL_A7
#ifndef C3D_WG2D_KERNEL_A7
#define C3D_WG2D_KERNEL_A7

#define c3d_wg2d_a7(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_A7\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, DH, DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.43063 msec, Performace = 13509.7 GFlop/s
__global__ void c3d_wg2d_kernel_A7(
	const float* __restrict__ D, int DH, int DW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8][16][36];
	__shared__ float Xs[8][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	G += tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	D += (dn*DH + dh)*DW + dw;//[ic, n, ih, iw]

	const int GM = N * DH * DW;
	for (int oic = 0; oic < IC; oic += 8) {
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float2 x[8];//[ic, n, ih, iw]
		const int X0 = ic * GM;
		const int X1 = X0 + DW;
		const int X2 = X0 + (DW << 1);
		const int X3 = X0 + DW * 3;

		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3

		Winograd2D_f22x33_transform_X((float*)x);

		Xs[ty][0][tx] = x[0].x; Xs[ty][1][tx] = x[0].y;
		Xs[ty][2][tx] = x[1].x; Xs[ty][3][tx] = x[1].y;
		Xs[ty][4][tx] = x[2].x; Xs[ty][5][tx] = x[2].y;
		Xs[ty][6][tx] = x[3].x; Xs[ty][7][tx] = x[3].y;
		Xs[ty][8][tx] = x[4].x; Xs[ty][9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		//======[load 1 group from G[OC, IC, 4, 4]]==============================
		float4 g[4];//[ic, 4, 4, oc]
		const int G0 = (ic << 4) * OC;
		const int G1 = G0 + OC;
		const int G2 = G0 + (OC << 1);
		const int G3 = G0 + OC * 3;

		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

		*(float4*)(&Gs[ty][tg][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
		__syncthreads();
	}

	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	//=================================================================================
	//const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16) 20?
	//turn0: x, [ux, uy, elem]
	*(float4*)(Ys0 + (ux * 17 + uy) * 16) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 17 + uy) * 16 + 4) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(Ys0 + (ux * 17 + uy) * 16 + 8) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(Ys0 + (ux * 17 + uy) * 16 + 12) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn1: y
	*(float4*)(Ys1 + (ux * 17 + uy) * 16) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 17 + uy) * 16 + 4) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(Ys1 + (ux * 17 + uy) * 16 + 8) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(Ys1 + (ux * 17 + uy) * 16 + 12) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//(elem, uy, ux)
	a[0] = Ys0[(0 * 17 + uy) * 16 + ux]; a[1] = Ys0[(1 * 17 + uy) * 16 + ux]; a[2] = Ys0[(2 * 17 + uy) * 16 + ux]; a[3] = Ys0[(3 * 17 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 17 + uy) * 16 + ux]; a[5] = Ys0[(5 * 17 + uy) * 16 + ux]; a[6] = Ys0[(6 * 17 + uy) * 16 + ux]; a[7] = Ys0[(7 * 17 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 17 + uy) * 16 + ux]; a[9] = Ys0[(9 * 17 + uy) * 16 + ux]; a[10] = Ys0[(10 * 17 + uy) * 16 + ux]; a[11] = Ys0[(11 * 17 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 17 + uy) * 16 + ux]; a[13] = Ys0[(13 * 17 + uy) * 16 + ux]; a[14] = Ys0[(14 * 17 + uy) * 16 + ux]; a[15] = Ys0[(15 * 17 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y0);

	a[0] = Ys1[(0 * 17 + uy) * 16 + ux]; a[1] = Ys1[(1 * 17 + uy) * 16 + ux]; a[2] = Ys1[(2 * 17 + uy) * 16 + ux]; a[3] = Ys1[(3 * 17 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 17 + uy) * 16 + ux]; a[5] = Ys1[(5 * 17 + uy) * 16 + ux]; a[6] = Ys1[(6 * 17 + uy) * 16 + ux]; a[7] = Ys1[(7 * 17 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 17 + uy) * 16 + ux]; a[9] = Ys1[(9 * 17 + uy) * 16 + ux]; a[10] = Ys1[(10 * 17 + uy) * 16 + ux]; a[11] = Ys1[(11 * 17 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 17 + uy) * 16 + ux]; a[13] = Ys1[(13 * 17 + uy) * 16 + ux]; a[14] = Ys1[(14 * 17 + uy) * 16 + ux]; a[15] = Ys1[(15 * 17 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z
	*(float4*)(Ys0 + (ux * 17 + uy) * 16) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 17 + uy) * 16 + 4) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + (ux * 17 + uy) * 16 + 8) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + (ux * 17 + uy) * 16 + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w
	*(float4*)(Ys1 + (ux * 17 + uy) * 16) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 17 + uy) * 16 + 4) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + (ux * 17 + uy) * 16 + 8) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + (ux * 17 + uy) * 16 + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[0] = Ys0[(0 * 17 + uy) * 16 + ux]; a[1] = Ys0[(1 * 17 + uy) * 16 + ux]; a[2] = Ys0[(2 * 17 + uy) * 16 + ux]; a[3] = Ys0[(3 * 17 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 17 + uy) * 16 + ux]; a[5] = Ys0[(5 * 17 + uy) * 16 + ux]; a[6] = Ys0[(6 * 17 + uy) * 16 + ux]; a[7] = Ys0[(7 * 17 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 17 + uy) * 16 + ux]; a[9] = Ys0[(9 * 17 + uy) * 16 + ux]; a[10] = Ys0[(10 * 17 + uy) * 16 + ux]; a[11] = Ys0[(11 * 17 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 17 + uy) * 16 + ux]; a[13] = Ys0[(13 * 17 + uy) * 16 + ux]; a[14] = Ys0[(14 * 17 + uy) * 16 + ux]; a[15] = Ys0[(15 * 17 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y2);

	a[0] = Ys1[(0 * 17 + uy) * 16 + ux]; a[1] = Ys1[(1 * 17 + uy) * 16 + ux]; a[2] = Ys1[(2 * 17 + uy) * 16 + ux]; a[3] = Ys1[(3 * 17 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 17 + uy) * 16 + ux]; a[5] = Ys1[(5 * 17 + uy) * 16 + ux]; a[6] = Ys1[(6 * 17 + uy) * 16 + ux]; a[7] = Ys1[(7 * 17 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 17 + uy) * 16 + ux]; a[9] = Ys1[(9 * 17 + uy) * 16 + ux]; a[10] = Ys1[(10 * 17 + uy) * 16 + ux]; a[11] = Ys1[(11 * 17 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 17 + uy) * 16 + ux]; a[13] = Ys1[(13 * 17 + uy) * 16 + ux]; a[14] = Ys1[(14 * 17 + uy) * 16 + ux]; a[15] = Ys1[(15 * 17 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y3);

	//======[write to Y]===================================================================
	const int yoc = boc + GIdx + (ux & 1) * 4;
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);

	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


//shared memory optimize:
//select: Size = 9, Time = 1.39493 msec, Performace = 13855.4 GFlop/s
//        Size = 9, Time = 1.22741 msec, Performace = 15746.5 GFlop/s
#define C3D_WG2D_KERNEL_A8
#ifndef C3D_WG2D_KERNEL_A8
#define C3D_WG2D_KERNEL_A8

#define c3d_wg2d_a8(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_A8\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, DH, DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.39493 msec, Performace = 13855.4 GFlop/s
__global__ void c3d_wg2d_kernel_A8(
	const float* __restrict__ D, int DH, int DW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	//
	__shared__ float Gs[10][16][36];
	__shared__ float Xs[10][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	G += tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	D += (dn*DH + dh)*DW + dw;//[ic, n, ih, iw]

	const int GM = N * DH * DW;
	for (int oic = 0; oic < IC; oic += 8) {
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float2 x[8];//[ic, n, ih, iw]
		const int X0 = ic * GM;
		const int X1 = X0 + DW;
		const int X2 = X0 + (DW << 1);
		const int X3 = X0 + DW * 3;

		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3

		Winograd2D_f22x33_transform_X((float*)x);

		Xs[ty][0][tx] = x[0].x; Xs[ty][1][tx] = x[0].y;
		Xs[ty][2][tx] = x[1].x; Xs[ty][3][tx] = x[1].y;
		Xs[ty][4][tx] = x[2].x; Xs[ty][5][tx] = x[2].y;
		Xs[ty][6][tx] = x[3].x; Xs[ty][7][tx] = x[3].y;
		Xs[ty][8][tx] = x[4].x; Xs[ty][9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		//======[load 1 group from G[OC, IC, 4, 4]]==============================
		float4 g[4];//[ic, 4, 4, oc]
		const int G0 = (ic << 4) * OC;
		const int G1 = G0 + OC;
		const int G2 = G0 + (OC << 1);
		const int G3 = G0 + OC * 3;

		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

		*(float4*)(&Gs[ty][tg][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
		__syncthreads();
	}

	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	//=================================================================================
	//const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16) 20?
	//turn0: x, [ux, uy, elem]
	*(float4*)(Ys0 + (ux * 17 + uy) * 20) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 17 + uy) * 20 + 4) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(Ys0 + (ux * 17 + uy) * 20 + 8) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(Ys0 + (ux * 17 + uy) * 20 + 12) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn1: y
	*(float4*)(Ys1 + (ux * 17 + uy) * 20) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 17 + uy) * 20 + 4) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(Ys1 + (ux * 17 + uy) * 20 + 8) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(Ys1 + (ux * 17 + uy) * 20 + 12) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//(elem, uy, ux)
	a[0] = Ys0[(0 * 17 + uy) * 20 + ux]; a[1] = Ys0[(1 * 17 + uy) * 20 + ux]; a[2] = Ys0[(2 * 17 + uy) * 20 + ux]; a[3] = Ys0[(3 * 17 + uy) * 20 + ux];
	a[4] = Ys0[(4 * 17 + uy) * 20 + ux]; a[5] = Ys0[(5 * 17 + uy) * 20 + ux]; a[6] = Ys0[(6 * 17 + uy) * 20 + ux]; a[7] = Ys0[(7 * 17 + uy) * 20 + ux];
	a[8] = Ys0[(8 * 17 + uy) * 20 + ux]; a[9] = Ys0[(9 * 17 + uy) * 20 + ux]; a[10] = Ys0[(10 * 17 + uy) * 20 + ux]; a[11] = Ys0[(11 * 17 + uy) * 20 + ux];
	a[12] = Ys0[(12 * 17 + uy) * 20 + ux]; a[13] = Ys0[(13 * 17 + uy) * 20 + ux]; a[14] = Ys0[(14 * 17 + uy) * 20 + ux]; a[15] = Ys0[(15 * 17 + uy) * 20 + ux];
	Winograd2D_f22x33_transform_Y(a, y0);

	a[0] = Ys1[(0 * 17 + uy) * 20 + ux]; a[1] = Ys1[(1 * 17 + uy) * 20 + ux]; a[2] = Ys1[(2 * 17 + uy) * 20 + ux]; a[3] = Ys1[(3 * 17 + uy) * 20 + ux];
	a[4] = Ys1[(4 * 17 + uy) * 20 + ux]; a[5] = Ys1[(5 * 17 + uy) * 20 + ux]; a[6] = Ys1[(6 * 17 + uy) * 20 + ux]; a[7] = Ys1[(7 * 17 + uy) * 20 + ux];
	a[8] = Ys1[(8 * 17 + uy) * 20 + ux]; a[9] = Ys1[(9 * 17 + uy) * 20 + ux]; a[10] = Ys1[(10 * 17 + uy) * 20 + ux]; a[11] = Ys1[(11 * 17 + uy) * 20 + ux];
	a[12] = Ys1[(12 * 17 + uy) * 20 + ux]; a[13] = Ys1[(13 * 17 + uy) * 20 + ux]; a[14] = Ys1[(14 * 17 + uy) * 20 + ux]; a[15] = Ys1[(15 * 17 + uy) * 20 + ux];
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z
	*(float4*)(Ys0 + (ux * 17 + uy) * 20) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 17 + uy) * 20 + 4) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + (ux * 17 + uy) * 20 + 8) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + (ux * 17 + uy) * 20 + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w
	*(float4*)(Ys1 + (ux * 17 + uy) * 20) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 17 + uy) * 20 + 4) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + (ux * 17 + uy) * 20 + 8) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + (ux * 17 + uy) * 20 + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[0] = Ys0[(0 * 17 + uy) * 20 + ux]; a[1] = Ys0[(1 * 17 + uy) * 20 + ux]; a[2] = Ys0[(2 * 17 + uy) * 20 + ux]; a[3] = Ys0[(3 * 17 + uy) * 20 + ux];
	a[4] = Ys0[(4 * 17 + uy) * 20 + ux]; a[5] = Ys0[(5 * 17 + uy) * 20 + ux]; a[6] = Ys0[(6 * 17 + uy) * 20 + ux]; a[7] = Ys0[(7 * 17 + uy) * 20 + ux];
	a[8] = Ys0[(8 * 17 + uy) * 20 + ux]; a[9] = Ys0[(9 * 17 + uy) * 20 + ux]; a[10] = Ys0[(10 * 17 + uy) * 20 + ux]; a[11] = Ys0[(11 * 17 + uy) * 20 + ux];
	a[12] = Ys0[(12 * 17 + uy) * 20 + ux]; a[13] = Ys0[(13 * 17 + uy) * 20 + ux]; a[14] = Ys0[(14 * 17 + uy) * 20 + ux]; a[15] = Ys0[(15 * 17 + uy) * 20 + ux];
	Winograd2D_f22x33_transform_Y(a, y2);

	a[0] = Ys1[(0 * 17 + uy) * 20 + ux]; a[1] = Ys1[(1 * 17 + uy) * 20 + ux]; a[2] = Ys1[(2 * 17 + uy) * 20 + ux]; a[3] = Ys1[(3 * 17 + uy) * 20 + ux];
	a[4] = Ys1[(4 * 17 + uy) * 20 + ux]; a[5] = Ys1[(5 * 17 + uy) * 20 + ux]; a[6] = Ys1[(6 * 17 + uy) * 20 + ux]; a[7] = Ys1[(7 * 17 + uy) * 20 + ux];
	a[8] = Ys1[(8 * 17 + uy) * 20 + ux]; a[9] = Ys1[(9 * 17 + uy) * 20 + ux]; a[10] = Ys1[(10 * 17 + uy) * 20 + ux]; a[11] = Ys1[(11 * 17 + uy) * 20 + ux];
	a[12] = Ys1[(12 * 17 + uy) * 20 + ux]; a[13] = Ys1[(13 * 17 + uy) * 20 + ux]; a[14] = Ys1[(14 * 17 + uy) * 20 + ux]; a[15] = Ys1[(15 * 17 + uy) * 20 + ux];
	Winograd2D_f22x33_transform_Y(a, y3);

	//======[write to Y]===================================================================
	const int yoc = boc + GIdx + (ux & 1) * 4;
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);

	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


#define C3D_WG2D_KERNEL_A9
#ifndef C3D_WG2D_KERNEL_A9
#define C3D_WG2D_KERNEL_A9

#define c3d_wg2d_a9(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_A9\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, DH, DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.39493 msec, Performace = 13855.4 GFlop/s
__global__ void c3d_wg2d_kernel_A9(
	const float* __restrict__ D, int DH, int DW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[10][16][36];
	__shared__ float Xs[10][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	G += tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	D += (dn*DH + dh)*DW + dw;//[ic, n, ih, iw]

	const int GM = N * DH * DW;
	for (int oic = 0; oic < IC; oic += 8)
	{
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float2 x[8];//[ic, n, ih, iw]
		const int X0 = ic * GM;
		const int X1 = X0 + DW;
		const int X2 = X0 + (DW << 1);
		const int X3 = X0 + DW * 3;

		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3

		Winograd2D_f22x33_transform_X((float*)x);

		Xs[ty][0][tx] = x[0].x; Xs[ty][1][tx] = x[0].y;
		Xs[ty][2][tx] = x[1].x; Xs[ty][3][tx] = x[1].y;
		Xs[ty][4][tx] = x[2].x; Xs[ty][5][tx] = x[2].y;
		Xs[ty][6][tx] = x[3].x; Xs[ty][7][tx] = x[3].y;
		Xs[ty][8][tx] = x[4].x; Xs[ty][9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		//======[load 1 group from G[OC, IC, 4, 4]]==============================
		float4 g[4];//[ic, 4, 4, oc]
		const int G0 = (ic << 4) * OC;
		const int G1 = G0 + OC;
		const int G2 = G0 + (OC << 1);
		const int G3 = G0 + OC * 3;

		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

		*(float4*)(&Gs[ty][tg][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
		__syncthreads();
	}

	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)

	//=================================================================================
	//turn1: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt + 4) = float4{ v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt + 8) = float4{ v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt + 4) = float4{ v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt + 8) = float4{ v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	a[0] = Ys0[Yrd]; a[1] = Ys0[Yrd + 340]; a[2] = Ys0[Yrd + 680]; a[3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[4] = Ys0[Yrd + 1360]; a[5] = Ys0[Yrd + 1700]; a[6] = Ys0[Yrd + 2040]; a[7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[8] = Ys0[Yrd + 2720]; a[9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	a[0] = Ys1[Yrd]; a[1] = Ys1[Yrd + 340]; a[2] = Ys1[Yrd + 680]; a[3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[4] = Ys1[Yrd + 1360]; a[5] = Ys1[Yrd + 1700]; a[6] = Ys1[Yrd + 2040]; a[7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[8] = Ys1[Yrd + 2720]; a[9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt + 4) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt + 8) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt + 4) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt + 8) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[0] = Ys0[Yrd]; a[1] = Ys0[Yrd + 340]; a[2] = Ys0[Yrd + 680]; a[3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[4] = Ys0[Yrd + 1360]; a[5] = Ys0[Yrd + 1700]; a[6] = Ys0[Yrd + 2040]; a[7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[8] = Ys0[Yrd + 2720]; a[9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	a[0] = Ys1[Yrd]; a[1] = Ys1[Yrd + 340]; a[2] = Ys1[Yrd + 680]; a[3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[4] = Ys1[Yrd + 1360]; a[5] = Ys1[Yrd + 1700]; a[6] = Ys1[Yrd + 2040]; a[7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[8] = Ys1[Yrd + 2720]; a[9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y3);

	//======[write to Y]===================================================================
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);

	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


#define C3D_WG2D_KERNEL_A5
#ifndef C3D_WG2D_KERNEL_A5
#define C3D_WG2D_KERNEL_A5

#define c3d_wg2d_a5(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_A5\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, DH, DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.49874 msec, Performace = 12895.7 GFlop/s
//Size = 9, Time = 2.06779 msec, Performace = 9346.88 GFlop/s
//Size = 9, Time = 1.46301 msec, Performace = 13210.7 GFlop/s
__global__ void c3d_wg2d_kernel_A5(
	const float* __restrict__ D, int DH, int DW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8][16][36];
	__shared__ float Xs[8][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	//const int oc = boc + tx;//[0 - 31]

	//[gt: 4, oc: 8]
	const int toc = (tx % 8) * 4;
	const int oc = boc + toc;//[4x: oc0 - oc3]
	const int gt = (tx / 8) * 4;

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int xn, ih, iw; {
		xn = xj / THW; int jr = xj - xn * THW;
		int th = jr / TW, tw = jr - th * TW;
		ih = (th << 1); iw = (tw << 1);
	}

	const int xoffset = (xn*DH + ih)*DW + iw;//[ic, n, ih, iw]
	for (int oic = 0; oic < IC; oic += 8) {
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float x[16]; {
			const int X00 = xoffset + ic * N * DH * DW;//[ic, n, ih, iw]
			const int X10 = X00 + DW;
			const int X20 = X10 + DW;
			const int X30 = X20 + DW;

			*(float2*)(x) = *(float2*)(D + X00);
			*(float2*)(x + 2) = *(float2*)(D + X00 + 2);
			*(float2*)(x + 4) = *(float2*)(D + X10);
			*(float2*)(x + 6) = *(float2*)(D + X10 + 2);
			*(float2*)(x + 8) = *(float2*)(D + X20);
			*(float2*)(x + 10) = *(float2*)(D + X20 + 2);
			*(float2*)(x + 12) = *(float2*)(D + X30);
			*(float2*)(x + 14) = *(float2*)(D + X30 + 2);

			Winograd2D_f22x33_transform_X(x);

			Xs[ty][0][tx] = x[0]; Xs[ty][1][tx] = x[1]; Xs[ty][2][tx] = x[2]; Xs[ty][3][tx] = x[3];
			Xs[ty][4][tx] = x[4]; Xs[ty][5][tx] = x[5]; Xs[ty][6][tx] = x[6]; Xs[ty][7][tx] = x[7];
			Xs[ty][8][tx] = x[8]; Xs[ty][9][tx] = x[9]; Xs[ty][10][tx] = x[10]; Xs[ty][11][tx] = x[11];
			Xs[ty][12][tx] = x[12]; Xs[ty][13][tx] = x[13]; Xs[ty][14][tx] = x[14]; Xs[ty][15][tx] = x[15];
		}

		//======[load 1 group from G[OC, IC, 4, 4]]==============================
		float g[16]; {
			const int goffset = (ic * 16 + gt) * OC + oc;//[ic, 4, 4, oc]
			*(float4*)(g) = *(float4*)(G + goffset);//[gt0: oc0 - oc3]
			*(float4*)(g + 4) = *(float4*)(G + goffset + OC);//[gt1: oc0 - oc3]
			*(float4*)(g + 8) = *(float4*)(G + goffset + OC * 2);//[gt2: oc0 - oc3]
			*(float4*)(g + 12) = *(float4*)(G + goffset + OC * 3);//[gt3: oc0 - oc3]

			*(float4*)(&Gs[ty][gt][toc]) = *(float4*)(g);
			*(float4*)(&Gs[ty][gt + 1][toc]) = *(float4*)(g + 4);
			*(float4*)(&Gs[ty][gt + 2][toc]) = *(float4*)(g + 8);
			*(float4*)(&Gs[ty][gt + 3][toc]) = *(float4*)(g + 12);
		}
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
		__syncthreads();
	}

	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	//=================================================================================
	//turn0: x, [ux, uy, elem]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn1: y
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//(elem, uy, ux)
	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y0);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z
	*(float4*)(Ys0 + (ux * 16 + uy) * 16) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 4) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 8) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + (ux * 16 + uy) * 16 + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w
	*(float4*)(Ys1 + (ux * 16 + uy) * 16) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 4) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 8) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + (ux * 16 + uy) * 16 + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[0] = Ys0[(0 * 16 + uy) * 16 + ux]; a[1] = Ys0[(1 * 16 + uy) * 16 + ux]; a[2] = Ys0[(2 * 16 + uy) * 16 + ux]; a[3] = Ys0[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys0[(4 * 16 + uy) * 16 + ux]; a[5] = Ys0[(5 * 16 + uy) * 16 + ux]; a[6] = Ys0[(6 * 16 + uy) * 16 + ux]; a[7] = Ys0[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys0[(8 * 16 + uy) * 16 + ux]; a[9] = Ys0[(9 * 16 + uy) * 16 + ux]; a[10] = Ys0[(10 * 16 + uy) * 16 + ux]; a[11] = Ys0[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys0[(12 * 16 + uy) * 16 + ux]; a[13] = Ys0[(13 * 16 + uy) * 16 + ux]; a[14] = Ys0[(14 * 16 + uy) * 16 + ux]; a[15] = Ys0[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y2);

	a[0] = Ys1[(0 * 16 + uy) * 16 + ux]; a[1] = Ys1[(1 * 16 + uy) * 16 + ux]; a[2] = Ys1[(2 * 16 + uy) * 16 + ux]; a[3] = Ys1[(3 * 16 + uy) * 16 + ux];
	a[4] = Ys1[(4 * 16 + uy) * 16 + ux]; a[5] = Ys1[(5 * 16 + uy) * 16 + ux]; a[6] = Ys1[(6 * 16 + uy) * 16 + ux]; a[7] = Ys1[(7 * 16 + uy) * 16 + ux];
	a[8] = Ys1[(8 * 16 + uy) * 16 + ux]; a[9] = Ys1[(9 * 16 + uy) * 16 + ux]; a[10] = Ys1[(10 * 16 + uy) * 16 + ux]; a[11] = Ys1[(11 * 16 + uy) * 16 + ux];
	a[12] = Ys1[(12 * 16 + uy) * 16 + ux]; a[13] = Ys1[(13 * 16 + uy) * 16 + ux]; a[14] = Ys1[(14 * 16 + uy) * 16 + ux]; a[15] = Ys1[(15 * 16 + uy) * 16 + ux];
	Winograd2D_f22x33_transform_Y(a, y3);
	__syncthreads();

	//======[write to Y]===================================================================
	const int yoc = boc + GIdx + (ux & 1) * 4;
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; {
		yn = yj / THW; int jr = yj - yn * THW;
		int th = jr / TW, tw = jr - th * TW;
		oh = (th << 1); ow = (tw << 1);
	}

	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif
