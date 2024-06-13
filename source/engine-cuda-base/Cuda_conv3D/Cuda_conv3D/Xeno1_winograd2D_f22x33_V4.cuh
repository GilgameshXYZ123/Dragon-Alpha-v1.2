

#define C3D_WG2D_KERNEL_B1
#ifndef C3D_WG2D_KERNEL_B1
#define C3D_WG2D_KERNEL_B1

#define c3d_wg2d_b1(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_B1\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, DH, DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.38038 msec, Performace = 14001.5 GFlop/s
__global__ void c3d_wg2d_kernel_B1(
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
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
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

	//-----[compute area]----------------------------------------------------------------------
	//preload 1 group from X[N, IH, IW, IC]
	float2 x[8]; {//[ic, n, ih, iw]
		const int X0 = ty * GM, X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		Winograd2D_f22x33_transform_X((float*)x);
	}

	//preload 1 group from G[OC, IC, 4, 4]
	float4 g[4]; {//[ic, 4, 4, oc]
		const int G0 = (ty << 4) * OC, G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
	}
	
	for (int oic = 8; oic < IC; oic += 8) 
	{
		//write to shared memory
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
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

		const int ic = oic + ty;

		//load 1 group from X[N, IH, IW, IC]
		{//[ic, n, ih, iw]
			const int X0 = ic * GM, X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
			x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
			x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
			x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
			x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
			Winograd2D_f22x33_transform_X((float*)x);
		}
		
		//load 1 group from G[OC, IC, 4, 4]
		{//[ic, 4, 4, oc]
			const int G0 = (ic << 4) * OC, G1 = G0 + OC;
			const int G2 = G0 + (OC << 1);
			const int G3 = G0 + OC * 3;
			g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
			g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
			g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
			g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
		}
		__syncthreads();
	}
	{
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];
		__syncthreads();

	}
#pragma unroll
	for (int ik = 0; ik < 8; ik++) {
		float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
		float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
	__syncthreads();


	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)

	//=================================================================================
	//turn1: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y3);

	//======[write to Y]===================================================================
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj  = bj + XIdx + (ux >> 1);
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


#define C3D_WG2D_KERNEL_B2
#ifndef C3D_WG2D_KERNEL_B2
#define C3D_WG2D_KERNEL_B2

#define c3d_wg2d_b2(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_B2\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.38038 msec, Performace = 14001.5 GFlop/s
__global__ void c3d_wg2d_kernel_B2(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[10][16][36];
	__shared__ float Xs[10][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
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
	D += dn * DH_DW + dh * DW + dw;//[ic, n, ih, iw]
	
	//-----[compute area]----------------------------------------------------------------------
	float2 x[8]; float4 g[4];
	const int GM = N * DH_DW;

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = ty * GM;//[ic, n, ih, iw]
	const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
	x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
	x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
	x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
	x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
	Winograd2D_f22x33_transform_X((float*)x);

	 //preload 1 group from G[OC, IC, 4, 4]
	const int G0 = (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
	
	for (int oic = 1; oic < (IC >> 3); ++oic) 
	{
		//write to shared memory
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
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

		const int ic = (oic << 3) + ty;

		//load 1 group from X[N, IH, IW, IC]
		const int X0 = ic * GM;//[ic, n, ih, iw]
		const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		Winograd2D_f22x33_transform_X((float*)x);
	
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
		__syncthreads();
	}
	{
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];
		__syncthreads();

	}
#pragma unroll
	for (int ik = 0; ik < 8; ik++) {
		float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
		float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
	__syncthreads();


	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)

	//=================================================================================
	//turn1: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y3);

	//======[write to Y]===================================================================
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj  = bj + XIdx + (ux >> 1);
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


//Size = 9, Time = 1.3624 msec, Performace = 14186.3 GFlop/s
#define C3D_WG2D_KERNEL_B3
#ifndef C3D_WG2D_KERNEL_B3
#define C3D_WG2D_KERNEL_B3

#define c3d_wg2d_b3(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_B3\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, OH, OW, N, IC, oc_index, j_index)\

//Size = 9, Time = 1.3624 msec, Performace = 14186.3 GFlop/s
//Size = 9, Time = 1.18539 msec, Performace = 16304.7 GFlop/s
template<int OC = 128>
__global__ void c3d_wg2d_kernel_B3(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, //ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[10][16][36];
	__shared__ float Xs[10][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for thread_offset
	const int ux = (ty << 1) + (tx & 1);//[0 - 16]
	const int uy = (tx >> 1);//[0 - 16]
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int XIdx = ((uy & 7) >> 1) << 3;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	const int goffset = tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	const int xoffset = dn * DH_DW + dh * DW + dw;//[ic, n, ih, iw]
	
	//-----[compute area]----------------------------------------------------------------------
	float2 x[8]; float4 g[4];
	const int GM = N * DH_DW;

	//preload 1 group from G[OC, IC, 4, 4]
	const int G0 = goffset + (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = xoffset + ty * GM;//[ic, n, ih, iw]
	const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
	x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
	x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
	x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
	x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
	Winograd2D_f22x33_transform_X((float*)x);
	
	for (int oic = 1; oic < (IC >> 3); ++oic) 
	{
		//write to shared memory
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
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

		const int ic = (oic << 3) + ty;
	
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = goffset + (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

		//load 1 group from X[N, IH, IW, IC]
		const int X0 = xoffset + ic * GM;//[ic, n, ih, iw]
		const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		Winograd2D_f22x33_transform_X((float*)x);
		__syncthreads();
	}
	*(float4*)(&Gs[ty][tg][toc]) = g[0];
	*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
	*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
	*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

	Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
	Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
	Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
	Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
	Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
	Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
	Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
	Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
	__syncthreads();

#pragma unroll
	for (int ik = 0; ik < 8; ik++) {
		float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
		float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
	__syncthreads();


	//write: [ux: a0 - a16][uy: <oc, yj>][16 elements per round]
	//read:  [    a0 - a16][uy: <oc, yj>][tx: <oc: (0, 4) + round, yj: 0 - 8>]
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3

	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)

	//=================================================================================
	//turn1: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//=================================================================================
	//turn0: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn1: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y3);

	//======[write to Y]===================================================================
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj  = bj + XIdx + (ux >> 1);
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


//[Standard] select: Size = 9, Time = 1.32725 msec, Performace = 14562 GFlop/s
#define C3D_WG2D_KERNEL_B4
#ifndef C3D_WG2D_KERNEL_B4
#define C3D_WG2D_KERNEL_B4

#define c3d_wg2d_b4(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_B4\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, (OH>>1)*(OW>>1), (OW>>1), (N*DH*DW), IC, oc_index, j_index)\

//Size = 9, Time = 1.32725 msec, Performace = 14562 GFlop/s
//Size = 9, Time = 1.15243 msec, Performace = 16770.9 GFlop/s
template<int OC = 128>
__global__ void c3d_wg2d_kernel_B4(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int THW, int TW,
	int GM, int IC,//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8][20][36];//[8, 16, 32]
	__shared__ float Xs[8][20][36];//[8, 16, 32]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	G += tg * OC + (boc + toc);//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	D += dn * DH_DW + dh * DW + dw;//[ic, n, ih, iw]

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for Y[N, OH, OW, OC]
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	THW = THW << 2; TW = TW << 1;//OH * OW, OW
	const int y00 = (yn*THW + oh*TW + ow)*OC + yoc;//(0, 0)
	
	//======[compute area1: local]===============================================
	float2 x[8]; float4 g[4];

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = ty * GM;//[ic, n, ih, iw]
	const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
	x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
	x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
	x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
	x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
	Winograd2D_f22x33_transform_X((float*)x);

	 //preload 1 group from G[OC, IC, 4, 4]
	const int G0 = (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
	
	for (int oic = 1; oic < (IC >> 3); ++oic) 
	{
		//write to shared memory Xs
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		//write to shared memory Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
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

		const int ic = (oic << 3) + ty;

		//load 1 group from X[N, IH, IW, IC]
		const int X0 = ic * GM;//[ic, n, ih, iw]
		const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		Winograd2D_f22x33_transform_X((float*)x);
	
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
		__syncthreads();
	}
	{
		//write to shared memory Xs
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		//write to shared memory Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];
		__syncthreads();

	}
#pragma unroll
	for (int ik = 0; ik < 8; ik++) {
		float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
		float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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
	

	//======[compute area12: block]===============================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//write-turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	//read-turn1: x, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	//read-turn3: z, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y3);

	//write to Y[N, OH, OW, OC]
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + TW * OC, y11 = y10 + OC;//(1, 1), (1, 0)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


//4 / 1 double buffer shared memory
#define C3D_WG2D_KERNEL_B5
#ifndef C3D_WG2D_KERNEL_B5
#define C3D_WG2D_KERNEL_B5

#define c3d_wg2d_b5(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_B5\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, (OH>>1)*(OW>>1), (OW>>1), (N*DH*DW), IC, oc_index, j_index)\

//Size = 9, Time = 1.32725 msec, Performace = 14562 GFlop/s
//Size = 9, Time = 1.15243 msec, Performace = 16770.9 GFlop/s
template<int OC = 128>
__global__ void c3d_wg2d_kernel_B5(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int THW, int TW,
	int GM, int IC,//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[8][20][36];//8 + 2
	__shared__ float Xs[8][20][36];//8 + 2

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	G += tg * OC + (boc + toc);//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	D += dn * DH_DW + dh * DW + dw;//[ic, n, ih, iw]

	//prepare for thread_offset
	const int Idx = (ty << 5) + tx;//ty * 32 + tx
	const int uy = Idx >> 4, ux = Idx & 15;//(Idx / 16, Idx % 16)
	const int GIdx = (uy >> 2) << 3, XIdx = (uy & 3) << 3;

	//prepare for Y[N, OH, OW, OC]
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	THW = THW << 2; TW = TW << 1;//OH * OW, OW
	const int y00 = (yn*THW + oh*TW + ow)*OC + yoc;//(0, 0)
	
	//======[compute area1: local]===============================================
	float2 x[8]; float4 g[4];
	
	//preload 1 group from G[OC, IC, 4, 4]
	const int G0 = (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
	*(float4*)(&Gs[ty][tg + buf * (16 - 3 * (tg >> 2))][toc]) = g[0];

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = ty * GM;//[ic, n, ih, iw]
	const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
	x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
	x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
	x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
	x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
	Winograd2D_f22x33_transform_X((float*)x);
	Xs[ty][0 + buf * 16][tx] = x[0].x; Xs[ty][1 + buf * 16][tx] = x[0].y;
	Xs[ty][2 + buf * 16][tx] = x[1].x; Xs[ty][3 + buf * 16][tx] = x[1].y;

	for (int oic = 1; oic < (IC >> 3); ++oic) 
	{
		//write to shared memory Xs
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;

		//write to shared memory Gs
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux + !(ux & 3) * buf * (16 - 3 * (ux >> 2))][GIdx]);
			float4 a1 = *(float4*)(&Gs[ik][ux + !(ux & 3) * buf * (16 - 3 * (ux >> 2))][GIdx + 4]);
			float4 b0 = *(float4*)(&Xs[ik][ux + buf * 16 * (ux < 4)][XIdx]);
			float4 b1 = *(float4*)(&Xs[ik][ux + buf * 16 * (ux < 4)][XIdx + 4]);

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

		const int ic = (oic << 3) + ty;

		//load 1 group from X[N, IH, IW, IC]
		const int X0 = ic * GM;//[ic, n, ih, iw]
		const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		Winograd2D_f22x33_transform_X((float*)x);
		Xs[ty][0 + buf * 16][tx] = x[0].x; Xs[ty][1 + buf * 16][tx] = x[0].y;
		Xs[ty][2 + buf * 16][tx] = x[1].x; Xs[ty][3 + buf * 16][tx] = x[1].y;

		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
		*(float4*)(&Gs[ty][tg + buf * (16 - 3 * (tg >> 2))][toc]) = g[0];
		__syncthreads();
	}
	{
		//write to shared memory Gs
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to shared memory Xs
		Xs[ty][4][tx] = x[2].x; Xs[ty][5][tx] = x[2].y;
		Xs[ty][6][tx] = x[3].x; Xs[ty][7][tx] = x[3].y;
		Xs[ty][8][tx] = x[4].x; Xs[ty][9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
		__syncthreads();

	}
#pragma unroll
	for (int ik = 0; ik < 8; ik++) {
		float4 a0 = *(float4*)(&Gs[ik][ux + (buf * 16 - 3 * (ux >> 2)) * (!(ux & 3))][GIdx]);
		float4 a1 = *(float4*)(&Gs[ik][ux + (buf * 16 - 3 * (ux >> 2)) * (!(ux & 3))][GIdx + 4]);
		float4 b0 = *(float4*)(&Xs[ik][ux + buf * 16 * (ux < 4)][XIdx]);
		float4 b1 = *(float4*)(&Xs[ik][ux + buf * 16 * (ux < 4)][XIdx + 4]);

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
	__syncthreads();

	//======[compute area12: block]===============================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)

	//group0----------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//write-turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	//read-turn1: x, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	//read-turn3: z, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y3);

	//write to Y[N, OH, OW, OC]
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + TW * OC, y11 = y10 + OC;//(1, 1), (1, 0)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


#define C3D_WG2D_KERNEL_B6
#ifndef C3D_WG2D_KERNEL_B6
#define C3D_WG2D_KERNEL_B6


#define c3d_wg2d_b6(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_B6\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, (OH>>1)*(OW>>1), (OW>>1), (N*DH*DW), IC, oc_index, j_index)\

//Size = 9, Time = 1.32725 msec, Performace = 14562 GFlop/s
//Size = 9, Time = 1.15243 msec, Performace = 16770.9 GFlop/s
template<int OC = 128>
__global__ void c3d_wg2d_kernel_B6(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int THW, int TW,
	int GM, int IC,//ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8][20][36];//[8: ty, 16: elem, 32: tx]
	__shared__ float Xs[8][20][36];//[8: ty, 16: elem, 32: tx]

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	G += tg * OC + (boc + toc);//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	D += dn * DH_DW + dh * DW + dw;//[ic, n, ih, iw]

	//prepare for thread_offset
	const int ux = (ty << 1) + (tx & 1);//[0 - 16]
	const int uy = (tx >> 1);//[0 - 16]
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int XIdx = ((uy & 7) >> 1) << 3;

	//prepare for Y[N, OH, OW, OC]
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	THW = THW << 2; TW = TW << 1;//OH * OW, OW
	const int y00 = (yn*THW + oh*TW + ow)*OC + yoc;//(0, 0)

	//======[compute area1: local]===============================================
	float2 x[8]; float4 g[4];
	
	 //preload 1 group from G[OC, IC, 4, 4]
	const int G0 = (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = ty * GM;//[ic, n, ih, iw]
	const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
	x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
	x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
	x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
	x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
	Winograd2D_f22x33_transform_X((float*)x);

	for (int oic = 1; oic < (IC >> 3); ++oic) 
	{
		//write to shared memory Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to shared memory Xs
		Xs[ty][0][tx] = x[0].x; Xs[ty][1][tx] = x[0].y;
		Xs[ty][2][tx] = x[1].x; Xs[ty][3][tx] = x[1].y;
		Xs[ty][4][tx] = x[2].x; Xs[ty][5][tx] = x[2].y;
		Xs[ty][6][tx] = x[3].x; Xs[ty][7][tx] = x[3].y;
		Xs[ty][8][tx] = x[4].x; Xs[ty][9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

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

		const int ic = (oic << 3) + ty;

		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

		//load 1 group from X[N, IH, IW, IC]
		const int X0 = ic * GM;//[ic, n, ih, iw]
		const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		Winograd2D_f22x33_transform_X((float*)x);
		__syncthreads();
	}
	{
		//write to shared memory Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to shared memory Xs
		Xs[ty][0][tx] = x[0].x; Xs[ty][1][tx] = x[0].y;
		Xs[ty][2][tx] = x[1].x; Xs[ty][3][tx] = x[1].y;
		Xs[ty][4][tx] = x[2].x; Xs[ty][5][tx] = x[2].y;
		Xs[ty][6][tx] = x[3].x; Xs[ty][7][tx] = x[3].y;
		Xs[ty][8][tx] = x[4].x; Xs[ty][9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
		__syncthreads();

	}
#pragma unroll
	for (int ik = 0; ik < 8; ik++) {
		float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);
		float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

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
	

	//======[compute area12: block]===============================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//write-turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	//read-turn1: x, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	//read-turn3: z, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y3);

	//write to Y[N, OH, OW, OC]
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + TW * OC, y11 = y10 + OC;//(1, 1), (1, 0)

	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


//[Standard] select: Size = 9, Time = 1.30719 msec, Performace = 14785.4 GFlop/s
#define C3D_WG2D_KERNEL_B7
#ifndef C3D_WG2D_KERNEL_B7
#define C3D_WG2D_KERNEL_B7

#define c3d_wg2d_b7(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_B7\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, OH, OW, N, IC, oc_index, j_index)\

//Size = 9, Time = 1.30719 msec, Performace = 14785.4 GFlop/s
//Size = 9, Time = 1.11962 msec, Performace = 17262.4 GFlop/s
template<int OC = 128>
__global__ void c3d_wg2d_kernel_B7(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, //ph = pw = sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[10][16][36];
	__shared__ float Xs[10][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	
	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	const int goffset = tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]  
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	const int xoffset = dn * DH_DW + dh * DW + dw;//[ic, n, ih, iw]

	//prepare for thread_offset
	//const int ux = (ty << 1) + (tx > 15);
	//const int uy = (tx & 15);

	const int ux = (ty << 1) + (tx & 1);//[0 - 16]
	const int uy = (tx >> 1);//[0 - 16]
	const int GIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int XIdx = ((uy & 7) >> 1) << 3;

	//prepare for Y[N, OH, OW, OC]
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	
	//======[compute area1: local]===============================================
	float2 x[8]; float4 g[4];
	const int GM = N * DH_DW;

	//preload 1 group from G[OC, IC, 4, 4]
	const int G0 = goffset + (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = xoffset + ty * GM;//[ic, n, ih, iw]
	const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
	x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
	x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
	x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
	x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
	Winograd2D_f22x33_transform_X((float*)x);
	
	for (int oic = 1; oic < (IC >> 3); ++oic) 
	{
		//write to Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to Xs
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) 
		{
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

		const int ic = (oic << 3) + ty;
	
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = goffset + (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

		//load 1 group from X[N, IH, IW, IC]
		const int X0 = xoffset + ic * GM;//[ic, n, ih, iw]
		const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		Winograd2D_f22x33_transform_X((float*)x);
		__syncthreads();
	}
	{
		//write to Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to Xs
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
	}
	__syncthreads();

#pragma unroll
	for (int ik = 0; ik < 8; ik++) {
		float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
		float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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

	//======[compute area12: block]===============================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//write-turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	//read-turn1: x, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	//read-turn3: z, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
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


#define C3D_WG2D_KERNEL_B8
#ifndef C3D_WG2D_KERNEL_B8
#define C3D_WG2D_KERNEL_B8

#define c3d_wg2d_b8(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_B8\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.29312 msec, Performace = 14946.3 GFlop/s
//Size = 9, Time = 1.21633 msec, Performace = 15889.8 GFlop/s
//Size = 9, Time = 1.05817 msec, Performace = 18264.9 GFlop/s
//template<int OC = 128>  
__global__ void c3d_wg2d_kernel_B8(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,//ph = pw = sh = sw = 1 
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[10][16][36];
	__shared__ float Xs[10][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	
	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	const int goffset = tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]  
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	const int xoffset = dn * DH_DW + dh * DW + dw;//[ic, n, ih, iw]

	//prepare for thread_offset
	const int ux = (ty << 1) + (tx > 15);
	const int uy = (tx & 15);
	const int XIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//prepare for Y[N, OH, OW, OC]
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	
	//======[compute area1: local]===============================================
	float2 x[8]; float4 g[4];
	const int GM = N * DH_DW;

	//preload 1 group from G[OC, IC, 4, 4]
	const int G0 = goffset + (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = xoffset + ty * GM;//[ic, n, ih, iw]
	const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
	x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
	x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
	x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
	x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
	Winograd2D_f22x33_transform_X((float*)x);
	
	for (int oic = 1; oic < (IC >> 3); ++oic) 
	{
		//write to Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to Xs
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) 
		{
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

		const int ic = (oic << 3) + ty;
	
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = goffset + (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

		//load 1 group from X[N, IH, IW, IC]
		const int X0 = xoffset + ic * GM;//[ic, n, ih, iw]
		const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		Winograd2D_f22x33_transform_X((float*)x);
		__syncthreads();
	}
	{
		//write to Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to Xs
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
	}
	__syncthreads();

#pragma unroll
	for (int ik = 0; ik < 8; ik++) {
		float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
		float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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

	//======[compute area12: block]===============================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//write-turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	//read-turn1: x, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	//read-turn3: z, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
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

//const int XIdx = (((uy & 1) << 1) + ((uy >> 3))) << 3;
//const int GIdx = (((uy & 7) >> 2) + ((uy & 3) >> 1) * 2) << 3;

#define C3D_WG2D_KERNEL_B9
#ifndef C3D_WG2D_KERNEL_B9
#define C3D_WG2D_KERNEL_B9

#define c3d_wg2d_b9(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_B9\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.28462 msec, Performace = 15045.2 GFlop/s
//Size = 9, Time = 1.21633 msec, Performace = 15889.8 GFlop/s
//Size = 9, Time = 1.05817 msec, Performace = 18264.9 GFlop/s
//template<int OC = 128>  
__global__ void c3d_wg2d_kernel_B9(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,//ph = pw = sh = sw = 1 
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[10][16][36];
	__shared__ float Xs[10][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	
	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	const int goffset = tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]  
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	const int xoffset = dn * DH_DW + dh * DW + dw;//[ic, n, ih, iw]

	//prepare for thread_offset
	const int ux = (ty << 1) + (tx > 15);
	const int uy = (tx & 15);
	
	const int XIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//const int XIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	//const int XIdx = (((uy & 1) << 1) + ((uy >> 3))) << 3;
	//const int GIdx = (((uy & 7) >> 2) + ((uy & 3) >> 1) * 2) << 3;

	//prepare for Y[N, OH, OW, OC]
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	
	//======[compute area1: local]===============================================
	float2 x[8]; float4 g[4];
	const int GM = N * DH_DW;

	//preload 1 group from G[OC, IC, 4, 4]
	const int G0 = goffset + (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = xoffset + ty * GM;//[ic, n, ih, iw]
	const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
	x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
	x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
	x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
	x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
	Winograd2D_f22x33_transform_X((float*)x);
	
	for (int oic = 1; oic < (IC >> 3); ++oic) 
	{
		//write to Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to Xs
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) 
		{
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

		const int ic = (oic << 3) + ty;
	
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = goffset + (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

		//load 1 group from X[N, IH, IW, IC]
		const int X0 = xoffset + ic * GM;//[ic, n, ih, iw]
		const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		Winograd2D_f22x33_transform_X((float*)x);
		__syncthreads();
	}
	{
		//write to Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to Xs
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
	}
	__syncthreads();

#pragma unroll
	for (int ik = 0; ik < 8; ik++) {
		float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
		float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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

	//======[compute area12: block]===============================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//write-turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	//read-turn1: x, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	//read-turn3: z, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
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


//[Standard:] Size = 9, Time = 1.25118 msec, Performace = 15447.3 GFlop/s
#define C3D_WG2D_KERNEL_B10
#ifndef C3D_WG2D_KERNEL_B10
#define C3D_WG2D_KERNEL_B10

#define c3d_wg2d_b10(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC)\
	c3d_wg2d_kernel_B10\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.25118 msec, Performace = 15447.3 GFlop/s
//Size = 9, Time = 1.21633 msec, Performace = 15889.8 GFlop/s
//Size = 9, Time = 1.028 msec, Performace = 18801 GFlop/s
//template<int OC = 128>  
__global__ void c3d_wg2d_kernel_B10(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,//ph = pw = sh = sw = 1 
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[10][16][36];
	__shared__ float Xs[10][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	
	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (tx >> 3) << 2, toc = (tx & 7) << 2;
	const int goffset = tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]  
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	const int xoffset = dn * DH_DW + dh * DW + dw;//[ic, n, ih, iw]

	//prepare for thread_offset
	const int ux = (ty << 1) + (tx > 15);
	const int uy = (tx & 15);
	const int XIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//prepare for Y[N, OH, OW, OC]
	//uy: oc -> GIdx = (uy / 4) << 3, j -> XIdx = (uy % 4) << 3;
	//ux: oc -> (tx & 1) * 4, j -> (tx >> 1)
	const int yoc = boc + GIdx + ((ux & 1) << 2);
	const int yj = bj + XIdx + (ux >> 1);
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	const int y00 = ((yn*OH + oh)*OW + ow)*OC + yoc;//(0, 0)
	
	//======[compute area1: local]===============================================
	float2 x[8]; float4 g[4];
	const int GM = N * DH_DW;

	//preload 1 group from G[OC, IC, 4, 4]
	const int G0 = goffset + (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = xoffset + ty * GM;//[ic, n, ih, iw]
	const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
	x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
	x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
	x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
	x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
	
	for (int oic = 1; oic < (IC >> 3); ++oic) 
	{
		//write to Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to Xs
		Winograd2D_f22x33_transform_X((float*)x);
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) 
		{
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

		const int ic = (oic << 3) + ty;
	
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = goffset + (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]

		//load 1 group from X[N, IH, IW, IC]
		const int X0 = xoffset + ic * GM;//[ic, n, ih, iw]
		const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		
		__syncthreads();
	}
	{
		//write to Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to Xs
		Winograd2D_f22x33_transform_X((float*)x);
		Xs[ty][ 0][tx] = x[0].x; Xs[ty][ 1][tx] = x[0].y;
		Xs[ty][ 2][tx] = x[1].x; Xs[ty][ 3][tx] = x[1].y;
		Xs[ty][ 4][tx] = x[2].x; Xs[ty][ 5][tx] = x[2].y;
		Xs[ty][ 6][tx] = x[3].x; Xs[ty][ 7][tx] = x[3].y;
		Xs[ty][ 8][tx] = x[4].x; Xs[ty][ 9][tx] = x[4].y;
		Xs[ty][10][tx] = x[5].x; Xs[ty][11][tx] = x[5].y;
		Xs[ty][12][tx] = x[6].x; Xs[ty][13][tx] = x[6].y;
		Xs[ty][14][tx] = x[7].x; Xs[ty][15][tx] = x[7].y;
	}
	__syncthreads();

#pragma unroll
	for (int ik = 0; ik < 8; ik++) {
		float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
		float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]), b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

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

	//======[compute area12: block]===============================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Xs[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn0: x, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.x,  v1.x,  v2.x,  v3.x };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.x,  v5.x,  v6.x,  v7.x };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.x,  v9.x, v10.x, v11.x };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.x, v13.x, v14.x, v15.x };//(ux, uy, 12 - 15)

	//write-turn1: y, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.y,  v1.y,  v2.y,  v3.y };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.y,  v5.y,  v6.y,  v7.y };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.y,  v9.y, v10.y, v11.y };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.y, v13.y, v14.y, v15.y };//(ux, uy, 12 - 15)
	__syncthreads();

	//read-turn0: x, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y0);

	//read-turn1: x, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys1[Yrd + 4080]; a[13] = Ys1[Yrd + 4420]; a[14] = Ys1[Yrd + 4760]; a[15] = Ys1[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y1);
	__syncthreads();

	//group0----------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };

	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//[oc0, oc4]
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//read-turn2: z, [oc0, oc4]
	a[ 0] = Ys0[Yrd       ]; a[ 1] = Ys0[Yrd +  340]; a[ 2] = Ys0[Yrd +  680]; a[ 3] = Ys0[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys0[Yrd + 1360]; a[ 5] = Ys0[Yrd + 1700]; a[ 6] = Ys0[Yrd + 2040]; a[ 7] = Ys0[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys0[Yrd + 2720]; a[ 9] = Ys0[Yrd + 3060]; a[10] = Ys0[Yrd + 3400]; a[11] = Ys0[Yrd + 3740];//( 8 - 11, uy, ux)
	a[12] = Ys0[Yrd + 4080]; a[13] = Ys0[Yrd + 4420]; a[14] = Ys0[Yrd + 4760]; a[15] = Ys0[Yrd + 5100];//(12 - 15. uy, ux)
	Winograd2D_f22x33_transform_Y(a, y2);

	//read-turn3: z, [oc0, oc4]
	a[ 0] = Ys1[Yrd       ]; a[ 1] = Ys1[Yrd +  340]; a[ 2] = Ys1[Yrd +  680]; a[ 3] = Ys1[Yrd + 1020];//( 0 -  3, uy, ux)
	a[ 4] = Ys1[Yrd + 1360]; a[ 5] = Ys1[Yrd + 1700]; a[ 6] = Ys1[Yrd + 2040]; a[ 7] = Ys1[Yrd + 2380];//( 4 -  7, uy, ux)
	a[ 8] = Ys1[Yrd + 2720]; a[ 9] = Ys1[Yrd + 3060]; a[10] = Ys1[Yrd + 3400]; a[11] = Ys1[Yrd + 3740];//( 8 - 11, uy, ux)
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


