

#ifndef WINOGRAD_2D_KERNEL0
#define WINOGRAD_2D_KERNEL0

#define winograd2d_k0(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC, GN, GM)\
	winograd2D_kenrel0\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, (OH*OW), OW, (N*DH*DW), IC, OC, oc_index, j_index)\

//Size = 18, Time = 2.168 msec, Performace = 17829.7 GFlop/s
//Size = 18, Time = 1.978 msec, Performace = 19542.3 GFlop/s
//Size = 18, Time = 2.896 msec, Performace = 13347.6 GFlop/s

__global__ void winograd2D_kenrel0(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH_OW, int OW,
	int N_DH_DW, int IC, int OC,//sh = sw = 1 
	int oc_index, int j_index) 
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 2][16][32 + 4];
	__shared__ float Ds[8 + 2][16][32 + 4];

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
	const int THW = (OH_OW >> 2), TW = (OW >> 1);
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	const int doffset = dn * DH_DW + dh * DW + dw;//[ic, n, dh, dw]

	//prepare for thread_offset
	//const int ux = ty + (tx > 15) * 8;
	//const int uy = (tx & 15);

	const int ux = (ty << 1) + (tx > 15);
	const int uy = (tx & 15);
	const int XIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//prepare for Y[N, OH, OW, OC]
	const int yoc = boc + GIdx + ((ux & 1) << 2);//uy: oc = GIdx,         j = XIdx;
	const int yj  = bj  + XIdx + (ux >> 1);      //ux: oc = (ux & 1) * 4, j = (ux >> 1)
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	const int y00 = (yn*OH_OW + oh*OW + ow)*OC + yoc;//(0, 0)
	
	//======[compute area1: local]======================================================
	float2 x[8]; float4 g[4];

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = doffset + ty * N_DH_DW;//[ic, n, ih, iw]
	const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
	x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
	x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
	x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
	x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3

	//preload 1 group from G[OC, IC, 4, 4]
	const int G0 = goffset + (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
	
	for (int oic = 1; oic < (IC >> 3); ++oic) {
		//write to shared memory Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];
		 
		//write to shared memory Xs
		Winograd2D_f22x33_transform_X_column((float*)x);//for each column
		Ds[ty][ 0][tx] = x[0].x - x[1].x;//ht0 - ht2
		Ds[ty][ 1][tx] = x[0].y + x[1].x;//ht1 + ht2
		Ds[ty][ 2][tx] = x[1].x - x[0].y;//ht2 - ht1
		Ds[ty][ 3][tx] = x[0].y - x[1].y;//ht1 - ht3

		Ds[ty][ 4][tx] = x[2].x - x[3].x;//ht0 - ht2
		Ds[ty][ 5][tx] = x[2].y + x[3].x;//ht1 + ht2
		Ds[ty][ 6][tx] = x[3].x - x[2].y;//ht2 - ht1
		Ds[ty][ 7][tx] = x[2].y - x[3].y;//ht1 - ht3

		Ds[ty][ 8][tx] = x[4].x - x[5].x;//ht0 - ht2
		Ds[ty][ 9][tx] = x[4].y + x[5].x;//ht1 + ht2
		Ds[ty][10][tx] = x[5].x - x[4].y;//ht2 - ht1
		Ds[ty][11][tx] = x[4].y - x[5].y;//ht1 - ht3

		Ds[ty][12][tx] = x[6].x - x[7].x;//ht0 - ht2
		Ds[ty][13][tx] = x[6].y + x[7].x;//ht1 + ht2 
		Ds[ty][14][tx] = x[7].x - x[6].y;//ht2 - ht1
		Ds[ty][15][tx] = x[6].y - x[7].y;//ht1 - ht3
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][XIdx]), b1 = *(float4*)(&Ds[ik][ux][XIdx + 4]);

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

		const int ic = (oic << 3) + ty;
	
		//load 1 group from X[N, IH, IW, IC]
		const int X0 = doffset + ic * N_DH_DW;//[ic, n, ih, iw]
		const int X1 = X0 + DW, X2 = X0 + (DW << 1), X3 = X0 + DW * 3;
		x[0] = *(float2*)(D + X0); x[1] = *(float2*)(D + X0 + 2);//ih = 0
		x[2] = *(float2*)(D + X1); x[3] = *(float2*)(D + X1 + 2);//ih = 1
		x[4] = *(float2*)(D + X2); x[5] = *(float2*)(D + X2 + 2);//ih = 2
		x[6] = *(float2*)(D + X3); x[7] = *(float2*)(D + X3 + 2);//ih = 3
		
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = goffset + (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
		__syncthreads();
	}
	{
		//write to shared memory Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to shared memory Xs
		Winograd2D_f22x33_transform_X_column((float*)x);//for each column
		Ds[ty][ 0][tx] = x[0].x - x[1].x;//ht0 - ht2
		Ds[ty][ 1][tx] = x[0].y + x[1].x;//ht1 + ht2
		Ds[ty][ 2][tx] = x[1].x - x[0].y;//ht2 - ht1
		Ds[ty][ 3][tx] = x[0].y - x[1].y;//ht1 - ht3

		Ds[ty][ 4][tx] = x[2].x - x[3].x;//ht0 - ht2
		Ds[ty][ 5][tx] = x[2].y + x[3].x;//ht1 + ht2
		Ds[ty][ 6][tx] = x[3].x - x[2].y;//ht2 - ht1
		Ds[ty][ 7][tx] = x[2].y - x[3].y;//ht1 - ht3

		Ds[ty][ 8][tx] = x[4].x - x[5].x;//ht0 - ht2
		Ds[ty][ 9][tx] = x[4].y + x[5].x;//ht1 + ht2
		Ds[ty][10][tx] = x[5].x - x[4].y;//ht2 - ht1
		Ds[ty][11][tx] = x[4].y - x[5].y;//ht1 - ht3

		Ds[ty][12][tx] = x[6].x - x[7].x;//ht0 - ht2
		Ds[ty][13][tx] = x[6].y + x[7].x;//ht1 + ht2 
		Ds[ty][14][tx] = x[7].x - x[6].y;//ht2 - ht1
		Ds[ty][15][tx] = x[6].y - x[7].y;//ht1 - ht3
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
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
	}
	
	//======[compute area12: block]======================================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Ds[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0-----------------------------------------------------------------------------
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
	
	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };//(ux, uy, 12 - 15)
	 
	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };//(ux, uy, 12 - 15)
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


#ifndef WINOGRAD_2D_KERNEL1
#define WINOGRAD_2D_KERNEL1

#define winograd2d_k1(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC, GN, GM)\
	winograd2D_kenrel1\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, (OH*OW), OW, (N*DH*DW), IC, OC, oc_index, j_index)\

//Size = 18, Time = 2.143 msec, Performace = 18037.7 GFlop/s
//Size = 18, Time = 1.978 msec, Performace = 19542.3 GFlop/s
//Size = 18, Time = 2.896 msec, Performace = 13347.6 GFlop/s

__global__ void winograd2D_kenrel1(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH_OW, int OW,
	int N_DH_DW, int IC, int OC,//sh = sw = 1 
	int oc_index, int j_index) 
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8 + 2][16][32 + 4];
	__shared__ float Ds[8 + 2][16][32 + 4];

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
	const int THW = (OH_OW >> 2), TW = (OW >> 1);
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	const int doffset = (dn * DH_DW + dh * DW + dw) * IC;//[ic, n, dh, dw] -> [n, dh, dw, ic]

	//prepare for thread_offset
	const int ux = (ty << 1) + (tx > 15);
	const int uy = (tx & 15);
	const int XIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//prepare for Y[N, OH, OW, OC]
	const int yoc = boc + GIdx + ((ux & 1) << 2);//uy: oc = GIdx,         j = XIdx;
	const int yj  = bj  + XIdx + (ux >> 1);      //ux: oc = (ux & 1) * 4, j = (ux >> 1)
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	const int y00 = (yn*OH_OW + oh*OW + ow)*OC + yoc;//(0, 0)
	
	//======[compute area1: local]======================================================
	float2 x[8]; float4 g[4];
	const int DW_IC = DW * IC;

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = doffset + ty;//[ic, n, ih, iw] -> [n, ih, iw, ic]
	const int X1 = DW_IC, X2 = X0 + DW_IC * 2, X3 = X0 + DW_IC * 3;
	x[0].x = D[X0]; x[0].y = D[X0 + IC]; x[1].x = D[X0 + IC * 2]; x[1].y = D[X0 + IC * 3];//ih = 0
	x[2].x = D[X1]; x[2].y = D[X1 + IC]; x[3].x = D[X1 + IC * 2]; x[3].y = D[X1 + IC * 3];//ih = 0
	x[4].x = D[X2]; x[4].y = D[X2 + IC]; x[5].x = D[X2 + IC * 2]; x[5].y = D[X2 + IC * 3];//ih = 0
	x[6].x = D[X3]; x[6].y = D[X3 + IC]; x[7].x = D[X3 + IC * 2]; x[7].y = D[X3 + IC * 3];//ih = 0

	//preload 1 group from G[OC, IC, 4, 4]
	const int G0 = goffset + (ty << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
	
	for (int oic = 1; oic < (IC >> 3); ++oic) {
		//write to shared memory Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];
		 
		//write to shared memory Xs
		Winograd2D_f22x33_transform_X_column((float*)x);//for each column
		Ds[ty][ 0][tx] = x[0].x - x[1].x;//ht0 - ht2
		Ds[ty][ 1][tx] = x[0].y + x[1].x;//ht1 + ht2
		Ds[ty][ 2][tx] = x[1].x - x[0].y;//ht2 - ht1
		Ds[ty][ 3][tx] = x[0].y - x[1].y;//ht1 - ht3

		Ds[ty][ 4][tx] = x[2].x - x[3].x;//ht0 - ht2
		Ds[ty][ 5][tx] = x[2].y + x[3].x;//ht1 + ht2
		Ds[ty][ 6][tx] = x[3].x - x[2].y;//ht2 - ht1
		Ds[ty][ 7][tx] = x[2].y - x[3].y;//ht1 - ht3

		Ds[ty][ 8][tx] = x[4].x - x[5].x;//ht0 - ht2
		Ds[ty][ 9][tx] = x[4].y + x[5].x;//ht1 + ht2
		Ds[ty][10][tx] = x[5].x - x[4].y;//ht2 - ht1
		Ds[ty][11][tx] = x[4].y - x[5].y;//ht1 - ht3

		Ds[ty][12][tx] = x[6].x - x[7].x;//ht0 - ht2
		Ds[ty][13][tx] = x[6].y + x[7].x;//ht1 + ht2 
		Ds[ty][14][tx] = x[7].x - x[6].y;//ht2 - ht1
		Ds[ty][15][tx] = x[6].y - x[7].y;//ht1 - ht3
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][XIdx]), b1 = *(float4*)(&Ds[ik][ux][XIdx + 4]);

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

		const int ic = (oic << 3) + ty;
	
		//load 1 group from X[N, IH, IW, IC]
		const int X0 = doffset + ic;//[ic, n, ih, iw] -> [n, ih, iw, ic]
		const int X1 = X0 + IC, X2 = X0 + (IC << 1), X3 = X0 + IC * 3;
		x[0].x = D[X0]; x[0].y = D[X0 + IC]; x[1].x = D[X0 + IC * 2]; x[1].y = D[X0 + IC * 3];//ih = 0
		x[2].x = D[X1]; x[2].y = D[X1 + IC]; x[3].x = D[X1 + IC * 2]; x[3].y = D[X1 + IC * 3];//ih = 0
		x[4].x = D[X2]; x[4].y = D[X2 + IC]; x[5].x = D[X2 + IC * 2]; x[5].y = D[X2 + IC * 3];//ih = 0
		x[6].x = D[X3]; x[6].y = D[X3 + IC]; x[7].x = D[X3 + IC * 2]; x[7].y = D[X3 + IC * 3];//ih = 0
		
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = goffset + (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
		__syncthreads();
	}
	{
		//write to shared memory Gs
		*(float4*)(&Gs[ty][tg    ][toc]) = g[0];
		*(float4*)(&Gs[ty][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[ty][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[ty][tg + 3][toc]) = g[3];

		//write to shared memory Xs
		Winograd2D_f22x33_transform_X_column((float*)x);//for each column
		Ds[ty][ 0][tx] = x[0].x - x[1].x;//ht0 - ht2
		Ds[ty][ 1][tx] = x[0].y + x[1].x;//ht1 + ht2
		Ds[ty][ 2][tx] = x[1].x - x[0].y;//ht2 - ht1
		Ds[ty][ 3][tx] = x[0].y - x[1].y;//ht1 - ht3

		Ds[ty][ 4][tx] = x[2].x - x[3].x;//ht0 - ht2
		Ds[ty][ 5][tx] = x[2].y + x[3].x;//ht1 + ht2
		Ds[ty][ 6][tx] = x[3].x - x[2].y;//ht2 - ht1
		Ds[ty][ 7][tx] = x[2].y - x[3].y;//ht1 - ht3

		Ds[ty][ 8][tx] = x[4].x - x[5].x;//ht0 - ht2
		Ds[ty][ 9][tx] = x[4].y + x[5].x;//ht1 + ht2
		Ds[ty][10][tx] = x[5].x - x[4].y;//ht2 - ht1
		Ds[ty][11][tx] = x[4].y - x[5].y;//ht1 - ht3

		Ds[ty][12][tx] = x[6].x - x[7].x;//ht0 - ht2
		Ds[ty][13][tx] = x[6].y + x[7].x;//ht1 + ht2 
		Ds[ty][14][tx] = x[7].x - x[6].y;//ht2 - ht1
		Ds[ty][15][tx] = x[6].y - x[7].y;//ht1 - ht3
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
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
	}
	
	//======[compute area12: block]======================================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Ds[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0-----------------------------------------------------------------------------
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
	
	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };//(ux, uy, 12 - 15)
	 
	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };//(ux, uy, 12 - 15)
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


#ifndef WINOGRAD_2D_KERNEL2
#define WINOGRAD_2D_KERNEL2

//(tx, ty)
#define winograd2d_k2(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC, GN, GM)\
	winograd2D_kenrel2\
		<<< dim3((GN>>5), (GM>>5)), dim3(8, 32), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, (OH*OW), OW, (N*DH*DW), IC, OC, oc_index, j_index)\

//Size = 18, Time = 2.143 msec, Performace = 18037.7 GFlop/s
//Size = 18, Time = 1.978 msec, Performace = 19542.3 GFlop/s
//Size = 18, Time = 2.896 msec, Performace = 13347.6 GFlop/s

__global__ void winograd2D_kenrel2(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH_OW, int OW,
	int N_DH_DW, int IC, int OC,//sh = sw = 1 
	int oc_index, int j_index) 
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float Gs[8 + 2][16][32 + 4];
	__shared__ float Ds[8 + 2][16][32 + 4];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	
	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (ty >> 3) << 2, toc = (ty & 7) << 2;
	const int goffset = tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]  
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + ty;//[0 - 31]
	const int THW = (OH_OW >> 2), TW = (OW >> 1);
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	const int doffset = (dn * DH_DW + dh * DW + dw) * IC;//[ic, n, dh, dw] -> [n, dh, dw, ic]

	//prepare for thread_offset
	const int ux = tx + (ty > 15) * 8;
	const int uy = (ty & 15);
	//const int ux = (tx << 1) + (ty > 15);
	//const int uy = (ty & 15);

	const int XIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//prepare for Y[N, OH, OW, OC]
	const int yoc = boc + GIdx + ((ux & 1) << 2);//uy: oc = GIdx,         j = XIdx;
	const int yj  = bj  + XIdx + (ux >> 1);      //ux: oc = (ux & 1) * 4, j = (ux >> 1)
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	const int y00 = (yn*OH_OW + oh*OW + ow)*OC + yoc;//(0, 0)
	
	//======[compute area1: local]======================================================
	float2 x[8]; float4 g[4];
	const int DW_IC = DW * IC;

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = doffset + tx ;//[ic, n, ih, iw] -> [n, ih, iw, ic]
	const int X1 = DW_IC, X2 = X0 + DW_IC * 2, X3 = X0 + DW_IC * 3;
	x[0].x = D[X0]; x[0].y = D[X0 + IC]; x[1].x = D[X0 + IC * 2]; x[1].y = D[X0 + IC * 3];//ih = 0
	x[2].x = D[X1]; x[2].y = D[X1 + IC]; x[3].x = D[X1 + IC * 2]; x[3].y = D[X1 + IC * 3];//ih = 0
	x[4].x = D[X2]; x[4].y = D[X2 + IC]; x[5].x = D[X2 + IC * 2]; x[5].y = D[X2 + IC * 3];//ih = 0
	x[6].x = D[X3]; x[6].y = D[X3 + IC]; x[7].x = D[X3 + IC * 2]; x[7].y = D[X3 + IC * 3];//ih = 0

	//preload 1 group from G[OC, IC, 4, 4]
	const int G0 = goffset + (tx << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
	
	for (int oic = 1; oic < (IC >> 3); ++oic) {
		//write to shared memory Gs
		*(float4*)(&Gs[tx][tg    ][toc]) = g[0];
		*(float4*)(&Gs[tx][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[tx][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[tx][tg + 3][toc]) = g[3];
		 
		//write to shared memory Xs
		Winograd2D_f22x33_transform_X_column((float*)x);//for each column
		Ds[tx][ 0][ty] = x[0].x - x[1].x;//ht0 - ht2
		Ds[tx][ 1][ty] = x[0].y + x[1].x;//ht1 + ht2
		Ds[tx][ 2][ty] = x[1].x - x[0].y;//ht2 - ht1
		Ds[tx][ 3][ty] = x[0].y - x[1].y;//ht1 - ht3

		Ds[tx][ 4][ty] = x[2].x - x[3].x;//ht0 - ht2
		Ds[tx][ 5][ty] = x[2].y + x[3].x;//ht1 + ht2
		Ds[tx][ 6][ty] = x[3].x - x[2].y;//ht2 - ht1
		Ds[tx][ 7][ty] = x[2].y - x[3].y;//ht1 - ht3

		Ds[tx][ 8][ty] = x[4].x - x[5].x;//ht0 - ht2
		Ds[tx][ 9][ty] = x[4].y + x[5].x;//ht1 + ht2
		Ds[tx][10][ty] = x[5].x - x[4].y;//ht2 - ht1
		Ds[tx][11][ty] = x[4].y - x[5].y;//ht1 - ht3

		Ds[tx][12][ty] = x[6].x - x[7].x;//ht0 - ht2
		Ds[tx][13][ty] = x[6].y + x[7].x;//ht1 + ht2 
		Ds[tx][14][ty] = x[7].x - x[6].y;//ht2 - ht1
		Ds[tx][15][ty] = x[6].y - x[7].y;//ht1 - ht3
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][XIdx]), b1 = *(float4*)(&Ds[ik][ux][XIdx + 4]);

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

		const int ic = (oic << 3) + tx;
	
		//load 1 group from X[N, IH, IW, IC]
		const int X0 = doffset + ic;//[ic, n, ih, iw] -> [n, ih, iw, ic]
		const int X1 = X0 + IC, X2 = X0 + (IC << 1), X3 = X0 + IC * 3;
		x[0].x = D[X0]; x[0].y = D[X0 + IC]; x[1].x = D[X0 + IC * 2]; x[1].y = D[X0 + IC * 3];//ih = 0
		x[2].x = D[X1]; x[2].y = D[X1 + IC]; x[3].x = D[X1 + IC * 2]; x[3].y = D[X1 + IC * 3];//ih = 0
		x[4].x = D[X2]; x[4].y = D[X2 + IC]; x[5].x = D[X2 + IC * 2]; x[5].y = D[X2 + IC * 3];//ih = 0
		x[6].x = D[X3]; x[6].y = D[X3 + IC]; x[7].x = D[X3 + IC * 2]; x[7].y = D[X3 + IC * 3];//ih = 0
		
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = goffset + (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
		__syncthreads();
	}
	{
		//write to shared memory Gs
		*(float4*)(&Gs[tx][tg    ][toc]) = g[0];
		*(float4*)(&Gs[tx][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[tx][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[tx][tg + 3][toc]) = g[3];

		//write to shared memory Xs
		Winograd2D_f22x33_transform_X_column((float*)x);//for each column
		Ds[tx][ 0][ty] = x[0].x - x[1].x;//ht0 - ht2
		Ds[tx][ 1][ty] = x[0].y + x[1].x;//ht1 + ht2
		Ds[tx][ 2][ty] = x[1].x - x[0].y;//ht2 - ht1
		Ds[tx][ 3][ty] = x[0].y - x[1].y;//ht1 - ht3

		Ds[tx][ 4][ty] = x[2].x - x[3].x;//ht0 - ht2
		Ds[tx][ 5][ty] = x[2].y + x[3].x;//ht1 + ht2
		Ds[tx][ 6][ty] = x[3].x - x[2].y;//ht2 - ht1
		Ds[tx][ 7][ty] = x[2].y - x[3].y;//ht1 - ht3

		Ds[tx][ 8][ty] = x[4].x - x[5].x;//ht0 - ht2
		Ds[tx][ 9][ty] = x[4].y + x[5].x;//ht1 + ht2
		Ds[tx][10][ty] = x[5].x - x[4].y;//ht2 - ht1
		Ds[tx][11][ty] = x[4].y - x[5].y;//ht1 - ht3

		Ds[tx][12][ty] = x[6].x - x[7].x;//ht0 - ht2
		Ds[tx][13][ty] = x[6].y + x[7].x;//ht1 + ht2 
		Ds[tx][14][ty] = x[7].x - x[6].y;//ht2 - ht1
		Ds[tx][15][ty] = x[6].y - x[7].y;//ht1 - ht3
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
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
	}
	
	//======[compute area12: block]======================================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Ds[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0-----------------------------------------------------------------------------
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
	
	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };//(ux, uy, 12 - 15)
	 
	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };//(ux, uy, 12 - 15)
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


#ifndef WINOGRAD_2D_KERNEL3
#define WINOGRAD_2D_KERNEL3

//(tx, ty)
#define winograd2d_k3(stream, oc_index, j_index, D, DH, DW, G, Y, OH, OW, N, IC, OC, GN, GM)\
	winograd2D_kenrel3\
		<<< dim3((GN>>5), (GM>>5)), dim3(8, 32), 0, stream >>>\
			(D, (DH*DW), DW, G, Y, (OH*OW), OW, (N*DH*DW), IC, OC, oc_index, j_index)\

//Size = 18, Time = 2.143 msec, Performace = 18037.7 GFlop/s
//Size = 18, Time = 1.978 msec, Performace = 19542.3 GFlop/s
//Size = 18, Time = 2.896 msec, Performace = 13347.6 GFlop/s

__global__ void winograd2D_kenrel3(
	const float* __restrict__ D, int DH_DW, int DW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH_OW, int OW,
	int N_DH_DW, int IC, int OC,//sh = sw = 1 
	int oc_index, int j_index) 
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float Gs[8 + 2][16][32 + 4];
	__shared__ float Ds[8 + 2][16][32 + 4];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;
	
	//prepare for G[IC, OC, 4, 4]
	const int boc = (blockIdx.x << 5) + oc_index;
	const int tg = (ty >> 3) << 2, toc = (ty & 7) << 2;
	const int goffset = tg * OC + boc + toc;//[4x: oc0 - oc3]

	//prepare for X[N, IH, IW, IC]  
	const int bj = (blockIdx.y << 5) + j_index;
	const int xj = bj + ty;//[0 - 31]
	const int THW = (OH_OW >> 2), TW = (OW >> 1);
	int dn, dh, dw; winnograd2D_f22x33_get_nhw(xj, dn, dh, dw);
	const int doffset = (dn * DH_DW + dh * DW + dw) * IC;//[ic, n, dh, dw] -> [n, dh, dw, ic]

	//prepare for thread_offset
	const int ux = tx + (ty > 15) * 8;
	const int uy = (ty & 15);
	//const int ux = (tx << 1) + (ty > 15);
	//const int uy = (ty & 15);

	const int XIdx = ((uy & 1) + ((uy >> 3) << 1)) << 3;
	const int GIdx = ((uy & 7) >> 1) << 3;

	//prepare for Y[N, OH, OW, OC]
	const int yoc = boc + GIdx + ((ux & 1) << 2);//uy: oc = GIdx,         j = XIdx;
	const int yj  = bj  + XIdx + (ux >> 1);      //ux: oc = (ux & 1) * 4, j = (ux >> 1)
	int yn, oh, ow; winnograd2D_f22x33_get_nhw(yj, yn, oh, ow);
	const int y00 = (yn*OH_OW + oh*OW + ow)*OC + yoc;//(0, 0)
	
	//======[compute area1: local]======================================================
	float2 x[8]; float4 g[4];
	const int DW_IC = DW * IC;

	//preload 1 group from X[N, IH, IW, IC]
	const int X0 = doffset + (tx >> 1) * DW_IC + ((tx & 1) << 2);
	*(float4*)(x    ) = *(float4*)(D + X0         );
	*(float4*)(x + 2) = *(float4*)(D + X0 + IC    );
	*(float4*)(x + 4) = *(float4*)(D + X0 + IC * 2);
	*(float4*)(x + 6) = *(float4*)(D + X0 + IC * 3);

	//preload 1 group from G[OC, IC, 4, 4]
	const int G0 = goffset + (tx << 4) * OC;//[ic, 4, 4, oc]
	const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
	g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
	g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
	g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
	g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
	
	for (int oic = 1; oic < (IC >> 3); ++oic) {
		//write to shared memory Gs
		*(float4*)(&Gs[tx][tg    ][toc]) = g[0];
		*(float4*)(&Gs[tx][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[tx][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[tx][tg + 3][toc]) = g[3];
		 
		//write to shared memory Xs
		Winograd2D_f22x33_transform_X_column((float*)x);//for each column
		Ds[tx][ 0][ty] = x[0].x - x[1].x;//ht0 - ht2
		Ds[tx][ 1][ty] = x[0].y + x[1].x;//ht1 + ht2
		Ds[tx][ 2][ty] = x[1].x - x[0].y;//ht2 - ht1
		Ds[tx][ 3][ty] = x[0].y - x[1].y;//ht1 - ht3

		Ds[tx][ 4][ty] = x[2].x - x[3].x;//ht0 - ht2
		Ds[tx][ 5][ty] = x[2].y + x[3].x;//ht1 + ht2
		Ds[tx][ 6][ty] = x[3].x - x[2].y;//ht2 - ht1
		Ds[tx][ 7][ty] = x[2].y - x[3].y;//ht1 - ht3

		Ds[tx][ 8][ty] = x[4].x - x[5].x;//ht0 - ht2
		Ds[tx][ 9][ty] = x[4].y + x[5].x;//ht1 + ht2
		Ds[tx][10][ty] = x[5].x - x[4].y;//ht2 - ht1
		Ds[tx][11][ty] = x[4].y - x[5].y;//ht1 - ht3

		Ds[tx][12][ty] = x[6].x - x[7].x;//ht0 - ht2
		Ds[tx][13][ty] = x[6].y + x[7].x;//ht1 + ht2 
		Ds[tx][14][ty] = x[7].x - x[6].y;//ht2 - ht1
		Ds[tx][15][ty] = x[6].y - x[7].y;//ht1 - ht3
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);
			float4 b0 = *(float4*)(&Ds[ik][ux][XIdx]), b1 = *(float4*)(&Ds[ik][ux][XIdx + 4]);

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

		const int ic = (oic << 3) + tx;
	
		//load 1 group from X[N, IH, IW, IC]
		const int X0 = doffset + (tx >> 1) * DW_IC + ((tx & 1) << 2) + (oic << 3);
		*(float4*)(x    ) = *(float4*)(D + X0);
		*(float4*)(x + 2) = *(float4*)(D + X0 + IC);
		*(float4*)(x + 4) = *(float4*)(D + X0 + IC * 2);
		*(float4*)(x + 6) = *(float4*)(D + X0 + IC * 3);
		
		//load 1 group from G[OC, IC, 4, 4]
		const int G0 = goffset + (ic << 4) * OC;//[ic, 4, 4, oc]
		const int G1 = G0 + OC, G2 = G0 + (OC << 1), G3 = G0 + OC * 3;
		g[0] = *(float4*)(G + G0);//[gt0: oc0 - oc3]
		g[1] = *(float4*)(G + G1);//[gt1: oc0 - oc3]
		g[2] = *(float4*)(G + G2);//[gt2: oc0 - oc3]
		g[3] = *(float4*)(G + G3);//[gt3: oc0 - oc3]
		__syncthreads();
	}
	{
		//write to shared memory Gs
		*(float4*)(&Gs[tx][tg    ][toc]) = g[0];
		*(float4*)(&Gs[tx][tg + 1][toc]) = g[1];
		*(float4*)(&Gs[tx][tg + 2][toc]) = g[2];
		*(float4*)(&Gs[tx][tg + 3][toc]) = g[3];

		//write to shared memory Xs
		Winograd2D_f22x33_transform_X_column((float*)x);//for each column
		Ds[tx][ 0][ty] = x[0].x - x[1].x;//ht0 - ht2
		Ds[tx][ 1][ty] = x[0].y + x[1].x;//ht1 + ht2
		Ds[tx][ 2][ty] = x[1].x - x[0].y;//ht2 - ht1
		Ds[tx][ 3][ty] = x[0].y - x[1].y;//ht1 - ht3

		Ds[tx][ 4][ty] = x[2].x - x[3].x;//ht0 - ht2
		Ds[tx][ 5][ty] = x[2].y + x[3].x;//ht1 + ht2
		Ds[tx][ 6][ty] = x[3].x - x[2].y;//ht2 - ht1
		Ds[tx][ 7][ty] = x[2].y - x[3].y;//ht1 - ht3

		Ds[tx][ 8][ty] = x[4].x - x[5].x;//ht0 - ht2
		Ds[tx][ 9][ty] = x[4].y + x[5].x;//ht1 + ht2
		Ds[tx][10][ty] = x[5].x - x[4].y;//ht2 - ht1
		Ds[tx][11][ty] = x[4].y - x[5].y;//ht1 - ht3

		Ds[tx][12][ty] = x[6].x - x[7].x;//ht0 - ht2
		Ds[tx][13][ty] = x[6].y + x[7].x;//ht1 + ht2 
		Ds[tx][14][ty] = x[7].x - x[6].y;//ht2 - ht1
		Ds[tx][15][ty] = x[6].y - x[7].y;//ht1 - ht3
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++) {
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
	}
	
	//======[compute area12: block]======================================================
	float *Ys0 = &Gs[0][0][0], *Ys1 = &Ds[0][0][0];
	float a[16], y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	const int Ywt = (ux * 17 + uy) * 20;//write: (ux, uy, 0)
	const int Yrd = (uy * 20) + ux;     //read:  (0, uy, ux)
	__syncthreads();

	//group0-----------------------------------------------------------------------------
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
	
	//group1-----------------------------------------------------------------------------
	//write-turn2: z, [oc0, oc4]
	*(float4*)(Ys0 + Ywt     ) = float4{  v0.z,  v1.z,  v2.z,  v3.z };//(ux, uy,  0 -  3)
	*(float4*)(Ys0 + Ywt +  4) = float4{  v4.z,  v5.z,  v6.z,  v7.z };//(ux, uy,  4 -  7)
	*(float4*)(Ys0 + Ywt +  8) = float4{  v8.z,  v9.z, v10.z, v11.z };//(ux, uy,  8 - 11)
	*(float4*)(Ys0 + Ywt + 12) = float4{ v12.z, v13.z, v14.z, v15.z };//(ux, uy, 12 - 15)
	 
	//write-turn3: w, [oc0, oc4]
	*(float4*)(Ys1 + Ywt     ) = float4{  v0.w,  v1.w,  v2.w,  v3.w };//(ux, uy,  0 -  3)
	*(float4*)(Ys1 + Ywt +  4) = float4{  v4.w,  v5.w,  v6.w,  v7.w };//(ux, uy,  4 -  7)
	*(float4*)(Ys1 + Ywt +  8) = float4{  v8.w,  v9.w, v10.w, v11.w };//(ux, uy,  8 - 11)
	*(float4*)(Ys1 + Ywt + 12) = float4{ v12.w, v13.w, v14.w, v15.w };//(ux, uy, 12 - 15)
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