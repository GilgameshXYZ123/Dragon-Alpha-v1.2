
//[0-3, 0-3] -> [ty % 4, ty / 4] = 16 threads
//[a0 - a15] -> [tx] = 16 threads

//(ty, tx)-------------------------------------------------
//[0]
//( 0, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 1, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 2, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 3, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 4, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 5, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 6, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//- - - - - - - - - - - - - - - -  - - - - - - - - - - - -
//[1]
//( 7, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 8, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 9, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(10, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(11, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(12, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(13, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(14, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(15, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//- - - - - - - - - - - - - - - -  - - - - - - - - - - - - 
//[2]
//( 0, 0), (, 1), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 1, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 2, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 3, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 4, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 5, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 6, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 7, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//- - - - - - - - - - - - - - - -  - - - - - - - - - - - - 
//[3]
//( 8, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//( 9, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(10, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(11, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(12, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(13, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(14, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//(15, 8), (, 9), (, 2), (, 3), (, 4), (, 5), (, 6), (, 7),
//-----------------------------------------------------------


#define C3D_WG2D_KERNEL1
#ifndef C3D_WG2D_KERNEL1
#define C3D_WG2D_KERNEL1

//GN = OC
//GM = N *(OH / 2) * (OW / 2)
//block(32, 32) -> (32, 128) -> (oc, (n, oh, ow))
//32 * 128

#define c3d_wg2d_k1(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel1\
		<<< dim3((GM>>5), (GN>>5)), dim3(16, 16), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)\

__global__ void c3d_wg2d_kernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Xs[16][16][16];//(ik, tx) with the same (n, ih , iw)
	__shared__ float Gs[16][16][16];//(ik, ty) with the same oc

	//prepare for G[IC, OC, 4, 4]
	//oc[0 -  15]: ty[0 - 15] + 0 
	//oc[16 - 31]: ty[0 - 15] + 16
	const int goc = ((blockIdx.y << 5) + ty) + (tx >= 8) * 16 + oc_index;

	//prepare for X[N, IH, IW, IC]
	const int xj = ((blockIdx.x << 5) + tx) + (ty >= 8) * 16 + j_index;
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int xn, ih0, iw0; {
		xn = xj / THW; int jr = xj % THW;
		int xth = jr / TW, xtw = jr % TW;
		ih0 = (xth << 1) - ph; iw0 = (xtw << 1) - pw;
	}

	//-----------------------------------------
	//gidx = ty / 4, xidx = ty % 4;
	//gidx -> Gs{0, 1, 2, 3} -> (0, 0) to (1, 1)
	//xidx -> Xs{0, 1, 2, 3} -> (0, 0) to (1, 1)
	//gidx ->  (gidx / 2, gidx % 2) -> oc
	//xidx -> (xidx / 2, xidx % 2)
	//gidx
	//(0, 0) -> (0, 0) -> oc 


	//idx -> (y, x)
	//0 -> (0, 0)
	//1 -> (0, 1)
	//2 -> (1, 0)
	//3 -> (1, 1)
	//-----------------------------------------
	//0 (0, 0)        | 1 (0, 1)
	//                |
	//8----------------------------------------
	//2 (1, 0)        | 3 (1, 1)
	//                |
	//-----------------------------------------

	const int gidx = ty / 4, xidx = ty % 4;
	const int Gsy = (gidx / 2) << 3, Gsx = (gidx % 2) << 3;
	const int Xsy = (xidx / 2) << 3, Xsx = (xidx / 2) << 3;

	//compute 8*8 accumulators
	//(yj, yoc)---------> (8*8) = 64
	//|
	//|
	//\/
	//所有tx相同的共享一组accumulator
	//按照ty分成16组
	//16 * 64 -> 4 * 64 = 256个Y中的元素，
	//总计256 * 16
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	for (int oic = 0; oic < IC; oic += 8)
	{
		//=======[load 1 group from filter, with the same tx]========================================
		const int gic = oic + (tx % 8);
		const int goffset = (gic*OC + goc) << 4;
		float g[16];
		*(float4*)(g) = *(float4*)(G + goffset);
		*(float4*)(g + 4) = *(float4*)(G + goffset + 4);
		*(float4*)(g + 8) = *(float4*)(G + goffset + 8);
		*(float4*)(g + 12) = *(float4*)(G + goffset + 12);

		Gs[0][tx][ty] = g[0]; Gs[1][tx][ty] = g[1]; Gs[2][tx][ty] = g[2]; Gs[3][tx][ty] = g[3];
		Gs[4][tx][ty] = g[4]; Gs[5][tx][ty] = g[5]; Gs[6][tx][ty] = g[6]; Gs[7][tx][ty] = g[7];
		Gs[8][tx][ty] = g[8]; Gs[9][tx][ty] = g[9]; Gs[10][tx][ty] = g[10]; Gs[11][tx][ty] = g[11];
		Gs[12][tx][ty] = g[12]; Gs[13][tx][ty] = g[13]; Gs[14][tx][ty] = g[13]; Gs[15][tx][ty] = g[15];

		//======[load 1 group from features maps, with the same ty]===================================
		const int xic = oic + (ty % 8);
		const int X00 = ((xn*IH + ih0)*IW + iw0)*IC + xic;//[tn, tih0, tiw0, xic]

		float x[16];//(i, j)
#pragma unroll
		for (int i = 0; i < 4; i++)
#pragma unroll
			for (int j = 0; j < 4; j++) {
				const int tih = ih0 + i;
				const int tiw = iw0 + j;
				const int lx = (tih >= 0) && (tih < IH) && (tiw >= 0) && (tiw < IW);
				const int xoffset = X00 + (i * IW + j) * IC;
				x[(i << 2) + j] = (lx ? X[xoffset] : 0);
			}

#pragma unroll
		for (int t = 0; t < 4; t++) {
			float x0 = x[t];//(0, t)
			float x1 = x[t + 4];//(1, t)
			float x2 = x[t + 8];//(2, t)
			float x3 = x[t + 12];//(3, t)
			x[0] = x0 - x2;//x0t - x2t
			x[1] = x1 + x2;//x1t + x2t
			x[2] = x2 - x1;//x2t - x1t
			x[3] = x1 - x3;//x1t - x3t
		}

#pragma unroll
		for (int t = 0; t < 4; t++) {
			float h0 = x[t * 4];//(t, 0)
			float h1 = x[t * 4 + 1];//(t, 1)
			float h2 = x[t * 4 + 2];//(t, 2)
			float h3 = x[t * 4 + 3];//(t, 3)
			x[0] = h0 - h2;//ht0 - ht2
			x[4] = h1 + h2;//ht1 + ht2
			x[8] = h2 - h1;//ht2 - ht1
			x[12] = h1 - h3;//ht1 - ht3
		}

		Xs[0][ty][tx] = x[0]; Xs[1][ty][tx] = x[1]; Xs[2][ty][tx] = x[2]; Xs[3][ty][tx] = x[3];
		Xs[4][ty][tx] = x[4]; Xs[5][ty][tx] = x[5]; Xs[6][ty][tx] = x[6]; Xs[7][ty][tx] = x[7];
		Xs[8][ty][tx] = x[8]; Xs[9][ty][tx] = x[9]; Xs[10][ty][tx] = x[10]; Xs[11][ty][tx] = x[11];
		Xs[12][ty][tx] = x[12]; Xs[13][ty][tx] = x[13]; Xs[14][ty][tx] = x[14]; Xs[15][ty][tx] = x[15];

		G += 8 * OC * 16;//[ic + 8, oc, 0, 0]
		X += 8;//[n, ih, iw, ic + 8]
		__syncthreads();



#pragma unroll
		for (int ik = 0; ik < 8; ik++)
		{
			float4 a0 = *(float4*)(&Gs[tx][ik + Gsy][Gsx]), a1 = *(float4*)(&Gs[tx][ik + Gsy][Gsx + 4]);
			float4 b0 = *(float4*)(&Xs[tx][ik + Xsy][Xsx]), b1 = *(float4*)(&Xs[tx][ik + Xsy][Gsx + 4]);

			//oc[0 - 3]              oc[4 - 7]
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

	//compute 8*8 accumulators
	//(yj, yoc)---------> (8*8) = 64
	//|
	//|
	//\/
	//所有tx相同的共享一组accumulator
	//按照ty分成16组
	//16 * 64 -> 4 * 64 = 256个Y中的元素，
	//总计256 * 16
	//64 * 16 * 16
	//共享内存： 16 * 16 * 16
	//分成四组写出
	//线程块写出：16 * 16 * 16个元素
	//每个线程写出 16个元素: 4 * 4 -> 2*8

	//总计256 * 16
	//8 * 8
	//oc[0-3], oc[4-7]
	// v0,  v1: j0 -> turn0 Gs
	// v2,  v3: j1 -> turn0 Gs
	// v4,  v5: j2 -> turn1 Xs
	// v6,  v7: j3 -> turn1 Xs
	// v9, v10: j4 -> turn2 Gs
	//v11, v12: j5 -> turn2 Gs
	//v13, v14: j6 -> turn3 Xs
	//v15, v16: j7 -> turn3 Xs
	//改写：
	//{v0 - v16}.x, oc0, oc4
	//{v0 - v16}.y, oc1, oc5
	//{v0 - v16}.z, oc2, oc6
	//{v0 - v16}.w, oc3, oc7


	//const int gidx = ty / 4, xidx = ty % 4;
	//const int Gsy = (gidx / 2) << 3, Gsx = (gidx % 2) << 3;
	//const int Xsy = (xidx / 2) << 3, Xsx = (xidx / 2) << 3;
	//const int yoc = (gidx << 3) + (blockIdx.y << 5);
	//const int yj = (xidx << 3) + (blockIdx.x << 5);

	//[16][16][16]
	//[ty (gidx, xidx) -> (yoc, yj)][tx from 0 - 16, 4 * 4 elements)][16 elements of a turn]

	//oc: 0 - 3;
	float y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	float a[16];

	//=================================================================================
	//turn0: x
	*(float4*)(&Gs[ty][tx][0]) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(&Gs[ty][tx][4]) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Gs[ty][tx][8]) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Gs[ty][tx][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn2: y
	*(float4*)(&Xs[ty][tx][0]) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc1, oc5]
	*(float4*)(&Xs[ty][tx][4]) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Xs[ty][tx][8]) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Xs[ty][tx][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//ty: (gidx, xidx) -> (yoc, yj):
	//(1) yoc = (gidx << 3) + (blockIdx.y << 5)
	//(2) yj  = (xidx << 3) + (blockIdx.x << 5)
	//tx: v0 -> v16 yj             : tx
	//(1) yoc = tx
	//=> yoc =  (gidx << 3) + (blockIdx.y << 5) + (tx >= 8)*4 + round (oc0 -> oc3)
	//=> yj  =(xidx << 3) + (blockIdx.x << 5)

	//trun0: x
	a[0] = Gs[ty][0][tx]; a[1] = Gs[ty][1][tx]; a[2] = Gs[ty][2][tx]; a[3] = Gs[ty][3][tx];
	a[4] = Gs[ty][4][tx]; a[5] = Gs[ty][5][tx]; a[6] = Gs[ty][6][tx]; a[7] = Gs[ty][7][tx];
	a[8] = Gs[ty][8][tx]; a[9] = Gs[ty][9][tx]; a[10] = Gs[ty][10][tx]; a[11] = Gs[ty][10][tx];
	a[12] = Gs[ty][12][tx]; a[13] = Gs[ty][13][tx]; a[14] = Gs[ty][14][tx]; a[15] = Gs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y0);

	//turn1; y
	a[0] = Xs[ty][0][tx]; a[1] = Xs[ty][1][tx]; a[2] = Xs[ty][2][tx]; a[3] = Xs[ty][3][tx];
	a[4] = Xs[ty][4][tx]; a[5] = Xs[ty][5][tx]; a[6] = Xs[ty][6][tx]; a[7] = Xs[ty][7][tx];
	a[8] = Xs[ty][8][tx]; a[9] = Xs[ty][9][tx]; a[10] = Xs[ty][10][tx]; a[11] = Xs[ty][10][tx];
	a[12] = Xs[ty][12][tx]; a[13] = Xs[ty][13][tx]; a[14] = Xs[ty][14][tx]; a[15] = Xs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y1);

	//=================================================================================
	//turn0: z
	*(float4*)(&Gs[ty][tx][0]) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(&Gs[ty][tx][4]) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Gs[ty][tx][8]) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Gs[ty][tx][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn2: w
	*(float4*)(&Xs[ty][tx][0]) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc1, oc5]
	*(float4*)(&Xs[ty][tx][4]) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Xs[ty][tx][8]) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Xs[ty][tx][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//trun0: x
	a[0] = Gs[ty][0][tx]; a[1] = Gs[ty][1][tx]; a[2] = Gs[ty][2][tx]; a[3] = Gs[ty][3][tx];
	a[4] = Gs[ty][4][tx]; a[5] = Gs[ty][5][tx]; a[6] = Gs[ty][6][tx]; a[7] = Gs[ty][7][tx];
	a[8] = Gs[ty][8][tx]; a[9] = Gs[ty][9][tx]; a[10] = Gs[ty][10][tx]; a[11] = Gs[ty][10][tx];
	a[12] = Gs[ty][12][tx]; a[13] = Gs[ty][13][tx]; a[14] = Gs[ty][14][tx]; a[15] = Gs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y2);

	//turn1; y
	a[0] = Xs[ty][0][tx]; a[1] = Xs[ty][1][tx]; a[2] = Xs[ty][2][tx]; a[3] = Xs[ty][3][tx];
	a[4] = Xs[ty][4][tx]; a[5] = Xs[ty][5][tx]; a[6] = Xs[ty][6][tx]; a[7] = Xs[ty][7][tx];
	a[8] = Xs[ty][8][tx]; a[9] = Xs[ty][9][tx]; a[10] = Xs[ty][10][tx]; a[11] = Xs[ty][10][tx];
	a[12] = Xs[ty][12][tx]; a[13] = Xs[ty][13][tx]; a[14] = Xs[ty][14][tx]; a[15] = Xs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y3);

	//=====[write to Y]=======================================================================
	//ty: (gidx, xidx) -> (yoc, yj):
	//(1) yoc = (gidx << 3) + (blockIdx.y << 5)
	//(2) yj  = (xidx << 3) + (blockIdx.x << 5)
	//tx: v0 -> v16 yj             : tx
	//(1) yoc = tx
	//=> yoc =  (gidx << 3) + (blockIdx.y << 5) + (tx >= 8)*4 + round (oc0 -> oc3)
	//=> yj = (xidx << 3) + (blockIdx.x << 5)

	const int yoc = (gidx << 3) + (blockIdx.y << 5) + (tx >= 8) * 4;
	const int yj = (xidx << 3) + (blockIdx.x << 5);
	int yn, yoh, yow; {
		yn = yj / THW; int jr = yj % THW;
		int yth = jr / TW, ytw = jr % TW;
		yoh = yth << 1, yow = ytw << 1;
	}

	const int y00 = ((yn*N + yoh)*OW + yow)*OC + yoc;
	const int y01 = y00 + OC;
	const int y10 = y00 + OW * OC;
	const int y11 = y10 + OC;

	//y00, y0, y1, y2, y3
	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };



}

#endif


#define C3D_WG2D_KERNEL2
#ifndef C3D_WG2D_KERNEL2
#define C3D_WG2D_KERNEL2

#define c3d_wg2d_k2(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel2\
		<<< dim3((GM>>5), (GN>>5)), dim3(16, 16), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)\

__global__ void c3d_wg2d_kernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Xs[16][16][16];
	__shared__ float Gs[16][16][16];

	//compute 8*8 accumulator
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for G[IC, OC, 4, 4]
	const int goc = ((blockIdx.y << 5) + ty) + (tx >= 8) * 16 + oc_index;

	//prepare for X[N, IH, IW, IC]
	const int xj = ((blockIdx.x << 5) + tx) + (ty >= 8) * 16 + j_index;
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int xn, ih0, iw0; {
		xn = xj / THW; int jr = xj % THW;
		int xth = jr / TW, xtw = jr % TW;
		ih0 = (xth << 1) - ph; iw0 = (xtw << 1) - pw;
	}

	const int Gidx = ty / 4, Xidx = ty % 4;//4 * 4 = 16
	const int Gsy = (Gidx / 2) << 3, Gsx = (Gidx % 2) << 3;
	const int Xsy = (Xidx / 2) << 3, Xsx = (Xidx / 2) << 3;

	for (int oic = 0; oic < IC; oic += 8)
	{
		//=======[load 1 group from filter: with the same tx]========================================
		const int gic = oic + (tx % 8);
		const int goffset = (gic*OC + goc) << 4;
		float g[16];
		*(float4*)(g) = *(float4*)(G + goffset);
		*(float4*)(g + 4) = *(float4*)(G + goffset + 4);
		*(float4*)(g + 8) = *(float4*)(G + goffset + 8);
		*(float4*)(g + 12) = *(float4*)(G + goffset + 12);

		Gs[0][tx][ty] = g[0]; Gs[1][tx][ty] = g[1]; Gs[2][tx][ty] = g[2]; Gs[3][tx][ty] = g[3];
		Gs[4][tx][ty] = g[4]; Gs[5][tx][ty] = g[5]; Gs[6][tx][ty] = g[6]; Gs[7][tx][ty] = g[7];
		Gs[8][tx][ty] = g[8]; Gs[9][tx][ty] = g[9]; Gs[10][tx][ty] = g[10]; Gs[11][tx][ty] = g[11];
		Gs[12][tx][ty] = g[12]; Gs[13][tx][ty] = g[13]; Gs[14][tx][ty] = g[14]; Gs[15][tx][ty] = g[15];

		//======[load 1 group from features maps, with the same ty]===================================
		const int xic = oic + (ty % 8);
		const int X00 = ((xn*IH + ih0)*IW + iw0)*IC + xic;//[tn, tih0, tiw0, xic]

		float x[16];//(i, j)
#pragma unroll
		for (int i = 0; i < 4; i++)
#pragma unroll
			for (int j = 0; j < 4; j++) {
				const int tih = ih0 + i;
				const int tiw = iw0 + j;
				const int lx = (tih >= 0) && (tih < IH) && (tiw >= 0) && (tiw < IW);
				const int xoffset = X00 + (i * IW + j) * IC;
				x[(i << 2) + j] = (lx ? X[xoffset] : 0);
			}

#pragma unroll
		for (int t = 0; t < 4; t++) {//for each column
			float x0 = x[t];//(0, t)
			float x1 = x[4 + t];//(1, t)
			float x2 = x[8 + t];//(2, t)
			float x3 = x[12 + t];//(3, t)
			x[t] = x0 - x2;//x0t - x2t
			x[4 + t] = x1 + x2;//x1t + x2t
			x[8 + t] = x2 - x1;//x2t - x1t
			x[12 + t] = x1 - x3;//x1t - x3t
		}

#pragma unroll
		for (int t = 0; t < 4; t++) {
			float h0 = x[t * 4];//(t, 0)
			float h1 = x[t * 4 + 1];//(t, 1)
			float h2 = x[t * 4 + 2];//(t, 2)
			float h3 = x[t * 4 + 3];//(t, 3)
			x[t * 4] = h0 - h2;//ht0 - ht2
			x[t * 4 + 1] = h1 + h2;//ht1 + ht2
			x[t * 4 + 2] = h2 - h1;//ht2 - ht1
			x[t * 4 + 3] = h1 - h3;//ht1 - ht3
		}

		Xs[0][ty][tx] = x[0]; Xs[1][ty][tx] = x[1]; Xs[2][ty][tx] = x[2]; Xs[3][ty][tx] = x[3];
		Xs[4][ty][tx] = x[4]; Xs[5][ty][tx] = x[5]; Xs[6][ty][tx] = x[6]; Xs[7][ty][tx] = x[7];
		Xs[8][ty][tx] = x[8]; Xs[9][ty][tx] = x[9]; Xs[10][ty][tx] = x[10]; Xs[11][ty][tx] = x[11];
		Xs[12][ty][tx] = x[12]; Xs[13][ty][tx] = x[13]; Xs[14][ty][tx] = x[14]; Xs[15][ty][tx] = x[15];

		G += 8 * OC * 16;//[ic + 8, oc, 0, 0]
		X += 8;//[n, ih, iw, ic + 8]
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++)
		{
			float4 a0 = *(float4*)(&Gs[tx][ik + Gsy][Gsx]), a1 = *(float4*)(&Gs[tx][ik + Gsy][Gsx + 4]);
			float4 b0 = *(float4*)(&Xs[tx][ik + Xsy][Xsx]), b1 = *(float4*)(&Xs[tx][ik + Xsy][Gsx + 4]);

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
	//oc: 0 - 3;
	float y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	float a[16];

	//=================================================================================
	//turn0: x
	*(float4*)(&Gs[ty][tx][0]) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(&Gs[ty][tx][4]) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Gs[ty][tx][8]) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Gs[ty][tx][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn2: y
	*(float4*)(&Xs[ty][tx][0]) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc1, oc5]
	*(float4*)(&Xs[ty][tx][4]) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Xs[ty][tx][8]) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Xs[ty][tx][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//ty: (gidx, xidx) -> (yoc, yj):
	//(1) yoc = (gidx << 3) + (blockIdx.y << 5)
	//(2) yj  = (xidx << 3) + (blockIdx.x << 5)
	//tx: v0 -> v16 yj             : tx
	//(1) yoc = tx
	//=> yoc =  (gidx << 3) + (blockIdx.y << 5) + (tx >= 8)*4 + round (oc0 -> oc3)
	//=> yj  = (xidx << 3) + (blockIdx.x << 5)

	//trun0: x
	a[0] = Gs[ty][0][tx]; a[1] = Gs[ty][1][tx]; a[2] = Gs[ty][2][tx]; a[3] = Gs[ty][3][tx];
	a[4] = Gs[ty][4][tx]; a[5] = Gs[ty][5][tx]; a[6] = Gs[ty][6][tx]; a[7] = Gs[ty][7][tx];
	a[8] = Gs[ty][8][tx]; a[9] = Gs[ty][9][tx]; a[10] = Gs[ty][10][tx]; a[11] = Gs[ty][10][tx];
	a[12] = Gs[ty][12][tx]; a[13] = Gs[ty][13][tx]; a[14] = Gs[ty][14][tx]; a[15] = Gs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y0);

	//turn1; y
	a[0] = Xs[ty][0][tx]; a[1] = Xs[ty][1][tx]; a[2] = Xs[ty][2][tx]; a[3] = Xs[ty][3][tx];
	a[4] = Xs[ty][4][tx]; a[5] = Xs[ty][5][tx]; a[6] = Xs[ty][6][tx]; a[7] = Xs[ty][7][tx];
	a[8] = Xs[ty][8][tx]; a[9] = Xs[ty][9][tx]; a[10] = Xs[ty][10][tx]; a[11] = Xs[ty][10][tx];
	a[12] = Xs[ty][12][tx]; a[13] = Xs[ty][13][tx]; a[14] = Xs[ty][14][tx]; a[15] = Xs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y1);

	//=================================================================================
	//turn0: z
	*(float4*)(&Gs[ty][tx][0]) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(&Gs[ty][tx][4]) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Gs[ty][tx][8]) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Gs[ty][tx][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn2: w
	*(float4*)(&Xs[ty][tx][0]) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc1, oc5]
	*(float4*)(&Xs[ty][tx][4]) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Xs[ty][tx][8]) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Xs[ty][tx][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//trun0: x
	a[0] = Gs[ty][0][tx]; a[1] = Gs[ty][1][tx]; a[2] = Gs[ty][2][tx]; a[3] = Gs[ty][3][tx];
	a[4] = Gs[ty][4][tx]; a[5] = Gs[ty][5][tx]; a[6] = Gs[ty][6][tx]; a[7] = Gs[ty][7][tx];
	a[8] = Gs[ty][8][tx]; a[9] = Gs[ty][9][tx]; a[10] = Gs[ty][10][tx]; a[11] = Gs[ty][10][tx];
	a[12] = Gs[ty][12][tx]; a[13] = Gs[ty][13][tx]; a[14] = Gs[ty][14][tx]; a[15] = Gs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y2);

	//turn1; y
	a[0] = Xs[ty][0][tx]; a[1] = Xs[ty][1][tx]; a[2] = Xs[ty][2][tx]; a[3] = Xs[ty][3][tx];
	a[4] = Xs[ty][4][tx]; a[5] = Xs[ty][5][tx]; a[6] = Xs[ty][6][tx]; a[7] = Xs[ty][7][tx];
	a[8] = Xs[ty][8][tx]; a[9] = Xs[ty][9][tx]; a[10] = Xs[ty][10][tx]; a[11] = Xs[ty][10][tx];
	a[12] = Xs[ty][12][tx]; a[13] = Xs[ty][13][tx]; a[14] = Xs[ty][14][tx]; a[15] = Xs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y3);

	//=====[write to Y]=======================================================================
	//ty: (gidx, xidx) -> (yoc, yj):
	const int yoc = (blockIdx.y << 5) + (Gidx << 3);
	const int yj = (blockIdx.x << 5) + (Xidx << 3);

	// +  + (tx >= 8) * 4;
	//const int yj =  + 



	int yn, yoh, yow; {
		yn = yj / THW; int jr = yj % THW;
		int yth = jr / TW, ytw = jr % TW;
		yoh = yth << 1, yow = ytw << 1;
	}


	const int y00 = ((yn*OH + yoh)*OW + yow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1）

	//int limit = N * OH * OW * OC;
	//if (y00 >= limit) printf("%d, %d, %d, %d\n", yn, yoh, yow, yoc);

	/*desv_print(y0, 4);
	desv_print(y1, 4);
	desv_print(y2, 4);
	desv_print(y3, 4);*/

	//y00, y0, y1, y2, y3
	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


#define C3D_WG2D_KERNEL3
#ifndef C3D_WG2D_KERNEL3
#define C3D_WG2D_KERNEL3

#define c3d_wg2d_k3(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel3\
		<<< dim3((GM>>5), (GN>>5)), dim3(16, 16), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)\

__global__ void c3d_wg2d_kernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Xs[16][16][16];
	__shared__ float Gs[16][16][16];

	//compute 8*8 accumulator: [ty][tx][64]
	//(1) ty -> (Gidx: oc, Yidx: n, oh, ow)
	//(2) tx -> a00 - a16
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	const int Gidx = ty / 4, Xidx = ty % 4;//4 * 4 = 16
	const int Gsy = (Gidx / 2) << 3, Gsx = (Gidx % 2) << 3;
	const int Xsy = (Xidx / 2) << 3, Xsx = (Xidx / 2) << 3;

	int u = ty + (tx >= 8) * 16;
	int v = tx + (ty >= 8) * 16;
	if (blockIdx.x == 0 && blockIdx.y == 0) printf("[tx, ty][%d, %d]\n", u, v);

	//prepare for G[IC, OC, 4, 4]
	const int goc = ((blockIdx.y << 5) + ty) + (tx >= 8) * 16 + oc_index;

	//prepare for X[N, IH, IW, IC]
	const int xj = ((blockIdx.x << 5) + tx) + (ty >= 8) * 16 + j_index;
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int xn, ih0, iw0; {//[n, oh, ow]
		xn = xj / THW; int jr = xj % THW;
		int xth = jr / TW, xtw = jr % TW;
		ih0 = (xth << 1) - ph; iw0 = (xtw << 1) - pw;
	}

	for (int oic = 0; oic < IC; oic += 8)
	{
		//=======[load 1 group from filter: with the same tx]========================================
		const int gic = oic + (tx % 8);
		const int goffset = (gic*OC + goc) << 4;//[gic, goc, 0, 0]
		float g[16];
		*(float4*)(g) = *(float4*)(G + goffset);
		*(float4*)(g + 4) = *(float4*)(G + goffset + 4);
		*(float4*)(g + 8) = *(float4*)(G + goffset + 8);
		*(float4*)(g + 12) = *(float4*)(G + goffset + 12);

		Gs[0][tx][ty] = g[0]; Gs[1][tx][ty] = g[1]; Gs[2][tx][ty] = g[2]; Gs[3][tx][ty] = g[3];
		Gs[4][tx][ty] = g[4]; Gs[5][tx][ty] = g[5]; Gs[6][tx][ty] = g[6]; Gs[7][tx][ty] = g[7];
		Gs[8][tx][ty] = g[8]; Gs[9][tx][ty] = g[9]; Gs[10][tx][ty] = g[10]; Gs[11][tx][ty] = g[11];
		Gs[12][tx][ty] = g[12]; Gs[13][tx][ty] = g[13]; Gs[14][tx][ty] = g[14]; Gs[15][tx][ty] = g[15];

		//======[load 1 group from features maps, with the same ty]===================================
		const int xic = oic + (ty % 8);//[xn, ih0, iw0, xic]
		const int X00 = ((xn*IH + ih0)*IW + iw0)*IC + xic;//[tn, tih0, tiw0, xic]

		float x[16];//(i, j)
#pragma unroll
		for (int i = 0; i < 4; i++)
#pragma unroll
			for (int j = 0; j < 4; j++) {
				const int tih = ih0 + i;
				const int tiw = iw0 + j;
				const int lx = (tih >= 0) && (tih < IH) && (tiw >= 0) && (tiw < IW);
				const int xoffset = X00 + (i * IW + j) * IC;
				x[(i << 2) + j] = (lx ? X[xoffset] : 0);
			}

#pragma unroll
		for (int t = 0; t < 4; t++) {//for each column
			float x0 = x[t];//(0, t)
			float x1 = x[4 + t];//(1, t)
			float x2 = x[8 + t];//(2, t)
			float x3 = x[12 + t];//(3, t)
			x[t] = x0 - x2;//x0t - x2t
			x[4 + t] = x1 + x2;//x1t + x2t
			x[8 + t] = x2 - x1;//x2t - x1t
			x[12 + t] = x1 - x3;//x1t - x3t
		}

#pragma unroll
		for (int t = 0; t < 4; t++) {
			float h0 = x[t * 4];//(t, 0)
			float h1 = x[t * 4 + 1];//(t, 1)
			float h2 = x[t * 4 + 2];//(t, 2)
			float h3 = x[t * 4 + 3];//(t, 3)
			x[t * 4] = h0 - h2;//ht0 - ht2
			x[t * 4 + 1] = h1 + h2;//ht1 + ht2
			x[t * 4 + 2] = h2 - h1;//ht2 - ht1
			x[t * 4 + 3] = h1 - h3;//ht1 - ht3
		}

		Xs[0][ty][tx] = x[0]; Xs[1][ty][tx] = x[1]; Xs[2][ty][tx] = x[2]; Xs[3][ty][tx] = x[3];
		Xs[4][ty][tx] = x[4]; Xs[5][ty][tx] = x[5]; Xs[6][ty][tx] = x[6]; Xs[7][ty][tx] = x[7];
		Xs[8][ty][tx] = x[8]; Xs[9][ty][tx] = x[9]; Xs[10][ty][tx] = x[10]; Xs[11][ty][tx] = x[11];
		Xs[12][ty][tx] = x[12]; Xs[13][ty][tx] = x[13]; Xs[14][ty][tx] = x[14]; Xs[15][ty][tx] = x[15];

		G += 8 * OC * 16;//[ic + 8, oc, 0, 0]
		X += 8;//[n, ih, iw, ic + 8]
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < 8; ik++)
		{
			float4 a0 = *(float4*)(&Gs[tx][ik + Gsy][Gsx]), a1 = *(float4*)(&Gs[tx][ik + Gsy][Gsx + 4]);
			float4 b0 = *(float4*)(&Xs[tx][ik + Xsy][Xsx]), b1 = *(float4*)(&Xs[tx][ik + Xsy][Gsx + 4]);

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
	//oc: 0 - 3;
	float y0[4], y1[4], y2[4], y3[4];//oc0, oc1, oc2, oc3
	float a[16];

	//=================================================================================
	//turn0: x
	*(float4*)(&Gs[ty][tx][0]) = float4{ v0.x,  v1.x,  v2.x,  v3.x };//[oc0, oc4]
	*(float4*)(&Gs[ty][tx][4]) = float4{ v4.x,  v5.x,  v6.x,  v7.x };
	*(float4*)(&Gs[ty][tx][8]) = float4{ v8.x,  v9.x, v10.x, v11.x };
	*(float4*)(&Gs[ty][tx][12]) = float4{ v12.x, v13.x, v14.x, v15.x };

	//turn2: y
	*(float4*)(&Xs[ty][tx][0]) = float4{ v0.y,  v1.y,  v2.y,  v3.y };//[oc1, oc5]
	*(float4*)(&Xs[ty][tx][4]) = float4{ v4.y,  v5.y,  v6.y,  v7.y };
	*(float4*)(&Xs[ty][tx][8]) = float4{ v8.y,  v9.y, v10.y, v11.y };
	*(float4*)(&Xs[ty][tx][12]) = float4{ v12.y, v13.y, v14.y, v15.y };
	__syncthreads();

	//ty: (gidx, xidx) -> (yoc, yj):
	//(1) yoc = (gidx << 3) + (blockIdx.y << 5)
	//(2) yj  = (xidx << 3) + (blockIdx.x << 5)
	//tx: v0 -> v16 yj             : tx
	//(1) yoc = tx
	//=> yoc =  (gidx << 3) + (blockIdx.y << 5) + (tx >= 8)*4 + round (oc0 -> oc3)
	//=> yj  = (xidx << 3) + (blockIdx.x << 5)

	//trun0: x
	a[0] = Gs[ty][0][tx]; a[1] = Gs[ty][1][tx]; a[2] = Gs[ty][2][tx]; a[3] = Gs[ty][3][tx];
	a[4] = Gs[ty][4][tx]; a[5] = Gs[ty][5][tx]; a[6] = Gs[ty][6][tx]; a[7] = Gs[ty][7][tx];
	a[8] = Gs[ty][8][tx]; a[9] = Gs[ty][9][tx]; a[10] = Gs[ty][10][tx]; a[11] = Gs[ty][10][tx];
	a[12] = Gs[ty][12][tx]; a[13] = Gs[ty][13][tx]; a[14] = Gs[ty][14][tx]; a[15] = Gs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y0);

	//turn1; y
	a[0] = Xs[ty][0][tx]; a[1] = Xs[ty][1][tx]; a[2] = Xs[ty][2][tx]; a[3] = Xs[ty][3][tx];
	a[4] = Xs[ty][4][tx]; a[5] = Xs[ty][5][tx]; a[6] = Xs[ty][6][tx]; a[7] = Xs[ty][7][tx];
	a[8] = Xs[ty][8][tx]; a[9] = Xs[ty][9][tx]; a[10] = Xs[ty][10][tx]; a[11] = Xs[ty][10][tx];
	a[12] = Xs[ty][12][tx]; a[13] = Xs[ty][13][tx]; a[14] = Xs[ty][14][tx]; a[15] = Xs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y1);

	//=================================================================================
	//turn0: z
	*(float4*)(&Gs[ty][tx][0]) = float4{ v0.z,  v1.z,  v2.z,  v3.z };//[oc0, oc4]
	*(float4*)(&Gs[ty][tx][4]) = float4{ v4.z,  v5.z,  v6.z,  v7.z };
	*(float4*)(&Gs[ty][tx][8]) = float4{ v8.z,  v9.z, v10.z, v11.z };
	*(float4*)(&Gs[ty][tx][12]) = float4{ v12.z, v13.z, v14.z, v15.z };

	//turn2: w
	*(float4*)(&Xs[ty][tx][0]) = float4{ v0.w,  v1.w,  v2.w,  v3.w };//[oc1, oc5]
	*(float4*)(&Xs[ty][tx][4]) = float4{ v4.w,  v5.w,  v6.w,  v7.w };
	*(float4*)(&Xs[ty][tx][8]) = float4{ v8.w,  v9.w, v10.w, v11.w };
	*(float4*)(&Xs[ty][tx][12]) = float4{ v12.w, v13.w, v14.w, v15.w };
	__syncthreads();

	//trun0: x
	a[0] = Gs[ty][0][tx]; a[1] = Gs[ty][1][tx]; a[2] = Gs[ty][2][tx]; a[3] = Gs[ty][3][tx];
	a[4] = Gs[ty][4][tx]; a[5] = Gs[ty][5][tx]; a[6] = Gs[ty][6][tx]; a[7] = Gs[ty][7][tx];
	a[8] = Gs[ty][8][tx]; a[9] = Gs[ty][9][tx]; a[10] = Gs[ty][10][tx]; a[11] = Gs[ty][10][tx];
	a[12] = Gs[ty][12][tx]; a[13] = Gs[ty][13][tx]; a[14] = Gs[ty][14][tx]; a[15] = Gs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y2);

	//turn1; y
	a[0] = Xs[ty][0][tx]; a[1] = Xs[ty][1][tx]; a[2] = Xs[ty][2][tx]; a[3] = Xs[ty][3][tx];
	a[4] = Xs[ty][4][tx]; a[5] = Xs[ty][5][tx]; a[6] = Xs[ty][6][tx]; a[7] = Xs[ty][7][tx];
	a[8] = Xs[ty][8][tx]; a[9] = Xs[ty][9][tx]; a[10] = Xs[ty][10][tx]; a[11] = Xs[ty][10][tx];
	a[12] = Xs[ty][12][tx]; a[13] = Xs[ty][13][tx]; a[14] = Xs[ty][14][tx]; a[15] = Xs[ty][15][tx];
	Winograd2D_f22x33_transform_Y(a, y3);

	//=====[write to Y]=======================================================================
	//ty: (gidx, xidx) -> (yoc, yj):
	const int yoc = (blockIdx.y << 5) + (Gidx << 3) + (tx & 1) * 4;
	const int yj = (blockIdx.x << 5) + (Xidx << 3) + (tx >> 1);

	int yn, yoh, yow; {
		yn = yj / THW; int jr = yj % THW;
		int yth = jr / TW, ytw = jr % TW;
		yoh = yth << 1, yow = ytw << 1;
	}


	const int y00 = ((yn*OH + yoh)*OW + yow)*OC + yoc;//(0, 0)
	const int y01 = y00 + OC;//(0, 1)
	const int y10 = y00 + OW * OC;//(1, 0)
	const int y11 = y10 + OC;//(1, 1）

	//int limit = N * OH * OW * OC;
	//if (y00 >= limit) printf("%d, %d, %d, %d\n", yn, yoh, yow, yoc);

	/*desv_print(y0, 4);
	desv_print(y1, 4);
	desv_print(y2, 4);
	desv_print(y3, 4);*/

	//y00, y0, y1, y2, y3
	*(float4*)(Y + y00) = float4{ y0[0], y1[0], y2[0], y3[0] };
	*(float4*)(Y + y01) = float4{ y0[1], y1[1], y2[1], y3[1] };
	*(float4*)(Y + y10) = float4{ y0[2], y1[2], y2[2], y3[2] };
	*(float4*)(Y + y11) = float4{ y0[3], y1[3], y2[3], y3[3] };
}

#endif


//correct
#define C3D_WG2D_KERNEL4
#ifndef C3D_WG2D_KERNEL4
#define C3D_WG2D_KERNEL4

#define c3d_wg2d_k4(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel4\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)\

//Size = 9, Time = 3.34401 msec, Performace = 5779.7 GFlop/s
__global__ void c3d_wg2d_kernel4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Gs[8][16][36];
	__shared__ float Xs[8][16][36];

	//compute 8*8 accumulators: 
	float4  v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4  v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4  v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for block_offset: dim3(GN>>5), (GM>>5)
	const int boc = (bx << 5) + oc_index;
	const int bj = (by << 5) + j_index;

	//prepare for thread_offset: dim3(tx: 32, ty: 8)
	const int Idx = ty * 32 + tx;
	const int uy = Idx / 16, ux = Idx % 16;
	const int GIdx = (uy / 4) << 3, XIdx = (uy % 4) << 3;

	const int oc = boc + tx;//[0 - 31]
	const int xj = bj + tx;//[0 - 31]
	const int TH = (OH >> 1), TW = (OW >> 1), THW = TH * TW;
	int xn, ih, iw; {
		xn = xj / THW; int jr = xj % THW;
		int th = jr / TW, tw = jr % TW;
		ih = (th << 1) - ph; iw = (tw << 1) - pw;
	}

	for (int oic = 0; oic < IC; oic += 8)
	{
		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float x[16]; {//load 4 * elements
			const int xic = oic + ty;
			const int X00 = ((xn*IH + ih)*IW + iw)*IC + xic;
#pragma unroll
			for (int i = 0; i < 4; i++) {
#pragma unroll
				for (int j = 0; j < 4; j++) {
					int tih = ih + i;
					int tiw = iw + j;
					const int lx = (tih >= 0) && (tih < IH) && (tiw >= 0) && (tiw < IW);
					const int xoffset = X00 + (i * IW + j) * IC;
					float v = (lx ? X[xoffset] : 0);
					x[i * 4 + j] = v;
				}
			}
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
			const int gic = oic + ty;
			const int goffset = (gic*OC + oc) << 4;//[gic, goc, 0, 0]

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
			//shared -> register: Xs[ik][ux][XIdx], Xs[ik][ux][XIdx + 4]
			//(1) ik = 0 -> 8
			//(2) ux = g0 -> g16
			//(3) XIdx -> j (n, oh, ow)
			float4 b0 = *(float4*)(&Xs[ik][ux][XIdx]);
			float4 b1 = *(float4*)(&Xs[ik][ux][XIdx + 4]);

			//shared -> register: Gs[ik][ux][GIdx], Gs[ik][ux][GIdx + 4], 8 * 16 * 32
			//(1) ik = 0 -> 8
			//(2) ux: g0 -> g16
			//(3) GIdx -> oc
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]);
			float4 a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

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
		yn = yj / THW; int jr = yj % THW;
		int th = jr / TW, tw = jr % TW;
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


//correct
#define C3D_WG2D_KERNEL5
#ifndef C3D_WG2D_KERNEL5
#define C3D_WG2D_KERNEL5

#define c3d_wg2d_k5(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel5\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)\

//Size = 9, Time = 3.05191 msec, Performace = 6332.87 GFlop/s
__global__ void c3d_wg2d_kernel5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
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
		ih = (th << 1) - ph; iw = (tw << 1) - pw;
	}
	const int X0 = ((xn*IH + ih)*IW + iw)*IC;//X[n, ih, iw, 0]

	for (int oic = 0; oic < IC; oic += 8)
	{
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float x[16]; {//load 4 * 4 elements
			const int xoffset = X0 + ic;
#pragma unroll
			for (int i = 0; i < 4; i++) {
				const int ly = (ih >= -i) && (ih < IH - i);
				const int lx0 = (iw >= 0) && (iw < IW);
				const int lx1 = (iw >= -1) && (iw < IW - 1);
				const int lx2 = (iw >= -2) && (iw < IW - 2);
				const int lx3 = (iw >= -3) && (iw < IW - 3);
				const int xoffset0 = xoffset + i * IW * IC;//(i, 0)
				const int xoffset1 = xoffset0 + IC;//(i, 1)
				const int xoffset2 = xoffset0 + (IC << 1);//(i, 2)
				const int xoffset3 = xoffset0 + IC * 3;//(i, 3)
				float x0 = (ly && lx0 ? X[xoffset0] : 0);
				float x1 = (ly && lx1 ? X[xoffset1] : 0);
				float x2 = (ly && lx2 ? X[xoffset2] : 0);
				float x3 = (ly && lx3 ? X[xoffset3] : 0);
				x[i * 4 + 0] = x0;
				x[i * 4 + 1] = x1;
				x[i * 4 + 2] = x2;
				x[i * 4 + 3] = x3;
			}
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
			//shared -> register: Gs[ik][ux][GIdx], Gs[ik][ux][GIdx + 4], 8 * 16 * 32
			//(1) ik = 0 -> 8
			//(2) ux: g0 -> g16
			//(3) GIdx -> oc
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

			//shared -> register: Xs[ik][ux][XIdx], Xs[ik][ux][XIdx + 4]
			//(1) ik = 0 -> 8
			//(2) ux = g0 -> g16
			//(3) XIdx -> j (n, oh, ow)
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


#define C3D_WG2D_KERNEL6
#ifndef C3D_WG2D_KERNEL6
#define C3D_WG2D_KERNEL6

#define c3d_wg2d_k6(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel6\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)\

//Size = 9, Time = 3.01955 msec, Performace = 6400.74 GFlop/s
__global__ void c3d_wg2d_kernel6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
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
		ih = (th << 1) - ph; iw = (tw << 1) - pw;
	}
	const int xoffset = ((xn*IH + ih)*IW + iw)*IC;//X[n, ih, iw, 0]

	for (int oic = 0; oic < IC; oic += 8)
	{
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float x[16]; {//load 4 * 4 elements
			const int ly0 = (ih >= 0) && (ih < IH);
			const int ly1 = (ih >= -1) && (ih < IH - 1);
			const int ly2 = (ih >= -2) && (ih < IH - 2);
			const int ly3 = (ih >= -3) && (ih < IH - 3);
			const int lx0 = (iw >= 0) && (iw < IW);
			const int lx1 = (iw >= -1) && (iw < IW - 1);
			const int lx2 = (iw >= -2) && (iw < IW - 2);
			const int lx3 = (iw >= -3) && (iw < IW - 3);
			// 0,  1,  2,   3
			// 4, [5,  6],  7
			// 8, [9, 10], 11
			//12, 13,  14, 15
			const int X00 = xoffset + ic, X01 = X00 + IC, X02 = X01 + IC, X03 = X02 + IC;
			const int X10 = X00 + IW * IC, X11 = X10 + IC, X12 = X11 + IC, X13 = X12 + IC;
			const int X20 = X10 + IW * IC, X21 = X20 + IC, X22 = X21 + IC, X23 = X22 + IC;
			const int X30 = X20 + IW * IC, X31 = X30 + IC, X32 = X31 + IC, X33 = X32 + IC;
			x[0] = (ly0 && lx0 ? X[X00] : 0);
			x[1] = (ly0 && lx1 ? X[X01] : 0);
			x[2] = (ly0 && lx2 ? X[X02] : 0);
			x[3] = (ly0 && lx3 ? X[X03] : 0);

			x[4] = (ly1 && lx0 ? X[X10] : 0);
			x[5] = (ly1 && lx1 ? X[X11] : 0);
			x[6] = (ly1 && lx2 ? X[X12] : 0);
			x[7] = (ly1 && lx3 ? X[X13] : 0);

			x[8] = (ly2 && lx0 ? X[X20] : 0);
			x[9] = (ly2 && lx1 ? X[X21] : 0);
			x[10] = (ly2 && lx2 ? X[X22] : 0);
			x[11] = (ly2 && lx3 ? X[X23] : 0);

			x[12] = (ly3 && lx0 ? X[X30] : 0);
			x[13] = (ly3 && lx1 ? X[X31] : 0);
			x[14] = (ly3 && lx2 ? X[X32] : 0);
			x[15] = (ly3 && lx3 ? X[X33] : 0);

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
			//shared -> register: Gs[ik][ux][GIdx], Gs[ik][ux][GIdx + 4], 8 * 16 * 32
			//(1) ik = 0 -> 8
			//(2) ux: g0 -> g16
			//(3) GIdx -> oc
			float4 a0 = *(float4*)(&Gs[ik][ux][GIdx]), a1 = *(float4*)(&Gs[ik][ux][GIdx + 4]);

			//shared -> register: Xs[ik][ux][XIdx], Xs[ik][ux][XIdx + 4]
			//(1) ik = 0 -> 8
			//(2) ux = g0 -> g16
			//(3) XIdx -> j (n, oh, ow)
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


//ph = pw = 1
#define C3D_WG2D_KERNEL7
#ifndef C3D_WG2D_KERNEL7
#define C3D_WG2D_KERNEL7

#define c3d_wg2d_k7(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel7\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 2.99124 msec, Performace = 6461.31 GFlop/s
__global__ void c3d_wg2d_kernel7(
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

	const int xoffset = ((xn*IH + ih)*IW + iw)*IC;//X[n, ih, iw, 0]
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

			const int X00 = xoffset + ic, X01 = X00 + IC, X02 = X00 + (IC << 1), X03 = X00 + IC * 3;
			const int X10 = X00 + IW * IC, X11 = X10 + IC, X12 = X10 + (IC << 1), X13 = X10 + IC * 3;
			const int X20 = X10 + IW * IC, X21 = X20 + IC, X22 = X20 + (IC << 1), X23 = X20 + IC * 3;
			const int X30 = X20 + IW * IC, X31 = X30 + IC, X32 = X30 + (IC << 1), X33 = X30 + IC * 3;

			x[5] = X[X11];
			x[6] = X[X12];
			x[9] = X[X21];
			x[10] = X[X22];

			x[1] = (ly0 ? X[X01] : 0);
			x[2] = (ly0 ? X[X02] : 0);
			x[13] = (ly3 ? X[X31] : 0);
			x[14] = (ly3 ? X[X32] : 0);

			x[4] = (lx0 ? X[X10] : 0);
			x[8] = (lx0 ? X[X20] : 0);
			x[7] = (lx3 ? X[X13] : 0);
			x[11] = (lx3 ? X[X23] : 0);

			x[0] = (ly0 && lx0 ? X[X00] : 0);
			x[3] = (ly0 && lx3 ? X[X03] : 0);
			x[12] = (ly3 && lx0 ? X[X30] : 0);
			x[15] = (ly3 && lx3 ? X[X33] : 0);

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



//[n, ih, iw]正确, [ih, iw, n]不合适
#define C3D_WG2D_KERNEL8
#ifndef C3D_WG2D_KERNEL8
#define C3D_WG2D_KERNEL8

#define c3d_wg2d_k8(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel8\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 2.99124 msec, Performace = 6461.31 GFlop/s
__global__ void c3d_wg2d_kernel8(
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
	const int TH = (OH >> 1), TW = (OW >> 1);
	const int THW = TH * TW;
	const int TWN = TW * N;

	int xn, ih, iw; {
		const int j = bj + tx;//[0 - 31]

		//======{ [n, ih, iw] = 6461.31 GFlop/s }======
		//xn = j / THW; int jr = j - xn * THW;
		//int th = jr / TW, tw = jr - th * TW;

		//======{ [ih, iw, n] }===============
		int th = j / TWN, jr = j % TWN;
		int tw = jr / N; xn = jr % N;

		ih = (th << 1) - 1; iw = (tw << 1) - 1;
	}

	const int xoffset = ((xn*IH + ih)*IW + iw)*IC;//X[n, ih, iw, 0]
	for (int oic = 0; oic < IC; oic += 8)
	{
		const int ic = oic + ty;

		//======[load 1 group from X[N, IH, IW, IC]]==============================
		float x[16]; {//load 4 * 4 elements
			const int ly0 = (ih >= 0) && (ih < IH);
			const int ly3 = (ih >= -3) && (ih < IH - 3);
			const int lx0 = (iw >= 0) && (iw < IW);
			const int lx3 = (iw >= -3) && (iw < IW - 3);

			const int X00 = xoffset + ic, X01 = X00 + IC, X02 = X00 + (IC << 1), X03 = X00 + IC * 3;
			const int X10 = X00 + IW * IC, X11 = X10 + IC, X12 = X10 + (IC << 1), X13 = X10 + IC * 3;
			const int X20 = X10 + IW * IC, X21 = X20 + IC, X22 = X20 + (IC << 1), X23 = X20 + IC * 3;
			const int X30 = X20 + IW * IC, X31 = X30 + IC, X32 = X30 + (IC << 1), X33 = X30 + IC * 3;

			x[5] = X[X11];
			x[6] = X[X12];
			x[9] = X[X21];
			x[10] = X[X22];

			x[1] = (ly0 ? X[X01] : 0);
			x[2] = (ly0 ? X[X02] : 0);
			x[13] = (ly3 ? X[X31] : 0);
			x[14] = (ly3 ? X[X32] : 0);

			x[4] = (lx0 ? X[X10] : 0);
			x[8] = (lx0 ? X[X20] : 0);
			x[7] = (lx3 ? X[X13] : 0);
			x[11] = (lx3 ? X[X23] : 0);

			x[0] = (ly0 && lx0 ? X[X00] : 0);
			x[3] = (ly0 && lx3 ? X[X03] : 0);
			x[12] = (ly3 && lx0 ? X[X30] : 0);
			x[15] = (ly3 && lx3 ? X[X33] : 0);

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
	int yn, oh, ow; {
		int j = bj + XIdx + (ux >> 1);

		//======[n, ih, iw]===========================
		//yn = j / THW; int jr = j - yn * THW;
		//int th = jr / TW, tw = jr - th * TW;

		//=======[ih, iw, n]=============================
		int th = j / TWN, jr = j % TWN;
		int tw = jr / N; yn = jr % N;

		//int th = yj / TWN, jr = yj % TWN;
		//int tw = jr / N, yn = jr %  N;

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


//[n, ih, iw, ic] -> [ic, n, ih, iw]
//Size = 9, Time = 1.55917 msec, Performace = 12395.9 GFlop/s
#define C3D_WG2D_KERNEL9
#ifndef C3D_WG2D_KERNEL9
#define C3D_WG2D_KERNEL9

#define c3d_wg2d_k9(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel9\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 1.53425 msec, Performace = 12597.3 GFlop/s
__global__ void c3d_wg2d_kernel9(
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

			x[5] = X[X11];
			x[6] = X[X12];
			x[9] = X[X21];
			x[10] = X[X22];

			x[1] = (ly0 ? X[X01] : 0);
			x[2] = (ly0 ? X[X02] : 0);
			x[13] = (ly3 ? X[X31] : 0);
			x[14] = (ly3 ? X[X32] : 0);

			x[4] = (lx0 ? X[X10] : 0);
			x[8] = (lx0 ? X[X20] : 0);
			x[7] = (lx3 ? X[X13] : 0);
			x[11] = (lx3 ? X[X23] : 0);

			x[0] = (ly0 && lx0 ? X[X00] : 0);
			x[3] = (ly0 && lx3 ? X[X03] : 0);
			x[12] = (ly3 && lx0 ? X[X30] : 0);
			x[15] = (ly3 && lx3 ? X[X33] : 0);

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


//[n, ih, iw, ic] -> [ic, ih, iw, n]
#define C3D_WG2D_KERNEL10
#ifndef C3D_WG2D_KERNEL10
#define C3D_WG2D_KERNEL10

#define c3d_wg2d_k10(stream, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw)\
	c3d_wg2d_kernel10\
		<<< dim3((GN>>5), (GM>>5)), dim3(32, 8), 0, stream >>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, oc_index, j_index)\

//Size = 9, Time = 2.99124 msec, Performace = 6461.31 GFlop/s
__global__ void c3d_wg2d_kernel10(
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

	//[ic, ih, iw, n]
	const int xoffset = (ih*IW + iw)*N + xn;
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

			//[ic, ih, iw, n]
			const int X00 = xoffset + ic * IH * IW * N;
			const int X01 = X00 + N, X02 = X00 + (IC << 1), X03 = X00 + IC * 3;
			const int X10 = X00 + IW * N, X11 = X10 + N, X12 = X10 + (N << 1), X13 = X10 + N * 3;
			const int X20 = X10 + IW * N, X21 = X20 + N, X22 = X20 + (N << 1), X23 = X20 + N * 3;
			const int X30 = X20 + IW * N, X31 = X30 + N, X32 = X30 + (N << 1), X33 = X30 + N * 3;

			x[5] = X[X11];
			x[6] = X[X12];
			x[9] = X[X21];
			x[10] = X[X22];

			x[1] = (ly0 ? X[X01] : 0);
			x[2] = (ly0 ? X[X02] : 0);
			x[13] = (ly3 ? X[X31] : 0);
			x[14] = (ly3 ? X[X32] : 0);

			x[4] = (lx0 ? X[X10] : 0);
			x[8] = (lx0 ? X[X20] : 0);
			x[7] = (lx3 ? X[X13] : 0);
			x[11] = (lx3 ? X[X23] : 0);

			x[0] = (ly0 && lx0 ? X[X00] : 0);
			x[3] = (ly0 && lx3 ? X[X03] : 0);
			x[12] = (ly3 && lx0 ? X[X30] : 0);
			x[15] = (ly3 && lx3 ? X[X33] : 0);

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


