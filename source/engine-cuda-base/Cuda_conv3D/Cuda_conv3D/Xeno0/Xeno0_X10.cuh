

//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef PPX_KERNEL1
#define PPX_KERNEL1

#define ppx_k1(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	PPX_Kernel1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//LB = 4: Size = 9, Time = 2.48976 msec, Performace = 7762.74 GFlop/s
template<int LB, int STEP>
__global__ void PPX_Kernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc1 = (oc0 + ((ty >= STEP) << 2) + 1) * GK;
	W += toc1;//W[toc1, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;
	const int sw_IC = sw << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx - ((tx >= STEP) << LB >> 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float4 xv;
	xv.x = (lx0 ? X[xoffset - sw_IC] : 0);
	xv.y = (lx1 ? X[xoffset] : 0);
	xv.z = (lx2 ? X[xoffset + sw_IC] : 0);
	xv.w = (lx3 ? X[xoffset + (sw_IC << 1)] : 0);
	Xs[buf][tx][ty] = xv;

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_k = ty - ((ty >= STEP) << LB >> 1);
	wv.x = W[W_k - GK];//W0
	wv.y = W[W_k];//W1
	wv.z = W[W_k + GK];//W2
	wv.w = W[W_k + (GK << 1)];//W3
	Ws[buf][ty][tx] = wv;
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float4 xv;
		xv.x = (lx0 ? X[xoffset - sw_IC] : 0);
		xv.y = (lx1 ? X[xoffset] : 0);
		xv.z = (lx2 ? X[xoffset + sw_IC] : 0);
		xv.w = (lx3 ? X[xoffset + (sw_IC << 1)] : 0);
		Xs[buf][tx][ty] = xv;

		//load 4 elements from W[OC, FH, FW, IC]
		float4 wv; int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		wv.x = W[W_k - GK];//W0
		wv.y = W[W_k];//W1
		wv.z = W[W_k + GK];//W2
		wv.w = W[W_k + (GK << 1)];//W3
		Ws[buf][ty][tx] = wv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef PPX_KERNEL2
#define PPX_KERNEL2

#define ppx_k2(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	PPX_Kernel2<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//LB = 4: Size = 9, Time = 2.23396 msec, Performace = 8651.59 GFlop/s
template<int LB, int STEP>
__global__ void PPX_Kernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	int toc1 = (oc0 + ((ty >= STEP) << 2) + 1) * GK;
	W += toc1;//W[toc1, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw, tow2 = tow1 + sw, tow3 = tow2 + sw;
	const int sw_IC = sw << LIC;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = X0 + sw_IC;
	const int X2 = X1 + sw_IC;
	const int X3 = X2 + sw_IC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = tx & (STEP - 1);
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float4 xv;
	xv.x = (lx0 ? X[xoffset + X0] : 0);
	xv.y = (lx1 ? X[xoffset + X1] : 0);
	xv.z = (lx2 ? X[xoffset + X2] : 0);
	xv.w = (lx3 ? X[xoffset + X3] : 0);
	Xs[buf][tx][ty] = xv;

	//load 4 elements from W[OC, FH, FW, IC]
	float4 wv; int W_k = (ty & STEP - 1);
	wv.x = W[W_k - GK];//W0
	wv.y = W[W_k];//W1
	wv.z = W[W_k + GK];//W2
	wv.w = W[W_k + (GK << 1)];//W3
	Ws[buf][ty][tx] = wv;
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB >> 1) + (tx & (STEP - 1));
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float4 xv;
		xv.x = (lx0 ? X[xoffset + X0] : 0);
		xv.y = (lx1 ? X[xoffset + X1] : 0);
		xv.z = (lx2 ? X[xoffset + X2] : 0);
		xv.w = (lx3 ? X[xoffset + X3] : 0);
		Xs[buf][tx][ty] = xv;

		//load 4 elements from W[OC, FH, FW, IC]
		float4 wv; int W_k = (ok << LB >> 1) + (ty & (STEP - 1));
		wv.x = W[W_k - GK];//W0
		wv.y = W[W_k];//W1
		wv.z = W[W_k + GK];//W2
		wv.w = W[W_k + (GK << 1)];//W3
		Ws[buf][ty][tx] = wv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];
		float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef PPX_KERNEL3
#define PPX_KERNEL3

#define ppx_k3(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	PPX_Kernel3<LB, (1<<LB>>2), (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//LB = 4: Size = 9, Time = 2.98428 msec, Performace = 6476.39 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void PPX_Kernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	const int toc0 = (oc0 + ((ty >= STEP2) << 2) + ((ty & 1) << 1)) * GK;
	const int toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP2) << 2) + ((tx & 1) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph; tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = X0 + (sw << LIC);

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & (STEP2 - 1)) >> 1;
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	float2 xv;
	xv.x = (lx0 ? X[xoffset + X0] : 0);
	xv.y = (lx1 ? X[xoffset + X1] : 0);
	Xs[buf][tx >> 1][(ty << 1) + (tx & 1)] = xv;//with the same ty

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = ((ty & (STEP2 - 1))) >> 1;
	float2 wv; 
	wv.x = W[W_k + toc0];
	wv.y = W[W_k + toc1];
	Ws[buf][ty >> 1][(tx << 1) + (ty & 1)] = wv;//with the same tx
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 2 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik       ][ty << 1]);
			float4 b1 = *(float4*)(&Xs[buf][ik + STEP][ty << 1]);
			float4 a0 = *(float4*)(&Ws[buf][ik       ][tx << 1]);
			float4 a1 = *(float4*)(&Ws[buf][ik + STEP][tx << 1]);

			simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB >> 2) + ((tx & (STEP2 - 1)) >> 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		float2 xv;
		xv.x = (lx0 ? X[xoffset + X0] : 0);
		xv.y = (lx1 ? X[xoffset + X1] : 0);
		Xs[buf][tx >> 1][(ty << 1) + (tx & 1)] = xv;//with the same ty

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (ok << LB >> 2) + (((ty & (STEP2 - 1))) >> 1);
		float2 wv; 
		wv.x = W[W_k + toc0];//W0
		wv.y = W[W_k + toc1];//W1
		Ws[buf][ty >> 1][(tx << 1) + (ty & 1)] = wv;//with the same tx
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]), a1 = *(float4*)(&Ws[buf][ik + STEP][tx << 1]);
		float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]), b1 = *(float4*)(&Xs[buf][ik + STEP][ty << 1]);

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif



//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef PPX_KERNEL4
#define PPX_KERNEL4

#define ppx_k4(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	PPX_Kernel4<LB, (1<<LB>>2), (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//LB = 4: Size = 9, Time = 2.29123 msec, Performace = 8435.37 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void PPX_Kernel4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB][(2 << LB) + 2];

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	const int toc0 = (oc0 + ((ty >= STEP2) << 2) + ((ty & 1) << 1)) * GK;
	const int toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP2) << 2) + ((tx & 1) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph; tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = X0 + (sw << LIC);

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & (STEP2 - 1)) >> 1 << 1;
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
	Xs[buf][(tx >> 1 << 1)    ][(ty << 1) + (tx & 1)] = float2{ x0.x, x1.x };//with the same ty
	Xs[buf][(tx >> 1 << 1) + 1][(ty << 1) + (tx & 1)] = float2{ x0.y, x1.y };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty & (STEP2 - 1)) >> 1 << 1;
	float2 w0 = *(float2*)(W + W_k + toc0);
	float2 w1 = *(float2*)(W + W_k + toc1);
	Ws[buf][(ty >> 1 << 1)    ][(tx << 1) + (ty & 1)] = float2{ w0.x, w1.x };//with the same tx
	Ws[buf][(ty >> 1 << 1) + 1][(tx << 1) + (ty & 1)] = float2{ w0.y, w1.y };
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]), b1 = *(float4*)(&Xs[buf][ik + STEP2][ty << 1]);
			float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]), a1 = *(float4*)(&Ws[buf][ik + STEP2][tx << 1]);

			simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB >> 1) + ((tx & (STEP2 - 1)) >> 1 << 1);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
		Xs[buf][(tx >> 1 << 1)    ][(ty << 1) + (tx & 1)] = float2{ x0.x, x1.x };//with the same ty
		Xs[buf][(tx >> 1 << 1) + 1][(ty << 1) + (tx & 1)] = float2{ x0.y, x1.y };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (ok << LB >> 1) + ((ty & (STEP2 - 1)) >> 1 << 1);
		float2 w0 = *(float2*)(W + W_k + toc0);
		float2 w1 = *(float2*)(W + W_k + toc1);
		Ws[buf][(ty >> 1 << 1)    ][(tx << 1) + (ty & 1)] = float2{ w0.x, w1.x };//with the same tx
		Ws[buf][(ty >> 1 << 1) + 1][(tx << 1) + (ty & 1)] = float2{ w0.y, w1.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]), a1 = *(float4*)(&Ws[buf][ik + STEP2][tx << 1]);
		float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]), b1 = *(float4*)(&Xs[buf][ik + STEP2][ty << 1]);

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif



//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef PPX_KERNEL5
#define PPX_KERNEL5

#define ppx_k5(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	PPX_Kernel5<LB, (1<<LB>>2), (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//LB = 4: Size = 9, Time = 2.29123 msec, Performace = 8435.37 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4>
__global__ void PPX_Kernel5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][2 << LB][(2 << LB) + 2];
	__shared__ float2 Xs[2][2 << LB][(2 << LB) + 2];

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	const int toc0 = (oc0 + ((ty >= STEP2) << 2) + ((ty & 1) << 1)) * GK;
	const int toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP2) << 2) + ((tx & 1) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph; tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw;
	const int X0 = ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = X0 + (sw << LIC);

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & (STEP2 - 1)) >> 1 << 2;
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	float4 x0 = (lx0 ? *(float4*)(X + X0 + xoffset) : F32_4_0);
	float4 x1 = (lx1 ? *(float4*)(X + X1 + xoffset) : F32_4_0);
	Xs[buf][(tx >> 1 << 2)    ][(ty << 1) + (tx & 1)] = float2{ x0.x, x1.x };//with the same ty
	Xs[buf][(tx >> 1 << 2) + 1][(ty << 1) + (tx & 1)] = float2{ x0.y, x1.y };
	Xs[buf][(tx >> 1 << 2) + 2][(ty << 1) + (tx & 1)] = float2{ x0.z, x1.z };
	Xs[buf][(tx >> 1 << 2) + 3][(ty << 1) + (tx & 1)] = float2{ x0.w, x1.w };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty & (STEP2 - 1)) >> 1 << 2;
	float4 w0 = *(float4*)(W + W_k + toc0);
	float4 w1 = *(float4*)(W + W_k + toc1);
	Ws[buf][(ty >> 1 << 2)    ][(tx << 1) + (ty & 1)] = float2{ w0.x, w1.x };//with the same tx
	Ws[buf][(ty >> 1 << 2) + 1][(tx << 1) + (ty & 1)] = float2{ w0.y, w1.y };
	Ws[buf][(ty >> 1 << 2) + 2][(tx << 1) + (ty & 1)] = float2{ w0.z, w1.z };
	Ws[buf][(ty >> 1 << 2) + 3][(tx << 1) + (ty & 1)] = float2{ w0.w, w1.w };
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP4; ik++)
		{
			float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]), a1 = *(float4*)(&Ws[buf][ik + STEP4][tx << 1]);
			float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]), b1 = *(float4*)(&Xs[buf][ik + STEP4][ty << 1]);

			simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ((tx & (STEP2 - 1)) >> 1 << 2);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		float4 x0 = (lx0 ? *(float4*)(X + X0 + xoffset) : F32_4_0);
		float4 x1 = (lx1 ? *(float4*)(X + X1 + xoffset) : F32_4_0);
		Xs[buf][(tx >> 1 << 2)    ][(ty << 1) + (tx & 1)] = float2{ x0.x, x1.x };
		Xs[buf][(tx >> 1 << 2) + 1][(ty << 1) + (tx & 1)] = float2{ x0.y, x1.y };
		Xs[buf][(tx >> 1 << 2) + 2][(ty << 1) + (tx & 1)] = float2{ x0.z, x1.z };
		Xs[buf][(tx >> 1 << 2) + 3][(ty << 1) + (tx & 1)] = float2{ x0.w, x1.w };

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + ((ty & (STEP2 - 1)) >> 1 << 2);
		float4 w0 = *(float4*)(W + W_k + toc0);
		float4 w1 = *(float4*)(W + W_k + toc1);
		Ws[buf][(ty >> 1 << 2)    ][(tx << 1) + (ty & 1)] = float2{ w0.x, w1.x };
		Ws[buf][(ty >> 1 << 2) + 1][(tx << 1) + (ty & 1)] = float2{ w0.y, w1.y };
		Ws[buf][(ty >> 1 << 2) + 2][(tx << 1) + (ty & 1)] = float2{ w0.z, w1.z };
		Ws[buf][(ty >> 1 << 2) + 3][(tx << 1) + (ty & 1)] = float2{ w0.w, w1.w };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP4; ik++)
	{
		float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]), a1 = *(float4*)(&Ws[buf][ik + STEP4][tx << 1]);
		float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]), b1 = *(float4*)(&Xs[buf][ik + STEP4][ty << 1]);

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, IC is power of 2
//LB = 4, GK % 8 == 0 
#ifndef PPX_KERNEL6
#define PPX_KERNEL6
#define ppx_k6(stream, LB, oc_index, j_index, X, IH, IW, W, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	PPX_Kernel6<LB, (1<<LB>>2), (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, W, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//LB = 4: Size = 9, Time = 2.22563 msec, Performace = 8684.01 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4>
__global__ void PPX_Kernel6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][2 << LB][(2 << LB) + 2];
	__shared__ float2 Xs[2][2 << LB][(2 << LB) + 2];

	//prepare for GK = FH * FW * IC
	const int GK = 9 << LIC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	W += (oc0 + ((ty >= STEP2) << 2) + ((ty & 1) << 1)) * GK;//W[toc0, 0, 0, 0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP2) << 2) + ((tx & 1) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph; tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw;
	X += ((tn0*IH + toh0)*IW + tow0) << LIC;
	const int sw_IC = sw << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = (tx & (STEP2 - 1)) >> 1 << 2;
	int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	int X_fh = fhw >> 2, X_fw = fhw & 3;
	int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	float4 x0 = (lx0 ? *(float4*)(X + xoffset) : F32_4_0);
	float4 x1 = (lx1 ? *(float4*)(X + xoffset + sw_IC) : F32_4_0);
	Xs[buf][(tx >> 1 << 2)    ][(ty << 1) + (tx & 1)] = float2{ x0.x, x1.x };
	Xs[buf][(tx >> 1 << 2) + 1][(ty << 1) + (tx & 1)] = float2{ x0.y, x1.y };
	Xs[buf][(tx >> 1 << 2) + 2][(ty << 1) + (tx & 1)] = float2{ x0.z, x1.z };
	Xs[buf][(tx >> 1 << 2) + 3][(ty << 1) + (tx & 1)] = float2{ x0.w, x1.w };

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = (ty & (STEP2 - 1)) >> 1 << 2;
	float4 w0 = *(float4*)(W + W_k);
	float4 w1 = *(float4*)(W + W_k + GK);
	Ws[buf][(ty >> 1 << 2)    ][(tx << 1) + (ty & 1)] = float2{ w0.x, w1.x };
	Ws[buf][(ty >> 1 << 2) + 1][(tx << 1) + (ty & 1)] = float2{ w0.y, w1.y };
	Ws[buf][(ty >> 1 << 2) + 2][(tx << 1) + (ty & 1)] = float2{ w0.z, w1.z };
	Ws[buf][(ty >> 1 << 2) + 3][(tx << 1) + (ty & 1)] = float2{ w0.w, w1.w };
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP4; ik++)
		{
			float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]), b1 = *(float4*)(&Xs[buf][ik + STEP4][ty << 1]);
			float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]), a1 = *(float4*)(&Ws[buf][ik + STEP4][tx << 1]);

			simdMM4(v0, b0.x, a0);  simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0);  simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0);  simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0);  simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0);  simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + ((ty & (STEP2 - 1)) >> 1 << 2);
		float4 w0 = *(float4*)(W + W_k);
		float4 w1 = *(float4*)(W + W_k + GK);
		Ws[buf][(ty >> 1 << 2)][(tx << 1) + (ty & 1)] = float2{ w0.x, w1.x };
		Ws[buf][(ty >> 1 << 2) + 1][(tx << 1) + (ty & 1)] = float2{ w0.y, w1.y };
		Ws[buf][(ty >> 1 << 2) + 2][(tx << 1) + (ty & 1)] = float2{ w0.z, w1.z };
		Ws[buf][(ty >> 1 << 2) + 3][(tx << 1) + (ty & 1)] = float2{ w0.w, w1.w };

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ((tx & (STEP2 - 1)) >> 1 << 2);
		int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		int X_fh = fhw >> 2, X_fw = fhw & 3;
		int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		float4 x0 = (lx0 ? *(float4*)(X + xoffset) : F32_4_0);
		float4 x1 = (lx1 ? *(float4*)(X + xoffset + sw_IC) : F32_4_0);
		Xs[buf][(tx >> 1 << 2)][(ty << 1) + (tx & 1)] = float2{ x0.x, x1.x };
		Xs[buf][(tx >> 1 << 2) + 1][(ty << 1) + (tx & 1)] = float2{ x0.y, x1.y };
		Xs[buf][(tx >> 1 << 2) + 2][(ty << 1) + (tx & 1)] = float2{ x0.z, x1.z };
		Xs[buf][(tx >> 1 << 2) + 3][(ty << 1) + (tx & 1)] = float2{ x0.w, x1.w };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP4; ik++)
	{
		float4 a0 = *(float4*)(&Ws[buf][ik][tx << 1]), a1 = *(float4*)(&Ws[buf][ik + STEP4][tx << 1]);
		float4 b0 = *(float4*)(&Xs[buf][ik][ty << 1]), b1 = *(float4*)(&Xs[buf][ik + STEP4][ty << 1]);

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


