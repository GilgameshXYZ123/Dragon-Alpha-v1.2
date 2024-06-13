


//(Y: BLOCK_SIZE*16, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef VOP_KERNEL1
#define VOP_KERNEL1

#define vop_kernel1(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)


//LB = 4: Size = 1.125, Time = 0.769 msec, Performace = 3141.64 GFlop/s
//LB = 3: Size = 1.125, Time = 1.241 msec, Performace = 1946.75 GFlop/s
template<int LB, int STEP>
__global__ void VOP_Kernel1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(4 << LB) + 1];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 4) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0)*GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;
	const int toc4 = toc3 + GK, toc5 = toc4 + GK, toc6 = toc5 + GK, toc7 = toc6 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0) * sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0) * sw - pw;
	const int Xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 1;
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];
	Ws[buf][Ws_x][Ws_y].z = W[toc2 + W_k];
	Ws[buf][Ws_x][Ws_y].w = W[toc3 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].x = W[toc4 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].y = W[toc5 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].z = W[toc6 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].w = W[toc7 + W_k];

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int dX_k = ty >> 1, dX_fh, dX_fw; getX_fh_fw(dX_k, dX_fh, dX_fw);
	bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
	const int IW_IC = IW * IC;
	int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	//compute area-----------------------------------------
	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy02 = *(float4*)(deltaY + j0 + 8);
	float4 dy03 = *(float4*)(deltaY + j0 + 12);

	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);
	float4 dy12 = *(float4*)(deltaY + j1 + 8);
	float4 dy13 = *(float4*)(deltaY + j1 + 12);

	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 2)];
			float4 w1 = Ws[buf][ik][(ty << 2) + 1];
			float4 w2 = Ws[buf][ik][(ty << 2) + 2];
			float4 w3 = Ws[buf][ik][(ty << 2) + 3];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) +
				CASM4(w1, dy01) +
				CASM4(w2, dy02) +
				CASM4(w3, dy03);
			float dx1 = CASM4(w0, dy10) +
				CASM4(w1, dy11) +
				CASM4(w2, dy12) +
				CASM4(w3, dy13);

			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
		if (Xaddr != -1) {
			atomicAdd(deltaX + Xaddr, dXs[buf][Xs_y][Xs_x]);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];
		Ws[buf][Ws_x][Ws_y].z = W[toc2 + W_k];
		Ws[buf][Ws_x][Ws_y].w = W[toc3 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].x = W[toc4 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].y = W[toc5 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].z = W[toc6 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].w = W[toc7 + W_k];

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int dX_k = ((ok << LB) + ty) >> 1; getX_fh_fw(dX_k, dX_fh, dX_fw);
		bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
		int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 2)];
		float4 w1 = Ws[buf][ik][(ty << 2) + 1];
		float4 w2 = Ws[buf][ik][(ty << 2) + 2];
		float4 w3 = Ws[buf][ik][(ty << 2) + 3];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) +
			CASM4(w1, dy01) +
			CASM4(w2, dy02) +
			CASM4(w3, dy03);
		float dx1 = CASM4(w0, dy10) +
			CASM4(w1, dy11) +
			CASM4(w2, dy12) +
			CASM4(w3, dy13);
		dx0 *= (xoffset.x != -1);
		dx1 *= (xoffset.y != -1);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
	if (Xaddr != -1) {
		atomicAdd(deltaX + Xaddr, dXs[buf][Xs_y][Xs_x]);
	}
}

#endif


//(Y: BLOCK_SIZE*16, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef VOP_KERNEL2
#define VOP_KERNEL2

#define vop_kernel2(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)


//LB = 4:Size = 1.125, Time = 0.802 msec, Performace = 3012.37 GFlop/s
//LB = 3: Size = 1.125, Time = 1.241 msec, Performace = 1946.75 GFlop/s
template<int LB, int STEP>
__global__ void VOP_Kernel2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(4 << LB) + 1];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 4) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0)*GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;
	const int toc4 = toc3 + GK, toc5 = toc4 + GK, toc6 = toc5 + GK, toc7 = toc6 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0) * sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0) * sw - pw;
	const int Xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//compute area-------------------------------------------------
	const int IW_IC = IW * IC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 1;
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];
	Ws[buf][Ws_x][Ws_y].z = W[toc2 + W_k];
	Ws[buf][Ws_x][Ws_y].w = W[toc3 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].x = W[toc4 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].y = W[toc5 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].z = W[toc6 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].w = W[toc7 + W_k];

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int dX_k = ty >> 1, dX_fh, dX_fw; getX_fh_fw(dX_k, dX_fh, dX_fw);
	bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
	const int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	//compute area-----------------------------------------
	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy02 = *(float4*)(deltaY + j0 + 8);
	float4 dy03 = *(float4*)(deltaY + j0 + 12);

	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);
	float4 dy12 = *(float4*)(deltaY + j1 + 8);
	float4 dy13 = *(float4*)(deltaY + j1 + 12);

	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 2)];
			float4 w1 = Ws[buf][ik][(ty << 2) + 1];
			float4 w2 = Ws[buf][ik][(ty << 2) + 2];
			float4 w3 = Ws[buf][ik][(ty << 2) + 3];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) +
				CASM4(w1, dy01) +
				CASM4(w2, dy02) +
				CASM4(w3, dy03);
			float dx1 = CASM4(w0, dy10) +
				CASM4(w1, dy11) +
				CASM4(w2, dy12) +
				CASM4(w3, dy13);

			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)    ]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
		atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];
		Ws[buf][Ws_x][Ws_y].z = W[toc2 + W_k];
		Ws[buf][Ws_x][Ws_y].w = W[toc3 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].x = W[toc4 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].y = W[toc5 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].z = W[toc6 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].w = W[toc7 + W_k];

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int dX_k = ((ok << LB) + ty) >> 1; getX_fh_fw(dX_k, dX_fh, dX_fw);
		bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
		int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 2)];
		float4 w1 = Ws[buf][ik][(ty << 2) + 1];
		float4 w2 = Ws[buf][ik][(ty << 2) + 2];
		float4 w3 = Ws[buf][ik][(ty << 2) + 3];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) +
			CASM4(w1, dy01) +
			CASM4(w2, dy02) +
			CASM4(w3, dy03);
		float dx1 = CASM4(w0, dy10) +
			CASM4(w1, dy11) +
			CASM4(w2, dy12) +
			CASM4(w3, dy13);
		dx0 *= (xoffset.x != -1);
		dx1 *= (xoffset.y != -1);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
	atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
}

#endif


//(Y: BLOCK_SIZE*16, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef VOP_KERNEL3
#define VOP_KERNEL3

#define vop_kernel3(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel3<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)


//LB = 4: Size = 2.25, Time = 1.47 msec, Performace = 3286.96 GFlop/s
template<int LB, int STEP>
__global__ void VOP_Kernel3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(4 << LB) + 1];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 4) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0)*GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;
	const int toc4 = toc3 + GK, toc5 = toc4 + GK, toc6 = toc5 + GK, toc7 = toc6 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0) * sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0) * sw - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//compute area-------------------------------------------------
	const int IW_IC = IW * IC;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 1;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);

	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy02 = *(float4*)(deltaY + j0 + 8);
	float4 dy03 = *(float4*)(deltaY + j0 + 12);

	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);
	float4 dy12 = *(float4*)(deltaY + j1 + 8);
	float4 dy13 = *(float4*)(deltaY + j1 + 12);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	float wv0 = W[toc0 + W_k];
	float wv1 = W[toc1 + W_k];
	float wv2 = W[toc2 + W_k];
	float wv3 = W[toc3 + W_k];
	float wv4 = W[toc4 + W_k];
	float wv5 = W[toc5 + W_k];
	float wv6 = W[toc6 + W_k];
	float wv7 = W[toc7 + W_k];
	Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3};
	Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int X_k = (ty >> 1);
	int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
	bool write0 = (tih0 >= -X_fh) && (tih0 < IH - X_fh) && (tiw0 >= -X_fw) && (tiw0 < IW - X_fw);
	const int xoffset = X0 + (X_fh*IW_IC) + X_k;
	
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 2)];
			float4 w1 = Ws[buf][ik][(ty << 2) + 1];
			float4 w2 = Ws[buf][ik][(ty << 2) + 2];
			float4 w3 = Ws[buf][ik][(ty << 2) + 3];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) +
				CASM4(w1, dy01) +
				CASM4(w2, dy02) +
				CASM4(w3, dy03);
			float dx1 = CASM4(w0, dy10) +
				CASM4(w1, dy11) +
				CASM4(w2, dy12) +
				CASM4(w3, dy13);

			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
		atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		float wv0 = W[toc0 + W_k];
		float wv1 = W[toc1 + W_k];
		float wv2 = W[toc2 + W_k];
		float wv3 = W[toc3 + W_k];
		float wv4 = W[toc4 + W_k];
		float wv5 = W[toc5 + W_k];
		float wv6 = W[toc6 + W_k];
		float wv7 = W[toc7 + W_k];
		Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
		Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
		bool write0 = (tih0 >= -X_fh) && (tih0 < IH - X_fh) && (tiw0 >= -X_fw) && (tiw0 < IW - X_fw);
		int xoffset = X0 + (X_fh*IW_IC) + X_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 2)];
		float4 w1 = Ws[buf][ik][(ty << 2) + 1];
		float4 w2 = Ws[buf][ik][(ty << 2) + 2];
		float4 w3 = Ws[buf][ik][(ty << 2) + 3];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) +
			CASM4(w1, dy01) +
			CASM4(w2, dy02) +
			CASM4(w3, dy03);
		float dx1 = CASM4(w0, dy10) +
			CASM4(w1, dy11) +
			CASM4(w2, dy12) +
			CASM4(w3, dy13);
		dx0 *= (xoffset.x != -1);
		dx1 *= (xoffset.y != -1);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
	atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
}

#endif


//(Y: BLOCK_SIZE*16, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef VOP_KERNEL4
#define VOP_KERNEL4

#define vop_kernel4(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,(OH*OW),OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)


//LB = 4: Size = 2.25, Time = 1.441 msec, Performace = 3353.11 GFlop/s
template<int LB, int STEP>
__global__ void VOP_Kernel4(
	const float* __restrict__ deltaY, int OH_OW, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(4 << LB) + 1];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 4) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0)*GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;
	const int toc4 = toc3 + GK, toc5 = toc4 + GK, toc6 = toc5 + GK, toc7 = toc6 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0)*sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0)*sw - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//compute area-------------------------------------------------
	const int IW_IC = IW * IC;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 1;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);

	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy02 = *(float4*)(deltaY + j0 + 8);
	float4 dy03 = *(float4*)(deltaY + j0 + 12);

	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);
	float4 dy12 = *(float4*)(deltaY + j1 + 8);
	float4 dy13 = *(float4*)(deltaY + j1 + 12);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	float wv0 = W[toc0 + W_k];
	float wv1 = W[toc1 + W_k];
	float wv2 = W[toc2 + W_k];
	float wv3 = W[toc3 + W_k];
	float wv4 = W[toc4 + W_k];
	float wv5 = W[toc5 + W_k];
	float wv6 = W[toc6 + W_k];
	float wv7 = W[toc7 + W_k];
	Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
	Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
	bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
	const int xoffset = X0 + X_fh * IW_IC + X_k;
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 2)];
			float4 w1 = Ws[buf][ik][(ty << 2) + 1];
			float4 w2 = Ws[buf][ik][(ty << 2) + 2];
			float4 w3 = Ws[buf][ik][(ty << 2) + 3];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
			float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);
			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
		atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		float wv0 = W[toc0 + W_k];
		float wv1 = W[toc1 + W_k];
		float wv2 = W[toc2 + W_k];
		float wv3 = W[toc3 + W_k];
		float wv4 = W[toc4 + W_k];
		float wv5 = W[toc5 + W_k];
		float wv6 = W[toc6 + W_k];
		float wv7 = W[toc7 + W_k];
		Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
		Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
		bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
		int xoffset = X0 + X_fh * IW_IC + X_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 2)];
		float4 w1 = Ws[buf][ik][(ty << 2) + 1];
		float4 w2 = Ws[buf][ik][(ty << 2) + 2];
		float4 w3 = Ws[buf][ik][(ty << 2) + 3];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
		float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);
		dx0 *= (xoffset.x != -1);
		dx1 *= (xoffset.y != -1);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
	atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
}

#endif


//(Y: BLOCK_SIZE*16, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0//The Best
#ifndef VOP_KERNEL5
#define VOP_KERNEL5

#define vop_kernel5(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel5<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,(OH*OW),OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)


//LB = 4: Size = 2.25, Time = 1.441 msec, Performace = 3353.11 GFlop/s
template<int LB, int STEP>
__global__ void VOP_Kernel5(
	const float* __restrict__ deltaY, int OH_OW, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(4 << LB) + 1];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 4) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0)*GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;
	const int toc4 = toc3 + GK, toc5 = toc4 + GK, toc6 = toc5 + GK, toc7 = toc6 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0)*sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0)*sw - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//compute area-------------------------------------------------
	const int IW_IC = IW * IC;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 1;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);

	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy02 = *(float4*)(deltaY + j0 + 8);
	float4 dy03 = *(float4*)(deltaY + j0 + 12);

	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);
	float4 dy12 = *(float4*)(deltaY + j1 + 8);
	float4 dy13 = *(float4*)(deltaY + j1 + 12);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	float wv0 = W[toc0 + W_k];
	float wv1 = W[toc1 + W_k];
	float wv2 = W[toc2 + W_k];
	float wv3 = W[toc3 + W_k];
	float wv4 = W[toc4 + W_k];
	float wv5 = W[toc5 + W_k];
	float wv6 = W[toc6 + W_k];
	float wv7 = W[toc7 + W_k];
	Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
	Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
	bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
	const int xoffset = X0 + X_fh * IW_IC + X_k;
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 2)];
			float4 w1 = Ws[buf][ik][(ty << 2) + 1];
			float4 w2 = Ws[buf][ik][(ty << 2) + 2];
			float4 w3 = Ws[buf][ik][(ty << 2) + 3];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
			float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);
			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
		atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		float wv0 = W[toc0 + W_k];
		float wv1 = W[toc1 + W_k];
		float wv2 = W[toc2 + W_k];
		float wv3 = W[toc3 + W_k];
		float wv4 = W[toc4 + W_k];
		float wv5 = W[toc5 + W_k];
		float wv6 = W[toc6 + W_k];
		float wv7 = W[toc7 + W_k];
		Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
		Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
		bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
		int xoffset = X0 + X_fh * IW_IC + X_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 2)];
		float4 w1 = Ws[buf][ik][(ty << 2) + 1];
		float4 w2 = Ws[buf][ik][(ty << 2) + 2];
		float4 w3 = Ws[buf][ik][(ty << 2) + 3];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
		float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);
		dx0 *= (xoffset.x != -1);
		dx1 *= (xoffset.y != -1);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
	atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
}

#endif


//(Y: BLOCK_SIZE*16, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef VOP_KERNEL6
#define VOP_KERNEL6

#define vop_kernel6(stream, LB, oc_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel6<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,(OH*OW),OW, CW,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)


//LB = 4: Size = 2.25, Time = 1.441 msec, Performace = 3353.11 GFlop/s
template<int LB, int STEP>
__global__ void VOP_Kernel6(
	const float* __restrict__ deltaY, int OH_OW, int OW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(4 << LB) + 1];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 4) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0);
	const int toc1 = toc0, toc4 = toc0 + 4;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0)*sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0)*sw - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//compute area-------------------------------------------------
	const int IW_IC = IW * IC;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 1;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);

	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy02 = *(float4*)(deltaY + j0 + 8);
	float4 dy03 = *(float4*)(deltaY + j0 + 12);

	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);
	float4 dy12 = *(float4*)(deltaY + j1 + 8);
	float4 dy13 = *(float4*)(deltaY + j1 + 12);

	//load 4 elements from CW[FH, FW, IC, OC]
	int W_k = tx >> 1;
	const int woffset = W_k * OC;
	Ws[buf][Ws_x][Ws_y] = *(float4*)(CW + toc0 + woffset);
	Ws[buf][Ws_x][Ws_y + 1] = *(float4*)(CW + toc4 + woffset);

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
	bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
	const int xoffset = X0 + X_fh * IW_IC + X_k;
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 2)];
			float4 w1 = Ws[buf][ik][(ty << 2) + 1];
			float4 w2 = Ws[buf][ik][(ty << 2) + 2];
			float4 w3 = Ws[buf][ik][(ty << 2) + 3];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
			float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);
			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
		atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		const int woffset = W_k * OC;
		Ws[buf][Ws_x][Ws_y] = *(float4*)(CW + toc0 + woffset);
		Ws[buf][Ws_x][Ws_y + 1] = *(float4*)(CW + toc4 + woffset);

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
		bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
		int xoffset = X0 + X_fh * IW_IC + X_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 2)];
		float4 w1 = Ws[buf][ik][(ty << 2) + 1];
		float4 w2 = Ws[buf][ik][(ty << 2) + 2];
		float4 w3 = Ws[buf][ik][(ty << 2) + 3];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
		float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);
		dx0 *= (xoffset.x != -1);
		dx1 *= (xoffset.y != -1);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
	atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
}

#endif


//(Y: BLOCK_SIZE*16, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef VOP_KERNEL7
#define VOP_KERNEL7

#define vop_kernel7(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel7<LB, (1<<LB>>1), FH, FW, FW*IC>\
		<<< dim3(GM>>LB>>1, GN>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,(OH*OW),OW, W, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)

//LB = 4: Size = 2.25, Time = 1.441 msec, Performace = 3353.11 GFlop/s
template<int LB, int STEP, int FH, int FW, int FW_IC>
__global__ void VOP_Kernel7(
	const float* __restrict__ deltaY, int OH_OW, int OW,
	const float* __restrict__ W,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(4 << LB) + 1];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 4) + oc_index;
	const int GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0)*GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;
	const int toc4 = toc3 + GK, toc5 = toc4 + GK, toc6 = toc5 + GK, toc7 = toc6 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0)*sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0)*sw - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//compute area-------------------------------------------------
	const int IW_IC = IW * IC;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 1;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);

	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy02 = *(float4*)(deltaY + j0 + 8);
	float4 dy03 = *(float4*)(deltaY + j0 + 12);

	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);
	float4 dy12 = *(float4*)(deltaY + j1 + 8);
	float4 dy13 = *(float4*)(deltaY + j1 + 12);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	float wv0 = W[toc0 + W_k];
	float wv1 = W[toc1 + W_k];
	float wv2 = W[toc2 + W_k];
	float wv3 = W[toc3 + W_k];
	float wv4 = W[toc4 + W_k];
	float wv5 = W[toc5 + W_k];
	float wv6 = W[toc6 + W_k];
	float wv7 = W[toc7 + W_k];
	Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
	Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
	bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
	const int xoffset = X0 + X_fh * IW_IC + X_k;
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 2)];
			float4 w1 = Ws[buf][ik][(ty << 2) + 1];
			float4 w2 = Ws[buf][ik][(ty << 2) + 2];
			float4 w3 = Ws[buf][ik][(ty << 2) + 3];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
			float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);
			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
		atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		float wv0 = W[toc0 + W_k];
		float wv1 = W[toc1 + W_k];
		float wv2 = W[toc2 + W_k];
		float wv3 = W[toc3 + W_k];
		float wv4 = W[toc4 + W_k];
		float wv5 = W[toc5 + W_k];
		float wv6 = W[toc6 + W_k];
		float wv7 = W[toc7 + W_k];
		Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
		Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
		bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
		int xoffset = X0 + X_fh * IW_IC + X_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 2)];
		float4 w1 = Ws[buf][ik][(ty << 2) + 1];
		float4 w2 = Ws[buf][ik][(ty << 2) + 2];
		float4 w3 = Ws[buf][ik][(ty << 2) + 3];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
		float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);
		dx0 *= (xoffset.x != -1);
		dx1 *= (xoffset.y != -1);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
	atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
}

#endif


//(Y: BLOCK_SIZE*16, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0//The Best
#ifndef VOP_KERNEL8
#define VOP_KERNEL8

#define vop_kernel8(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel8<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,(OH*OW),OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)


//LB = 4: Size = 2.25, Time = 1.441 msec, Performace = 3353.11 GFlop/s
template<int LB, int STEP>
__global__ void VOP_Kernel8(
	const float* __restrict__ deltaY, int OH_OW, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(4 << LB) + 1];
	__shared__ float   Xs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 4) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0) * GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;
	const int toc4 = toc3 + GK, toc5 = toc4 + GK, toc6 = toc5 + GK, toc7 = toc6 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0)*sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0)*sw - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//compute area-------------------------------------------------
	const int IW_IC = IW * IC;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 1;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);

	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy02 = *(float4*)(deltaY + j0 + 8);
	float4 dy03 = *(float4*)(deltaY + j0 + 12);

	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);
	float4 dy12 = *(float4*)(deltaY + j1 + 8);
	float4 dy13 = *(float4*)(deltaY + j1 + 12);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	float wv0 = W[toc0 + W_k];
	float wv1 = W[toc1 + W_k];
	float wv2 = W[toc2 + W_k];
	float wv3 = W[toc3 + W_k];
	float wv4 = W[toc4 + W_k];
	float wv5 = W[toc5 + W_k];
	float wv6 = W[toc6 + W_k];
	float wv7 = W[toc7 + W_k];
	Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
	Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
	const int xoffset = X0 + X_fh * IW_IC + X_k;
	bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	Xs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 2)];
			float4 w1 = Ws[buf][ik][(ty << 2) + 1];
			float4 w2 = Ws[buf][ik][(ty << 2) + 2];
			float4 w3 = Ws[buf][ik][(ty << 2) + 3];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
			float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);
			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(Xs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(Xs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
		atomicAdd(deltaX + Xaddr * wrx, Xs[buf][Xs_y][Xs_x] * wrx);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		float wv0 = W[toc0 + W_k];
		float wv1 = W[toc1 + W_k];
		float wv2 = W[toc2 + W_k];
		float wv3 = W[toc3 + W_k];
		float wv4 = W[toc4 + W_k];
		float wv5 = W[toc5 + W_k];
		float wv6 = W[toc6 + W_k];
		float wv7 = W[toc7 + W_k];
		Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
		Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X0 + X_fh * IW_IC + X_k;
		bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		Xs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 2)];
		float4 w1 = Ws[buf][ik][(ty << 2) + 1];
		float4 w2 = Ws[buf][ik][(ty << 2) + 2];
		float4 w3 = Ws[buf][ik][(ty << 2) + 3];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
		float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);
		dx0 *= (xoffset.x != -1);
		dx1 *= (xoffset.y != -1);

		atomicAdd_block(&(Xs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(Xs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
	atomicAdd(deltaX + Xaddr * wrx, Xs[buf][Xs_y][Xs_x] * wrx);
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef VOP_KERNEL9
#define VOP_KERNEL9

#define vop_kernel9(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel9<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,(OH*OW),OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)

//LB = 4: IC % 8 == 0
//LB = 3: IC % 4 == 0
//LB = 4: Size = 2.25, Time = 2.139 msec, Performace = 2258.92 GFlop/s
template<int LB, int STEP>
__global__ void VOP_Kernel9(
	const float* __restrict__ deltaY, int OH_OW, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 2) + oc0)*GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0) * sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0) * sw - pw;
	const int Xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//compute area-------------------------------------------------
	const int IW_IC = IW * IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);

	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);

	//load 4 elements from W[OC, FH, FW, IC]	
	int W_k = tx >> 1;
	float wv0 = W[toc0 + W_k];
	float wv1 = W[toc1 + W_k];
	float wv2 = W[toc2 + W_k];
	float wv3 = W[toc3 + W_k];
	Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
	const int xoffset = Xoffset0 + (X_fh*IW_IC) + X_k;
	bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 1)];
			float4 w1 = Ws[buf][ik][(ty << 1) + 1];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01);
			float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11);
			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
		atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		float wv0 = W[toc0 + W_k];
		float wv1 = W[toc1 + W_k];
		float wv2 = W[toc2 + W_k];
		float wv3 = W[toc3 + W_k];
		Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
		const int xoffset = Xoffset0 + (X_fh*IW_IC) + X_k;
		bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 1)];
		float4 w1 = Ws[buf][ik][(ty << 1) + 1];
		int2 xaddr = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01);
		float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11);
		dx0 *= (xaddr.x != -1);
		dx1 *= (xaddr.y != -1);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
	atomicAdd(deltaX + Xaddr * wrx, dXs[buf][Xs_y][Xs_x] * wrx);
}

#endif


//(Y: BLOCK_SIZE*32, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0//The Best
#ifndef VOP_KERNEL10
#define VOP_KERNEL10

#define vop_kernel10(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel10<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>5), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,(OH*OW),OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)

//LB = 4: Size = 2.25, Time = 1.441 msec, Performace = 3353.11 GFlop/s
template<int LB, int STEP>
__global__ void VOP_Kernel10(
	const float* __restrict__ deltaY, int OH_OW, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(8 << LB) + 1];
	__shared__ float   Xs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 5) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0) * GK;
	const int toc1  = toc0  + GK, toc2  = toc1  + GK, toc3  = toc2  + GK;
	const int toc4  = toc3  + GK, toc5  = toc4  + GK, toc6  = toc5  + GK, toc7  = toc6  + GK;
	const int toc8  = toc7  + GK, toc9  = toc8  + GK, toc10 = toc9  + GK, toc11 = toc10 + GK;
	const int toc12 = toc10 + GK, toc13 = toc12 + GK, toc14 = toc13 + GK, toc15 = toc14 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0)*sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0)*sw - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//compute area-------------------------------------------------
	const int IW_IC = IW * IC;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 2;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);

	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy02 = *(float4*)(deltaY + j0 + 8);
	float4 dy03 = *(float4*)(deltaY + j0 + 12);
	float4 dy04 = *(float4*)(deltaY + j0 + 16);
	float4 dy05 = *(float4*)(deltaY + j0 + 20);
	float4 dy06 = *(float4*)(deltaY + j0 + 24);
	float4 dy07 = *(float4*)(deltaY + j0 + 28);

	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);
	float4 dy12 = *(float4*)(deltaY + j1 + 8);
	float4 dy13 = *(float4*)(deltaY + j1 + 12);
	float4 dy14 = *(float4*)(deltaY + j1 + 16);
	float4 dy15 = *(float4*)(deltaY + j1 + 20);
	float4 dy16 = *(float4*)(deltaY + j1 + 24);
	float4 dy17 = *(float4*)(deltaY + j1 + 28);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	float wv0  = W[toc0  + W_k];
	float wv1  = W[toc1  + W_k];
	float wv2  = W[toc2  + W_k];
	float wv3  = W[toc3  + W_k];
	float wv4  = W[toc4  + W_k];
	float wv5  = W[toc5  + W_k];
	float wv6  = W[toc6  + W_k];
	float wv7  = W[toc7  + W_k];
	float wv8  = W[toc8  + W_k];
	float wv9  = W[toc9  + W_k];
	float wv10 = W[toc10 + W_k];
	float wv11 = W[toc11 + W_k];
	float wv12 = W[toc12 + W_k];
	float wv13 = W[toc13 + W_k];
	float wv14 = W[toc14 + W_k];
	float wv15 = W[toc15 + W_k];
	Ws[buf][Ws_x][Ws_y    ] = float4{  wv0,  wv1,  wv2,  wv3 };
	Ws[buf][Ws_x][Ws_y + 1] = float4{  wv4,  wv5,  wv6,  wv7 };
	Ws[buf][Ws_x][Ws_y + 2] = float4{  wv8,  wv9, wv10, wv11 };
	Ws[buf][Ws_x][Ws_y + 3] = float4{ wv12, wv13, wv14, wv15 };

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
	const int xoffset = X0 + X_fh * IW_IC + X_k;
	bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	Xs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 3)];
			float4 w1 = Ws[buf][ik][(ty << 3) + 1];
			float4 w2 = Ws[buf][ik][(ty << 3) + 2];
			float4 w3 = Ws[buf][ik][(ty << 3) + 3];
			float4 w4 = Ws[buf][ik][(ty << 3) + 4];
			float4 w5 = Ws[buf][ik][(ty << 3) + 5];
			float4 w6 = Ws[buf][ik][(ty << 3) + 6];
			float4 w7 = Ws[buf][ik][(ty << 3) + 7];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03) +
						CASM4(w4, dy04) + CASM4(w5, dy05) + CASM4(w6, dy06) + CASM4(w7, dy07);
			float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13) + 
						CASM4(w4, dy14) + CASM4(w5, dy15) + CASM4(w6, dy16) + CASM4(w7, dy17);
			
			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(Xs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(Xs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
		atomicAdd(deltaX + Xaddr * wrx, Xs[buf][Xs_y][Xs_x] * wrx);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		float wv0  = W[toc0  + W_k];
		float wv1  = W[toc1  + W_k];
		float wv2  = W[toc2  + W_k];
		float wv3  = W[toc3  + W_k];
		float wv4  = W[toc4  + W_k];
		float wv5  = W[toc5  + W_k];
		float wv6  = W[toc6  + W_k];
		float wv7  = W[toc7  + W_k];
		float wv8  = W[toc8  + W_k];
		float wv9  = W[toc9  + W_k];
		float wv10 = W[toc10 + W_k];
		float wv11 = W[toc11 + W_k];
		float wv12 = W[toc12 + W_k];
		float wv13 = W[toc13 + W_k];
		float wv14 = W[toc14 + W_k];
		float wv15 = W[toc15 + W_k];
		Ws[buf][Ws_x][Ws_y    ] = float4{  wv0,  wv1,  wv2,  wv3 };
		Ws[buf][Ws_x][Ws_y + 1] = float4{  wv4,  wv5,  wv6,  wv7 };
		Ws[buf][Ws_x][Ws_y + 2] = float4{  wv8,  wv9, wv10, wv11 };
		Ws[buf][Ws_x][Ws_y + 3] = float4{ wv12, wv13, wv14, wv15 };

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X0 + X_fh * IW_IC + X_k;
		bool write0 = WRITE_X(tih0, tiw0, X_fh, X_fw);
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		Xs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 3)];
		float4 w1 = Ws[buf][ik][(ty << 3) + 1];
		float4 w2 = Ws[buf][ik][(ty << 3) + 2];
		float4 w3 = Ws[buf][ik][(ty << 3) + 3];
		float4 w4 = Ws[buf][ik][(ty << 3) + 4];
		float4 w5 = Ws[buf][ik][(ty << 3) + 5];
		float4 w6 = Ws[buf][ik][(ty << 3) + 6];
		float4 w7 = Ws[buf][ik][(ty << 3) + 7];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03) +
					CASM4(w4, dy04) + CASM4(w5, dy05) + CASM4(w6, dy06) + CASM4(w7, dy07);
		float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13) + 
					CASM4(w4, dy14) + CASM4(w5, dy15) + CASM4(w6, dy16) + CASM4(w7, dy17);

		dx0 *= (xoffset.x != -1);
		dx1 *= (xoffset.y != -1);

		atomicAdd_block(&(Xs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(Xs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
	atomicAdd(deltaX + Xaddr * wrx, Xs[buf][Xs_y][Xs_x] * wrx);
}

#endif


//(Y: BLOCK_SIZE*16, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0//The Best
#ifndef VOP_KERNEL11
#define VOP_KERNEL11

#define vop_kernel11(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	VOP_Kernel11<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,(OH*OW),OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)

//FH = FW = 1
//LB = 4: Size = 0.5, Time = 0.508 msec, Performace = 2113.66 GFlop/s
//FH = FW = 3
//LB = 4: Size = 2.25, Time = 1.441 msec, Performace = 3353.11 GFlop/s
template<int LB, int STEP>
__global__ void VOP_Kernel11(
	const float* __restrict__ deltaY, int OH_OW, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(4 << LB) + 1];
	__shared__ float   Xs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 4) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0) * GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;
	const int toc4 = toc3 + GK, toc5 = toc4 + GK, toc6 = toc5 + GK, toc7 = toc6 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0)*sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0)*sw - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//compute area-------------------------------------------------
	const int IW_IC = IW * IC;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 1;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);

	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(deltaY + j0);
	float4 dy01 = *(float4*)(deltaY + j0 + 4);
	float4 dy02 = *(float4*)(deltaY + j0 + 8);
	float4 dy03 = *(float4*)(deltaY + j0 + 12);

	float4 dy10 = *(float4*)(deltaY + j1);
	float4 dy11 = *(float4*)(deltaY + j1 + 4);
	float4 dy12 = *(float4*)(deltaY + j1 + 8);
	float4 dy13 = *(float4*)(deltaY + j1 + 12);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	float wv0 = W[toc0 + W_k];
	float wv1 = W[toc1 + W_k];
	float wv2 = W[toc2 + W_k];
	float wv3 = W[toc3 + W_k];
	float wv4 = W[toc4 + W_k];
	float wv5 = W[toc5 + W_k];
	float wv6 = W[toc6 + W_k];
	float wv7 = W[toc7 + W_k];
	Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
	Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int X_k = ty >> 1;
	int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
	const int xoffset = X0 + X_fh * IW_IC + X_k;
	Xaddrs[buf][Xs_y][Xs_x] = xoffset;
	Xs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 2)];
			float4 w1 = Ws[buf][ik][(ty << 2) + 1];
			float4 w2 = Ws[buf][ik][(ty << 2) + 2];
			float4 w3 = Ws[buf][ik][(ty << 2) + 3];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
			float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);

			atomicAdd_block(&(Xs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(Xs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x]; bool wrx = (Xaddr != -1);
		atomicAdd(deltaX + Xaddr, Xs[buf][Xs_y][Xs_x]);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		float wv0 = W[toc0 + W_k];
		float wv1 = W[toc1 + W_k];
		float wv2 = W[toc2 + W_k];
		float wv3 = W[toc3 + W_k];
		float wv4 = W[toc4 + W_k];
		float wv5 = W[toc5 + W_k];
		float wv6 = W[toc6 + W_k];
		float wv7 = W[toc7 + W_k];
		Ws[buf][Ws_x][Ws_y] = float4{ wv0, wv1, wv2, wv3 };
		Ws[buf][Ws_x][Ws_y + 1] = float4{ wv4, wv5, wv6, wv7 };

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int X_k = ((ok << LB) + ty) >> 1;
		int X_fh, X_fw; getX_fh_fw(X_k, X_fh, X_fw);
		const int xoffset = X0 + X_fh * IW_IC + X_k;
		Xaddrs[buf][Xs_y][Xs_x] = xoffset;
		Xs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 2)];
		float4 w1 = Ws[buf][ik][(ty << 2) + 1];
		float4 w2 = Ws[buf][ik][(ty << 2) + 2];
		float4 w3 = Ws[buf][ik][(ty << 2) + 3];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CASM4(w0, dy00) + CASM4(w1, dy01) + CASM4(w2, dy02) + CASM4(w3, dy03);
		float dx1 = CASM4(w0, dy10) + CASM4(w1, dy11) + CASM4(w2, dy12) + CASM4(w3, dy13);

		atomicAdd_block(&(Xs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(Xs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
	atomicAdd(deltaX + Xaddr, Xs[buf][Xs_y][Xs_x]);
}

#endif
