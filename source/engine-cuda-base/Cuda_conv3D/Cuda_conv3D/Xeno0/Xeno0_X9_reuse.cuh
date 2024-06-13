

//(OH, OW) % 4 == 0
//sh = sw = 1;
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE/2) == 0
//LB = 4, IC % 8 == 0 
#ifndef PXKERNEL14
#define PXKERNEL14

#define pxkernel14(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	PXkernel14<LB, (1<<LB>>1), FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.52079 msec, Performace = 1588.6 GFlop/s(1000)
template<int LB, int STEP, int FH, int FW>
__global__ void PXkernel14(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++) {
		for (int fh = 0; fh < FH; fh++)//[fh, fw, STEP] sub from CW
		{
			//load 4 elements from CW[FH, FW, IC, OC]
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int woffset0 = ((fh*FW*IC) + wic)*OC;
			Ws[buf][ty][tx] = *(float4*)(CW + woffset0);

			//load 4 elements from X[N, IH, IW, IC]
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
			bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
			bool lx1 = lx && (tow1 >= 0) && (tow1 < IW);
			bool lx2 = lx && (tow2 >= 0) && (tow2 < IW);
			bool lx3 = lx && (tow3 >= 0) && (tow3 < IW);
			float4 xv; int xoffset0 = (fh*IW*IC) + xic;
			xv.x = (lx0 ? X[xoffset0 - IC] : 0);
			xv.y = (lx1 ? X[xoffset0] : 0);
			xv.z = (lx2 ? X[xoffset0 + IC] : 0);
			xv.w = (lx3 ? X[xoffset0 + (IC << 1)] : 0);
			Xs[buf][tx][ty] = xv;
			__syncthreads();

#pragma unroll
			for (int fw = 1; fw < FW; fw++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
					float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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
				float4 xv0 = Xs[buf ^ 1][tx][ty];
				bool lx3 = lx && (tow3 >= -fw) && (tow3 < IW - fw);
				int xoffset3 = xoffset0 + (fw + 2)*IC;
				float x3 = (lx3 ? X[xoffset3] : 0);
				Xs[buf][tx][ty] = { xv0.y, xv0.z, xv0.w,  x3 };

				//load 4 elements from CW[FH, FW, IC, OC]
				int woffset = woffset0 + (fw*IC*OC);
				Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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
			buf ^= 1;
		}
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


//1650
#ifndef QXKERNEL1
#define QXKERNEL1

#define qxkernel1(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	QXkernel1<LB, (1<<LB>>1), FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.5136 msec, Performace = 1596.14 GFlop/s
template<int LB, int STEP, int FH, int FW>
__global__ void QXkernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area-----------------------------------------------------------
	float4  v0 = F32_4_0,  v1 = F32_4_0; float4  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0; float4  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;
	for (int gk = 0, GK = (IC << 1 >> LB) * FH; gk < GK; gk++)//oic * FH
	{
		int oic = gk / FH, fh = gk - oic * FH; oic = (oic << LB >> 1);

		//load 4 elements from CW[FH, FW, IC, OC]
		int wic = (ty & (STEP - 1)) + oic;
		int woffset0 = ((fh*FW*IC) + wic)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset0);

		//load 4 elements from X[N, IH, IW, IC]
		int xic = (tx & (STEP - 1)) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >=  0) && (tow0 < IW    );
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		float4 xv; int xoffset0 = (fh*IW*IC) + xic;
		xv.x = (lx0 ? X[xoffset0 - IC] : 0);
		xv.y = (lx1 ? X[xoffset0] : 0);
		xv.z = (lx2 ? X[xoffset0 + IC] : 0);
		xv.w = (lx3 ? X[xoffset0 + (IC << 1)] : 0);
		Xs[buf][tx][ty] = xv;
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < FW; fw++) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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
			bool lx3 = lx && (tow0 >= -fw - 3) && (tow0 < IW - fw - 3);
			int xoffset3 = xoffset0 + (fw + 2)*IC;
			float x3 = (lx3 ? X[xoffset3] : 0);
			xv = { xv.y, xv.z, xv.w, x3 };//read last result
			Xs[buf][tx][ty] = xv;

			//load 4 elements from CW[FH, FW, IC, OC]
			int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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
		buf ^= 1;
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


#ifndef QXKERNEL2
#define QXKERNEL2

#define qxkernel2(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	QXkernel2<LB, (1<<LB>>1), (1<<LB), FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 1.125, Time = 1.5136 msec, Performace = 1596.14 GFlop/s
template<int LB, int STEP, int STEP2, int FH, int FW>
__global__ void QXkernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	int tow1 = tow0 + 1, tow2 = tow0 + 2, tow3 = tow0 + 3;
	X += ((tn0*IH + toh0)*IW + tow1)*IC;//X += X1;

	//compute area-----------------------------------------------------------
	float4  v0 = F32_4_0,  v1 = F32_4_0; float4  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0; float4  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;
	for (int gk = 0, GK = (IC >> LB) * FH; gk < GK; gk++)//oic * FH
	{
		int oic = gk / FH, fh = gk - oic * FH; oic = (oic << LB >> 1);

		 //load 4 elements from X[N, IH, IW, IC]
		int xic = ((tx & (STEP - 1)) + oic) << 1;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow1 >= 0) && (tow1 < IW);
		bool lx2 = lx && (tow2 >= 0) && (tow2 < IW);
		bool lx3 = lx && (tow3 >= 0) && (tow3 < IW);
		int xoffset0 = (fh*IW*IC) + xic;
		float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		float4 xv0 = float4{ x0.x, x1.x, x2.x, x3.x };
		float4 xv1 = float4{ x0.y, x1.y, x2.y, x3.y };
		Xs[buf][(tx << 1)][ty] = xv0;
		Xs[buf][(tx << 1) + 1][ty] = xv1;

		//load 4 elements from CW[FH, FW, IC, OC]
		int wic = ((ty & (STEP - 1)) + oic) << 1;
		int woffset0 = ((fh*FW*IC) + wic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < FW; fw++)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
			}
			for (int ik = STEP; ik < STEP2; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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

			//load 4 elements from CW[FH, FW, IC, OC]
			int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

			//load 4 elements from X[N, IH, IW, IC]
			bool lx3 = lx && (tow3 >= -fw) && (tow3 < IW - fw);
			int xoffset3 = xoffset0 + (fw + 2)*IC;
			float2 x3 = (lx3 ? *(float2*)(X + xoffset3) : F32_2_0);
			Xs[buf][(tx << 1)][ty] = xv0 = { xv0.y, xv0.z, xv0.w, x3.x };
			Xs[buf][(tx << 1) + 1][ty] = xv1 = { xv1.y, xv1.z, xv1.w, x3.y };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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


#ifndef QXKERNEL3
#define QXKERNEL3

#define qxkernel3(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	QXkernel3<LB, (1<<LB>>1), FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//target: Size = 9, Time = 1.93616 msec, Performace = 9982.32 GFlop/s
//LB = 4: Size = 9, Time = 1.90242 msec, Performace = 10159.3 GFlop/s
template<int LB, int STEP, int FH, int FW>
__global__ void QXkernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;

	//compute area-----------------------------------------------------------
	float4  v0 = F32_4_0,  v1 = F32_4_0; float4  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0; float4  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;
	for (int gk = 0, GK = (IC << 1 >> LB) * FH; gk < GK; gk++)//oic * FH
	{
		int oic = gk / FH, fh = gk - oic * FH; oic = (oic << LB >> 1);

		//load 4 elements from CW[FH, FW, IC, OC]
		int wic = (ty & (STEP - 1)) + oic;
		int woffset0 = ((fh*FW*IC) + wic)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset0);

		//load 4 elements from X[N, IH, IW, IC]
		int xic = (tx & (STEP - 1)) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		float4 xv; int xoffset0 = (fh*IW*IC) + xic;
		xv.x = (lx0 ? X[xoffset0 - IC] : 0);
		xv.y = (lx1 ? X[xoffset0] : 0);
		xv.z = (lx2 ? X[xoffset0 + IC] : 0);
		xv.w = (lx3 ? X[xoffset0 + (IC << 1)] : 0);
		Xs[buf][tx][ty] = xv;
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < FW; fw++) {
			//pragama unroll ik = 0-------------------------------	
			{
				float4 b0 = Xs[buf][0][ty], b1 = Xs[buf][STEP][ty];
				float4 a0 = Ws[buf][0][tx], a1 = Ws[buf][STEP][tx];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
			}
#pragma unroll 
			for (int ik = 1; ik < STEP; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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
			bool lx3 = lx && (tow0 >= -fw - 3) && (tow0 < IW - fw - 3);
			float x3 = (lx3 ? X[xoffset0 + (fw + 2)*IC] : 0);
			Xs[buf][tx][ty] = xv = { xv.y, xv.z, xv.w, x3 };//read last result

			//load 4 elements from CW[FH, FW, IC, OC]
			int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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
		buf ^= 1;
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


#ifndef QXKERNEL4
#define QXKERNEL4

#define qxkernel4(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	QXkernel4<LB, (1<<LB>>1), (1<<LB>>1)-1, FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 9, Time = 1.86284 msec, Performace = 10375.2 GFlop/s
template<int LB, int STEP, int STEP_m1, int FH, int FW>
__global__ void QXkernel4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0; float4  v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0; float4  v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC << 1 >> LB) * FH; gk < GK; gk++)//oic_group * FH
	{
		const int oic_group = gk / FH, fh = gk - oic_group * FH;
		const int oic = (oic_group << LB >> 1);

		//load 4 elements from CW[FH, FW, IC, OC]
		int wic = (ty & STEP_m1) + oic;
		int woffset0 = ((fh*FW*IC) + wic)*OC;
		Ws[buf][ty][tx] = *(float4*)(CW + woffset0);

		//load 4 elements from X[N, IH, IW, IC]
		int xic = (tx & STEP_m1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		float4 xv; int xoffset0 = (fh*IW*IC) + xic;
		xv.x = (lx0 ? X[xoffset0 - IC] : 0);
		xv.y = (lx1 ? X[xoffset0] : 0);
		xv.z = (lx2 ? X[xoffset0 + IC] : 0);
		xv.w = (lx3 ? X[xoffset0 + (IC << 1)] : 0);
		Xs[buf][tx][ty] = xv;
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < FW; fw++) {
			//pragama unroll ik = 0-------------------------------	
			{
				float4 b0 = Xs[buf][0][ty], b1 = Xs[buf][STEP][ty];
				float4 a0 = Ws[buf][0][tx], a1 = Ws[buf][STEP][tx];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
			}
#pragma unroll 
			for (int ik = 1; ik < STEP; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP][ty];
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP][tx];

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
			bool lx3 = lx && (tow0 >= -fw - 3) && (tow0 < IW - fw - 3);
			float x3 = (lx3 ? X[xoffset0 + (fw + 2)*IC] : 0);
			Xs[buf][tx][ty] = xv = { xv.y, xv.z, xv.w, x3 };//read last result

			//load 4 elements from CW[FH, FW, IC, OC]
			int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][ty][tx] = *(float4*)(CW + woffset);
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
		buf ^= 1;
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


#ifndef QXKERNEL5
#define QXKERNEL5

#define qxkernel5(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	QXkernel5<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//LB = 4: Size = 9, Time = 1.86284 msec, Performace = 10375.2 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int FH, int FW>
__global__ void QXkernel5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0; float4  v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0; float4  v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC >> LB)*FH; gk < GK; gk++)//oic_group * FH
	{
		const int oic_group = gk / FH, fh = gk - oic_group * FH;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		const int woffset0 = ((fh*FW*IC) + wic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		const int xoffset0 = (fh*IW*IC) + xic;
		float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = { x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < FW; fw++) {
			{//pragama unroll ik = 0-------------------------------	
				float4 b0 = Xs[buf][0][ty], b1 = Xs[buf][STEP2][ty];
				float4 a0 = Ws[buf][0][tx], a1 = Ws[buf][STEP2][tx];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
			}
#pragma unroll 
			for (int ik = 1; ik < STEP2; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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
			bool lx3 = lx && (tow0 >= -fw - 3) && (tow0 < IW - fw - 3);
			x0 = x1;
			x1 = x2;
			x2 = x3;
			x3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
			Xs[buf][(tx << 1)][ty] = { x0.x, x1.x, x2.x, x3.x };//read last result
			Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };

			//load 4 elements from CW[FH, FW, IC, OC]
			const int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
	}

	
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


#ifndef QXKERNEL6
#define QXKERNEL6

#define qxkernel6(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	QXkernel6<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//IC % LB == 0
//OH, OW % 4 == 0

//LB = 4: Size = 9, Time = 1.83038 msec, Performace = 10559.2 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int FH, int FW>
__global__ void QXkernel6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0; float4  v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0; float4  v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC >> LB)*FH; gk < GK; gk++)//oic_group (32 channels) * FH
	{
		const int oic_group = gk / FH, fh = gk - oic_group * FH;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		const int woffset0 = ((fh*FW*IC) + wic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		const int xoffset0 = (fh*IW*IC) + xic;
		float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = { x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < FW; fw++) {
			{//pragama unroll ik = 0-------------------------------	
				float4 b0 = Xs[buf][0][ty], b1 = Xs[buf][STEP2][ty];
				float4 a0 = Ws[buf][0][tx], a1 = Ws[buf][STEP2][tx];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
			}
#pragma unroll 
			for (int ik = 1; ik < STEP2; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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
			bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
			x0 = x1, x1 = x2, x2 = x3;
			x3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
			Xs[buf][(tx << 1)][ty] = { x0.x, x1.x, x2.x, x3.x };//read last result
			Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };

			//load 4 elements from CW[FH, FW, IC, OC]
			const int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
	}

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


#ifndef QXKERNEL7
#define QXKERNEL7

#define qxkernel7(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	QXkernel7<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//IC % LB == 0
//OH, OW % 4 == 0

//for: Feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 9, Time = 1.79476 msec, Performace = 10768.8 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 13.7812, Time = 2.73474 msec, Performace = 10821.9 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [256, 128, 128]
//LB = 4: Size = 9, Time = 1.78218 msec, Performace = 10844.8 GFlop/s
//for: Feature = (8, 8), [N, IC, OC] = [256, 256, 256]
//LB = 4: Size = 9, Time = 1.7712 msec, Performace = 10912 GFlop/s
//for: Feature = (4, 4), [N, IC, OC] = [256, 512, 512]
//LB = 4: Size = 9, Time = 1.98168 msec, Performace = 9752.99 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int FH, int FW>
__global__ void QXkernel7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0; float4  v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0; float4  v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC >> LB)*FH; gk < GK; gk++)//oic_group (32 channels) * FH
	{
		const int oic_group = gk / FH, fh = gk - oic_group * FH;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		const int woffset0 = ((fh*FW*IC) + wic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		const int xoffset0 = (fh*IW*IC) + xic;
		float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = { x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < FW; fw++) {
			{//pragama unroll ik = 0-------------------------------	
				float4 b0 = Xs[buf][0][ty], b1 = Xs[buf][STEP2][ty];
				float4 a0 = Ws[buf][0][tx], a1 = Ws[buf][STEP2][tx];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
			}
#pragma unroll 
			for (int ik = 1; ik < STEP2; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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
			bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
			x3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
			float4 xv0 = Xs[buf ^ 1][(tx << 1)][ty];
			float4 xv1 = Xs[buf ^ 1][(tx << 1) + 1][ty];
			Xs[buf][(tx << 1)][ty] = { xv0.y, xv0.z, xv0.w, x3.x };//read last result
			Xs[buf][(tx << 1) + 1][ty] = { xv1.y, xv1.z, xv1.w, x3.y };

			//load 4 elements from CW[FH, FW, IC, OC]
			const int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
	}

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


//false
#define FQXKERNEL8
#ifndef FQXKERNEL8
#define FQXKERNEL8

#define Fqxkernel8(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	FQXkernel8<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//IC % LB == 0
//OH, OW % 4 == 0

//for: Feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 9, Time = 1.79476 msec, Performace = 10768.8 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 13.7812, Time = 2.73474 msec, Performace = 10821.9 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [256, 128, 128]
//LB = 4: Size = 9, Time = 1.78218 msec, Performace = 10844.8 GFlop/s
//for: Feature = (8, 8), [N, IC, OC] = [256, 256, 256]
//LB = 4: Size = 9, Time = 1.7712 msec, Performace = 10912 GFlop/s
//for: Feature = (4, 4), [N, IC, OC] = [256, 512, 512]
//LB = 4: Size = 9, Time = 1.98168 msec, Performace = 9752.99 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int FH, int FW>
__global__ void FQXkernel8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0; float4  v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0; float4  v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC >> LB)*FH; gk < GK; gk++)//oic_group (32 channels) * FH
	{
		const int oic_group = gk / FH, fh = gk - oic_group * FH;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		const int woffset0 = ((fh*FW*IC) + wic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		const int xoffset0 = (fh*IW*IC) + xic;
		float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = { x0.x, x1.x, x2.x, x3.x };//0, 1, 2, 3
		Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < FW; fw++) {
#pragma unroll 
			for (int ik = 0; ik < STEP2; ik++) {
				float b0[4], b1[4]; *(float4*)b0 = Xs[buf][ik][ty]; *(float4*)b1 = Xs[buf][ik + STEP2][ty];
				float a0[4], a1[4]; *(float4*)a0 = Ws[buf][ik][tx]; *(float4*)a1 = Ws[buf][ik + STEP2][tx];

				shuf_simdMM4( v0, b0[(3 * fw - 3) & 3], a0, fw); shuf_simdMM4( v1, b0[(3 * fw - 3) & 3], a1, fw);
				shuf_simdMM4( v2, b0[(3 * fw - 2) & 3], a0, fw); shuf_simdMM4( v3, b0[(3 * fw - 2) & 3], a1, fw);
				shuf_simdMM4( v4, b0[(3 * fw - 1) & 3], a0, fw); shuf_simdMM4( v5, b0[(3 * fw - 1) & 3], a1, fw);
				shuf_simdMM4( v6, b0[(3 * fw    ) & 3], a0, fw); shuf_simdMM4( v7, b0[(3 * fw    ) & 3], a1, fw);
				shuf_simdMM4( v8, b1[(3 * fw - 3) & 3], a0, fw); shuf_simdMM4( v9, b1[(3 * fw - 3) & 3], a1, fw);
				shuf_simdMM4(v10, b1[(3 * fw - 2) & 3], a0, fw); shuf_simdMM4(v11, b1[(3 * fw - 2) & 3], a1, fw);
				shuf_simdMM4(v12, b1[(3 * fw - 1) & 3], a0, fw); shuf_simdMM4(v13, b1[(3 * fw - 1) & 3], a1, fw);
				shuf_simdMM4(v14, b1[(3 * fw    ) & 3], a0, fw); shuf_simdMM4(v15, b1[(3 * fw    ) & 3], a1, fw);
			}
			buf ^= 1;
			
			//load 4 elements from X[N, IH, IW, IC]
			bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
			float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
			*((float*)(&Xs[buf][(tx << 1)    ][ty]) + ((fw - 1) & 3)) = x3.x;//update shared memory
			*((float*)(&Xs[buf][(tx << 1) + 1][ty]) + ((fw - 1) & 3)) = x3.y;
		
			//load 4 elements from CW[FH, FW, IC, OC]
			const int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float b0[4], b1[4]; *(float4*)(b0) = Xs[buf][ik][ty]; *(float4*)(b1) = Xs[buf][ik + STEP2][ty];
			float a0[4], a1[4]; *(float4*)(a0) = Ws[buf][ik][tx]; *(float4*)(a1) = Ws[buf][ik + STEP2][tx];

			shuf_simdMM4( v0, b0[(3 * FW - 3) & 3], a0, FW); shuf_simdMM4( v1, b0[(3 * FW - 3) & 3], a1, FW);
			shuf_simdMM4( v2, b0[(3 * FW - 2) & 3], a0, FW); shuf_simdMM4( v3, b0[(3 * FW - 2) & 3], a1, FW);
			shuf_simdMM4( v4, b0[(3 * FW - 1) & 3], a0, FW); shuf_simdMM4( v5, b0[(3 * FW - 1) & 3], a1, FW);
			shuf_simdMM4( v6, b0[(3 * FW    ) & 3], a0, FW); shuf_simdMM4( v7, b0[(3 * FW    ) & 3], a1, FW);
			shuf_simdMM4( v8, b1[(3 * FW - 3) & 3], a0, FW); shuf_simdMM4( v9, b1[(3 * FW - 3) & 3], a1, FW);
			shuf_simdMM4(v10, b1[(3 * FW - 2) & 3], a0, FW); shuf_simdMM4(v11, b1[(3 * FW - 2) & 3], a1, FW);
			shuf_simdMM4(v12, b1[(3 * FW - 1) & 3], a0, FW); shuf_simdMM4(v13, b1[(3 * FW - 1) & 3], a1, FW);
			shuf_simdMM4(v14, b1[(3 * FW    ) & 3], a0, FW); shuf_simdMM4(v15, b1[(3 * FW    ) & 3], a1, FW);
		}
		buf ^= 1;
	}

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

//false
#define FQXKERNEL9
#ifndef FQXKERNEL9
#define FQXKERNEL9

#define Fqxkernel9(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	FQXkernel9<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//IC % LB == 0
//OH, OW % 4 == 0

//for: Feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 9, Time = 1.79476 msec, Performace = 10768.8 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 13.7812, Time = 2.73474 msec, Performace = 10821.9 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [256, 128, 128]
//LB = 4: Size = 9, Time = 1.78218 msec, Performace = 10844.8 GFlop/s
//for: Feature = (8, 8), [N, IC, OC] = [256, 256, 256]
//LB = 4: Size = 9, Time = 1.7712 msec, Performace = 10912 GFlop/s
//for: Feature = (4, 4), [N, IC, OC] = [256, 512, 512]
//LB = 4: Size = 9, Time = 1.98168 msec, Performace = 9752.99 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int FH, int FW>
__global__ void FQXkernel9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0; float4  v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0; float4  v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC >> LB)*FH; gk < GK; gk++)//oic_group (32 channels) * FH
	{
		const int oic_group = gk / FH, fh = gk - oic_group * FH;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		const int woffset0 = ((fh*FW*IC) + wic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		const int xoffset0 = (fh*IW*IC) + xic;
		float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = { x0.x, x1.x, x2.x, x3.x };//0, 1, 2, 3
		Xs[buf][(tx << 1) + 1][ty] = Xs[buf ^ 1][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };
		Xs[buf ^ 1][(tx << 1)][ty] = Xs[buf][(tx << 1)][ty];
		Xs[buf ^ 1][(tx << 1) + 1][ty] = Xs[buf][(tx << 1) + 1][ty];
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < FW; fw++) {
#pragma unroll 
			for (int ik = 0; ik < STEP2; ik++) {
				float b0[4], b1[4]; *(float4*)b0 = Xs[buf][ik][ty]; *(float4*)b1 = Xs[buf][ik + STEP2][ty];
				float a0[4], a1[4]; *(float4*)a0 = Ws[buf][ik][tx]; *(float4*)a1 = Ws[buf][ik + STEP2][tx];

				shuf_simdMM4(v0, b0[(3 * fw - 3) & 3], a0, fw); shuf_simdMM4(v1, b0[(3 * fw - 3) & 3], a1, fw);
				shuf_simdMM4(v2, b0[(3 * fw - 2) & 3], a0, fw); shuf_simdMM4(v3, b0[(3 * fw - 2) & 3], a1, fw);
				shuf_simdMM4(v4, b0[(3 * fw - 1) & 3], a0, fw); shuf_simdMM4(v5, b0[(3 * fw - 1) & 3], a1, fw);
				shuf_simdMM4(v6, b0[(3 * fw) & 3], a0, fw); shuf_simdMM4(v7, b0[(3 * fw) & 3], a1, fw);
				shuf_simdMM4(v8, b1[(3 * fw - 3) & 3], a0, fw); shuf_simdMM4(v9, b1[(3 * fw - 3) & 3], a1, fw);
				shuf_simdMM4(v10, b1[(3 * fw - 2) & 3], a0, fw); shuf_simdMM4(v11, b1[(3 * fw - 2) & 3], a1, fw);
				shuf_simdMM4(v12, b1[(3 * fw - 1) & 3], a0, fw); shuf_simdMM4(v13, b1[(3 * fw - 1) & 3], a1, fw);
				shuf_simdMM4(v14, b1[(3 * fw) & 3], a0, fw); shuf_simdMM4(v15, b1[(3 * fw) & 3], a1, fw);
			}
			__syncthreads();
			buf ^= 1;

			//load 4 elements from X[N, IH, IW, IC]
			bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
			float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
			*((float*)(&Xs[buf][(tx << 1)][ty]) + ((fw - 1) & 3)) = x3.x;//update shared memory
			*((float*)(&Xs[buf][(tx << 1) + 1][ty]) + ((fw - 1) & 3)) = x3.y;
			Xs[buf ^ 1][(tx << 1)][ty] = Xs[buf][(tx << 1)][ty];
			Xs[buf ^ 1][(tx << 1) + 1][ty] = Xs[buf][(tx << 1) + 1][ty];

			//load 4 elements from CW[FH, FW, IC, OC]
			const int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float b0[4], b1[4]; *(float4*)(b0) = Xs[buf][ik][ty]; *(float4*)(b1) = Xs[buf][ik + STEP2][ty];
			float a0[4], a1[4]; *(float4*)(a0) = Ws[buf][ik][tx]; *(float4*)(a1) = Ws[buf][ik + STEP2][tx];

			shuf_simdMM4(v0, b0[(3 * FW - 3) & 3], a0, FW); shuf_simdMM4(v1, b0[(3 * FW - 3) & 3], a1, FW);
			shuf_simdMM4(v2, b0[(3 * FW - 2) & 3], a0, FW); shuf_simdMM4(v3, b0[(3 * FW - 2) & 3], a1, FW);
			shuf_simdMM4(v4, b0[(3 * FW - 1) & 3], a0, FW); shuf_simdMM4(v5, b0[(3 * FW - 1) & 3], a1, FW);
			shuf_simdMM4(v6, b0[(3 * FW) & 3], a0, FW); shuf_simdMM4(v7, b0[(3 * FW) & 3], a1, FW);
			shuf_simdMM4(v8, b1[(3 * FW - 3) & 3], a0, FW); shuf_simdMM4(v9, b1[(3 * FW - 3) & 3], a1, FW);
			shuf_simdMM4(v10, b1[(3 * FW - 2) & 3], a0, FW); shuf_simdMM4(v11, b1[(3 * FW - 2) & 3], a1, FW);
			shuf_simdMM4(v12, b1[(3 * FW - 1) & 3], a0, FW); shuf_simdMM4(v13, b1[(3 * FW - 1) & 3], a1, FW);
			shuf_simdMM4(v14, b1[(3 * FW) & 3], a0, FW); shuf_simdMM4(v15, b1[(3 * FW) & 3], a1, FW);
		}
		buf ^= 1;
		__syncthreads();
	}

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


#ifndef QXKERNEL8
#define QXKERNEL8

#define qxkernel8(stream, LB, oc_index, j_index, X, IH, IW, CW, FH, FW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	QXkernel8<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//IC % LB == 0
//OH, OW % 4 == 0

//for: 3 * 3 kernel
//for: Feature = (32, 32), [N, IC, OC] = [64, 128, 128]
//LB = 4: Size = 9, Time = 1.79476 msec, Performace = 10768.8 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 13.7812, Time = 2.73474 msec, Performace = 10821.9 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [256, 128, 128]
//LB = 4: Size = 9, Time = 1.78218 msec, Performace = 10844.8 GFlop/s
//for: Feature = (8, 8), [N, IC, OC] = [256, 256, 256]
//LB = 4: Size = 9, Time = 1.7712 msec, Performace = 10912 GFlop/s
//for: Feature = (4, 4), [N, IC, OC] = [256, 512, 512]
//LB = 4: Size = 9, Time = 1.98168 msec, Performace = 9752.99 GFlop/s

//for 5 * 5 kernel
//
template<int LB, int STEP, int STEP2, int STEP_m1, int FH, int FW>
__global__ void QXkernel8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0; float4  v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0; float4  v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC >> LB)*FH; gk < GK; gk++)//oic_group (32 channels) * FH
	{
		const int oic_group = gk / FH, fh = gk - oic_group * FH;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		const int woffset0 = ((fh*FW*IC) + wic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		const int xoffset0 = (fh*IW*IC) + xic;
		float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = { x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < FW; fw++) {
#pragma unroll 
			for (int ik = 0; ik < STEP2; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

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
			bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
			float2 nx3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
			float4 ox0 = Xs[buf ^ 1][(tx << 1)][ty];//update_shared_memory
			float4 ox1 = Xs[buf ^ 1][(tx << 1) + 1][ty];
			Xs[buf][(tx << 1)][ty] = { ox0.y, ox0.z, ox0.w, nx3.x };
			Xs[buf][(tx << 1) + 1][ty] = { ox1.y, ox1.z, ox1.w, nx3.y };

			//load 4 elements from CW[FH, FW, IC, OC]
			const int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
	}

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


#ifndef QXKERNEL9
#define QXKERNEL9

#define qxkernel9(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	QXkernel9<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//IC % LB == 0
//OH, OW % 4 == 0

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void QXkernel9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0; float4  v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0; float4  v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0; float4 v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0; float4 v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC >> LB)*3; gk < GK; gk++)//oic_group (32 channels) * FH
	{
		const int oic_group = gk / 3, fh = gk - oic_group * 3;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		const int woffset0 = ((fh*3*IC) + wic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		const int xoffset0 = (fh*IW*IC) + xic;
		float2 x0 = (lx0 ? *(float2*)(X + xoffset0 - IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty] = { x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = { x0.y, x1.y, x2.y, x3.y };
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < 3; fw++) {
#pragma unroll 
			for (int ik = 0; ik < STEP2; ik++)
			{
				float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];
				float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
			}
			float4 ox0 = Xs[buf][(tx << 1)][ty];//update_shared_memory
			float4 ox1 = Xs[buf][(tx << 1) + 1][ty];
			buf ^= 1;

			//load 4 elements from CW[FH, FW, IC, OC]
			const int woffset = woffset0 + (fw*IC*OC);
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

			//load 4 elements from X[N, IH, IW, IC]
			bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
			float2 nx3 = (lx3 ? *(float2*)(X + xoffset0 + (fw + 2)*IC) : F32_2_0);
			Xs[buf][(tx << 1)][ty] = { ox0.y, ox0.z, ox0.w, nx3.x };
			Xs[buf][(tx << 1) + 1][ty] = { ox1.y, ox1.z, ox1.w, nx3.y };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = Ws[buf][ik][tx], a1 = Ws[buf][ik + STEP2][tx];
			float4 b0 = Xs[buf][ik][ty], b1 = Xs[buf][ik + STEP2][ty];

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
	}

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



