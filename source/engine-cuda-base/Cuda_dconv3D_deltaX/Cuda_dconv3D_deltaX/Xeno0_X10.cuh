

#ifndef OPP_KERNEL1
#define OPP_KERNEL1

#define opp_kernel1(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	OPP_Kernel1<LB, (1<<LB>>1),(1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,LOC,\
			 ic_index,n_index)

//LB = 4: Size = 9, Time = 1.636 msec, Performace = 11813.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OPP_Kernel1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,//oph = opw = 1
	int ic_index, int n_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((2 - fhs) * 3 + (2 - fws))*IC + ic0 + ((ty >= STEP) << 2);//Wr[0, fhs, frws, tic0]

	//prepare for GM = N
	const int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	const int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	const int Xstride = IH * IW * IC;
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + fws + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	int FW = 3 * IC;//FW -> FW * IC = 3*IC
	int FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = ((9 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 

	//compute area----------------------------------------------------
	const int SY = (OW - tFW) << LOC;

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = (tx & STEP_m1) << 1;
	const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
	const int fh = fhw >> 2, fw = fhw & 3;
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
		const int fh = fhw >> 2, fw = fhw & 3;
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


#ifndef OPP_KERNEL2
#define OPP_KERNEL2

#define opp_kernel2(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	OPP_Kernel2<LB, (1<<LB>>1),(1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,LOC,\
			 ic_index,n_index)

//LB = 4: Size = 9, Time = 1.636 msec, Performace = 11813.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OPP_Kernel2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,//oph = opw = 1
	int ic_index, int n_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((2 - fhs) * 3 + (2 - fws))*IC + ic0 + ((ty >= STEP) << 2);//Wr[0, fhs, frws, tic0]

	//prepare for GM = N
	const int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	const int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	const int Xstride = IH * IW * IC;
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + fws + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//compute area----------------------------------------------------
	const int SY = (OW - tFW) << LOC, OC_m1 = (1 << LOC) - 1;
	const int GK0 = 9 * IC;//GK0 = FH * FW * IC

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = (tx & STEP_m1) << 1;
	const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
	int fh = fhw >> 2, fw = fhw & 3;
	const int yoffset = fh * SY + Y_k;
	const int yoffset0 = yoffset + Y0;
	const int yoffset1 = yoffset + Y1;
	const int yoffset2 = yoffset + Y2;
	const int yoffset3 = yoffset + Y3;
	float2 x0 = *(float2*)(deltaY + yoffset0);
	float2 x1 = *(float2*)(deltaY + yoffset1);
	float2 x2 = *(float2*)(deltaY + yoffset2);
	float2 x3 = *(float2*)(deltaY + yoffset3);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = (((W_k & OC_m1)*3 - fh) * 3 - fw)*IC;
	const int woffset1 = woffset0 + GK0;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset0);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
		int fh = fhw >> 2, fw = fhw & 3;
		const int yoffset = fh * SY + Y_k;
		const int yoffset0 = yoffset + Y0;
		const int yoffset1 = yoffset + Y1;
		const int yoffset2 = yoffset + Y2;
		const int yoffset3 = yoffset + Y3;
		float2 x0 = *(float2*)(deltaY + yoffset0);
		float2 x1 = *(float2*)(deltaY + yoffset1);
		float2 x2 = *(float2*)(deltaY + yoffset2);
		float2 x3 = *(float2*)(deltaY + yoffset3);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (((W_k & OC_m1) * 3 - fh) * 3 - fw)*IC;
		const int woffset1 = woffset0 + GK0;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


#ifndef OPP_KERNEL3
#define OPP_KERNEL3

#define opp_kernel3(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	OPP_Kernel3<LB, (1<<LB>>1),(1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,LOC,\
			 ic_index,n_index)

//LB = 4: Size = 9, Time = 1.636 msec, Performace = 11813.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OPP_Kernel3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,//oph = opw = 1
	int ic_index, int n_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((2 - fhs) * 3 + (2 - fws))*IC + ic0 + ((ty >= STEP) << 2);//Wr[0, fhs, frws, tic0]

	//prepare for GM = N
	const int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	const int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	const int Xstride = IH * IW * IC;
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + fws + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//compute area----------------------------------------------------
	int FW = 3 * IC;//FW -> FW * IC = 3*IC
	int FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = ((9 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 

	const int SY = (OW - tFW) << LOC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const char fhw = YIDX_V2_W3P1[(W_k >> LOC) + fhw_offset];
	const int fh = fhw >> 2, fw = fhw & 3;
	const int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = (tx & STEP_m1) << 1;
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const char fhw = YIDX_V2_W3P1[(W_k >> LOC) + fhw_offset];
		const int fh = fhw >> 2, fw = fhw & 3;
		const int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


#ifndef OPP_KERNEL4
#define OPP_KERNEL4

#define opp_kernel4(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	OPP_Kernel4<LB, (1<<LB>>1),(1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, N>>LB>>3, IH*IW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,LOC,\
			 ic_index,n_index)

//LB = 4: Size = 9, Time = 1.642 msec, Performace = 11770.6 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OPP_Kernel4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int LOC,//oph = opw = 1
	int ic_index, int n_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for bz -> IH * IW
	int bz = blockIdx.z;
	int ih = bz / IW, iw = bz - ih * IW;
	int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ((2 - fhs) * 3 + (2 - fws))*IC + ic0 + ((ty >= STEP) << 2);//Wr[0, fhs, frws, tic0]

	//prepare for GM = N
	const int n0 = (((blockIdx.y << LB) + ty) << 3) + n_index;
	const int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	const int Xstride = IH * IW * IC;
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + fws + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	int FW = 3 * IC;//FW -> FW * IC = 3*IC
	int FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = ((9 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 

	//compute area----------------------------------------------------
	const int SY = (OW - tFW) << LOC;

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = (tx & STEP_m1) << 1;
	const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
	const int fh = fhw >> 2;
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = W_k * FH - fh * FW - (fhw & 3) * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
		const int fh = fhw >> 2;
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = W_k * FH - fh * FW - (fhw & 3) * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


#ifndef OPP_KERNEL5
#define OPP_KERNEL5

#define opp_kernel5(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	OPP_Kernel5<LB, (1<<LB>>1),(1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), IW, IH), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX, IC,LOC,ic_index,n_index,(N>>LB>>3))

//LB = 4:Size = 9, Time = 1.636 msec, Performace = 11813.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OPP_Kernel5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX,
	int IC, int LOC,//oph = opw = 1
	int ic_index, int n_index, int GX)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for bz -> IH * IW
	const int IH = gridDim.z, IW = gridDim.y;
	const int ih = blockIdx.z, iw = blockIdx.y;
	const int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;

	//prepare for GN = IC
	const int ic0 = (((bx << LB) + tx) << 3) + ic_index;
	W += ((8 - fhs * 3 - fws))*IC + ic0 + ((ty >= STEP) << 2);

	//prepare for GM = N
	const int n0 = (((by << LB) + ty) << 3) + n_index;
	const int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	const int Xstride = IH * IW * IC;
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + fws + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	int FW = 3 * IC;//FW -> FW * IC = 3*IC
	int FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = ((9 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 

	//compute area----------------------------------------------------
	const int SY = (OW - tFW) << LOC;

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = (tx & STEP_m1) << 1;
	const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
	const int fh = fhw >> 2;
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = W_k * FH - fh * FW - (fhw & 3) * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
		const int fh = fhw >> 2;
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = W_k * FH - fh * FW - (fhw & 3) * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


#ifndef OPP_KERNEL6
#define OPP_KERNEL6

#define opp_kernel6(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	OPP_Kernel6<LB, (1<<LB>>1),(1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), IW, IH), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX, IC,LOC,ic_index,n_index,(N>>LB>>3))

//start<4>: Size = 9, Time = 1.663 msec, Performace = 11622 GFlop/s
//LB = 4:   Size = 9, Time = 1.636 msec, Performace = 11813.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OPP_Kernel6(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX,
	int IC, int LOC,//oph = opw = 1
	int ic_index, int n_index, int GX)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for bz -> IH * IW
	const int IH = gridDim.z, IW = gridDim.y;
	const int ih = blockIdx.z, iw = blockIdx.y;
	const int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;

	//prepare for GN = IC
	const int ic0 = (((bx << LB) + tx) << 3) + ic_index;
	W += (8 - fhs * 3 - fws)*IC + ic0 + ((ty >= STEP) << 2);

	//prepare for GM = N
	const int n0 = (((by << LB) + ty) << 3) + n_index;
	const int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	const int Xstride = IH * IW * IC;
	const int tn0 = n0 + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + tih + fhs)*OW + fws + tiw) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//compute area----------------------------------------------------
	int FW = 3 * IC;//FW -> FW * IC = 3*IC
	int FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = ((9 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 
	
	const int SY = (OW - tFW) << LOC;

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = (tx & STEP_m1) << 1;
	const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
	const int fh = fhw >> 2, fw = (fhw & 3);
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
		const int fh = fhw >> 2, fw = (fhw & 3);
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 x1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 x2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 x3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


#ifndef OPP_KERNEL7
#define OPP_KERNEL7

#define opp_kernel7(stream, LB, ic_index, n_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, GN, N) \
	OPP_Kernel7<LB, (1<<LB>>1),(1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3)*(N>>LB>>3), IW, IH), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX, IC,LOC,ic_index,n_index,(N>>LB>>3))

//start<4>: Size = 9, Time = 1.663 msec, Performace = 11622 GFlop/s
//LB = 4:   Size = 9, Time = 1.636 msec, Performace = 11813.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void OPP_Kernel7(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//tFH = tFW = 2 or 3
	float* __restrict__ deltaX,
	int IC, int LOC,//oph = opw = 1
	int ic_index, int n_index, int GX)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for bz -> IH * IW
	const int IH = gridDim.z, IW = gridDim.y;
	const int ih = blockIdx.z, iw = blockIdx.y;
	const int tih = ih - 1, tiw = iw - 1;

	//prepare for GK = FH * FW * OC
	const int fhs = -IF_int((tih < 0), tih, 0);
	const int fws = -IF_int((tiw < 0), tiw, 0);
	int fhe = OH - tih; fhe = IF_int((fhe > 3), 3, fhe);
	int fwe = OW - tiw; fwe = IF_int((fwe > 3), 3, fwe);
	const int tFH = fhe - fhs;
	const int tFW = fwe - fws;
	const int tFW_OC = tFW << LOC, GK = tFH * tFW_OC;
	const int fhw_offset = (((tFH == 3) << 1) + (tFW == 3)) * 9;

	const int bidx = blockIdx.x;
	const int by = bidx / GX, bx = bidx - by * GX;

	//prepare for GN = IC
	const int ic0 = (((bx << LB) + tx) << 3) + ic_index;
	W += ((8 - fhs * 3 - fws))*IC + ic0 + ((ty >= STEP) << 2);

	//prepare for GM = N
	const int n0 = (((by << LB) + ty) << 3) + n_index;
	const int X0 = ((n0*IH + ih)*IW + iw)*IC + ic0;
	const int Xstride = IH * IW * IC;
	const int tn1 = n0 + ((tx >= STEP) << 2) + 1;
	deltaY += ((tn1*OH + tih + fhs)*OW + tiw + fws) << LOC;//deltaY += Y1
	const int Ystride = (OH * OW) << LOC;

	//compute area----------------------------------------------------
	int FW = 3 * IC;//FW -> FW * IC = 3*IC
	int FH = 9 * IC;//FH -> FH * FW * IC = 9*IC
	IC = ((9 << LOC) + 1)*IC;//IC -> OC * FH * FW * IC + IC = (9*OC + 1)*IC

	//W_k -= fh*tFW_OC, max(fh*FW) = FH * FH * FW * IC * FW * OC = 81*IC*OC
	FW = tFW_OC * FH + FW;//sqrt(IC*OC) <= 3640, It's hard to out of bound of memory 

	const int SY = (OW - tFW) << LOC;

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = (tx & STEP_m1) << 1;
	const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
	const int fh = fhw >> 2, fw = (fhw & 3);
	const int yoffset = fh * SY + Y_k;
	float2 x0 = *(float2*)(deltaY + yoffset - Ystride);
	float2 x1 = *(float2*)(deltaY + yoffset);
	float2 x2 = *(float2*)(deltaY + yoffset + Ystride);
	float2 x3 = *(float2*)(deltaY + yoffset + (Ystride << 1));
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = W_k * FH - fh * FW - fw * IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const char fhw = YIDX_V2_W3P1[(Y_k >> LOC) + fhw_offset];
		const int fh = fhw >> 2, fw = (fhw & 3);
		const int yoffset = fh * SY + Y_k;
		float2 x0 = *(float2*)(deltaY + yoffset - Ystride);
		float2 x1 = *(float2*)(deltaY + yoffset);
		float2 x2 = *(float2*)(deltaY + yoffset + Ystride);
		float2 x3 = *(float2*)(deltaY + yoffset + (Ystride << 1));
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elem from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = W_k * FH - fh * FW - fw * IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + FH);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

		simdMM4(v0, y0.x, w0);  simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0);  simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0);  simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0);  simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0);  simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif