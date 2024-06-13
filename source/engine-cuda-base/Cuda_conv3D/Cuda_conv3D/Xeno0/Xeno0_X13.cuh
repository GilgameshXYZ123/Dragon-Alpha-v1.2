


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2, sh = sw = 1
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef UVV_KERNEL1
#define UVV_KERNEL1

#define UVVk1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	uvv_kernel1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//for: Feature = (64, 64), [N, IC, OC] = [32, 64, 128]
//LB = 4: Size = 9, Time = 1.79052 msec, Performace = 10794.2 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 64, 128]
//LB = 4: Size = 9, Time = 1.79476 msec, Performace = 10768.8 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 13.7812, Time = 2.73474 msec, Performace = 10821.9 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [256, 128, 128]
//LB = 4: Size = 9, Time = 1.78218 msec, Performace = 10844.8 GFlop/s
//for: Feature = (8, 8), [N, IC, OC] = [256, 256, 256]
//LB = 4: Size = 9, Time = 1.7712 msec, Performace = 10912 GFlop/s
//for: Feature = (4, 4), [N, IC, OC] = [256, 512, 512]
//LB = 4: Size = 9, Time = 1.98168 msec, Performace = 9752.99 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uvv_kernel1(
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
	float4 v0 = F32_4_0, v1 = F32_4_0; float4 v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0; float4 v6 = F32_4_0, v7 = F32_4_0;
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
	for (int gk = 0, GK = (IC >> LB) * 3; gk < GK; gk++)//oic_group (32 channels) * FH
	{
		const int oic_group = gk / 3, fh = gk - oic_group * 3;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		const int woffset0 = ((fh * 3 * IC) + wic)*OC;
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



//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2, sh = sw = 1
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef UVV_KERNEL2
#define UVV_KERNEL2

#define UVVk2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	uvv_kernel2<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//Size = 27.5625, Time = 5.57763 msec, Performace = 10612 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uvv_kernel2(
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
	for (int gk = 0, GK = (IC >> LB) * 3; gk < GK; gk++)//oic_group (32 channels) * FH
	{
		const int oic_group = gk / 3, fh = gk - oic_group * 3;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		const int woffset0 = ((fh * 3 * IC) + wic)*OC;
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
		Xs[buf][(tx << 1)][ty].x = x0.x;
		Xs[buf][(tx << 1) + 1][ty].x = x0.y;
		float2 x1 = (lx1 ? *(float2*)(X + xoffset0) : F32_2_0);
		Xs[buf][(tx << 1)][ty].y = x1.x;
		Xs[buf][(tx << 1) + 1][ty].y = x1.y;
		float2 x2 = (lx2 ? *(float2*)(X + xoffset0 + IC) : F32_2_0);
		Xs[buf][(tx << 1)][ty].z = x2.x;
		Xs[buf][(tx << 1) + 1][ty].z = x2.y;
		float2 x3 = (lx3 ? *(float2*)(X + xoffset0 + (IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)][ty].w = x3.x;
		Xs[buf][(tx << 1) + 1][ty].w = x3.y;
		

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0 sh = sw = 1
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef UVV_KERNEL3
#define UVV_KERNEL3

#define UVVk3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	uvv_kernel3<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//[IH = 224, IW = 224, N = 16, IC = 32, OC = 128]
//LB = 4: Size = 27.5625, Time = 5.55208 msec, Performace = 10660.9 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uvv_kernel3(
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
	for (int gk = 0, GK = (IC >> LB) * 3; gk < GK; gk++)//oic_group (16channels) * FH
	{
		const int oic_group = gk / 3, fh = gk - oic_group * 3;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		int woffset0 = ((fh * 3 * IC) + wic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		int xoffset0 = (fh*IW*IC) + xic;
		float2 x0 = F32_2_0; if (lx0) x0 = *(float2*)(X + xoffset0 - IC);
		float2 x1 = F32_2_0; if (lx1) x1 = *(float2*)(X + xoffset0);
		float2 x2 = F32_2_0; if (lx2) x2 = *(float2*)(X + xoffset0 + IC);
		float2 x3 = F32_2_0; if (lx3) x3 = *(float2*)(X + xoffset0 + (IC << 1));
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
			woffset0 += IC * OC;
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

			//load 4 elements from X[N, IH, IW, IC]
			bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < IW - (fw + 3));
			float2 nx3 = F32_2_0; if (lx3) nx3 = *(float2*)(X + xoffset0 + (fw + 2)*IC);
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2, sh = sw = 1
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef UVV_KERNEL4
#define UVV_KERNEL4

#define UVVk4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	uvv_kernel4<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//[IH = 224, IW = 224, N = 16, IC = 32, OC = 128]
//Start: 
//LB = 4:Size = 13.7812, Time = 2.68332 msec, Performace = 11029.2 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uvv_kernel4(
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
	const int IC_OC = IC * OC, IW_IC = IW * IC;

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC >> LB) * 3; gk < GK; gk++)//oic_group (16channels) * FH
	{
		const int oic_group = gk / 3, fh = gk - oic_group * 3;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		const int wic = ((ty & STEP_m1) << 1) + oic;
		int woffset0 = ((fh * 3 * IC) + wic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		int xoffset0 = fh * IW_IC + xic;
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
			float4 ox0 = Xs[buf][(tx << 1)    ][ty];//update_shared_memory
			float4 ox1 = Xs[buf][(tx << 1) + 1][ty];
			buf ^= 1;

			//load 4 elements from CW[FH, FW, IC, OC]
			woffset0 += IC_OC;
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2, sh = sw = 1
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef UVV_KERNEL5
#define UVV_KERNEL5

#define UVVk5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	uvv_kernel5<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, IC,OC, ph,pw,\
			oc_index,j_index)

//[IH = 224, IW = 224, N = 16, IC = 32, OC = 128]
//Start: 
//LB = 4:Size = 13.7812, Time = 2.68332 msec, Performace = 11029.2 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uvv_kernel5(
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
	CW += oc0 + ((ty >= STEP) << 2) + ((ty & STEP_m1) << 1)*OC;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*IH + toh0)*IW + tow0 + 1)*IC;//X += X1;
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int IC_OC = IC * OC, IW_IC = IW * IC;

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC >> LB) * 3; gk < GK; gk++)//oic_group (16channels) * FH
	{
		const int oic_group = gk / 3, fh = gk - oic_group * 3;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		int woffset0 = ((fh * 3 * IC) + oic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < IH - fh);
		bool lx0 = lx && (tow0 >= 0) && (tow0 < IW);
		bool lx1 = lx && (tow0 >= -1) && (tow0 < IW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < IW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < IW - 3);
		const int xoffset0 = fh * IW_IC + xic;
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
			woffset0 += IC_OC;
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2, sh = sw = 1
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef UVV_KERNEL6
#define UVV_KERNEL6

#define UVVk6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	uvv_kernel6<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, CW, Y,OH,OW, IC,OC, ph,pw,\
			oc_index,j_index)

//[IH = 224, IW = 224, N = 16, IC = 32, OC = 128]
//Start: 
//LB = 4:Size = 13.7812, Time = 2.68332 msec, Performace = 11029.2 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uvv_kernel6(
	const float* __restrict__ X,
	const float* __restrict__ CW,
	float* __restrict__ Y, int OH, int OW,
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
	CW += oc0 + ((ty >= STEP) << 2) + ((ty & STEP_m1) << 1)*OC;//CW[0, 0, 0, toc0]

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	const int OH_OW = OH * OW; get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 - ph, tow0 = tow0 - pw;
	X += ((tn0*OH + toh0)*OW + tow0 + 1)*IC;//X += X1;
	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	const int IC_OC = IC * OC, IW_IC = OW * IC;

	//compute area-----------------------------------------------------------
	for (int gk = 0, GK = (IC >> LB) * 3; gk < GK; gk++)//oic_group (16channels) * FH
	{
		const int oic_group = gk / 3, fh = gk - oic_group * 3;
		const int oic = (oic_group << LB);

		//load 4 elements from CW[FH, FW, IC, OC]
		int woffset0 = ((fh * 3 * IC) + oic)*OC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int xic = ((tx & STEP_m1) << 1) + oic;
		bool lx = (toh0 >= -fh) && (toh0 < OH - fh);
		bool lx0 = lx && (tow0 >=  0) && (tow0 < OW    );
		bool lx1 = lx && (tow0 >= -1) && (tow0 < OW - 1);
		bool lx2 = lx && (tow0 >= -2) && (tow0 < OW - 2);
		bool lx3 = lx && (tow0 >= -3) && (tow0 < OW - 3);
		int xoffset0 = fh * IW_IC + xic;
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
			woffset0 += IC_OC;
			Ws[buf][(ty << 1)][tx] = *(float4*)(CW + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset0 + OC);

			//load 4 elements from X[N, IH, IW, IC]
			bool lx3 = lx && (tow0 >= -(fw + 3)) && (tow0 < OW - (fw + 3));
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



