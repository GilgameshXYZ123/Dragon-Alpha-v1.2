

//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, (IH, IW) % 4 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef CC_KERNEL1
#define CC_KERNEL1

#define cc_kernel1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	CC_Kernel1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,(IH*IW),IW, IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

//Size = 9, Time = 1.772 msec, Performace = 10907.1 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void CC_Kernel1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int LOC,
	int oph, int opw,
	int ic_index, int j_index)
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

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	deltaY += ((tn0*OH + tih0)*OW + tiw1) << LOC;
	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic

	//compute area----------------------------------------------------
	const int GK = 9 << LOC;//GK = FH * FW * OC
	const int OC = (1 << LOC), OC_m1 = OC - 1;

	//load 4 elements from deltaY[N, OH, OW, OC]
	const int Y_k = (tx & STEP_m1) << 1;
	const int Idx = Y_k >> LOC; char fhw = YIDX_W33[Idx];
	const int fh = fhw >> 2, fw = fhw & 3;
	const int yoffset = ((fh*OW + fw - Idx) << LOC) + Y_k;
	bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
	bool ly0 = ly && (tiw0 >= -fw) && (tiw0 < OW - fw);
	bool ly1 = ly && (tiw1 >= -fw) && (tiw1 < OW - fw);
	bool ly2 = ly && (tiw2 >= -fw) && (tiw2 < OW - fw);
	bool ly3 = ly && (tiw3 >= -fw) && (tiw3 < OW - fw);
	float2 x0 = (ly0 ? *(float2*)(deltaY + yoffset - OC) : F32_2_0);
	float2 x1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);
	float2 x2 = (ly2 ? *(float2*)(deltaY + yoffset + OC) : F32_2_0);
	float2 x3 = (ly3 ? *(float2*)(deltaY + yoffset + (OC << 1)) : F32_2_0);
	Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = ((W_k & OC_m1) * 9 - fh * 3 - fw)*IC;
	Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
	Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = Y_k >> LOC; char fhw = YIDX_W33[Idx];
		const int fh = fhw >> 2, fw = fhw & 3;
		const int yoffset = ((fh*OW + fw - Idx) << LOC) + Y_k;
		bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
		bool ly0 = ly && (tiw0 >= -fw) && (tiw0 < OW - fw);
		bool ly1 = ly && (tiw1 >= -fw) && (tiw1 < OW - fw);
		bool ly2 = ly && (tiw2 >= -fw) && (tiw2 < OW - fw);
		bool ly3 = ly && (tiw3 >= -fw) && (tiw3 < OW - fw);
		float2 x0 = (ly0 ? *(float2*)(deltaY + yoffset - OC) : F32_2_0);//Y0
		float2 x1 = (ly1 ? *(float2*)(deltaY + yoffset) : F32_2_0);//Y1
		float2 x2 = (ly2 ? *(float2*)(deltaY + yoffset + OC) : F32_2_0);//Y2
		float2 x3 = (ly3 ? *(float2*)(deltaY + yoffset + (OC << 1)) : F32_2_0);//Y3
		Ys[buf][(tx << 1)][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = ((W_k & OC_m1) * 9 - fh * 3 - fw)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + 9 * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, (IH, IW) % 4 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef CC_KERNEL2
#define CC_KERNEL2

#define cc_kernel2(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	CC_Kernel2<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,(IH*IW),IW, IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

//Size = 9, Time = 1.782 msec, Performace = 10845.9 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void CC_Kernel2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
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

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	deltaY += ((tn0*OH + tih0)*OW + tiw0 + 1)*OC;//deltaY += Y1
	j0 = j0 * IC + ic0;//j0 = ((n*OH + oh)*OW + ow)*IC + ic

	//compute area----------------------------------------------------
	for (int gk = 0, GK = (OC >> LB) * 3; gk < GK; gk++)//ooc_group (32 channels) * FH
	{
		const int ooc_group = gk / 3, fh = gk - ooc_group * 3;
		const int ooc = (ooc_group << LB);

		//load 4 elements from W[OC, FH, FW, IC]
		const int woc = ((ty & STEP_m1) << 1) + ooc;
		const int woffset0 = (woc*9 - fh*3)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset0 + (9 * IC));

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int yoc = ((tx & STEP_m1) << 1) + ooc;
		bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
		bool ly0 = ly && (tiw0 >= 0) && (tiw0 < OW);
		bool ly1 = ly && (tiw0 >= -1) && (tiw0 < OW - 1);
		bool ly2 = ly && (tiw0 >= -2) && (tiw0 < OW - 2);
		bool ly3 = ly && (tiw0 >= -3) && (tiw0 < OW - 3);
		const int yoffset0 = (fh*OW*OC) + yoc;//[n, oh, ow, oc]
		float2 y0 = (ly0 ? *(float2*)(deltaY + yoffset0 - OC) : F32_2_0);
		float2 y1 = (ly1 ? *(float2*)(deltaY + yoffset0) : F32_2_0);
		float2 y2 = (ly2 ? *(float2*)(deltaY + yoffset0 + OC) : F32_2_0);
		float2 y3 = (ly3 ? *(float2*)(deltaY + yoffset0 + (OC << 1)) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < 3; fw++) {
#pragma unroll 
			for (int ik = 0; ik < STEP2; ik++)
			{
				float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
				float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
				simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
				simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
				simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
				simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
				simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
				simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
				simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
			}
			float4 oy0 = Ys[buf][(tx << 1)][ty];//update_shared_memory
			float4 oy1 = Ys[buf][(tx << 1) + 1][ty];
			buf ^= 1;

			//load 4 elements from W[OC, FH, FW, IC]
			const int woffset = woffset0 - fw * IC;
			Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + (9 * IC));

			//load 4 elements from deltaY[N, OH, OW, OC]
			bool ly3 = ly && (tiw0 >= -(fw + 3)) && (tiw0 < OW - (fw + 3));
			float2 ny3 = (ly3 ? *(float2*)(deltaY + yoffset0 + (fw + 2)*OC) : F32_2_0);
			Ys[buf][(tx << 1)    ][ty] = float4{ oy0.y, oy0.z, oy0.w, ny3.x };
			Ys[buf][(tx << 1) + 1][ty] = float4{ oy1.y, oy1.z, oy1.w, ny3.y };
			__syncthreads();
		}
#pragma unroll 
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;
	}

	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, (IH, IW) % 4 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef CC_KERNEL3
#define CC_KERNEL3

#define cc_kernel3(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, LOC, ph, pw, GN, GM) \
	CC_Kernel3<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,(IH*IW),IW, IC,LOC, (2-ph),(2-pw),\
			ic_index,j_index)

//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 64]
//LB = 4: Size = 9, Time = 1.734 msec, Performace = 11146.1 GFlop/s
//LB = 3: Size = 9, Time = 1.812 msec, Performace = 10666.3 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.742 msec, Performace = 11094.9 GFlop/s
//LB = 3: Size = 9, Time = 1.888 msec, Performace = 10236.9 GFlop/s
//for: Feature = (12, 12), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 10.125, Time = 1.944 msec, Performace = 11184.8 GFlop/s
//LB = 3: Size = 10.125, Time = 2.126 msec, Performace = 10227.3 GFlop/s
//for: Feature = (8, 8), [N, IC, OC] = [128, 256, 512]
//LB = 4: Size = 9, Time = 1.85  msec, Performace = 10447.2 GFlop/s
//LB = 3: Size = 9, Time = 1.816 msec, Performace = 10642.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void CC_Kernel3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
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

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph; tiw0 = tiw0 - opw;
	deltaY += ((tn0*OH + tih0)*OW + tiw0 + 1)*OC;//deltaY += Y1
	j0 = j0 * IC + ic0;//j0 = ((n*OH + oh)*OW + ow)*IC + ic

	//compute area----------------------------------------------------
	for (int gk = 0, GK = (OC >> LB) * 3; gk < GK; gk++)//ooc_group (32 channels) * FH
	{
		const int ooc_group = gk / 3, fh = gk - ooc_group * 3;
		const int ooc = (ooc_group << LB);

		//load 4 elements from W[OC, FH, FW, IC]
		const int woc = ((ty & STEP_m1) << 1) + ooc;
		const int woffset0 = (woc * 9 - fh * 3)*IC;
		Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset0 + (9 * IC));

		//load 4 elements from deltaY[N, OH, OW, OC]
		const int yoc = ((tx & STEP_m1) << 1) + ooc;
		bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
		bool ly0 = ly && (tiw0 >= 0) && (tiw0 < OW);
		bool ly1 = ly && (tiw0 >= -1) && (tiw0 < OW - 1);
		bool ly2 = ly && (tiw0 >= -2) && (tiw0 < OW - 2);
		bool ly3 = ly && (tiw0 >= -3) && (tiw0 < OW - 3);
		const int yoffset0 = (fh*OW*OC) + yoc;
		float2 y0 = (ly0 ? *(float2*)(deltaY + yoffset0 - OC) : F32_2_0);
		float2 y1 = (ly1 ? *(float2*)(deltaY + yoffset0) : F32_2_0);
		float2 y2 = (ly2 ? *(float2*)(deltaY + yoffset0 + OC) : F32_2_0);
		float2 y3 = (ly3 ? *(float2*)(deltaY + yoffset0 + (OC << 1)) : F32_2_0);
		Ys[buf][(tx << 1)][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < 3; fw++) {
#pragma unroll 
			for (int ik = 0; ik < STEP2; ik++)
			{
				float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
				float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
				simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
				simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
				simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
				simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
				simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
				simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
				simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
			}
			float4 oy0 = Ys[buf][(tx << 1)][ty];//update_shared_memory
			float4 oy1 = Ys[buf][(tx << 1) + 1][ty];
			buf ^= 1;

			//load 4 elements from W[OC, FH, FW, IC]
			const int woffset = woffset0 - fw * IC;
			Ws[buf][(ty << 1)][tx] = *(float4*)(W + woffset);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + (9 * IC));

			//load 4 elements from deltaY[N, OH, OW, OC]
			bool ly3 = ly && (tiw0 >= -(fw + 3)) && (tiw0 < OW - (fw + 3));
			float2 ny3 = (ly3 ? *(float2*)(deltaY + yoffset0 + (fw + 2)*OC) : F32_2_0);
			Ys[buf][(tx << 1)][ty] = float4{ oy0.y, oy0.z, oy0.w, ny3.x };
			Ys[buf][(tx << 1) + 1][ty] = float4{ oy1.y, oy1.z, oy1.w, ny3.y };
			__syncthreads();
		}
#pragma unroll 
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP2][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP2][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;
	}

	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif
