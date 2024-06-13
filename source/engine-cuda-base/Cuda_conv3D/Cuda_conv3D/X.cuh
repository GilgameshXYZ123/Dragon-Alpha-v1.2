


 


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), IC % BLOCK_SIZE == 0, IC is power of 2
//LB = 4, IC % 16 == 0  
//LB = 3, IC %  8 == 0
#ifndef U3V3O3_KERNEL1
#define U3V3O3_KERNEL1

#define u3v3o3_k1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	u3v3o3_kernel1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X,IH,IW, CW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index,j_index)

//for: (64, 64) -> (32, 32), [FH, FW] = 3, [N, IC, OC] = [128, 64, 128]
//LB = 4: Size = 9, Time = 1.87748 msec, Performace = 10294.3 GFlop/s
//LB = 3: Size = 9, Time = 2.15023 msec, Performace =  8988.5 GFlop/s
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 9, Time = 1.81573 msec, Performace = 10644.4 GFlop/s
//LB = 3: Size = 9, Time = 2.05602 msec, Performace =  9400.4 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void u3v3o3_kernel1(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//FH = FW = 3
	      float* __restrict__  Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 results
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << LB << 3) + oc_index;
	const int toc0 = boc0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += toc0;//CW[0, 0, 0, toc0]

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	const int tow1 = tow0 + sw;
	const int tow2 = tow1 + sw;
	const int tow3 = tow2 + sw;
	X += ((tn0*IH + toh0)*IW + tow1) << LIC;//X += X1;

	//prepare for Y[N, OH, OW, OC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 3);
	const int yj0  = bj0  + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//oc = f(by), j = f(bx) -> (n, oh, ow)

	//compute area------------------------------------------------------
	const int GK = 9 << LIC;//GK = FH * FW * IC
	const int sw_IC = sw << LIC;

	//load 4 elements from X[N, IH, IW, IC]
	const int X_k = (tx & STEP_m1) << 1;
	const int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
	const int X_fh = fhw >> 2, X_fw = fhw & 3;
	const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
	float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
	float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
	float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
	Xs[0][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
	Xs[0][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };

	//load 4 elements from W[OC, FH, FW, IC]
	const int woffset = ((ty & STEP_m1) << 1) * OC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];
			float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];

			simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);
			simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);
			simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);
			simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);
			simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		const int woffset = (ok + ((ty & STEP_m1) << 1))*OC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + OC);

		//load 4 elements from X[N, IH, IW, IC]
		const int X_k = ok + ((tx & STEP_m1) << 1);
		const int Idx = X_k >> LIC; char fhw = XIDX_W3[Idx];
		const int X_fh = fhw >> 2, X_fw = fhw & 3;
		const int xoffset = ((X_fh * IW + X_fw - Idx) << LIC) + X_k;
		bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
		bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		float2 x0 = (lx0 ? *(float2*)(X + xoffset - sw_IC) : F32_2_0);
		float2 x1 = (lx1 ? *(float2*)(X + xoffset) : F32_2_0);
		float2 x2 = (lx2 ? *(float2*)(X + xoffset + sw_IC) : F32_2_0);
		float2 x3 = (lx3 ? *(float2*)(X + xoffset + (sw_IC << 1)) : F32_2_0);
		Xs[buf][(tx << 1)    ][ty] = float4{ x0.x, x1.x, x2.x, x3.x };
		Xs[buf][(tx << 1) + 1][ty] = float4{ x0.y, x1.y, x2.y, x3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 b0 = Xs[buf][ik][uy], b1 = Xs[buf][ik + STEP2][uy];
		float4 a0 = Ws[buf][ik][ux], a1 = Ws[buf][ik + STEP2][ux];

		simdMM4( v0, b0.x, a0); simdMM4( v1, b0.x, a1);
		simdMM4( v2, b0.y, a0); simdMM4( v3, b0.y, a1);
		simdMM4( v4, b0.z, a0); simdMM4( v5, b0.z, a1);
		simdMM4( v6, b0.w, a0); simdMM4( v7, b0.w, a1);
		simdMM4( v8, b1.x, a0); simdMM4( v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif



