#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_IMSR_UERNEL_H
#define DCONV3D_DX_KERNEL_SPLIT_IMSR_UERNEL_H

//IMS: INPUT_MOD_STEP
//R: remode: W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]
//(IH, IW) % (sh, sw) == 0
//CWstride = CFH * CFW * OC * IC
#ifndef DCONV3D_DX_UERNEL_SPLIT_IMSR_CALL
#define DCONV3D_DX_UERNEL_SPLIT_IMSR_CALL

//LB = log2(BLOCK_SIZE)

#define ksIms_u88R8_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOC, sh, sw, ph, pw, GN, GM) \
	ksIms_uernel_8_8R8_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, LOC, sh, sw, ph, pw, ic_index, j_index)

#define ksIms_u88R4_oc2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOC, sh, sw, ph, pw, GN, GM) \
	ksIms_uernel_8_8R4_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice * IW_slice), IW_slice, IC, LOC, sh, sw, ph, pw, ic_index, j_index)

#endif


//(IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE) == 0, OC is power of 2
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_UERNEL_8_8R8_OC_2POW
#define DCONV3D_DX_KERNEL_SPLIT_IMS_UERNEL_8_8R8_OC_2POW

//for: (64, 64) -> (32, 32), [FH, FW] = 3, [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 72, Time = 3.867 msec, Performace = 39984.2 GFlop/s
//LB = 3: Size = 72, Time = 4.288 msec, Performace = 36058.5 GFlop/s
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 72, Time = 3.666 msec, Performace = 42176.4 GFlop/s
//LB = 3: Size = 72, Time = 4.051 msec, Performace = 38168.1 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 72, Time = 3.6   msec, Performace = 42949.7 GFlop/s
//LB = 3: Size = 72, Time = 4.014 msec, Performace = 38519.9 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void ksIms_uernel_8_8R8_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__     CW, int FH, int FW, int CWstride,
	      float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int LOC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GZ = sh*sw 
	const int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	const int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	const int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	const int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	const int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	const int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW << LOC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += tic0;//CW[y, x, 0, 0, tic0]

	//prepare for CW[y, x, CFH, CFW, OC, IC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3); Ims_n_ih_iw(tj0, yn0, yih0, yiw0);
	const int tohs0 = (yih0 + ph) / sh - oph;
	const int tows0 = (yiw0 + ((tx >= STEP) << 3) + pw) / sw - opw;
	const int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	const int Y1 = ((yn0*OH + tohs0)*OW + tows1) << LOC;
	OH -= tohs0;//OH = OH - tohs0 = OH_m_tohs0

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0 = bj0  + (uy << 3); Ims_n_ih_iw(xj0, xn0, xih0, xiw0);
	const int X0 = ((xn0*IH_IW_slice*sh*sw) + xih0 * IW_slice*sw + xiw0)*IC + xic0;

	//compute area-------------------------------------------------
	//load 4 elem from deltaY
	int Y_k = (tx & STEP_m1) << 1;
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	const int yoffset = (Y_fhr << LOC) * OW + Y_k;
	float2 y0 = (ly0 ? *(float2*)(deltaY + Y1 - (1 << LOC) + yoffset) : F32_2_0);
	float2 y1 = (ly1 ? *(float2*)(deltaY + Y1              + yoffset) : F32_2_0);
	float2 y2 = (ly2 ? *(float2*)(deltaY + Y1 + (1 << LOC) + yoffset) : F32_2_0);
	float2 y3 = (ly3 ? *(float2*)(deltaY + Y1 + (2 << LOC) + yoffset) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

	//load 4 elem from W
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = W_k * IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

			simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
			simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
			simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
			simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
			simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = W_k * IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);

		//load 4 elem from Y
		int Y_k = ok + ((tx & STEP_m1) << 1);
		Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
		bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH);
		bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		const int yoffset = (Y_fhr << LOC) * OW + Y_k;
		float2 y0 = (ly0 ? *(float2*)(deltaY + Y1 - (1 << LOC) + yoffset) : F32_2_0);
		float2 y1 = (ly1 ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 y2 = (ly2 ? *(float2*)(deltaY + Y1 + (1 << LOC) + yoffset) : F32_2_0);
		float2 y3 = (ly3 ? *(float2*)(deltaY + Y1 + (2 << LOC) + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)    ][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
		simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
		simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
		simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
		simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	IC *= sw;
	const int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	const int X4 = X3 + IC, X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

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


//(IH_slice, IW_slice) % 4 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE) == 0, OC is power of 2
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_UERNEL_8_8R4_OC_2POW
#define DCONV3D_DX_KERNEL_SPLIT_IMS_UERNEL_8_8R4_OC_2POW

//for: (64, 64) -> (32, 32), [FH, FW] = 3, [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 72, Time = 3.875 msec, Performace = 39901.6 GFlop/s
//LB = 3: Size = 72, Time = 4.332 msec, Performace = 35692.2 GFlop/s
//for: (32, 32) -> (16, 16), [FH, FW] = 3, [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 72, Time = 3.665 msec, Performace = 42187.9 GFlop/s
//LB = 3: Size = 72, Time = 4.089 msec, Performace = 37813.4 GFlop/s
//for: (16, 16) -> ( 8,  8), [FH, FW] = 3, [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 72, Time = 3.624 msec, Performace = 42665.2 GFlop/s
//LB = 3: Size = 72, Time = 4.074 msec, Performace = 37952.6 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void ksIms_uernel_8_8R4_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__     CW, int FH, int FW, int CWstride,
	      float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int LOC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for GZ = sh*sw 
	const int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	const int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	const int ihs = -ph + (-(ph > 0) & ((ph + sh - 1) / sh * sh));
	const int iws = -pw + (-(pw > 0) & ((pw + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	const int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	const int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW << LOC, GK = CFH * CFW_OC;

	//prepare for CW[y, x, CFH, CFW, OC, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += tic0;//CW[y, x, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3), tj4 = tj0 + 4;
	Ims_n_ih_iw(tj0, yn0, yih0, yiw0);
	Ims_n_ih_iw(tj4, yn4, yih4, yiw4);
	bool flagX = (tx >= STEP);
	const int tohs0 = (IF_int(flagX, yih4, yih0) + ph) / sh - oph;
	const int tows0 = (IF_int(flagX, yiw4, yiw0) + pw) / sw - opw;
	const int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	const int Y1 = ((yn0*OH + tohs0)*OW + tows1) << LOC;
	OH -= tohs0;//OH = OH - tohs0 = OH_m_tohs0

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3), xj4 = xj0 + 4;
	Ims_n_ih_iw(xj0, xn0, xih0, xiw0);
	Ims_n_ih_iw(xj4, xn4, xih4, xiw4);
	IH_IW_slice *= sh * sw; //IH_IW_slice -> IH * IW
	IW_slice    *= sw;      //IW_slice    -> IW
	const int X0 = ((xn0*IH_IW_slice) + xih0 * IW_slice + xiw0)*IC + xic0;
	const int X4 = ((xn4*IH_IW_slice) + xih4 * IW_slice + xiw4)*IC + xic0;

	//compute area-------------------------------------------------
	//load 4 elem from deltaY
	int Y_k = (tx & STEP_m1) << 1;
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	const int yoffset = (Y_fhr << LOC) * OW + Y_k;
	float2 y0 = (ly0 ? *(float2*)(deltaY + Y1 - (1 << LOC) + yoffset) : F32_2_0);
	float2 y1 = (ly1 ? *(float2*)(deltaY + Y1              + yoffset) : F32_2_0);
	float2 y2 = (ly2 ? *(float2*)(deltaY + Y1 + (1 << LOC) + yoffset) : F32_2_0);
	float2 y3 = (ly3 ? *(float2*)(deltaY + Y1 + (2 << LOC) + yoffset) : F32_2_0);
	Ys[0][(tx << 1)    ][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

	//load 4 elem from W
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = W_k * IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

			simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
			simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
			simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
			simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
			simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = W_k * IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);

		//load 4 elem from Y
		int Y_k = ok + ((tx & STEP_m1) << 1);
		Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
		bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH);
		bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		const int yoffset = (Y_fhr << LOC) * OW + Y_k;
		float2 y0 = (ly0 ? *(float2*)(deltaY + Y1 - (1 << LOC) + yoffset) : F32_2_0);
		float2 y1 = (ly1 ? *(float2*)(deltaY + Y1 + yoffset) : F32_2_0);
		float2 y2 = (ly2 ? *(float2*)(deltaY + Y1 + (1 << LOC) + yoffset) : F32_2_0);
		float2 y3 = (ly3 ? *(float2*)(deltaY + Y1 + (2 << LOC) + yoffset) : F32_2_0);
		Ys[buf][(tx << 1)    ][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	IC *= sw;
	const int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	const int X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

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


#endif