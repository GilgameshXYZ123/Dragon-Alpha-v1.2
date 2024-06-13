#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_UERNEL_H
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_UERNEL_H

//IMS: INPUT_MOD_STEP
//R: remode: W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]
//S2: sh = sw = 2
//(IH, IW) % (sh, sw) == 0
//CWstride = CFH * CFW * OC * IC
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_UERNEL_CALL
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2R_UERNEL_CALL

//======[Common]============================================================
#define ksV2_Ims2_u88R(stream, LB, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, OC, ph, pw, GN, N) \
	ksV2_Ims2_uernel_8_8R<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, OC, ph, pw,\
			ic_index, n_index)

//======[FW = 4 or 3 or 2] -> [CFW is power of 2]===========================
#define ksV2_Ims2_u88R_CFW_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, CW, FH, FW, deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, N) \
	ksV2_Ims2_uernel_8_8R_CFW_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, FH, FW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, LOC, ph, pw,\
			ic_index, n_index, (GN>>LB>>3))

//======[FW = 5] -> [CFW = 3 or 2], [OH, OW >= 2]===========================
#define ksV2_Ims2_u88R_W5_oc2pow(stream, LB, ic_index, n_index, deltaY, OH, OW, CW, deltaX, IH_slice, IW_slice, IC, LOC, ph, pw, GN, N) \
	ksV2_Ims2_uernel_8_8R_W5_OC2pow<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3((GN>>LB>>3), (N>>LB>>3), (4*IH_slice*IW_slice)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, OH, OW, CW, CWstride, deltaX, (IH_slice*IW_slice), IW_slice, IC, LOC, ph, pw,\
			ic_index, n_index)

#endif


//======[Common]==============================================================
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % BLOCK_SIZE == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R

//(IH, IW) = 64, (OH, OW) = 32, (N, IC, OC) = (128, 128, 128):
//LB = 4: Size = 72, Time = 3.506 msec, Performace = 44101.2 GFlop/s
//LB = 3: Size = 72, Time = 4.401 msec, Performace = 35132.7 GFlop/s
//(IH, IW) = 32, (OH, OW) = 16, (N, IC, OC) = (128, 256, 256):
//LB = 4: Size = 72, Time = 3.189 msec, Performace = 48485 GFlop/s
//LB = 3: Size = 72, Time = 3.565 msec, Performace = 43371.3 GFlop/s
//(IH, IW) = 28, (OH, OW) = 14, (N, IC, OC) = (128, 256, 256):
//LB = 4: Size = 55.125, Time = 2.473 msec, Performace = 47869 GFlop/s
//(IH, IW) = 16, (OH, OW) =  8, (N, IC, OC) = (128, 512, 512):
//LB = 4: Size = 72, Time = 3.018 msec, Performace = 51232.2 GFlop/s
//LB = 3: Size = 72, Time = 3.432 msec, Performace = 45052.1 GFlop/s
//(IH, IW) = 12, (OH, OW) =  6, (N, IC, OC) = (512, 256, 256):
//LB = 4: Size = 40.5, Time = 1.749 msec, Performace = 49727.3 GFlop/s
//(IH, IW) =  8, (OH, OW) =  4, (N, IC, OC) = (512, 512, 512):
//LB = 4: Size = 72, Time = 2.772 msec, Performace = 55778.8 GFlop/s
//LB = 3: Size = 72, Time = 3.099 msec, Performace = 49893.1 GFlop/s
//(IH, IW) =  4, (OH, OW) =  2, (N, IC, OC) = (512, 512, 1024):
//LB = 4: Size = 36, Time = 1.302 msec, Performace = 59377.4 GFlop/s
//LB = 3: Size = 36, Time = 1.41  msec, Performace = 54829.4 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void ksV2_Ims2_uernel_8_8R(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__     CW, int FH, int FW, int CWstride,
	      float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int n_index)
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

	//prepare for GZ((y, x), ih, iw) = sh * sw * IH_slice * IW_slice 
	int bz = blockIdx.z;
	int yx = bz / IH_IW_slice; bz -= yx * IH_IW_slice;//bz %= IH_IW_slice
	int ih = bz / IW_slice, iw = bz - ih * IW_slice;//bz % IW_slice
	CW += yx * CWstride;//CW[y, x]

	const int y = yx >> 1, x = yx & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	const int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	const int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));
	ih = (ih << 1) + ihs;
	iw = (iw << 1) + iws;

	//prepare for GK(bz(y, x)) = tCFW * tCFH * OC
	const int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	const int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int ohs = ((ih + ph) >> 1) - oph;
	const int ows = ((iw + pw) >> 1) - opw;

	const int fhs = -IF_int((ohs < 0), ohs, 0);
	const int fws = -IF_int((ows < 0), ows, 0);
	int fhe = OH - ohs; fhe = IF_int((fhe > CFH), CFH, fhe);
	int fwe = OW - ows; fwe = IF_int((fwe > CFW), CFW, fwe);
	const int tCFH = fhe - fhs;
	const int tCFW = fwe - fws;
	const int tCFW_OC = tCFW * OC, GK = tCFH * tCFW_OC;
	CW += (fhs*CFW + fws)*OC*IC;//CW[y, x, fhs, fws, 0, 0]
	deltaY += (fhs*OW + fws)*OC;//deltaY[0, fhs, fws, 0]

	//prepare for CW[y, x, CFH, CFW, OC, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += tic0;//CW[y, x, fhs, fws, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bn0 = (blockIdx.y << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + ohs)*OW + ows) * OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride, Y2 = Y1 + Ystride, Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xn0  = bn0  + (uy << 3);
	const int xic0 = bic0 + (ux << 3);
	IH_IW_slice <<= 2;//IH_IW_slice * sh * sw -> IH * IW
	IW_slice    <<= 1;//IW_slice * sw         -> IW
	const int X0 = ((xn0*IH_IW_slice) + ih * IW_slice + iw)*IC + xic0;

	//compute area-------------------------------------------------
	const int SY = (OW - tCFW)*OC;
	const int SW = (CFW - tCFW)*OC;

	//load 4 elem from deltaY
	const int Y_k = (tx & STEP_m1) << 1;
	const int fhr = Y_k / tCFW_OC;
	const int yoffset = fhr * SY + Y_k;
	float2 dy0 = *(float2*)(deltaY + Y0 + yoffset);
	float2 dy1 = *(float2*)(deltaY + Y1 + yoffset);
	float2 dy2 = *(float2*)(deltaY + Y2 + yoffset);
	float2 dy3 = *(float2*)(deltaY + Y3 + yoffset);
	Ys[0][(tx << 1)    ][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

	//load 4 elem from W
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset = (fhr * SW + W_k)*IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

			//transposed compute core: (W * dY)^T
			simdMM4(v0,  y0.x, w0); simdMM4(v1,  y0.x, w1);
			simdMM4(v2,  y0.y, w0); simdMM4(v3,  y0.y, w1);
			simdMM4(v4,  y0.z, w0); simdMM4(v5,  y0.z, w1);
			simdMM4(v6,  y0.w, w0); simdMM4(v7,  y0.w, w1);
			simdMM4(v8,  y1.x, w0); simdMM4(v9,  y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from Y
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int fhr = Y_k / tCFW_OC;
		const int yoffset = fhr * SY + Y_k;
		float2 dy0 = *(float2*)(deltaY + Y0 + yoffset);
		float2 dy1 = *(float2*)(deltaY + Y1 + yoffset);
		float2 dy2 = *(float2*)(deltaY + Y2 + yoffset);
		float2 dy3 = *(float2*)(deltaY + Y3 + yoffset);
		Ys[buf][(tx << 1)    ][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

		//load 4 elem from W
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset = (fhr * SW + W_k)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset + IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		//transposed compute core: (W * dY)^T
		simdMM4(v0,  y0.x, w0); simdMM4(v1,  y0.x, w1);
		simdMM4(v2,  y0.y, w0); simdMM4(v3,  y0.y, w1);
		simdMM4(v4,  y0.z, w0); simdMM4(v5,  y0.z, w1);
		simdMM4(v6,  y0.w, w0); simdMM4(v7,  y0.w, w1);
		simdMM4(v8,  y1.x, w0); simdMM4(v9,  y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Xstride = IH_IW_slice * IC;//IH * IW * IC
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


//======[FW = 4 or 3 or 2] -> [CFW is power of 2]=============================
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, OC is power of 2
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R_CFW_OC2POW
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R_CFW_OC2POW

//(IH, IW) = 64, (OH, OW) = 32, (N, IC, OC) = (128, 128, 128):
//LB = 4: Size = 72, Time = 3.519 msec, Performace = 43938.3 GFlop/s
//LB = 3: Size = 72, Time = 4.269 msec, Performace = 36219   GFlop/s
//(IH, IW) = 32, (OH, OW) = 16, (N, IC, OC) = (128, 256, 256):
//LB = 4: Size = 72, Time = 3.151 msec, Performace = 49069.8 GFlop/s
//LB = 3: Size = 72, Time = 3.547 msec, Performace = 43591.4 GFlop/s
//(IH, IW) = 28, (OH, OW) = 14, (N, IC, OC) = (128, 256, 256):
//LB = 4: Size = 55.125, Time = 2.473 msec, Performace = 47869 GFlop/s
//(IH, IW) = 24, (OH, OW) = 12, (N, IC, OC) = (128, 256, 256):
//LB = 4: Size = 40.5, Time = 1.848 msec, Performace = 47063.4 GFlop/s
//(IH, IW) = 16, (OH, OW) =  8, (N, IC, OC) = (128, 512, 512):
//LB = 4: Size = 72, Time = 3.02  msec, Performace = 51198.3 GFlop/s
//LB = 3: Size = 72, Time = 3.432 msec, Performace = 45052.1 GFlop/s
//(IH, IW) = 12, (OH, OW) =  6, (N, IC, OC) = (512, 256, 256):
//LB = 4: Size = 40.5, Time = 1.76 msec, Performace = 49416.5 GFlop/s
//(IH, IW) =  8, (OH, OW) =  4, (N, IC, OC) = (512, 512, 512):
//LB = 4: Size = 72, Time = 2.79  msec, Performace = 55418.9 GFlop/s
//LB = 3: Size = 72, Time = 3.165 msec, Performace = 48852.7 GFlop/s
//(IH, IW) =  4, (OH, OW) =  2, (N, IC, OC) = (512, 512, 1024):
//LB = 4: Size = 36, Time = 1.291 msec, Performace = 59883.4 GFlop/s
//LB = 3: Size = 36, Time = 1.419 msec, Performace = 54481.6 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void ksV2_Ims2_uernel_8_8R_CFW_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__     CW, int FH, int FW, int CWstride,
	      float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int LOC,
	int ph, int pw,
	int ic_index, int n_index, int GX)
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

	//prepare for GZ((y, x), ih, iw) = sh * sw * IH_slice * IW_slice 
	int bz = blockIdx.z;
	int yx = bz / IH_IW_slice; bz -= yx * IH_IW_slice;//bz %= IH_IW_slice
	int ih = bz / IW_slice, iw = bz - ih * IW_slice;//bz % IW_slice
	CW += yx * CWstride;//CW[y, x]

	const int y = yx >> 1, x = yx & 1;//x = yx % sw
	ph = ph - y; pw = pw - x;
	const int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	const int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));
	ih = (ih << 1) + ihs;
	iw = (iw << 1) + iws;

	//prepare for GK(bz) = tCFW * tCFH * OC
	const int CFH = (FH - y + 1) >> 1, oph = CFH - 1;
	const int CFW = (FW - x + 1) >> 1, opw = CFW - 1;
	const int ohs = ((ih + ph) >> 1) - oph;
	const int ows = ((iw + pw) >> 1) - opw;

	const int fhs = -IF_int((ohs < 0), ohs, 0);
	const int fws = -IF_int((ows < 0), ows, 0);
	int fhe = OH - ohs; fhe = IF_int((fhe > CFH), CFH, fhe);
	int fwe = OW - ows; fwe = IF_int((fwe > CFW), CFW, fwe);
	const int tCFH = fhe - fhs;
	const int tCFW = fwe - fws;

	//tCFW = 2 or 1, so (tCFW >> 1) == log2(tCFW)
	const int LtCFW_OC = (tCFW >> 1) + LOC, GK = tCFH << LtCFW_OC;

	//prepare for CW[y, x, CFH, CFW, OC, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += (((fhs*CFW + fws)*IC) << LOC) + tic0;//CW[y, x, fhs, fws, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bn0 = (blockIdx.y << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + ohs + fhs)*OW + ows + fws) << LOC;//Y[tn0, fhs + ohs, fws + owss, 0]
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xn0  = bn0  + (uy << 3);
	IH_IW_slice <<= 2;//IH_IW_slice -> IH * IW
	IW_slice    <<= 1;//IW_slice    -> IW
	const int X0 = ((xn0*IH_IW_slice) + ih * IW_slice + iw)*IC + xic0;
	
	//compute area-------------------------------------------------
	const int SY = (OW - tCFW) << LOC;
	const int SW = (CFW - tCFW) << LOC;

	//load 4 elem from deltaY
	const int Y_k = (tx & STEP_m1) << 1;
	const int fhr = Y_k >> LtCFW_OC;
	const int yoffset = fhr * SY + Y_k;
	const int yoffset0 = yoffset + Y0;
	const int yoffset1 = yoffset + Y1;
	const int yoffset2 = yoffset + Y2;
	const int yoffset3 = yoffset + Y3;
	float2 dy0 = *(float2*)(deltaY + yoffset0);
	float2 dy1 = *(float2*)(deltaY + yoffset1);
	float2 dy2 = *(float2*)(deltaY + yoffset2);
	float2 dy3 = *(float2*)(deltaY + yoffset3);
	Ys[0][(tx << 1)    ][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

	//load 4 elem from W
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = (fhr * SW + W_k)*IC;
	const int woffset1 = woffset0 + IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

			//transposed compute core: (W * dY)^T
			simdMM4(v0,  y0.x, w0); simdMM4(v1,  y0.x, w1);
			simdMM4(v2,  y0.y, w0); simdMM4(v3,  y0.y, w1);
			simdMM4(v4,  y0.z, w0); simdMM4(v5,  y0.z, w1);
			simdMM4(v6,  y0.w, w0); simdMM4(v7,  y0.w, w1);
			simdMM4(v8,  y1.x, w0); simdMM4(v9,  y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from Y
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const int fhr = Y_k >> LtCFW_OC;
		const int yoffset = fhr * SY + Y_k;
		const int yoffset0 = yoffset + Y0;
		const int yoffset1 = yoffset + Y1;
		const int yoffset2 = yoffset + Y2;
		const int yoffset3 = yoffset + Y3;
		float2 dy0 = *(float2*)(deltaY + yoffset0);
		float2 dy1 = *(float2*)(deltaY + yoffset1);
		float2 dy2 = *(float2*)(deltaY + yoffset2);
		float2 dy3 = *(float2*)(deltaY + yoffset3);
		Ys[buf][(tx << 1)    ][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

		//load 4 elem from W
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fhr * SW + W_k)*IC;
		const int woffset1 = woffset0 + IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

		//transposed compute core: (W * dY)^T
		simdMM4(v0,  y0.x, w0); simdMM4(v1,  y0.x, w1);
		simdMM4(v2,  y0.y, w0); simdMM4(v3,  y0.y, w1);
		simdMM4(v4,  y0.z, w0); simdMM4(v5,  y0.z, w1);
		simdMM4(v6,  y0.w, w0); simdMM4(v7,  y0.w, w1);
		simdMM4(v8,  y1.x, w0); simdMM4(v9,  y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Xstride = IH_IW_slice * IC;//IH * IW * IC
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


//======[FW = 5] -> [CFW = 3 or 2], [OH, OW >= 2]=============================
//LB = 4: OC % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R_W5_OC2POW
#define DCONV3D_DX_KERNEL_SPLIT_V2_IMS2_UERNEL_8_8R_W5_OC2POW

//(IH, IW) = 64, (OH, OW) = 32, (N, IC, OC) = (128, 128, 64):
//LB = 4: Size = 100, Time = 4.317 msec, Performace = 49744.8 GFlop/s
//LB = 3: Size = 100, Time = 4.854 msec, Performace = 44241.5 GFlop/s
//(IH, IW) = 32, (OH, OW) = 16, (N, IC, OC) = (128, 128, 128):
//LB = 4: Size = 50, Time = 2.102 msec, Performace = 51081.9 GFlop/s
//LB = 3: Size = 50, Time = 2.515 msec, Performace = 42693.5 GFlop/s
//(IH, IW) = 28, (OH, OW) = 14, (N, IC, OC) = (128, 128, 128):
//LB = 4: Size = 38.2812, Time = 1.703 msec, Performace = 48272.7 GFlop/s
//LB = 3: Size = 38.2812, Time = 1.971 msec, Performace = 41709   GFlop/s
//(IH, IW) = 24, (OH, OW) = 12, N = 128, (IC, OC) = (128, 256):
//LB = 4: Size = 56.25, Time = 2.538 msec, Performace = 47594.9 GFlop/s
//(IH, IW) = 16, (OH, OW) =  8, (N, IC, OC) = (128, 256, 256):
//LB = 4: Size = 50, Time = 1.974 msec, Performace = 54394.2 GFlop/s
//LB = 3: Size = 50, Time = 2.307 msec, Performace = 46542.8 GFlop/s
//(IH, IW) = 12, (OH, OW) =  6, (N, IC, OC) = (128, 256, 256):
//LB = 4: Size = 28.125, Time = 1.136 msec, Performace = 53167.2 GFlop/s
//(IH, IW) =  8, (OH, OW) =  4, (N, IC, OC) = (128, 512, 512):
//LB = 4: Size = 50, Time = 1.814 msec, Performace = 59191.9 GFlop/s
//LB = 3: Size = 50, Time = 2.101 msec, Performace = 51106.2 GFlop/s
//(IH, IW) =  4, (OH, OW) =  2, (N, IC, OC) = (512, 512, 512):
//LB = 4: Size = 50, Time = 1.291 msec, Performace = 83171.3 GFlop/s
//LB = 3: Size = 50, Time = 1.422 msec, Performace = 75509.3 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void ksV2_Ims2_uernel_8_8R_W5_OC2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__     CW, int CWstride,//FH = FW = 5
	      float* __restrict__ deltaX, int IH_IW_slice, int IW_slice,
	int IC, int LOC,
	int ph, int pw,
	int ic_index, int n_index)
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

	//prepare for GZ((y, x), ih, iw) = sh * sw * IH_slice * IW_slice 
	int bz = blockIdx.z;
	int yx = bz / IH_IW_slice; bz -= yx * IH_IW_slice;//bz %= IH_IW_slice
	int ih = bz / IW_slice, iw = bz - ih * IW_slice;//bz % IW_slice
	CW += yx * CWstride;//CW[y, x]

	const int y = yx >> 1, x = yx & 1;//x = bz % sw
	ph = ph - y; pw = pw - x;
	const int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	const int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));
	ih = (ih << 1) + ihs;
	iw = (iw << 1) + iws;

	//prepare for GK(bz(y, x)) = tCFW * tCFH * OC
	const int CFH = (6 - y) >> 1, oph = CFH - 1;
	const int CFW = (6 - x) >> 1, opw = CFW - 1;
	const int ohs = ((ih + ph) >> 1) - oph;
	const int ows = ((iw + pw) >> 1) - opw;

	const int fhs = -IF_int((ohs < 0), ohs, 0);
	const int fws = -IF_int((ows < 0), ows, 0);
	int fhe = OH - ohs; fhe = IF_int((fhe > CFH), CFH, fhe);
	int fwe = OW - ows; fwe = IF_int((fwe > CFW), CFW, fwe);
	const int tCFH = fhe - fhs;
	const int tCFW = fwe - fws;
	const int tCFW_OC = tCFW << LOC, GK = tCFH * tCFW_OC;
	CW += ((fhs*CFW + fws)*IC) << LOC;//CW[y, x, fhs, fws, 0, 0]
	deltaY += (fhs*OW + fws) << LOC;//deltaY[0, fhs, fws, 0]

	const int fh_idx = (tCFH == 2) + ((tCFH == 3) << 1);
	const int fw_idx = (tCFW == 2) + ((tCFW == 3) << 1);
	const int fhw_offset = (fh_idx * 3 + fw_idx) * 9;

	//prepare for CW[y, x, CFH, CFW, OC, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	CW += tic0;//CW[y, x, fhs, fws, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bn0 = (blockIdx.y << LB << 3) + n_index;
	const int tn0 = bn0 + (ty << 3) + ((tx >= STEP) << 2);
	const int Y0 = ((tn0*OH + ohs)*OW + ows) << LOC;
	const int Ystride = (OH * OW) << LOC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xn0  = bn0  + (uy << 3);
	const int xic0 = bic0 + (ux << 3);
	IH_IW_slice <<= 2;//IH_IW_slice * sh * sw -> IH * IW
	IW_slice    <<= 1;//IW_slice * sw         -> IW
	const int X0 = ((xn0*IH_IW_slice) + ih * IW_slice + iw)*IC + xic0;

	//compute area-------------------------------------------------
	const int SY = (OW - tCFW) << LOC;
	const int SW = (CFW - tCFW) << LOC;

	//load 4 elem from deltaY
	const int Y_k = (tx & STEP_m1) << 1;
	const char fhr = YIDX_V2_W3P2_FH[fhw_offset + (Y_k >> LOC)];
	const int yoffset = fhr * SY + Y_k;
	const int yoffset0 = yoffset + Y0;
	const int yoffset1 = yoffset + Y1;
	const int yoffset2 = yoffset + Y2;
	const int yoffset3 = yoffset + Y3;
	float2 dy0 = *(float2*)(deltaY + yoffset0);
	float2 dy1 = *(float2*)(deltaY + yoffset1);
	float2 dy2 = *(float2*)(deltaY + yoffset2);
	float2 dy3 = *(float2*)(deltaY + yoffset3);
	Ys[0][(tx << 1)    ][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
	Ys[0][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

	//load 4 elem from W
	const int W_k = (ty & STEP_m1) << 1;
	const int woffset0 = (fhr * SW + W_k)*IC;
	const int woffset1 = woffset0 + IC;
	Ws[0][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
	Ws[0][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
	__syncthreads();

	for (int ok = STEP2; ok < GK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
			float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

			//transposed compute core: (W * dY)^T
			simdMM4(v0,  y0.x, w0); simdMM4(v1,  y0.x, w1);
			simdMM4(v2,  y0.y, w0); simdMM4(v3,  y0.y, w1);
			simdMM4(v4,  y0.z, w0); simdMM4(v5,  y0.z, w1);
			simdMM4(v6,  y0.w, w0); simdMM4(v7,  y0.w, w1);
			simdMM4(v8,  y1.x, w0); simdMM4(v9,  y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from Y
		const int Y_k = ok + ((tx & STEP_m1) << 1);
		const char fhr = YIDX_V2_W3P2_FH[fhw_offset + (Y_k >> LOC)];
		const int yoffset = fhr * SY + Y_k;
		const int yoffset0 = yoffset + Y0;
		const int yoffset1 = yoffset + Y1;
		const int yoffset2 = yoffset + Y2;
		const int yoffset3 = yoffset + Y3;
		float2 dy0 = *(float2*)(deltaY + yoffset0);
		float2 dy1 = *(float2*)(deltaY + yoffset1);
		float2 dy2 = *(float2*)(deltaY + yoffset2);
		float2 dy3 = *(float2*)(deltaY + yoffset3);
		Ys[buf][(tx << 1)    ][ty] = float4{ dy0.x, dy1.x, dy2.x, dy3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ dy0.y, dy1.y, dy2.y, dy3.y };

		//load 4 elem from W
		const int W_k = ok + ((ty & STEP_m1) << 1);
		const int woffset0 = (fhr * SW + W_k)*IC;
		const int woffset1 = woffset0 + IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(CW + woffset0);
		Ws[buf][(ty << 1) + 1][tx] = *(float4*)(CW + woffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];
		float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];

		//transposed compute core: (W * dY)^T
		simdMM4(v0,  y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2,  y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4,  y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6,  y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8,  y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	const int Xstride = IH_IW_slice * IC;//IH * IW * IC
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

#endif