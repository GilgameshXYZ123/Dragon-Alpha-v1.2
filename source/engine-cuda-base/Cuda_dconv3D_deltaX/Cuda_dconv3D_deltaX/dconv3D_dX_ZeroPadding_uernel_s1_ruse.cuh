#pragma once

#ifndef DECONV_3D_DX_ZERO_PADDING_UERNEL_S1_RUSE_H
#define DECONV_3D_DX_ZERO_PADDING_UERNEL_S1_RUSE_H

//Unsparse Matrix Method:
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
#ifndef DECONV3D_DX_ZERO_PADDING_UERNEL_S1_RUSE_CALL
#define DECONV3D_DX_ZERO_PADDING_UERNEL_S1_RUSE_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]===========================================================
#define u88s1x4_ruse(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8_s1x4_ruse<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, FH, FW>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC), deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//======[FH = FW = 3]======================================================
//IW % 4 == 0
#define u88s1W3x4_ruse(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_uernel_8_8_s1W3x4_ruse<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + (IC<<3)), deltaX,(IH*IW),IW, IC,OC, (2-ph),(2-pw),\
			ic_index,j_index)

#endif


//======[Common]===========================================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, IW % 4 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV_3D_DX_ZERO_PADDING_UERNEL_8_8_S1X4_RUSE
#define DECONV_3D_DX_ZERO_PADDING_UERNEL_8_8_S1X4_RUSE

//when [FH, FW] = 3, [ph, pw] = 1:
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 64]
//LB = 4: Size = 18, Time = 3.21 msec, Performace = 12042 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 18, Time = 3.256 msec, Performace = 11871.8 GFlop/s
//LB = 3: Size = 18, Time = 3.524 msec, Performace = 10969   GFlop/s
//when [FH, FW] = 9, [ph, pw] = 4:
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 162, Time = 28.37 msec, Performace = 12262.7 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1, int FH, int FW>
__global__ void zeroPadding_uernel_8_8_s1x4_ruse(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,
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
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph; tiw0 = tiw0 - opw;
	deltaY += ((tn0*OH + tih0)*OW + tiw0 + 1)*OC;//deltaY += Y1

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3);
	const int Y0 = xj0 * IC + xic0;//j0 = ((n*OH + oh)*OW + ow)*IC + ic

	//compute area----------------------------------------------------
	for (int fh = 0; fh < FH; fh++) {
		for (int ooc = 0; ooc < OC; ooc += STEP2) {
			//load 4 elements from deltaY[N, OH, OW, OC]
			const int yoc = ((tx & STEP_m1) << 1) + ooc;
			bool ly = (tih0 >= -fh) && (tih0 < OH - fh);
			bool ly0 = ly && (tiw0 >=  0) && (tiw0 < OW    );
			bool ly1 = ly && (tiw0 >= -1) && (tiw0 < OW - 1);
			bool ly2 = ly && (tiw0 >= -2) && (tiw0 < OW - 2);
			bool ly3 = ly && (tiw0 >= -3) && (tiw0 < OW - 3);
			const int yoffset0 = (fh*OW*OC) + yoc;
			float2 y0 = (ly0 ? *(float2*)(deltaY + yoffset0 - OC) : F32_2_0);
			float2 y1 = (ly1 ? *(float2*)(deltaY + yoffset0) : F32_2_0);
			float2 y2 = (ly2 ? *(float2*)(deltaY + yoffset0 + OC) : F32_2_0);
			float2 y3 = (ly3 ? *(float2*)(deltaY + yoffset0 + (OC << 1)) : F32_2_0);
			Ys[buf][(tx << 1)    ][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
			Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };

			//load 4 elements from W[OC, FH, FW, IC]
			const int woc = ((ty & STEP_m1) << 1) + ooc;
			const int woffset0 = (woc*FH - fh)*FW*IC;
			Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset0);
			Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset0 + (FH * FW * IC));
			__syncthreads();

			for (int fw = 1; fw < FW; fw++) {
#pragma unroll 
				for (int ik = 0; ik < STEP2; ik++) {
					float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
					float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

					//transposed compute core: (W * dY)^T
					simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
					simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
					simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
					simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
					simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
					simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
					simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
					simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
				}
				float4 oy0 = Ys[buf][(tx << 1)    ][ty];//update_shared_memory
				float4 oy1 = Ys[buf][(tx << 1) + 1][ty];
				buf ^= 1;

				//load 4 elements from deltaY[N, OH, OW, OC]
				bool ly3 = ly && (tiw0 >= -(fw + 3)) && (tiw0 < OW - (fw + 3));
				float2 ny3 = (ly3 ? *(float2*)(deltaY + yoffset0 + (fw + 2)*OC) : F32_2_0);
				Ys[buf][(tx << 1)][ty] = float4{ oy0.y, oy0.z, oy0.w, ny3.x };
				Ys[buf][(tx << 1) + 1][ty] = float4{ oy1.y, oy1.z, oy1.w, ny3.y };

				//load 4 elements from W[OC, FH, FW, IC]
				const int woffset = woffset0 - fw * IC;
				Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset);
				Ws[buf][(ty << 1) + 1][tx] = *(float4*)(W + woffset + (FH * FW * IC));
				__syncthreads();
			}
#pragma unroll 
			for (int ik = 0; ik < STEP2; ik++) {
				float4 w0 = Ws[buf][ik][ux], w1 = Ws[buf][ik + STEP2][ux];
				float4 y0 = Ys[buf][ik][uy], y1 = Ys[buf][ik + STEP2][uy];

				//transposed compute core: (W * dY)^T
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
		}
	}

	const int Y1 = Y0 + IC, Y2 = Y1 + IC, Y3 = Y2 + IC;
	const int Y4 = Y3 + IC, Y5 = Y4 + IC, Y6 = Y5 + IC, Y7 = Y6 + IC;

	*(float4*)(deltaX + Y0) = v0;  *(float4*)(deltaX + Y0 + 4) = v1;
	*(float4*)(deltaX + Y1) = v2;  *(float4*)(deltaX + Y1 + 4) = v3;
	*(float4*)(deltaX + Y2) = v4;  *(float4*)(deltaX + Y2 + 4) = v5;
	*(float4*)(deltaX + Y3) = v6;  *(float4*)(deltaX + Y3 + 4) = v7;
	*(float4*)(deltaX + Y4) = v8;  *(float4*)(deltaX + Y4 + 4) = v9;
	*(float4*)(deltaX + Y5) = v10; *(float4*)(deltaX + Y5 + 4) = v11;
	*(float4*)(deltaX + Y6) = v12; *(float4*)(deltaX + Y6 + 4) = v13;
	*(float4*)(deltaX + Y7) = v14; *(float4*)(deltaX + Y7 + 4) = v15;
}

#endif


//======[FH = FW = 3]======================================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), OC % BLOCK_SIZE == 0, IW % 4 == 0
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DECONV_3D_DX_ZERO_PADDING_UERNEL_8_8_S1W3X4_RUSE
#define DECONV_3D_DX_ZERO_PADDING_UERNEL_8_8_S1W3X4_RUSE

//for: Feature = (64, 64), [N, IC, OC] = [32, 128, 128]
//LB = 4: Size = 18, Time = 3.173 msec, Performace = 12182.4 GFlop/s
//LB = 3: Size = 18, Time = 3.45  msec, Performace = 11204.3 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 64]
//LB = 4: Size = 18, Time = 3.163 msec, Performace = 12220.9 GFlop/s
//LB = 3: Size = 9, Time = 1.812 msec, Performace = 10666.3 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 128, 256]
//LB = 4: Size = 18, Time = 3.197 msec, Performace = 12090.9 GFlop/s
//LB = 3: Size = 18, Time = 3.451 msec, Performace = 11201 GFlop/s
//for: Feature = (12, 12), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 10.125, Time = 1.822 msec, Performace = 11933.7 GFlop/s
//LB = 3: Size = 10.125, Time = 1.958 msec, Performace = 11104.8 GFlop/s
//for: Feature = (8, 8), [N, IC, OC] = [128, 512, 512]
//LB = 4: Size = 18, Time = 3.217 msec, Performace = 12015.8 GFlop/s
//LB = 3: Size = 18, Time = 3.527 msec, Performace = 10959.7 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void zeroPadding_uernel_8_8_s1W3x4_ruse(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,//FH = FW = 3
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
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 3) + ic_index;
	const int tic0 = bic0 + (tx << 3) + ((ty >= STEP) << 2);
	W += tic0;//W[0, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph; tiw0 = tiw0 - opw;
	deltaY += ((tn0*OH + tih0)*OW + tiw0 + 1)*OC;//deltaY += Y1

	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 3);
	const int xj0  = bj0  + (uy << 3);
	const int Y0 = xj0 * IC + xic0;//j0 = ((n*OH + oh)*OW + ow)*IC + ic

	//compute area----------------------------------------------------
	for (int gk = 0, GK = (OC >> LB) * 3; gk < GK; gk++) {//ooc_group (32 channels) * FH
		const int ooc_group = gk / 3, fh = gk - ooc_group * 3;
		const int ooc = (ooc_group << LB);

		//load 4 elements from W[OC, FH, FW, IC]
		const int woc = ((ty & STEP_m1) << 1) + ooc;
		const int woffset0 = (woc * 9 - fh * 3)*IC;
		Ws[buf][(ty << 1)    ][tx] = *(float4*)(W + woffset0);
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
		Ys[buf][(tx << 1)    ][ty] = float4{ y0.x, y1.x, y2.x, y3.x };
		Ys[buf][(tx << 1) + 1][ty] = float4{ y0.y, y1.y, y2.y, y3.y };
		__syncthreads();

#pragma unroll
		for (int fw = 1; fw < 3; fw++) {
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
			float4 oy0 = Ys[buf][(tx << 1)    ][ty];//update_shared_memory
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
	}

	const int Y1 = Y0 + IC, Y2 = Y1 + IC, Y3 = Y2 + IC;
	const int Y4 = Y3 + IC, Y5 = Y4 + IC, Y6 = Y5 + IC, Y7 = Y6 + IC;

	*(float4*)(deltaX + Y0) = v0;  *(float4*)(deltaX + Y0 + 4) = v1;
	*(float4*)(deltaX + Y1) = v2;  *(float4*)(deltaX + Y1 + 4) = v3;
	*(float4*)(deltaX + Y2) = v4;  *(float4*)(deltaX + Y2 + 4) = v5;
	*(float4*)(deltaX + Y3) = v6;  *(float4*)(deltaX + Y3 + 4) = v7;
	*(float4*)(deltaX + Y4) = v8;  *(float4*)(deltaX + Y4 + 4) = v9;
	*(float4*)(deltaX + Y5) = v10; *(float4*)(deltaX + Y5 + 4) = v11;
	*(float4*)(deltaX + Y6) = v12; *(float4*)(deltaX + Y6 + 4) = v13;
	*(float4*)(deltaX + Y7) = v14; *(float4*)(deltaX + Y7 + 4) = v15;
}

#endif


#endif
