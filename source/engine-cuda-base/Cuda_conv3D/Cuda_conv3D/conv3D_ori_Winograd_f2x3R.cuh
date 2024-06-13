#pragma once

#pragma once

#ifndef CONV_3D_WINOGRAD_F2X3R_ORIGINAL_H
#define CONV_3D_WINOGRAD_F2X3R_ORIGINAL_H

//(1) sh = sw = 1
//(2) FW = 3
//(3) OW % 4 == 0: group = 2 elements
//(3) Remode the kernel: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
//<1> GN = OC
//<2> GM = N * OH * OW
#ifndef CONV_3D_WINOGRAD_F2X3R_ORIGINAL_CALL
#define CONV_3D_WINOGRAD_F2X3R_ORIGINAL_CALL

//LB = log2(BLOCK_SIZE)

//================[IC_template: 4 * 8 kernel]============================
#define conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinograd_f2x3_kernel_4_8R_p1_ICT<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, oc_index, j_index)

#define conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinograd_f2x3_kernel_4_8R_ICT_texture<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

#endif


//================[IC_template: 4 * 8 kernel]============================
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE / 2) == 0
//LB = 4, IC % 8 == 0
#ifndef CONV_3D_WINOGRAD_F2X3_KERNEL_4_8R_ICT_TEXTURE
#define CONV_3D_WINOGRAD_F2X3_KERNEL_4_8R_ICT_TEXTURE

//for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 18, Time = 3.10639 msec, Performace = 12443.6 GFlop/s
//for: Feature = (64, 64), [N, IC, OC] = [32, 128, 128]
//LB = 4: Size = 18, Time = 3.14332 msec, Performace = 12297.4 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [32, 256, 256]
//LB = 4: Size = 18, Time = 3.04253 msec, Performace = 12704.8 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 18, Time = 3.12615 msec, Performace = 12364.9 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 192, 192]
//LB = 4: Size = 40.5, Time = 6.51195 msec, Performace = 13355.9 GFlop/s
//for: Feature = (28, 28), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 27.5625, Time = 4.67175 msec, Performace = 12669.8 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [32, 512, 512]
//LB = 4: Size = 18, Time = 3.0699 msec, Performace = 12591.5 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 18, Time = 2.95833 msec, Performace = 13066.4 GFlop/s
//LB = 3: Size = 18, Time = 3.60787 msec, Performace = 10714    GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 320, 320]
//LB = 4: Size = 28.125, Time = 4.55438 msec, Performace = 13261.5 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [ 64, 384, 384]
//LB = 4: Size = 20.25, Time = 3.40069 msec, Performace = 12787.5 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [ 64, 448, 448]
//LB = 4: Size = 27.5625, Time = 4.46192 msec, Performace = 13265.6 GFlop/s
//for: Feature = ( 8,  8), [N, IC, OC] = [32, 1024, 1024]
//LB = 4: Size = 18, Time = 3.20664 msec, Performace = 12054.6 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void conv3dWinograd_f2x3_kernel_4_8R_ICT_texture(
	cudaTextureObject_t        X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Gst[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dst[2][1 << LB][(1 << LB) + 1];

	//compute 4*8 results: 8*8 accumulators
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << LB << 2) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, (ty & STEP_m1), toc0]
	const int Wstride = IC * OC;

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for Y[N, OH, OW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 2);//oc = f(by)
	const int yj0 = bj0 + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)

	//compute area------------------------------------------------------------------
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset      , -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC , -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

		//load 2 group from CW
		const int woffset = fh * IC3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

		//Winograd transform: W(3) -> G(4); X(4) -> D(4)
		float4 d0 = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
		float4 d1 = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, w2.x, w0.y, w2.y };
		float4 g1 = float4{ gst0, gst1, gst2, gst3 };

		//write to shread memory
		Dsv[buf][tx][ty] = d0; Dst[buf][tx][ty] = d1;
		Gsv[buf][ty][tx] = g0; Gst[buf][ty][tx] = g1;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float4 g0 = Gsv[buf][ik][ux], g2 = Gsv[buf][ik + STEP][ux];//{x, w} for v;
				float4 d0 = Dsv[buf][ik][uy], d2 = Dsv[buf][ik + STEP][uy];//{x, w} for v;

				winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
				winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

				float4 g1 = Gst[buf][ik][ux], g3 = Gst[buf][ik + STEP][ux];//{y, z} for t
				float4 d1 = Dst[buf][ik][uy], d3 = Dst[buf][ik + STEP][uy];//{y, z} for t

				winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
				winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
			}
			buf ^= 1;

			//load 2 group from X
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset      , -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC , -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));

			//load 2 group from CW
			const int woffset = (fh*IC3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//Winograd transform: W(3) -> G(4); X[4] -> D[4]
			float4 d0 = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
			float4 d1 = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, w2.x, w0.y, w2.y };
			float4 g1 = float4{ gst0, gst1, gst2, gst3 };

			//write to shread memory
			Dsv[buf][tx][ty] = d0; Dst[buf][tx][ty] = d1;
			Gsv[buf][ty][tx] = g0; Gst[buf][ty][tx] = g1;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 g0 = Gsv[buf][ik][ux], g2 = Gsv[buf][ik + STEP][ux];//{x, w} for v;
			float4 d0 = Dsv[buf][ik][uy], d2 = Dsv[buf][ik + STEP][uy];//{x, w} for v;

			winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
			winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

			float4 g1 = Gst[buf][ik][ux], g3 = Gst[buf][ik + STEP][ux];//{y, z} for t
			float4 d1 = Dst[buf][ik][uy], d3 = Dst[buf][ik + STEP][uy];//{y, z} for t

			winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
			winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2; *(float4*)(Y + Y3) = v3;
	*(float4*)(Y + Y4) = v4; *(float4*)(Y + Y5) = v5;
	*(float4*)(Y + Y6) = v6; *(float4*)(Y + Y7) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE / 2) == 0
//LB = 4, IC % 8 == 0
//ph = pw = 1
#ifndef CONV_3D_WINOGRAD_F2X3_KERNEL_4_8R_P1_ICT
#define CONV_3D_WINOGRAD_F2X3_KERNEL_4_8R_P1_ICT

//for: Feature = (128, 128), [N, IC, OC] = [32, 64, 64]
//LB = 4: Size = 18, Time = 3.1566 msec, Performace = 12245.7 GFlop/s
//for: Feature = (64, 64), [N, IC, OC] = [32, 128, 128]
//LB = 4: Size = 18, Time = 3.04327 msec, Performace = 12701.7 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [32, 256, 256]
//LB = 4: Size = 18, Time = 3.04442 msec, Performace = 12696.9 GFlop/s
//for: Feature = (32, 32), [N, IC, OC] = [128, 128, 128]
//LB = 4: Size = 18, Time = 3.05372 msec, Performace = 12658.2 GFlop/s
//for: Feature = (28, 28), [N, IC, OC] = [64, 256, 256]
//LB = 4: Size = 27.5625, Time = 4.66477 msec, Performace = 12688.7 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [32, 512, 512]
//LB = 4: Size = 18, Time = 3.08271 msec, Performace = 12539.2 GFlop/s
//for: Feature = (16, 16), [N, IC, OC] = [128, 256, 256]
//LB = 4: Size = 18, Time = 3.01697 msec, Performace = 12812.4 GFlop/s
//for: Feature = ( 8,  8), [N, IC, OC] = [32, 1024, 1024]
//LB = 4: Size = 18, Time = 3.20961 msec, Performace = 12043.4 GFlop/s
//for: Feature = ( 8,  8), [N, IC, OC] = [32, 1024, 1024]
//LB = 4: Size = 9, Time = 1.55151 msec, Performace = 12457.2 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void conv3dWinograd_f2x3_kernel_4_8R_p1_ICT(
	const float* __restrict__  X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__  Y, int OH_OW, int OW, int OC,
	//ph = pw =sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Gst[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dst[2][1 << LB][(1 << LB) + 1];

	//compute 4*8 results: 8*8 accumulators
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for CW[FH, FW, IC, OC]
	const int boc0 = (blockIdx.x << LB << 2) + oc_index;
	const int toc0 = boc0 + (tx << 2) + ((ty >= STEP) << 1);
	CW += (ty & STEP_m1)*OC + toc0;//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for X[N, IH, IW, IC]
	const int bj0 = (blockIdx.y << LB << 3) + j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - 1, tiw0 = tow0 - 1;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);

	//prepare for Y[N, OH, OW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int yoc0 = boc0 + (ux << 2);//oc = f(by)
	const int yj0 = bj0 + (uy << 3);
	const int Y0 = yj0 * OC + yoc0;//j = f(bx) -> (n, oh, ow)

#pragma once //compute area---------------------------------------------
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool lw0 = lh0 && (tiw0    >=  0);
		bool lw5 = lh0 && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = (lw0 ? X[xoffset      ] : 0.0f);
		float x1 = (lh0 ? X[xoffset + IC ] : 0.0f);
		float x2 = (lh0 ? X[xoffset + IC2] : 0.0f);
		float x3 = (lh0 ? X[xoffset + IC3] : 0.0f);
		float x4 = (lh0 ? X[xoffset + IC4] : 0.0f);
		float x5 = (lw5 ? X[xoffset + IC5] : 0.0f);

		//load 2 group from CW
		const int woffset = fh * IC3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

		//Winograd transform: W(3) -> G(4); X[4] -> D[4]
		float4 d0 = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
		float4 d1 = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, w2.x, w0.y, w2.y };
		float4 g1 = float4{ gst0, gst1, gst2, gst3 };

		//write to shread memory
		Dsv[buf][tx][ty] = d0; Dst[buf][tx][ty] = d1;
		Gsv[buf][ty][tx] = g0; Gst[buf][ty][tx] = g1;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float4 g0 = Gsv[buf][ik][ux], g2 = Gsv[buf][ik + STEP][ux];//{x, w} for v;
				float4 d0 = Dsv[buf][ik][uy], d2 = Dsv[buf][ik + STEP][uy];//{x, w} for v;

				winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
				winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

				float4 g1 = Gst[buf][ik][ux], g3 = Gst[buf][ik + STEP][ux];//{y, z} for t
				float4 d1 = Dst[buf][ik][uy], d3 = Dst[buf][ik + STEP][uy];//{y, z} for t

				winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
				winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
			}
			buf ^= 1;

			//load 2 group from X
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = (lw0 ? X[xoffset      ] : 0.0f);
			float x1 = (lh0 ? X[xoffset + IC ] : 0.0f);
			float x2 = (lh0 ? X[xoffset + IC2] : 0.0f);
			float x3 = (lh0 ? X[xoffset + IC3] : 0.0f);
			float x4 = (lh0 ? X[xoffset + IC4] : 0.0f);
			float x5 = (lw5 ? X[xoffset + IC5] : 0.0f);

			//load 2 group from CW
			const int woffset = (fh*IC3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));

			//Winograd transform: W(3) -> G(4); X[4] -> D[4]
			float4 d0 = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
			float4 d1 = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			float4 g0 = float4{ w0.x, w2.x, w0.y, w2.y };
			float4 g1 = float4{ gst0, gst1, gst2, gst3 };

			//write to shread memory
			Dsv[buf][tx][ty] = d0; Dst[buf][tx][ty] = d1;
			Gsv[buf][ty][tx] = g0; Gst[buf][ty][tx] = g1;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 g0 = Gsv[buf][ik][ux], g2 = Gsv[buf][ik + STEP][ux];//{x, w} for v;
			float4 d0 = Dsv[buf][ik][uy], d2 = Dsv[buf][ik + STEP][uy];//{x, w} for v;

			winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
			winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

			float4 g1 = Gst[buf][ik][ux], g3 = Gst[buf][ik + STEP][ux];//{y, z} for t
			float4 d1 = Dst[buf][ik][uy], d3 = Dst[buf][ik + STEP][uy];//{y, z} for t

			winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
			winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2; *(float4*)(Y + Y3) = v3;
	*(float4*)(Y + Y4) = v4; *(float4*)(Y + Y5) = v5;
	*(float4*)(Y + Y6) = v6; *(float4*)(Y + Y7) = v7;
}

#endif


//================[integrated functions]=================================
#ifndef CONV_3D_WINOGRAD_F2X3_4_8R_TEXTURE
#define CONV_3D_WINOGRAD_F2X3_4_8R_TEXTURE

template<int LB>
inline bool conv3D_Winograd_f2x3_k48R_tex(cudaStream_t stream, int oc_index, int j_index,
	cudaTextureObject_t X, int IH, int IW,
	const float*       CW,//FH = FW = 3
	      float*        Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int GN, int GM)
{
	bool flag = !(OW & 3) && (IC >= 64 && IC <= 1024) && !(IC & 7); if (!flag) return false;

	//------[2^x channels]---------------------------------------------------------------------------------------------------------------------------
	if (IC ==   64) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,   64, OC, ph, pw, GN, GM); return true; }
	if (IC ==  128) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  128, OC, ph, pw, GN, GM); return true; }
	if (IC ==  256) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  256, OC, ph, pw, GN, GM); return true; }
	if (IC ==  512) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  512, OC, ph, pw, GN, GM); return true; }
	if (IC == 1024) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 1024, OC, ph, pw, GN, GM); return true; }

	//------[64x channels]---------------------------------------------------------------------------------------------------------------------------
	if (IC == 192) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 192, OC, ph, pw, GN, GM); return true; }
	if (IC == 320) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 320, OC, ph, pw, GN, GM); return true; }
	if (IC == 384) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 384, OC, ph, pw, GN, GM); return true; }
	if (IC == 448) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 448, OC, ph, pw, GN, GM); return true; }

	//------[8x channels]----------------------------------------------------------------------------------------------------------------------------
	if (IC ==  72) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  72, OC, ph, pw, GN, GM); return true; }
	if (IC ==  80) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  80, OC, ph, pw, GN, GM); return true; }
	if (IC ==  88) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  88, OC, ph, pw, GN, GM); return true; }
	if (IC ==  96) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  96, OC, ph, pw, GN, GM); return true; }
	if (IC == 104) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 104, OC, ph, pw, GN, GM); return true; }
	if (IC == 112) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 112, OC, ph, pw, GN, GM); return true; }
	if (IC == 120) { conv3dWinograd_f2x3_k48R_ICT_tex(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 120, OC, ph, pw, GN, GM); return true; }

	return false;
}

#endif


//ph = pw = 1
#ifndef CONV_3D_WINOGRAD_F2X3_4_8R_P1
#define CONV_3D_WINOGRAD_F2X3_4_8R_P1

template<int LB>
inline bool conv3D_Winograd_f2x3_k48R_p1(cudaStream_t stream, int oc_index, int j_index,
	const float*  X, int IH, int IW,
	const float* CW,//FH = FW = 3
	      float*  Y, int OH, int OW,
	int IC, int OC,//ph = pw = sh = sw = 1
	int GN, int GM)
{
	bool flag = !(OW & 3) && (IC >= 64 && IC <= 1024) && !(IC & 7); if (!flag) return false;

	//------[2^x channels]---------------------------------------------------------------------------------------------------------------------------
	if (IC ==   64) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,   64, OC, ph, pw, GN, GM); return true; }
	if (IC ==  128) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  128, OC, ph, pw, GN, GM); return true; }
	if (IC ==  256) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  256, OC, ph, pw, GN, GM); return true; }
	if (IC ==  512) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  512, OC, ph, pw, GN, GM); return true; }
	if (IC == 1024) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 1024, OC, ph, pw, GN, GM); return true; }

	//------[8x channels]----------------------------------------------------------------------------------------------------------------------------
	if (IC ==  72) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  72, OC, ph, pw, GN, GM); return true; }
	if (IC ==  80) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  80, OC, ph, pw, GN, GM); return true; }
	if (IC ==  88) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  88, OC, ph, pw, GN, GM); return true; }
	if (IC ==  96) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW,  96, OC, ph, pw, GN, GM); return true; }
	if (IC == 104) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 104, OC, ph, pw, GN, GM); return true; }
	if (IC == 112) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 112, OC, ph, pw, GN, GM); return true; }
	if (IC == 120) { conv3dWinograd_f2x3_k48R_p1_ICT(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, 120, OC, ph, pw, GN, GM); return true; }

	return false;
}

#endif

#endif