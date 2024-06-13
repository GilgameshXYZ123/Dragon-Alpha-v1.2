#pragma once

#ifndef DECONV3D_DX_WINOGRAD_F2X3_KERNEL_H
#define DECONV3D_DX_WINOGRAD_F2X3_KERNEL_H

//Im2col-Winograd Convolution 3D:
//(1) sh = sw = 1
//(2) FH = FW = 3
//(3) (IH, IW) % 4 == 0
#ifndef DECONV3D_DX_WINOGRAD_F2X3_CALL
#define DECONV3D_DX_WINOGRAD_F2X3_CALL

//LB = log2(BLOCK_SIZE)

#define CAN_WINOGRAD_F2X3_k48 \
	((FH == 3) && (FW == 3) &&\
	!(IH & 3) && !(IW & 3) &&\
	(OC >= 64 && OC <= 1024) && !(OC & 7))

#define winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	winograd_f2x3_k48_texture<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, OC,(OC<<1),(OC*3),(OC<<2),(OC*5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W+(6*IC)), deltaX,(IH*IW), IW,IC, (2-ph),(2-pw), ic_index,j_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0 
#ifndef DECONV3D_DX_WINOGRAD_F2X3_KERNEL_4_8_OCT_TEXTURE
#define DECONV3D_DX_WINOGRAD_F2X3_KERNEL_4_8_OCT_TEXTURE

//for: Feature = [64, 64], [N, IC] = [32, 128], LB = 4, Vs. u88s1W3x4
//(1) OC =  64: Size = 9    , Time = 1.453 msec, Performace = 13301.7 GFlop/s (11107.7 GFlop/s)
//(3) OC =  80: Size = 11.25, Time = 1.791 msec, Performace = 13489.2 GFlop/s (11299.9 GFlop/s)
//(5) OC =  96: Size = 13.5 , Time = 2.126 msec, Performace = 13636.4 GFlop/s (11289.3 GFlop/s)
//(7) OC = 112: Size = 15.75, Time = 2.478 msec, Performace = 13649.3 GFlop/s (11403.5 GFlop/s)
//(9) OC = 128: Size = 18    , Time = 2.973 msec, Performace = 13001.9 GFlop/s (11463.4 GFlop/s)
//for: Feature = [32, 32], [N, IC, OC] = [32, 256, 256]:
//LB = 4: Size = 18, Time = 2.885 msec, Performace = 13398.5 GFlop/s
//for: Feature = [32, 32], [N, IC, OC] = [128, 128, 128]:
//LB = 4: Size = 18, Time = 2.961 msec, Performace = 13054.6 GFlop/s
//for: Feature = [16, 16], [N, IC, OC] = [ 32, 512, 512]:
//LB = 4: Size = 18, Time = 2.892 msec, Performace = 13366.1 GFlop/s
//for: Feature = [16, 16], [N, IC, OC] = [128, 256, 256]:
//LB = 4: Size = 18, Time = 2.852 msec, Performace = 13553.5 GFlop/s
//for: Feature = [ 8,  8], [N, IC, OC] = [32, 1024, 1024]:
//Size = 18, Time = 2.889 msec, Performace = 13380 GFlop/s
//for: Feature = [ 8,  8], [N, IC, OC] = [128, 512, 512]:
//LB = 4: Size = 18, Time = 2.912 msec, Performace = 13274.3 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1, int OC, int OC2, int OC3, int OC4, int OC5>
__global__ void winograd_f2x3_k48_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,
	      float* __restrict__ deltaX, int IH_IW, int IW, int IC,
	int oph, int opw,//sh = sw = 1
	int ic_index, int j_index)
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

	//prepare for W[OC, FH, FW, IC]
	const int bic0 = (blockIdx.x << LB << 2) + ic_index;
	const int tic0 = bic0 + (tx << 2) + ((ty >= STEP) << 1);
	W += (ty & STEP_m1) * 9 * IC + tic0;//W[oic_start, 0, 0, tic0]

	//prepare for deltaY[N, OH, OW, OC]
	const int bj0 = (blockIdx.y << LB << 3) + +j_index;
	const int tj0 = bj0 + (ty << 3) + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph; tiw0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC + (tx & STEP_m1);//Y[tn0, tih0, tiw0, oc_start]
	
	//prepare for deltaX[N, IH, IW, IC]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int xic0 = bic0 + (ux << 2);
	const int xj0  = bj0  + (uy << 3);
	const int X0 = xj0 * IC + xic0;//j0 = ((n*OH + oh)*OW + ow)*IC + ic

	//compute area--------------------------------------------------------
	for (int fh = 0; fh < 3; fh++) {
		//load 2 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < OH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < OW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < OW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < OW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < OW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < OW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < OW);
		const int yoffset = Y0 + fh * OC * OW;//with the same ty
		float y0 = tex1Dfetch<float>(deltaY, IF_int(ly0, yoffset      , -1));
		float y1 = tex1Dfetch<float>(deltaY, IF_int(ly1, yoffset + OC , -1));
		float y2 = tex1Dfetch<float>(deltaY, IF_int(ly2, yoffset + OC2, -1));
		float y3 = tex1Dfetch<float>(deltaY, IF_int(ly3, yoffset + OC3, -1));
		float y4 = tex1Dfetch<float>(deltaY, IF_int(ly4, yoffset + OC4, -1));
		float y5 = tex1Dfetch<float>(deltaY, IF_int(ly5, yoffset + OC5, -1));
	
		//load 2 group from W
		const int woffset = -3 * fh * IC;//[oc, 2-fh, 2, ic]
		float2 w2 = *(float2*)(W + woffset);
		float2 w1 = *(float2*)(W + woffset + IC);
		float2 w0 = *(float2*)(W + woffset + (IC << 1));
		
		//Winograd transform: W(3) -> G(4); X[4] -> D[4]
		float4 d0 = float4{ y0 - y2, y3 - y1, y2 - y4, y5 - y3 };
		float4 d1 = float4{ y1 + y2, y2 - y1, y3 + y4, y4 - y3 };
	
		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		float4 g0 = float4{ w0.x, w2.x, w0.y, w2.y };
		float4 g1 = float4{ gst0, gst1, gst2, gst3 };

		//write to shread memory
		Dsv[buf][tx][ty] = d0; Dst[buf][tx][ty] = d1;
		Gsv[buf][ty][tx] = g0; Gst[buf][ty][tx] = g1;
		__syncthreads();

		for (int ooc = STEP; ooc < OC; ooc += STEP) {
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
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

			//load 2 group from Y
			const int yoffset = Y0 + fh * OC * OW + ooc;
			float y0 = tex1Dfetch<float>(deltaY, IF_int(ly0, yoffset      , -1));
			float y1 = tex1Dfetch<float>(deltaY, IF_int(ly1, yoffset + OC , -1));
			float y2 = tex1Dfetch<float>(deltaY, IF_int(ly2, yoffset + OC2, -1));
			float y3 = tex1Dfetch<float>(deltaY, IF_int(ly3, yoffset + OC3, -1));
			float y4 = tex1Dfetch<float>(deltaY, IF_int(ly4, yoffset + OC4, -1));
			float y5 = tex1Dfetch<float>(deltaY, IF_int(ly5, yoffset + OC5, -1));
		
			//load 2 group from W
			const int woffset = (ooc * 9 - 3 * fh)*IC;//[ooc, 2-fh, 2, 0]
			float2 w2 = *(float2*)(W + woffset);
			float2 w1 = *(float2*)(W + woffset + IC);
			float2 w0 = *(float2*)(W + woffset + (IC << 1));

			//Winograd transform: W(3) -> G(4); X[4] -> D[4]
			float4 d0 = float4{ y0 - y2, y3 - y1, y2 - y4, y5 - y3 };
			float4 d1 = float4{ y1 + y2, y2 - y1, y3 + y4, y4 - y3 };

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
		for (int ik = 0; ik < STEP; ik++)
		{
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

	const int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	const int X4 = X3 + IC, X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

	*(float4*)(deltaX + X0) = v0; *(float4*)(deltaX + X1) = v1;
	*(float4*)(deltaX + X2) = v2; *(float4*)(deltaX + X3) = v3;
	*(float4*)(deltaX + X4) = v4; *(float4*)(deltaX + X5) = v5;
	*(float4*)(deltaX + X6) = v6; *(float4*)(deltaX + X7) = v7;
}

#endif


//======[integrated function]====================================================
#ifndef DECONV3D_DX_WINOGRAD_F2X3_KERNEL_4_8_OCT_TEXTURE_FUNCTION
#define DECONV3D_DX_WINOGRAD_F2X3_KERNEL_4_8_OCT_TEXTURE_FUNCTION

template<int LB> //OC = 64, 72, 80, 88, 96, 104, 112, 120
inline bool deconv3D_dX_winograd_f2x3_k48_tex(cudaStream_t stream, int ic_index, int j_index,
	cudaTextureObject_t deltaY, int OH, int OW,
	const float*            W,//FH = FW = 3
	      float*       deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int GN, int GM)
{
	bool flag = !(IW & 3) && (OC >= 64 && OC <= 1024) && !(OC & 7); if (!flag) return false;

	//=====[2^x channels]============================================================================================================================
	if (OC ==   64) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC,   64, ph, pw, GN, GM); return true; }
	if (OC ==  128) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC,  128, ph, pw, GN, GM); return true; }
	if (OC ==  256) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC,  256, ph, pw, GN, GM); return true; }
	if (OC ==  512) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC,  512, ph, pw, GN, GM); return true; }
	if (OC == 1024) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, 1024, ph, pw, GN, GM); return true; }

	//=====[64x channels]=============================================================================================================================
	if (OC == 192) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, 192, ph, pw, GN, GM); return true; }
	if (OC == 320) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, 320, ph, pw, GN, GM); return true; }
	if (OC == 384) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, 384, ph, pw, GN, GM); return true; }
	if (OC == 448) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, 448, ph, pw, GN, GM); return true; }

	//=====[8x channels]=============================================================================================================================
	if (OC ==  72) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC,  72, ph, pw, GN, GM); return true; }
	if (OC ==  80) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC,  80, ph, pw, GN, GM); return true; }
	if (OC ==  88) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC,  88, ph, pw, GN, GM); return true; }
	if (OC ==  96) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC,  96, ph, pw, GN, GM); return true; }
	if (OC == 104) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, 104, ph, pw, GN, GM); return true; }
	if (OC == 112) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, 112, ph, pw, GN, GM); return true; }
	if (OC == 120) { winograd_f2x3_k48_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, 120, ph, pw, GN, GM); return true; }
	return false;
}

#endif

#endif