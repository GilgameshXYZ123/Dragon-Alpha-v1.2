

//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE / 2) == 0
//LB = 4, IC % 8 == 0
#ifndef XENO_WINOGRAD_S1_KERNEL_A1
#define XENO_WINOGRAD_S1_KERNEL_A1

#define xeno_winograd_s1_a1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_s1_kernel_a1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, OC,(OC<<1),(OC*3),(OC<<2),(OC*5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW,IC, (2-ph),(2-pw), ic_index,j_index)

//OC = 128:
//Target: Size = 18, Time = 3.344 msec, Performace = 11559.4 GFlop/s
//LB = 4: Size = 18, Time = 3.438 msec, Performace = 11243.4 GFlop/s
//OC = 64:
//Target: Size = 9, Time = 1.744 msec, Performace = 11082.2 GFlop/s
//LB = 4: Size = 9, Time = 1.838 msec, Performace = 10515.4 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int OC, int OC2, int OC3, int OC4, int OC5>
__global__ void xeno_winograd_s1_kernel_a1(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,//[OC, FH, FW, IC]
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

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += (ty & STEP_m1) * 9 * IC + ic0 + ((ty >= STEP) << 1);//W[oic_start, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph; tiw0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC + (tx & STEP_m1);//Y[tn0, tih0, tiw0, oc_start]
	j0 = j0 * IC + ic0;//j0 = ((n*OH + oh)*OW + ow)*IC + ic

#pragma once //compute area------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < OH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < OW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < OW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < OW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < OW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < OW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < OW);
		const int yoffset = Y0 + fh * OW * OC;//with the same ty
		float y0 = tex1Dfetch<float>(deltaY, IF_int(ly0, yoffset, -1));
		float y1 = tex1Dfetch<float>(deltaY, IF_int(ly1, yoffset + OC, -1));
		float y2 = tex1Dfetch<float>(deltaY, IF_int(ly2, yoffset + OC2, -1));
		float y3 = tex1Dfetch<float>(deltaY, IF_int(ly3, yoffset + OC3, -1));
		float y4 = tex1Dfetch<float>(deltaY, IF_int(ly4, yoffset + OC4, -1));
		float y5 = tex1Dfetch<float>(deltaY, IF_int(ly5, yoffset + OC5, -1));
		Dsv[buf][tx][ty] = float4{ y0 - y2, y3 - y1, y2 - y4, y5 - y3 };
		Dst[buf][tx][ty] = float4{ y1 + y2, y2 - y1, y3 + y4, y4 - y3 };

		//load 2 group from CW[oc, fh, fw, ic]
		const int woffset = ((2 - fh) * 3 + 2)*IC;//[oc, 2-fh, 2, ic]
		float2 w0 = *(float2*)(W + woffset);
		float2 w1 = *(float2*)(W + woffset - IC);
		float2 w2 = *(float2*)(W + woffset - (IC << 1));
		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
		Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };
		__syncthreads();

		for (int ooc = STEP; ooc < OC; ooc += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
				float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

				winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
				winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

				float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
				float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

				winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
				winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
			}
			buf ^= 1;

			//load 2 group from Y
			const int yoffset = Y0 + fh * OW * OC + ooc;
			float y0 = tex1Dfetch<float>(deltaY, IF_int(ly0, yoffset, -1));
			float y1 = tex1Dfetch<float>(deltaY, IF_int(ly1, yoffset + OC, -1));
			float y2 = tex1Dfetch<float>(deltaY, IF_int(ly2, yoffset + OC2, -1));
			float y3 = tex1Dfetch<float>(deltaY, IF_int(ly3, yoffset + OC3, -1));
			float y4 = tex1Dfetch<float>(deltaY, IF_int(ly4, yoffset + OC4, -1));
			float y5 = tex1Dfetch<float>(deltaY, IF_int(ly5, yoffset + OC5, -1));
			Dsv[buf][tx][ty] = float4{ y0 - y2, y3 - y1, y2 - y4, y5 - y3 };
			Dst[buf][tx][ty] = float4{ y1 + y2, y2 - y1, y3 + y4, y4 - y3 };

			//load 2 group from CW[oc, fh, fw, ic]
			const int woffset = ((ooc * 3 + (2 - fh)) * 3 + 2)*IC;//[ooc, 2-fh, 2, 0]
			float2 w0 = *(float2*)(W + woffset);
			float2 w1 = *(float2*)(W + woffset - IC);
			float2 w2 = *(float2*)(W + woffset - (IC << 1));
			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
			Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
			float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

			winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
			winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

			float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
			float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

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

	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2; *(float4*)(deltaX + j3) = v3;
	*(float4*)(deltaX + j4) = v4; *(float4*)(deltaX + j5) = v5;
	*(float4*)(deltaX + j6) = v6; *(float4*)(deltaX + j7) = v7;
}



#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE / 2) == 0
//LB = 4, IC % 8 == 0
#ifndef XENO_WINOGRAD_S1_KERNEL_A2
#define XENO_WINOGRAD_S1_KERNEL_A2

#define xeno_winograd_s1_a2(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_s1_kernel_a2<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, OC,(OC<<1),(OC*3),(OC<<2),(OC*5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW,IC, (2-ph),(2-pw), ic_index,j_index)

//OC = 128:
//Target: Size = 18, Time = 3.344 msec, Performace = 11559.4 GFlop/s
//LB = 4: Size = 18, Time = 3.438 msec, Performace = 11243.4 GFlop/s
//OC = 64:
//Target: Size = 9, Time = 1.744 msec, Performace = 11082.2 GFlop/s
//LB = 4: Size = 9, Time = 1.694 msec, Performace = 11409.3 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int OC, int OC2, int OC3, int OC4, int OC5>
__global__ void xeno_winograd_s1_kernel_a2(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,//[OC, FH, FW, IC]
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

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += (ty & STEP_m1) * 9 * IC + ic0 + ((ty >= STEP) << 1);//W[oic_start, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph; tiw0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC + (tx & STEP_m1);//Y[tn0, tih0, tiw0, oc_start]
	j0 = j0 * IC + ic0;//j0 = ((n*OH + oh)*OW + ow)*IC + ic

#pragma once //compute area------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < OH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < OW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < OW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < OW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < OW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < OW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < OW);
		const int yoffset = Y0 + fh * OC * OW;//with the same ty
		float y0 = tex1Dfetch<float>(deltaY, IF_int(ly0, yoffset, -1));
		float y1 = tex1Dfetch<float>(deltaY, IF_int(ly1, yoffset + OC, -1));
		float y2 = tex1Dfetch<float>(deltaY, IF_int(ly2, yoffset + OC2, -1));
		float y3 = tex1Dfetch<float>(deltaY, IF_int(ly3, yoffset + OC3, -1));
		float y4 = tex1Dfetch<float>(deltaY, IF_int(ly4, yoffset + OC4, -1));
		float y5 = tex1Dfetch<float>(deltaY, IF_int(ly5, yoffset + OC5, -1));
		Dsv[buf][tx][ty] = float4{ y0 - y2, y3 - y1, y2 - y4, y5 - y3 };
		Dst[buf][tx][ty] = float4{ y1 + y2, y2 - y1, y3 + y4, y4 - y3 };

		//load 2 group from CW[oc, fh, fw, ic]
		const int woffset = ((2 - fh) * 3)*IC;//[oc, 2-fh, 2, ic]
		float2 w2 = *(float2*)(W + woffset);
		float2 w1 = *(float2*)(W + woffset + IC);
		float2 w0 = *(float2*)(W + woffset + (IC << 1));
		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
		Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };
		__syncthreads();

		for (int ooc = STEP; ooc < OC; ooc += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
				float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

				winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
				winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

				float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
				float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

				winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
				winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
			}
			buf ^= 1;

			//load 2 group from Y
			const int yoffset = Y0 + fh * OC * OW + ooc;
			float y0 = tex1Dfetch<float>(deltaY, IF_int(ly0, yoffset, -1));
			float y1 = tex1Dfetch<float>(deltaY, IF_int(ly1, yoffset + OC, -1));
			float y2 = tex1Dfetch<float>(deltaY, IF_int(ly2, yoffset + OC2, -1));
			float y3 = tex1Dfetch<float>(deltaY, IF_int(ly3, yoffset + OC3, -1));
			float y4 = tex1Dfetch<float>(deltaY, IF_int(ly4, yoffset + OC4, -1));
			float y5 = tex1Dfetch<float>(deltaY, IF_int(ly5, yoffset + OC5, -1));
			Dsv[buf][tx][ty] = float4{ y0 - y2, y3 - y1, y2 - y4, y5 - y3 };
			Dst[buf][tx][ty] = float4{ y1 + y2, y2 - y1, y3 + y4, y4 - y3 };

			//load 2 group from CW[oc, fh, fw, ic]
			const int woffset = ((ooc * 3 + (2 - fh)) * 3)*IC;//[ooc, 2-fh, 2, 0]
			float2 w2 = *(float2*)(W + woffset);
			float2 w1 = *(float2*)(W + woffset + IC);
			float2 w0 = *(float2*)(W + woffset + (IC << 1));
			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
			Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
			float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

			winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
			winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

			float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
			float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

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

	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2; *(float4*)(deltaX + j3) = v3;
	*(float4*)(deltaX + j4) = v4; *(float4*)(deltaX + j5) = v5;
	*(float4*)(deltaX + j6) = v6; *(float4*)(deltaX + j7) = v7;
}

#endif


#ifndef XENO_WINOGRAD_S1_KERNEL_A3
#define XENO_WINOGRAD_S1_KERNEL_A3

#define xeno_winograd_s1_a3(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_s1_kernel_a3<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, OC,(OC<<1),(OC*3),(OC<<2),(OC*5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W+(6*IC)), deltaX,(IH*IW),IW,IC, (2-ph),(2-pw), ic_index,j_index)

//OC = 128:
//Target: Size = 18, Time = 3.344 msec, Performace = 11559.4 GFlop/s
//LB = 4: Size = 18, Time = 3.438 msec, Performace = 11243.4 GFlop/s
//OC = 64:
//Target: Size = 9, Time = 1.744 msec, Performace = 11082.2 GFlop/s
//LB = 4: Size = 9, Time = 1.694 msec, Performace = 11409.3 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int OC, int OC2, int OC3, int OC4, int OC5>
__global__ void xeno_winograd_s1_kernel_a3(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,//[OC, FH, FW, IC]
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

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += (ty & STEP_m1) * 9 * IC + ic0 + ((ty >= STEP) << 1);//W[oic_start, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph; tiw0 = tiw0 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC + (tx & STEP_m1);//Y[tn0, tih0, tiw0, oc_start]
	j0 = j0 * IC + ic0;//j0 = ((n*OH + oh)*OW + ow)*IC + ic

#pragma once //compute area------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < OH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < OW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < OW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < OW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < OW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < OW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < OW);
		const int yoffset = Y0 + fh * OC * OW;//with the same ty
		float y0 = tex1Dfetch<float>(deltaY, IF_int(ly0, yoffset, -1));
		float y1 = tex1Dfetch<float>(deltaY, IF_int(ly1, yoffset + OC, -1));
		float y2 = tex1Dfetch<float>(deltaY, IF_int(ly2, yoffset + OC2, -1));
		float y3 = tex1Dfetch<float>(deltaY, IF_int(ly3, yoffset + OC3, -1));
		float y4 = tex1Dfetch<float>(deltaY, IF_int(ly4, yoffset + OC4, -1));
		float y5 = tex1Dfetch<float>(deltaY, IF_int(ly5, yoffset + OC5, -1));
		Dsv[buf][tx][ty] = float4{ y0 - y2, y3 - y1, y2 - y4, y5 - y3 };
		Dst[buf][tx][ty] = float4{ y1 + y2, y2 - y1, y3 + y4, y4 - y3 };

		//load 2 group from CW[oc, fh, fw, ic]
		const int woffset = -3 * fh * IC;//[oc, 2-fh, 2, ic]
		float2 w2 = *(float2*)(W + woffset);
		float2 w1 = *(float2*)(W + woffset + IC);
		float2 w0 = *(float2*)(W + woffset + (IC << 1));
		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
		Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };
		__syncthreads();

		for (int ooc = STEP; ooc < OC; ooc += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
				float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

				winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
				winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

				float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
				float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

				winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
				winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
			}
			buf ^= 1;

			//load 2 group from Y
			const int yoffset = Y0 + fh * OC * OW + ooc;
			float y0 = tex1Dfetch<float>(deltaY, IF_int(ly0, yoffset, -1));
			float y1 = tex1Dfetch<float>(deltaY, IF_int(ly1, yoffset + OC, -1));
			float y2 = tex1Dfetch<float>(deltaY, IF_int(ly2, yoffset + OC2, -1));
			float y3 = tex1Dfetch<float>(deltaY, IF_int(ly3, yoffset + OC3, -1));
			float y4 = tex1Dfetch<float>(deltaY, IF_int(ly4, yoffset + OC4, -1));
			float y5 = tex1Dfetch<float>(deltaY, IF_int(ly5, yoffset + OC5, -1));
			Dsv[buf][tx][ty] = float4{ y0 - y2, y3 - y1, y2 - y4, y5 - y3 };
			Dst[buf][tx][ty] = float4{ y1 + y2, y2 - y1, y3 + y4, y4 - y3 };

			//load 2 group from CW[oc, fh, fw, ic]
			const int woffset = (ooc * 9 - 3 * fh)*IC;//[ooc, 2-fh, 2, 0]
			float2 w2 = *(float2*)(W + woffset);
			float2 w1 = *(float2*)(W + woffset + IC);
			float2 w0 = *(float2*)(W + woffset + (IC << 1));
			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
			Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
			float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

			winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
			winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

			float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
			float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

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

	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2; *(float4*)(deltaX + j3) = v3;
	*(float4*)(deltaX + j4) = v4; *(float4*)(deltaX + j5) = v5;
	*(float4*)(deltaX + j6) = v6; *(float4*)(deltaX + j7) = v7;
}

#endif

