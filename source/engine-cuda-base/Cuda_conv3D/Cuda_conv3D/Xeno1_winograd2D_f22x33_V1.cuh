

#ifndef WG2D23_KERNEL1
#define WG2D23_KERNEL1

//GN = OC
//GM = (OH >> 1) * (OW >> 1) * N

#define wg2d23_k1(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, GN, GM)\
	wg2d23_kernel1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB),  dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)


//Size = 9, Time = 18.0587 msec, Performace = 1070.25 GFlop / s
template<int LB, int STEP>
__global__ void wg2d23_kernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, IC, 4, 4]
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_sindex)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Gs[1 << LB][1 << LB][4][4];//with the same ty
	__shared__ float Ds[1 << LB][1 << LB][4][4];//with the same tx

	float4 a0 = F32_4_0;
	float4 a1 = F32_4_0;
	float4 a2 = F32_4_0;
	float4 a3 = F32_4_0;

	//prepare for G[OC, IC, FH, FW]
	const int oc0 = (blockIdx.y << LB) + ty + oc_index;
	const int j0 = (blockIdx.x << LB) + tx + j_sindex;
	const int OH_slice = OH >> 1, OW_slice = OW >> 1;
	const int OHW_slice = OH_slice * OW_slice;
	int n = j0 / OHW_slice, jr = j0 % OHW_slice;
	int oh = jr / OW_slice, ow = jr % OW_slice;
	oh <<= 1; ow <<= 1;

	int ih = oh - ph, iw = ow - pw;

	for (int oic = 0; oic < IC; oic += STEP)
	{
		int gic = oic + tx;
		for (int t = 0; t < 4; t++) {//(OC, IC, FH, FW)
			Gs[tx][ty][t][0] = get4d(G, oc0, gic, t, 0, IC, 4, 4);//[oc0, gic, t, 0]
			Gs[tx][ty][t][1] = get4d(G, oc0, gic, t, 1, IC, 4, 4);//[oc0, gic, t, 1]
			Gs[tx][ty][t][2] = get4d(G, oc0, gic, t, 2, IC, 4, 4);//[oc0, gic, t, 2]
			Gs[tx][ty][t][3] = get4d(G, oc0, gic, t, 3, IC, 4, 4);//[oc0, gic, t, 3]
		}

		int xic = oic + ty;
		for (int t = 0; t < 4; t++) {
			bool lx0 = ((ih) >= 0) && ((ih) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx1 = ((ih + 1) >= 0) && ((ih + 1) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx2 = ((ih + 2) >= 0) && ((ih + 2) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx3 = ((ih + 3) >= 0) && ((ih + 3) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);

			float x0 = (lx0 ? get4d(X, n, ih, iw + t, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
			float x1 = (lx1 ? get4d(X, n, ih + 1, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
			float x2 = (lx2 ? get4d(X, n, ih + 2, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
			float x3 = (lx3 ? get4d(X, n, ih + 3, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)

			Ds[ty][tx][0][t] = x0 - x2;
			Ds[ty][tx][1][t] = x1 + x2;
			Ds[ty][tx][2][t] = x2 - x1;
			Ds[ty][tx][3][t] = x1 - x3;
		}
		__syncthreads();

		for (int t = 0; t < 4; t++) {
			float b0 = Ds[ty][tx][t][0];
			float b1 = Ds[ty][tx][t][1];
			float b2 = Ds[ty][tx][t][2];
			float b3 = Ds[ty][tx][t][3];

			Ds[ty][tx][t][0] = b0 - b2;
			Ds[ty][tx][t][1] = b1 + b2;
			Ds[ty][tx][t][2] = b2 - b1;
			Ds[ty][tx][t][3] = b1 - b3;
		}
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 g0 = *(float4*)(&Gs[ik][ty][0][0]);
			float4 d0 = *(float4*)(&Ds[ik][tx][0][0]);
			a0.x += g0.x * d0.x;//a00 - a03
			a0.y += g0.y * d0.y;
			a0.z += g0.z * d0.z;
			a0.w += g0.w * d0.w;

			float4 g1 = *(float4*)(&Gs[ik][ty][1][0]);
			float4 d1 = *(float4*)(&Ds[ik][tx][1][0]);
			a1.x += g1.x * d1.x;
			a1.y += g1.y * d1.y;
			a1.z += g1.z * d1.z;
			a1.w += g1.w * d1.w;

			float4 g2 = *(float4*)(&Gs[ik][ty][2][0]);
			float4 d2 = *(float4*)(&Ds[ik][tx][2][0]);
			a2.x += g2.x * d2.x;
			a2.y += g2.y * d2.y;
			a2.z += g2.z * d2.z;
			a2.w += g2.w * d2.w;

			float4 g3 = *(float4*)(&Gs[ik][ty][3][0]);
			float4 d3 = *(float4*)(&Ds[ik][tx][3][0]);
			a3.x += g3.x * d3.x;
			a3.y += g3.y * d3.y;
			a3.z += g3.z * d3.z;
			a3.w += g3.w * d3.w;
		}
		__syncthreads();
	}

	float b00 = a0.x + a1.x + a2.x, b10 = a1.x - a2.x - a3.x;
	float b01 = a0.y + a1.y + a2.y, b11 = a1.y - a2.y - a3.y;
	float b02 = a0.z + a1.z + a2.z, b12 = a1.z - a2.z - a3.z;
	float b03 = a0.w + a1.w + a2.w, b13 = a1.w - a2.w - a3.w;

	float y00 = b00 + b01 + b02, y01 = b01 - b02 - b03;
	float y10 = b10 + b11 + b12, y11 = b11 - b12 - b13;

	get4d(Y, n, oh, ow, oc0, OH, OW, OC) = y00;
	get4d(Y, n, oh, ow + 1, oc0, OH, OW, OC) = y01;
	get4d(Y, n, oh + 1, ow, oc0, OH, OW, OC) = y10;
	get4d(Y, n, oh + 1, ow + 1, oc0, OH, OW, OC) = y11;
}

#endif



#ifndef WG2D23_KERNEL2
#define WG2D23_KERNEL2

//GN = OC
//GM = (OH >> 1) * (OW >> 1) * N

#define wg2d23_k2(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, GN, GM)\
	wg2d23_kernel2<LB, (1<<LB>>2)>\
		<<< dim3(GM>>LB, GN>>LB),  dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 16.3978 msec, Performace = 1178.66 GFlop/s
template<int LB, int STEP>
__global__ void wg2d23_kernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, IC, 4, 4]
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_sindex)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//[ik][ty/tx]
	__shared__ float Gs[1 << LB >> 2][1 << LB][4][4];//with the same ty
	__shared__ float Ds[1 << LB >> 2][1 << LB][4][4];//with the same tx

	float4 a0 = F32_4_0;
	float4 a1 = F32_4_0;
	float4 a2 = F32_4_0;
	float4 a3 = F32_4_0;

	//prepare for G[OC, IC, FH, FW]
	const int oc0 = (blockIdx.y << LB) + ty + oc_index;
	const int j0 = (blockIdx.x << LB) + tx + j_sindex;
	const int OH_slice = OH >> 1, OW_slice = OW >> 1;
	const int OHW_slice = OH_slice * OW_slice;
	int n = j0 / OHW_slice, jr = j0 % OHW_slice;
	int oh = jr / OW_slice, ow = jr % OW_slice;
	oh <<= 1; ow <<= 1;

	int ih = oh - ph, iw = ow - pw;

	for (int oic = 0; oic < IC; oic += STEP)
	{
		int gic = oic + (tx >> 2); {//with the same ty 
			int t = tx & 3;
			Gs[tx >> 2][ty][t][0] = get4d(G, oc0, gic, t, 0, IC, 4, 4);//[oc0, gic, t, 0]
			Gs[tx >> 2][ty][t][1] = get4d(G, oc0, gic, t, 1, IC, 4, 4);//[oc0, gic, t, 1]
			Gs[tx >> 2][ty][t][2] = get4d(G, oc0, gic, t, 2, IC, 4, 4);//[oc0, gic, t, 2]
			Gs[tx >> 2][ty][t][3] = get4d(G, oc0, gic, t, 3, IC, 4, 4);//[oc0, gic, t, 3]
		}

		int xic = oic + (ty >> 2); {//
			int t = ty & 3;
			bool lx0 = ((ih) >= 0) && ((ih) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx1 = ((ih + 1) >= 0) && ((ih + 1) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx2 = ((ih + 2) >= 0) && ((ih + 2) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx3 = ((ih + 3) >= 0) && ((ih + 3) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);

			float x0 = (lx0 ? get4d(X, n, ih, iw + t, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
			float x1 = (lx1 ? get4d(X, n, ih + 1, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
			float x2 = (lx2 ? get4d(X, n, ih + 2, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
			float x3 = (lx3 ? get4d(X, n, ih + 3, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)

			Ds[ty >> 2][tx][0][t] = x0 - x2;
			Ds[ty >> 2][tx][1][t] = x1 + x2;
			Ds[ty >> 2][tx][2][t] = x2 - x1;
			Ds[ty >> 2][tx][3][t] = x1 - x3;
			__syncthreads();

			float b0 = Ds[ty >> 2][tx][t][0];
			float b1 = Ds[ty >> 2][tx][t][1];
			float b2 = Ds[ty >> 2][tx][t][2];
			float b3 = Ds[ty >> 2][tx][t][3];

			Ds[ty >> 2][tx][t][0] = b0 - b2;
			Ds[ty >> 2][tx][t][1] = b1 + b2;
			Ds[ty >> 2][tx][t][2] = b2 - b1;
			Ds[ty >> 2][tx][t][3] = b1 - b3;
			__syncthreads();
		}

#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 g0 = *(float4*)(&Gs[ik][ty][0][0]);
			float4 d0 = *(float4*)(&Ds[ik][tx][0][0]);
			a0.x += g0.x * d0.x;//a00 - a03
			a0.y += g0.y * d0.y;
			a0.z += g0.z * d0.z;
			a0.w += g0.w * d0.w;

			float4 g1 = *(float4*)(&Gs[ik][ty][1][0]);
			float4 d1 = *(float4*)(&Ds[ik][tx][1][0]);
			a1.x += g1.x * d1.x;
			a1.y += g1.y * d1.y;
			a1.z += g1.z * d1.z;
			a1.w += g1.w * d1.w;

			float4 g2 = *(float4*)(&Gs[ik][ty][2][0]);
			float4 d2 = *(float4*)(&Ds[ik][tx][2][0]);
			a2.x += g2.x * d2.x;
			a2.y += g2.y * d2.y;
			a2.z += g2.z * d2.z;
			a2.w += g2.w * d2.w;

			float4 g3 = *(float4*)(&Gs[ik][ty][3][0]);
			float4 d3 = *(float4*)(&Ds[ik][tx][3][0]);
			a3.x += g3.x * d3.x;
			a3.y += g3.y * d3.y;
			a3.z += g3.z * d3.z;
			a3.w += g3.w * d3.w;
		}
		__syncthreads();
	}

	float b00 = a0.x + a1.x + a2.x, b10 = a1.x - a2.x - a3.x;
	float b01 = a0.y + a1.y + a2.y, b11 = a1.y - a2.y - a3.y;
	float b02 = a0.z + a1.z + a2.z, b12 = a1.z - a2.z - a3.z;
	float b03 = a0.w + a1.w + a2.w, b13 = a1.w - a2.w - a3.w;

	float y00 = b00 + b01 + b02, y01 = b01 - b02 - b03;
	float y10 = b10 + b11 + b12, y11 = b11 - b12 - b13;

	get4d(Y, n, oh, ow, oc0, OH, OW, OC) = y00;
	get4d(Y, n, oh, ow + 1, oc0, OH, OW, OC) = y01;
	get4d(Y, n, oh + 1, ow, oc0, OH, OW, OC) = y10;
	get4d(Y, n, oh + 1, ow + 1, oc0, OH, OW, OC) = y11;
}

#endif


#ifndef WG2D23_KERNEL3
#define WG2D23_KERNEL3

//GN = OC
//GM = (OH >> 1) * (OW >> 1) * N

#define wg2d23_k3(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, GN, GM)\
	wg2d23_kernel3<LB, (1<<LB>>2)>\
		<<< dim3(GM>>LB, GN>>LB),  dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 14.8106 msec, Performace = 1304.97 GFlop/s
template<int LB, int STEP>
__global__ void wg2d23_kernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, IC, 4, 4]
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_sindex)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//[ik][ty/tx]
	__shared__ float Gs[1 << LB >> 2][1 << LB][4][4];//with the same ty
	__shared__ float Ds[1 << LB >> 2][1 << LB][4][4];//with the same tx

	float4 a0 = F32_4_0;
	float4 a1 = F32_4_0;
	float4 a2 = F32_4_0;
	float4 a3 = F32_4_0;

	//prepare for G[OC, IC, FH, FW]
	const int oc0 = (blockIdx.y << LB) + ty + oc_index;
	const int j0 = (blockIdx.x << LB) + tx + j_sindex;
	const int OH_slice = OH >> 1, OW_slice = OW >> 1;
	const int OHW_slice = OH_slice * OW_slice;
	int n = j0 / OHW_slice, jr = j0 % OHW_slice;
	int oh = jr / OW_slice, ow = jr % OW_slice;
	oh <<= 1; ow <<= 1;

	int ih = oh - ph, iw = ow - pw;
	const int gt = tx & 3;

	for (int oic = 0; oic < IC; oic += STEP)
	{
		int gic = oic + (tx >> 2);
		*(float4*)(&Gs[tx >> 2][ty][gt]) = *(float4*)(&get4d(G, oc0, gic, gt, 0, IC, 4, 4));//[oc0, gic, t, 0-3]

		int xic = oic + (ty >> 2); {//
			int t = ty & 3;
			bool lx0 = ((ih) >= 0) && ((ih) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx1 = ((ih + 1) >= 0) && ((ih + 1) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx2 = ((ih + 2) >= 0) && ((ih + 2) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx3 = ((ih + 3) >= 0) && ((ih + 3) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);

			float x0 = (lx0 ? get4d(X, n, ih, iw + t, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
			float x1 = (lx1 ? get4d(X, n, ih + 1, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
			float x2 = (lx2 ? get4d(X, n, ih + 2, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
			float x3 = (lx3 ? get4d(X, n, ih + 3, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)

			Ds[ty >> 2][tx][0][t] = x0 - x2;
			Ds[ty >> 2][tx][1][t] = x1 + x2;
			Ds[ty >> 2][tx][2][t] = x2 - x1;
			Ds[ty >> 2][tx][3][t] = x1 - x3;
			__syncthreads();

			float4 b = *(float4*)(&Ds[ty >> 2][tx][t][0]);
			float b0 = b.x - b.z;
			float b1 = b.y + b.z;
			float b2 = b.z - b.y;
			float b3 = b.y - b.w;
			*(float4*)(&Ds[ty >> 2][tx][t][0]) = float4{ b0, b1, b2, b3 };
			__syncthreads();
		}

#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 g0 = *(float4*)(&Gs[ik][ty][0][0]), d0 = *(float4*)(&Ds[ik][tx][0][0]);
			a0.x += g0.x * d0.x;
			a0.y += g0.y * d0.y;
			a0.z += g0.z * d0.z;
			a0.w += g0.w * d0.w;

			float4 g1 = *(float4*)(&Gs[ik][ty][1][0]), d1 = *(float4*)(&Ds[ik][tx][1][0]);
			a1.x += g1.x * d1.x;
			a1.y += g1.y * d1.y;
			a1.z += g1.z * d1.z;
			a1.w += g1.w * d1.w;

			float4 g2 = *(float4*)(&Gs[ik][ty][2][0]), d2 = *(float4*)(&Ds[ik][tx][2][0]);
			a2.x += g2.x * d2.x;
			a2.y += g2.y * d2.y;
			a2.z += g2.z * d2.z;
			a2.w += g2.w * d2.w;

			float4 g3 = *(float4*)(&Gs[ik][ty][3][0]), d3 = *(float4*)(&Ds[ik][tx][3][0]);
			a3.x += g3.x * d3.x;
			a3.y += g3.y * d3.y;
			a3.z += g3.z * d3.z;
			a3.w += g3.w * d3.w;
		}
		__syncthreads();
	}

	float b00 = a0.x + a1.x + a2.x, b10 = a1.x - a2.x - a3.x;
	float b01 = a0.y + a1.y + a2.y, b11 = a1.y - a2.y - a3.y;
	float b02 = a0.z + a1.z + a2.z, b12 = a1.z - a2.z - a3.z;
	float b03 = a0.w + a1.w + a2.w, b13 = a1.w - a2.w - a3.w;

	float y00 = b00 + b01 + b02, y01 = b01 - b02 - b03;
	float y10 = b10 + b11 + b12, y11 = b11 - b12 - b13;

	get4d(Y, n, oh, ow, oc0, OH, OW, OC) = y00;
	get4d(Y, n, oh, ow + 1, oc0, OH, OW, OC) = y01;
	get4d(Y, n, oh + 1, ow, oc0, OH, OW, OC) = y10;
	get4d(Y, n, oh + 1, ow + 1, oc0, OH, OW, OC) = y11;
}

#endif


#ifndef WG2D23_KERNEL4
#define WG2D23_KERNEL4

//GN = OC
//GM = (OH >> 1) * (OW >> 1) * N

#define wg2d23_k4(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, GN, GM)\
	wg2d23_kernel4<LB, (1<<LB>>2)>\
		<<< dim3(GM>>LB, GN>>LB),  dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 14.8106 msec, Performace = 1304.97 GFlop/s
template<int LB, int STEP>
__global__ void wg2d23_kernel4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, IC, 4, 4]
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_sindex)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//[ik][ty/tx]
	__shared__ float Gs[STEP][1 << LB][16];//with the same ty
	__shared__ float Ds[STEP][1 << LB][16];//with the same tx

	float4 a0 = F32_4_0;
	float4 a1 = F32_4_0;
	float4 a2 = F32_4_0;
	float4 a3 = F32_4_0;

	//prepare for G[OC, IC, FH, FW]
	const int oc0 = (blockIdx.y << LB) + ty + oc_index;
	const int j0 = (blockIdx.x << LB) + tx + j_sindex;
	const int OH_slice = OH >> 1, OW_slice = OW >> 1;
	const int OHW_slice = OH_slice * OW_slice;
	int n = j0 / OHW_slice, jr = j0 % OHW_slice;
	int oh = jr / OW_slice, ow = jr % OW_slice;
	oh <<= 1; ow <<= 1;

	int ih = oh - ph, iw = ow - pw;
	const int gt = tx & 3;

	for (int oic = 0; oic < IC; oic += STEP)
	{
		int gic = oic + (tx >> 2);
		*(float4*)(&Gs[tx >> 2][ty][gt << 2]) = *(float4*)(&get4d(G, oc0, gic, gt, 0, IC, 4, 4));//[oc0, gic, t, 0-3]

		int xic = oic + (ty >> 2); {//
			int t = ty & 3;
			bool lx0 = ((ih) >= 0) && ((ih) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx1 = ((ih + 1) >= 0) && ((ih + 1) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx2 = ((ih + 2) >= 0) && ((ih + 2) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);
			bool lx3 = ((ih + 3) >= 0) && ((ih + 3) < IH) && ((iw + t) >= 0) && ((iw + t) < IW);

			float x0 = (lx0 ? get4d(X, n, ih, iw + t, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
			float x1 = (lx1 ? get4d(X, n, ih + 1, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
			float x2 = (lx2 ? get4d(X, n, ih + 2, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
			float x3 = (lx3 ? get4d(X, n, ih + 3, iw + t, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)

			Ds[ty >> 2][tx][t] = x0 - x2;
			Ds[ty >> 2][tx][t + 4] = x1 + x2;
			Ds[ty >> 2][tx][t + 8] = x2 - x1;
			Ds[ty >> 2][tx][t + 12] = x1 - x3;
			__syncthreads();

			float4 b = *(float4*)(&Ds[ty >> 2][tx][t << 2]);
			float b0 = b.x - b.z;
			float b1 = b.y + b.z;
			float b2 = b.z - b.y;
			float b3 = b.y - b.w;
			*(float4*)(&Ds[ty >> 2][tx][t << 2]) = float4{ b0, b1, b2, b3 };
			__syncthreads();
		}

#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 g0 = *(float4*)(&Gs[ik][ty][0]), d0 = *(float4*)(&Ds[ik][tx][0]);
			a0.x += g0.x * d0.x;
			a0.y += g0.y * d0.y;
			a0.z += g0.z * d0.z;
			a0.w += g0.w * d0.w;

			float4 g1 = *(float4*)(&Gs[ik][ty][4]), d1 = *(float4*)(&Ds[ik][tx][4]);
			a1.x += g1.x * d1.x;
			a1.y += g1.y * d1.y;
			a1.z += g1.z * d1.z;
			a1.w += g1.w * d1.w;

			float4 g2 = *(float4*)(&Gs[ik][ty][8]), d2 = *(float4*)(&Ds[ik][tx][8]);
			a2.x += g2.x * d2.x;
			a2.y += g2.y * d2.y;
			a2.z += g2.z * d2.z;
			a2.w += g2.w * d2.w;

			float4 g3 = *(float4*)(&Gs[ik][ty][12]), d3 = *(float4*)(&Ds[ik][tx][12]);
			a3.x += g3.x * d3.x;
			a3.y += g3.y * d3.y;
			a3.z += g3.z * d3.z;
			a3.w += g3.w * d3.w;
		}
		__syncthreads();
	}

	float b00 = a0.x + a1.x + a2.x, b10 = a1.x - a2.x - a3.x;
	float b01 = a0.y + a1.y + a2.y, b11 = a1.y - a2.y - a3.y;
	float b02 = a0.z + a1.z + a2.z, b12 = a1.z - a2.z - a3.z;
	float b03 = a0.w + a1.w + a2.w, b13 = a1.w - a2.w - a3.w;

	float y00 = b00 + b01 + b02, y01 = b01 - b02 - b03;
	float y10 = b10 + b11 + b12, y11 = b11 - b12 - b13;

	get4d(Y, n, oh, ow, oc0, OH, OW, OC) = y00;
	get4d(Y, n, oh, ow + 1, oc0, OH, OW, OC) = y01;
	get4d(Y, n, oh + 1, ow, oc0, OH, OW, OC) = y10;
	get4d(Y, n, oh + 1, ow + 1, oc0, OH, OW, OC) = y11;
}

#endif


#ifndef WG2D23_KERNEL5
#define WG2D23_KERNEL5

//GN = OC
//GM = (OH >> 1) * (OW >> 1) * N

#define wg2d23_k5(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, GN, GM)\
	wg2d23_kernel5<LB, (1<<LB>>2)>\
		<<< dim3(GM>>LB, GN>>LB),  dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 14.8106 msec, Performace = 1304.97 GFlop/s
template<int LB, int STEP>
__global__ void wg2d23_kernel5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, IC, 4, 4]
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_sindex)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//[ik][ty/tx]
	__shared__ float Gs[STEP][1 << LB][16];//with the same ty
	__shared__ float Ds[STEP][1 << LB][16];//with the same tx

	float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
	float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
	float a20 = 0.0f, a21 = 0.0f, a22 = 0.0f, a23 = 0.0f;
	float a30 = 0.0f, a31 = 0.0f, a32 = 0.0f, a33 = 0.0f;

	//prepare for G[OC, IC, FH, FW]
	const int oc0 = (blockIdx.y << LB) + ty + oc_index;
	const int j0 = (blockIdx.x << LB) + tx + j_sindex;
	const int OH_slice = OH >> 1, OW_slice = OW >> 1;
	const int OHW_slice = OH_slice * OW_slice;
	int n = j0 / OHW_slice, jr = j0 % OHW_slice;
	int oh = jr / OW_slice, ow = jr % OW_slice;
	oh <<= 1; ow <<= 1;

	int ih = oh - ph, iw = ow - pw;
	const int gt = tx & 3;
	const int xt = ty & 3;

	for (int oic = 0; oic < IC; oic += STEP)
	{
		const int gic = oic + (tx >> 2);
		const int Goffset = ((oc0*IC + gic) * 4 + gt) * 4;
		*(float4*)(&Gs[tx >> 2][ty][gt << 2]) = *(float4*)(G + Goffset);//[oc0, gic, t, 0-3]

		const int xic = oic + (ty >> 2);
		bool lx0 = ((ih) >= 0) && ((ih) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);
		bool lx1 = ((ih + 1) >= 0) && ((ih + 1) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);
		bool lx2 = ((ih + 2) >= 0) && ((ih + 2) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);
		bool lx3 = ((ih + 3) >= 0) && ((ih + 3) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);

		float x0 = (lx0 ? get4d(X, n, ih, iw + xt, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
		float x1 = (lx1 ? get4d(X, n, ih + 1, iw + xt, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
		float x2 = (lx2 ? get4d(X, n, ih + 2, iw + xt, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
		float x3 = (lx3 ? get4d(X, n, ih + 3, iw + xt, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)

		Ds[ty >> 2][tx][xt] = x0 - x2;
		Ds[ty >> 2][tx][xt + 4] = x1 + x2;
		Ds[ty >> 2][tx][xt + 8] = x2 - x1;
		Ds[ty >> 2][tx][xt + 12] = x1 - x3;
		__syncthreads();

		float4 b = *(float4*)(&Ds[ty >> 2][tx][xt << 2]);
		float b0 = b.x - b.z;
		float b1 = b.y + b.z;
		float b2 = b.z - b.y;
		float b3 = b.y - b.w;
		*(float4*)(&Ds[ty >> 2][tx][xt << 2]) = float4{ b0, b1, b2, b3 };
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 g0 = *(float4*)(&Gs[ik][ty][0]), d0 = *(float4*)(&Ds[ik][tx][0]);
			a00 += g0.x * d0.x;
			a01 += g0.y * d0.y;
			a02 += g0.z * d0.z;
			a03 += g0.w * d0.w;

			float4 g1 = *(float4*)(&Gs[ik][ty][4]), d1 = *(float4*)(&Ds[ik][tx][4]);
			a10 += g1.x * d1.x;
			a11 += g1.y * d1.y;
			a12 += g1.z * d1.z;
			a13 += g1.w * d1.w;

			float4 g2 = *(float4*)(&Gs[ik][ty][8]), d2 = *(float4*)(&Ds[ik][tx][8]);
			a20 += g2.x * d2.x;
			a21 += g2.y * d2.y;
			a22 += g2.z * d2.z;
			a23 += g2.w * d2.w;

			float4 g3 = *(float4*)(&Gs[ik][ty][12]), d3 = *(float4*)(&Ds[ik][tx][12]);
			a30 += g3.x * d3.x;
			a31 += g3.y * d3.y;
			a32 += g3.z * d3.z;
			a33 += g3.w * d3.w;
		}
		__syncthreads();
	}

	float b00 = a00 + a10 + a20, b10 = a10 - a20 - a30;
	float b01 = a01 + a11 + a21, b11 = a11 - a21 - a31;
	float b02 = a02 + a12 + a22, b12 = a12 - a22 - a32;
	float b03 = a03 + a13 + a23, b13 = a13 - a23 - a33;

	float y00 = b00 + b01 + b02, y01 = b01 - b02 - b03;
	float y10 = b10 + b11 + b12, y11 = b11 - b12 - b13;

	get4d(Y, n, oh, ow, oc0, OH, OW, OC) = y00;
	get4d(Y, n, oh, ow + 1, oc0, OH, OW, OC) = y01;
	get4d(Y, n, oh + 1, ow, oc0, OH, OW, OC) = y10;
	get4d(Y, n, oh + 1, ow + 1, oc0, OH, OW, OC) = y11;
}

#endif


#ifndef WG2D23_KERNEL6
#define WG2D23_KERNEL6

//GN = OC
//GM = (OH >> 1) * (OW >> 1) * N

#define wg2d23_k6(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, GN, GM)\
	wg2d23_kernel6<LB, (1<<LB>>2)>\
		<<< dim3(GM>>LB, (GN>>LB>>1)),  dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 8.60486 msec, Performace = 2246.1 GFlop/s
template<int LB, int STEP>
__global__ void wg2d23_kernel6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, IC, 4, 4]
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_sindex)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//[ik][ty/tx]
	__shared__ float Gs[STEP][2 << LB][16];//with the same ty
	__shared__ float Ds[STEP][1 << LB][16];//with the same tx

	float2 a00 = F32_2_0, a01 = F32_2_0, a02 = F32_2_0, a03 = F32_2_0;
	float2 a10 = F32_2_0, a11 = F32_2_0, a12 = F32_2_0, a13 = F32_2_0;
	float2 a20 = F32_2_0, a21 = F32_2_0, a22 = F32_2_0, a23 = F32_2_0;
	float2 a30 = F32_2_0, a31 = F32_2_0, a32 = F32_2_0, a33 = F32_2_0;

	//prepare for G[OC, IC, FH, FW]
	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int oc1 = oc0 + 1;

	//prepare for Y[N, OH, OW, OC]
	const int j0 = ((blockIdx.x << LB) + tx) + j_sindex;
	const int OH_slice = OH >> 1, OW_slice = OW >> 1;
	const int OHW_slice = OH_slice * OW_slice;
	int n = j0 / OHW_slice, jr = j0 % OHW_slice;
	int oh = jr / OW_slice, ow = jr % OW_slice;
	oh <<= 1; ow <<= 1;

	const int ih = oh - ph, iw = ow - pw;
	const int gt = tx & 3;
	const int xt = ty & 3;

	for (int oic = 0; oic < IC; oic += STEP)
	{
		const int gic = oic + (tx >> 2);//with the same ty
		const int goffset0 = ((oc0*IC + gic) * 4 + gt) * 4;//[oc0, gic, t, 0-3]
		const int goffset1 = ((oc1*IC + gic) * 4 + gt) * 4;//[oc0, gic, t, 0-3]
		*(float4*)(&Gs[tx >> 2][(ty << 1)][gt << 2]) = *(float4*)(G + goffset0);
		*(float4*)(&Gs[tx >> 2][(ty << 1) + 1][gt << 2]) = *(float4*)(G + goffset1);

		const int xic = oic + (ty >> 2);//with the same tx
		bool lx0 = ((ih) >= 0) && ((ih) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);
		bool lx1 = ((ih + 1) >= 0) && ((ih + 1) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);
		bool lx2 = ((ih + 2) >= 0) && ((ih + 2) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);
		bool lx3 = ((ih + 3) >= 0) && ((ih + 3) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);

		float x0 = (lx0 ? get4d(X, n, ih, iw + xt, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
		float x1 = (lx1 ? get4d(X, n, ih + 1, iw + xt, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
		float x2 = (lx2 ? get4d(X, n, ih + 2, iw + xt, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
		float x3 = (lx3 ? get4d(X, n, ih + 3, iw + xt, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)

		Ds[ty >> 2][tx][xt] = x0 - x2;
		Ds[ty >> 2][tx][xt + 4] = x1 + x2;
		Ds[ty >> 2][tx][xt + 8] = x2 - x1;
		Ds[ty >> 2][tx][xt + 12] = x1 - x3;
		__syncthreads();

		float4 b = *(float4*)(&Ds[ty >> 2][tx][xt << 2]);
		float b0 = b.x - b.z;
		float b1 = b.y + b.z;
		float b2 = b.z - b.y;
		float b3 = b.y - b.w;
		*(float4*)(&Ds[ty >> 2][tx][xt << 2]) = float4{ b0, b1, b2, b3 };
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 g0 = *(float4*)(&Gs[ik][(ty << 1)][0]);
			float4 g1 = *(float4*)(&Gs[ik][(ty << 1) + 1][0]);
			float4 d0 = *(float4*)(&Ds[ik][tx][0]);
			a00.x += g0.x * d0.x; a00.y += g1.x * d0.x;
			a01.x += g0.y * d0.y; a01.y += g1.y * d0.y;
			a02.x += g0.z * d0.z; a02.y += g1.z * d0.z;
			a03.x += g0.w * d0.w; a03.y += g1.w * d0.w;

			float4 g2 = *(float4*)(&Gs[ik][(ty << 1)][4]);
			float4 g3 = *(float4*)(&Gs[ik][(ty << 1) + 1][4]);
			float4 d1 = *(float4*)(&Ds[ik][tx][4]);
			a10.x += g2.x * d1.x; a10.y += g3.x * d1.x;
			a11.x += g2.y * d1.y; a11.y += g3.y * d1.y;
			a12.x += g2.z * d1.z; a12.y += g3.z * d1.z;
			a13.x += g2.w * d1.w; a13.y += g3.w * d1.w;

			float4 g4 = *(float4*)(&Gs[ik][(ty << 1)][8]);
			float4 g5 = *(float4*)(&Gs[ik][(ty << 1) + 1][8]);
			float4 d2 = *(float4*)(&Ds[ik][tx][8]);
			a20.x += g4.x * d2.x; a20.y += g5.x * d2.x;
			a21.x += g4.y * d2.y; a21.y += g5.y * d2.y;
			a22.x += g4.z * d2.z; a22.y += g5.z * d2.z;
			a23.x += g4.w * d2.w; a23.y += g5.w * d2.w;

			float4 g6 = *(float4*)(&Gs[ik][(ty << 1)][12]);
			float4 g7 = *(float4*)(&Gs[ik][(ty << 1) + 1][12]);
			float4 d3 = *(float4*)(&Ds[ik][tx][12]);
			a30.x += g6.x * d3.x; a30.y += g7.x * d3.x;
			a31.x += g6.y * d3.y; a31.y += g7.y * d3.y;
			a32.x += g6.z * d3.z; a32.y += g7.z * d3.z;
			a33.x += g6.w * d3.w; a33.y += g7.w * d3.w;
		}
		__syncthreads();
	}

	float2 b00, b10;
	b00.x = a00.x + a10.x + a20.x, b10.x = a10.x - a20.x - a30.x;
	b00.y = a00.y + a10.y + a20.y, b10.y = a10.y - a20.y - a30.y;

	float2 b01, b11;
	b01.x = a01.x + a11.x + a21.x, b11.x = a11.x - a21.x - a31.x;
	b01.y = a01.y + a11.y + a21.y, b11.y = a11.y - a21.y - a31.y;

	float2 b02, b12;
	b02.x = a02.x + a12.x + a22.x, b12.x = a12.x - a22.x - a32.x;
	b02.y = a02.y + a12.y + a22.y, b12.y = a12.y - a22.y - a32.y;

	float2 b03, b13;
	b03.x = a03.x + a13.x + a23.x, b13.x = a13.x - a23.x - a33.x;
	b03.y = a03.y + a13.y + a23.y, b13.y = a13.y - a23.y - a33.y;

	float2 y00, y01;
	y00.x = b00.x + b01.x + b02.x, y01.x = b01.x - b02.x - b03.x;
	y00.y = b00.y + b01.y + b02.y, y01.y = b01.y - b02.y - b03.y;

	float2 y10, y11;
	y10.x = b10.x + b11.x + b12.x, y11.x = b11.x - b12.x - b13.x;
	y10.y = b10.y + b11.y + b12.y, y11.y = b11.y - b12.y - b13.y;

	*(float2*)(&get4d(Y, n, oh, ow, oc0, OH, OW, OC)) = y00;
	*(float2*)(&get4d(Y, n, oh, ow + 1, oc0, OH, OW, OC)) = y01;
	*(float2*)(&get4d(Y, n, oh + 1, ow, oc0, OH, OW, OC)) = y10;
	*(float2*)(&get4d(Y, n, oh + 1, ow + 1, oc0, OH, OW, OC)) = y11;
}

#endif


#ifndef WG2D23_KERNEL7
#define WG2D23_KERNEL7

//GN = OC
//GM = (OH >> 1) * (OW >> 1) * N

#define wg2d23_k7(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, GN, GM)\
	wg2d23_kernel7<LB, (1<<LB>>2)>\
		<<< dim3(GM>>LB, (GN>>LB>>1)),  dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 8.60486 msec, Performace = 2246.1 GFlop/s
template<int LB, int STEP>
__global__ void wg2d23_kernel7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, IC, 4, 4]
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_sindex)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//[ik][ty/tx]
	__shared__ float4 Gs[STEP][2 << LB][4];//with the same ty
	__shared__ float Ds[STEP][1 << LB][16];//with the same tx

	float2 a00 = F32_2_0, a01 = F32_2_0, a02 = F32_2_0, a03 = F32_2_0;
	float2 a10 = F32_2_0, a11 = F32_2_0, a12 = F32_2_0, a13 = F32_2_0;
	float2 a20 = F32_2_0, a21 = F32_2_0, a22 = F32_2_0, a23 = F32_2_0;
	float2 a30 = F32_2_0, a31 = F32_2_0, a32 = F32_2_0, a33 = F32_2_0;

	//prepare for G[OC, IC, FH, FW]
	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int oc1 = oc0 + 1;

	//prepare for Y[N, OH, OW, OC]
	const int j0 = ((blockIdx.x << LB) + tx) + j_sindex;
	const int OH_slice = OH >> 1, OW_slice = OW >> 1;
	const int OHW_slice = OH_slice * OW_slice;
	int n = j0 / OHW_slice, jr = j0 % OHW_slice;
	int oh = jr / OW_slice, ow = jr % OW_slice;
	oh <<= 1; ow <<= 1;

	const int ih = oh - ph, iw = ow - pw;
	const int gt = tx & 3;
	const int xt = ty & 3;

	for (int oic = 0; oic < IC; oic += STEP)
	{
		const int gic = oic + (tx >> 2);//with the same ty
		const int goffset0 = ((oc0*IC + gic) * 4 + gt) * 4;//[oc0, gic, t, 0-3]
		const int goffset1 = ((oc1*IC + gic) * 4 + gt) * 4;//[oc0, gic, t, 0-3]
		Gs[tx >> 2][(ty << 1)][gt] = *(float4*)(G + goffset0);
		Gs[tx >> 2][(ty << 1) + 1][gt] = *(float4*)(G + goffset1);

		const int xic = oic + (ty >> 2);//with the same tx
		bool lx0 = ((ih) >= 0) && ((ih) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);
		bool lx1 = ((ih + 1) >= 0) && ((ih + 1) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);
		bool lx2 = ((ih + 2) >= 0) && ((ih + 2) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);
		bool lx3 = ((ih + 3) >= 0) && ((ih + 3) < IH) && ((iw + xt) >= 0) && ((iw + xt) < IW);

		float x0 = (lx0 ? get4d(X, n, ih, iw + xt, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
		float x1 = (lx1 ? get4d(X, n, ih + 1, iw + xt, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
		float x2 = (lx2 ? get4d(X, n, ih + 2, iw + xt, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
		float x3 = (lx3 ? get4d(X, n, ih + 3, iw + xt, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)

		Ds[ty >> 2][tx][xt] = x0 - x2;
		Ds[ty >> 2][tx][xt + 4] = x1 + x2;
		Ds[ty >> 2][tx][xt + 8] = x2 - x1;
		Ds[ty >> 2][tx][xt + 12] = x1 - x3;
		__syncthreads();

		float4 b = *(float4*)(&Ds[ty >> 2][tx][xt << 2]);
		float b0 = b.x - b.z;
		float b1 = b.y + b.z;
		float b2 = b.z - b.y;
		float b3 = b.y - b.w;
		*(float4*)(&Ds[ty >> 2][tx][xt << 2]) = float4{ b0, b1, b2, b3 };
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[ik][(ty << 1)][0], g1 = Gs[ik][(ty << 1) + 1][0];
			float4 d0 = *(float4*)(&Ds[ik][tx][0]);
			a00.x += g0.x * d0.x; a00.y += g1.x * d0.x;
			a01.x += g0.y * d0.y; a01.y += g1.y * d0.y;
			a02.x += g0.z * d0.z; a02.y += g1.z * d0.z;
			a03.x += g0.w * d0.w; a03.y += g1.w * d0.w;

			float4 g2 = Gs[ik][(ty << 1)][1], g3 = Gs[ik][(ty << 1) + 1][1];
			float4 d1 = *(float4*)(&Ds[ik][tx][4]);
			a10.x += g2.x * d1.x; a10.y += g3.x * d1.x;
			a11.x += g2.y * d1.y; a11.y += g3.y * d1.y;
			a12.x += g2.z * d1.z; a12.y += g3.z * d1.z;
			a13.x += g2.w * d1.w; a13.y += g3.w * d1.w;

			float4 g4 = Gs[ik][(ty << 1)][2], g5 = Gs[ik][(ty << 1) + 1][2];
			float4 d2 = *(float4*)(&Ds[ik][tx][8]);
			a20.x += g4.x * d2.x; a20.y += g5.x * d2.x;
			a21.x += g4.y * d2.y; a21.y += g5.y * d2.y;
			a22.x += g4.z * d2.z; a22.y += g5.z * d2.z;
			a23.x += g4.w * d2.w; a23.y += g5.w * d2.w;

			float4 g6 = Gs[ik][(ty << 1)][3], g7 = Gs[ik][(ty << 1) + 1][3];
			float4 d3 = *(float4*)(&Ds[ik][tx][12]);
			a30.x += g6.x * d3.x; a30.y += g7.x * d3.x;
			a31.x += g6.y * d3.y; a31.y += g7.y * d3.y;
			a32.x += g6.z * d3.z; a32.y += g7.z * d3.z;
			a33.x += g6.w * d3.w; a33.y += g7.w * d3.w;
		}
		__syncthreads();
	}

	//-----------------------------------------------------------------
	float2 b00, b01, b02, b03;
	b00.x = a00.x + a10.x + a20.x;
	b00.y = a00.y + a10.y + a20.y;

	b01.x = a01.x + a11.x + a21.x;
	b01.y = a01.y + a11.y + a21.y;

	b02.x = a02.x + a12.x + a22.x;
	b02.y = a02.y + a12.y + a22.y;

	b03.x = a03.x + a13.x + a23.x;
	b03.y = a03.y + a13.y + a23.y;

	float2 y00, y01;
	y00.x = b00.x + b01.x + b02.x, y01.x = b01.x - b02.x - b03.x;
	y00.y = b00.y + b01.y + b02.y, y01.y = b01.y - b02.y - b03.y;
	*(float2*)(&get4d(Y, n, oh, ow, oc0, OH, OW, OC)) = y00;
	*(float2*)(&get4d(Y, n, oh, ow + 1, oc0, OH, OW, OC)) = y01;


	//-----------------------------------------------------------------
	float2 b10, b11, b12, b13;
	b10.x = a10.x - a20.x - a30.x;
	b10.y = a10.y - a20.y - a30.y;

	b11.x = a11.x - a21.x - a31.x;
	b11.y = a11.y - a21.y - a31.y;

	b12.x = a12.x - a22.x - a32.x;
	b12.y = a12.y - a22.y - a32.y;

	b13.x = a13.x - a23.x - a33.x;
	b13.y = a13.y - a23.y - a33.y;

	float2 y10, y11;
	y10.x = b10.x + b11.x + b12.x, y11.x = b11.x - b12.x - b13.x;
	y10.y = b10.y + b11.y + b12.y, y11.y = b11.y - b12.y - b13.y;
	*(float2*)(&get4d(Y, n, oh + 1, ow, oc0, OH, OW, OC)) = y10;
	*(float2*)(&get4d(Y, n, oh + 1, ow + 1, oc0, OH, OW, OC)) = y11;
}

#endif


//(32, 16)
#ifndef WG2D23_KERNEL8
#define WG2D23_KERNEL8

//GN = OC
//GM = (OH >> 1) * (OW >> 1) * N

#define wg2d23_k8(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, GN, GM)\
	wg2d23_kernel8<LB, (1<<LB>>2)>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>1)),  dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)

//Size = 9, Time = 8.60486 msec, Performace = 2246.1 GFlop/s
template<int LB, int STEP>
__global__ void wg2d23_kernel8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, IC, 4, 4]
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_sindex)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//[ik][ty/tx]
	__shared__ float4 Gs[STEP][2 << LB][4];//with the same ty
	__shared__ float  Ds[STEP][2 << LB][16];//with the same tx

	float2 a00 = F32_2_0, a01 = F32_2_0, a02 = F32_2_0, a03 = F32_2_0;//group0
	float2 a10 = F32_2_0, a11 = F32_2_0, a12 = F32_2_0, a13 = F32_2_0;
	float2 a20 = F32_2_0, a21 = F32_2_0, a22 = F32_2_0, a23 = F32_2_0;
	float2 a30 = F32_2_0, a31 = F32_2_0, a32 = F32_2_0, a33 = F32_2_0;

	float2 a40 = F32_2_0, a41 = F32_2_0, a42 = F32_2_0, a43 = F32_2_0;//group1
	float2 a50 = F32_2_0, a51 = F32_2_0, a52 = F32_2_0, a53 = F32_2_0;
	float2 a60 = F32_2_0, a61 = F32_2_0, a62 = F32_2_0, a63 = F32_2_0;
	float2 a70 = F32_2_0, a71 = F32_2_0, a72 = F32_2_0, a73 = F32_2_0;

	//prepare for G[OC, IC, FH, FW]
	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int oc1 = oc0 + 1;

	//prepare for Y[N, OH, OW, OC]
	const int j0 = (((blockIdx.x << LB) + tx) << 1) + j_sindex;
	const int j1 = j0 + 1;

	const int OH_slice = OH >> 1, OW_slice = OW >> 1;
	const int OHW_slice = OH_slice * OW_slice;
	int n0, oh0, ow0; { n0 = j0 / OHW_slice; int jr = j0 % OHW_slice; oh0 = jr / OW_slice; ow0 = jr % OW_slice; }
	int n1, oh1, ow1; { n1 = j1 / OHW_slice; int jr = j1 % OHW_slice; oh1 = jr / OW_slice; ow1 = jr % OW_slice; }
	oh0 <<= 1; ow0 <<= 1; const int ih0 = oh0 - ph, iw0 = ow0 - pw;
	oh1 <<= 1; ow1 <<= 1; const int ih1 = oh1 - ph, iw1 = ow1 - pw;

	const int gt = tx & 3;
	const int xt = ty & 3;
	for (int oic = 0; oic < IC; oic += STEP)
	{
		const int gic = oic + (tx >> 2);//with the same ty
		const int goffset0 = ((oc0*IC + gic) * 4 + gt) * 4;//[oc0, gic, t, 0-3]
		const int goffset1 = ((oc1*IC + gic) * 4 + gt) * 4;//[oc0, gic, t, 0-3]
		Gs[tx >> 2][(ty << 1)][gt] = *(float4*)(G + goffset0);
		Gs[tx >> 2][(ty << 1) + 1][gt] = *(float4*)(G + goffset1);

		const int xic = oic + (ty >> 2);//with the same tx
		{
			bool lx0 = ((ih0) >= 0) && ((ih0) < IH) && ((iw0 + xt) >= 0) && ((iw0 + xt) < IW);
			bool lx1 = ((ih0 + 1) >= 0) && ((ih0 + 1) < IH) && ((iw0 + xt) >= 0) && ((iw0 + xt) < IW);
			bool lx2 = ((ih0 + 2) >= 0) && ((ih0 + 2) < IH) && ((iw0 + xt) >= 0) && ((iw0 + xt) < IW);
			bool lx3 = ((ih0 + 3) >= 0) && ((ih0 + 3) < IH) && ((iw0 + xt) >= 0) && ((iw0 + xt) < IW);

			float x0 = (lx0 ? get4d(X, n0, ih0, iw0 + xt, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
			float x1 = (lx1 ? get4d(X, n0, ih0 + 1, iw0 + xt, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
			float x2 = (lx2 ? get4d(X, n0, ih0 + 2, iw0 + xt, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
			float x3 = (lx3 ? get4d(X, n0, ih0 + 3, iw0 + xt, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)

			Ds[ty >> 2][tx << 1][xt] = x0 - x2;
			Ds[ty >> 2][tx << 1][xt + 4] = x1 + x2;
			Ds[ty >> 2][tx << 1][xt + 8] = x2 - x1;
			Ds[ty >> 2][tx << 1][xt + 12] = x1 - x3;
		}

		{
			bool lx0 = ((ih1) >= 0) && ((ih1) < IH) && ((iw1 + xt) >= 0) && ((iw1 + xt) < IW);
			bool lx1 = ((ih1 + 1) >= 0) && ((ih1 + 1) < IH) && ((iw1 + xt) >= 0) && ((iw1 + xt) < IW);
			bool lx2 = ((ih1 + 2) >= 0) && ((ih1 + 2) < IH) && ((iw1 + xt) >= 0) && ((iw1 + xt) < IW);
			bool lx3 = ((ih1 + 3) >= 0) && ((ih1 + 3) < IH) && ((iw1 + xt) >= 0) && ((iw1 + xt) < IW);

			float x0 = (lx0 ? get4d(X, n0, ih1, iw1 + xt, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
			float x1 = (lx1 ? get4d(X, n0, ih1 + 1, iw1 + xt, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
			float x2 = (lx2 ? get4d(X, n0, ih1 + 2, iw1 + xt, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
			float x3 = (lx3 ? get4d(X, n0, ih1 + 3, iw1 + xt, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)

			Ds[ty >> 2][(tx << 1) + 1][xt] = x0 - x2;
			Ds[ty >> 2][(tx << 1) + 1][xt + 4] = x1 + x2;
			Ds[ty >> 2][(tx << 1) + 1][xt + 8] = x2 - x1;
			Ds[ty >> 2][(tx << 1) + 1][xt + 12] = x1 - x3;
		}
		__syncthreads();

		{
			float4 b = *(float4*)(&Ds[ty >> 2][(tx << 1)][xt << 2]);
			float b0 = b.x - b.z;
			float b1 = b.y + b.z;
			float b2 = b.z - b.y;
			float b3 = b.y - b.w;
			*(float4*)(&Ds[ty >> 2][(tx << 1)][xt << 2]) = float4{ b0, b1, b2, b3 };
		}

		{
			float4 b = *(float4*)(&Ds[ty >> 2][(tx << 1) + 1][xt << 2]);
			float b0 = b.x - b.z;
			float b1 = b.y + b.z;
			float b2 = b.z - b.y;
			float b3 = b.y - b.w;
			*(float4*)(&Ds[ty >> 2][(tx << 1) + 1][xt << 2]) = float4{ b0, b1, b2, b3 };
		}
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			//------------------------------------------------------------
			float4 g0 = Gs[ik][(ty << 1)][0], g1 = Gs[ik][(ty << 1) + 1][0];
			float4 d0 = *(float4*)(&Ds[ik][(tx << 1)][0]);
			float4 d1 = *(float4*)(&Ds[ik][(tx << 1) + 1][0]);
			a00.x += g0.x * d0.x; a00.y += g1.x * d0.x;
			a01.x += g0.y * d0.y; a01.y += g1.y * d0.y;
			a02.x += g0.z * d0.z; a02.y += g1.z * d0.z;
			a03.x += g0.w * d0.w; a03.y += g1.w * d0.w;

			a40.x += g0.x * d1.x; a40.y += g1.x * d1.x;
			a41.x += g0.y * d1.y; a41.y += g1.y * d1.y;
			a42.x += g0.z * d1.z; a42.y += g1.z * d1.z;
			a43.x += g0.w * d1.w; a43.y += g1.w * d1.w;

			//
			float4 g2 = Gs[ik][(ty << 1)][1], g3 = Gs[ik][(ty << 1) + 1][1];
			float4 d2 = *(float4*)(&Ds[ik][(tx << 1)][4]);
			float4 d3 = *(float4*)(&Ds[ik][(tx << 1) + 1][4]);
			a10.x += g2.x * d2.x; a10.y += g3.x * d2.x;
			a11.x += g2.y * d2.y; a11.y += g3.y * d2.y;
			a12.x += g2.z * d2.z; a12.y += g3.z * d2.z;
			a13.x += g2.w * d2.w; a13.y += g3.w * d2.w;

			a50.x += g2.x * d3.x; a50.y += g3.x * d3.x;
			a51.x += g2.y * d3.y; a51.y += g3.y * d3.y;
			a52.x += g2.z * d3.z; a52.y += g3.z * d3.z;
			a53.x += g2.w * d3.w; a53.y += g3.w * d3.w;

			//
			float4 g4 = Gs[ik][(ty << 1)][2], g5 = Gs[ik][(ty << 1) + 1][2];
			float4 d4 = *(float4*)(&Ds[ik][(tx << 1)][8]);
			float4 d5 = *(float4*)(&Ds[ik][(tx << 1) + 1][8]);
			a20.x += g4.x * d4.x; a20.y += g5.x * d4.x;
			a21.x += g4.y * d4.y; a21.y += g5.y * d4.y;
			a22.x += g4.z * d4.z; a22.y += g5.z * d4.z;
			a23.x += g4.w * d4.w; a23.y += g5.w * d4.w;

			a60.x += g4.x * d5.x; a60.y += g5.x * d5.x;
			a61.x += g4.y * d5.y; a61.y += g5.y * d5.y;
			a62.x += g4.z * d5.z; a62.y += g5.z * d5.z;
			a63.x += g4.w * d5.w; a63.y += g5.w * d5.w;

			//
			float4 g6 = Gs[ik][(ty << 1)][3], g7 = Gs[ik][(ty << 1) + 1][3];
			float4 d6 = *(float4*)(&Ds[ik][(tx << 1)][12]);
			float4 d7 = *(float4*)(&Ds[ik][(tx << 1) + 1][12]);
			a30.x += g6.x * d6.x; a30.y += g7.x * d6.x;
			a31.x += g6.y * d6.y; a31.y += g7.y * d6.y;
			a32.x += g6.z * d6.z; a32.y += g7.z * d6.z;
			a33.x += g6.w * d6.w; a33.y += g7.w * d6.w;

			a70.x += g6.x * d7.x; a70.y += g7.x * d7.x;
			a71.x += g6.y * d7.y; a71.y += g7.y * d7.y;
			a72.x += g6.z * d7.z; a72.y += g7.z * d7.z;
			a73.x += g6.w * d7.w; a73.y += g7.w * d7.w;
		}
		__syncthreads();
	}

	//group0-----------------------------------------------------------------
	float2 b00, b01, b02, b03;
	b00.x = a00.x + a10.x + a20.x; b00.y = a00.y + a10.y + a20.y;
	b01.x = a01.x + a11.x + a21.x; b01.y = a01.y + a11.y + a21.y;
	b02.x = a02.x + a12.x + a22.x; b02.y = a02.y + a12.y + a22.y;
	b03.x = a03.x + a13.x + a23.x; b03.y = a03.y + a13.y + a23.y;

	float2 y00, y01;
	y00.x = b00.x + b01.x + b02.x, y01.x = b01.x - b02.x - b03.x;
	y00.y = b00.y + b01.y + b02.y, y01.y = b01.y - b02.y - b03.y;
	*(float2*)(&get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC)) = y00;
	*(float2*)(&get4d(Y, n0, oh0, ow0 + 1, oc0, OH, OW, OC)) = y01;

	//-----------------------------------------------------------------
	float2 b10, b11, b12, b13;
	b10.x = a10.x - a20.x - a30.x; b10.y = a10.y - a20.y - a30.y;
	b11.x = a11.x - a21.x - a31.x; b11.y = a11.y - a21.y - a31.y;
	b12.x = a12.x - a22.x - a32.x; b12.y = a12.y - a22.y - a32.y;
	b13.x = a13.x - a23.x - a33.x; b13.y = a13.y - a23.y - a33.y;

	float2 y10, y11;
	y10.x = b10.x + b11.x + b12.x, y11.x = b11.x - b12.x - b13.x;
	y10.y = b10.y + b11.y + b12.y, y11.y = b11.y - b12.y - b13.y;
	*(float2*)(&get4d(Y, n0, oh0 + 1, ow0, oc0, OH, OW, OC)) = y10;
	*(float2*)(&get4d(Y, n0, oh0 + 1, ow0 + 1, oc0, OH, OW, OC)) = y11;

	//group1-----------------------------------------------------------------
	float2 b40, b41, b42, b43;
	b40.x = a40.x + a50.x + a60.x; b40.y = a40.y + a50.y + a60.y;
	b41.x = a41.x + a51.x + a61.x; b41.y = a41.y + a51.y + a61.y;
	b42.x = a42.x + a52.x + a62.x; b42.y = a42.y + a52.y + a62.y;
	b43.x = a43.x + a53.x + a63.x; b43.y = a43.y + a53.y + a63.y;

	float2 y40, y41;
	y40.x = b40.x + b41.x + b42.x, y41.x = b41.x - b42.x - b43.x;
	y40.y = b40.y + b41.y + b42.y, y41.y = b41.y - b42.y - b43.y;
	*(float2*)(&get4d(Y, n0, oh1, ow1, oc0, OH, OW, OC)) = y40;
	*(float2*)(&get4d(Y, n0, oh1, ow1 + 1, oc0, OH, OW, OC)) = y41;

	//-----------------------------------------------------------------
	float2 b50, b51, b52, b53;
	b50.x = a50.x - a60.x - a70.x; b50.y = a50.y - a60.y - a70.y;
	b51.x = a51.x - a61.x - a71.x; b51.y = a51.y - a61.y - a71.y;
	b52.x = a52.x - a62.x - a72.x; b52.y = a52.y - a62.y - a72.y;
	b53.x = a53.x - a63.x - a73.x; b53.y = a53.y - a63.y - a73.y;

	float2 y50, y51;
	y50.x = b50.x + b51.x + b52.x, y51.x = b51.x - b52.x - b53.x;
	y50.y = b50.y + b51.y + b52.y, y51.y = b51.y - b52.y - b53.y;
	*(float2*)(&get4d(Y, n0, oh1 + 1, ow1, oc0, OH, OW, OC)) = y50;
	*(float2*)(&get4d(Y, n0, oh1 + 1, ow1 + 1, oc0, OH, OW, OC)) = y51;
}

#endif


#ifndef WG2D23_KERNEL9
#define WG2D23_KERNEL9

//GN = OC
//GM = (OH >> 1) * (OW >> 1) * N

#define wg2d23_k9(stream, LB, oc_index, j_index, X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, GN, GM)\
	wg2d23_kernel9<LB, (1<<LB>>2)>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>1)),  dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, G, Y, OH, OW, N, IC, OC, ph, pw, oc_index, j_index)


//Size = 9, Time = 8.60486 msec, Performace = 2246.1 GFlop/s
template<int LB, int STEP>
__global__ void wg2d23_kernel9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,//[OC, IC, 4, 4]
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_sindex)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;//[ik][ty/tx]
	__shared__ float4 Gs[STEP][2 << LB][4];//with the same ty
	__shared__ float  Ds[STEP][2 << LB][16];//with the same tx

	//2 * 16
	float2 a00 = F32_2_0, a01 = F32_2_0, a02 = F32_2_0, a03 = F32_2_0;//group0
	float2 a10 = F32_2_0, a11 = F32_2_0, a12 = F32_2_0, a13 = F32_2_0;
	float2 a20 = F32_2_0, a21 = F32_2_0, a22 = F32_2_0, a23 = F32_2_0;
	float2 a30 = F32_2_0, a31 = F32_2_0, a32 = F32_2_0, a33 = F32_2_0;

	float2 a40 = F32_2_0, a41 = F32_2_0, a42 = F32_2_0, a43 = F32_2_0;//group1
	float2 a50 = F32_2_0, a51 = F32_2_0, a52 = F32_2_0, a53 = F32_2_0;
	float2 a60 = F32_2_0, a61 = F32_2_0, a62 = F32_2_0, a63 = F32_2_0;
	float2 a70 = F32_2_0, a71 = F32_2_0, a72 = F32_2_0, a73 = F32_2_0;

	//prepare for G[OC, IC, FH, FW]
	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int oc1 = oc0 + 1;

	//prepare for Y[N, OH, OW, OC]
	const int j0 = (((blockIdx.x << LB) + tx) << 1) + j_sindex;
	const int j1 = j0 + 1;

	const int OH_slice = OH >> 1, OW_slice = OW >> 1;
	const int OHW_slice = OH_slice * OW_slice;
	int n0, oh0, ow0; { n0 = j0 / OHW_slice; int jr = j0 % OHW_slice; oh0 = jr / OW_slice; ow0 = jr % OW_slice; }
	int n1, oh1, ow1; { n1 = j1 / OHW_slice; int jr = j1 % OHW_slice; oh1 = jr / OW_slice; ow1 = jr % OW_slice; }
	oh0 <<= 1; ow0 <<= 1; const int ih0 = oh0 - ph, iw0 = ow0 - pw;
	oh1 <<= 1; ow1 <<= 1; const int ih1 = oh1 - ph, iw1 = ow1 - pw;

	const int gt = tx & 3;
	const int xt = ty & 3;
	for (int oic = 0; oic < IC; oic += STEP)
	{
		const int gic = oic + (tx >> 2);//with the same ty
		const int goffset0 = ((oc0*IC + gic) * 4 + gt) * 4;//[oc0, gic, t, 0-3]
		const int goffset1 = ((oc1*IC + gic) * 4 + gt) * 4;//[oc0, gic, t, 0-3]
		Gs[tx >> 2][(ty << 1) + 1][gt] = *(float4*)(G + goffset1);
		Gs[tx >> 2][(ty << 1)][gt] = *(float4*)(G + goffset0);

		const int xic = oic + (ty >> 2);//with the same tx
		bool lx0 = ((ih0) >= 0) && ((ih0) < IH) && ((iw0 + xt) >= 0) && ((iw0 + xt) < IW);
		bool lx1 = ((ih0 + 1) >= 0) && ((ih0 + 1) < IH) && ((iw0 + xt) >= 0) && ((iw0 + xt) < IW);
		bool lx2 = ((ih0 + 2) >= 0) && ((ih0 + 2) < IH) && ((iw0 + xt) >= 0) && ((iw0 + xt) < IW);
		bool lx3 = ((ih0 + 3) >= 0) && ((ih0 + 3) < IH) && ((iw0 + xt) >= 0) && ((iw0 + xt) < IW);
		float x0 = (lx0 ? get4d(X, n0, ih0, iw0 + xt, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
		float x1 = (lx1 ? get4d(X, n0, ih0 + 1, iw0 + xt, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
		float x2 = (lx2 ? get4d(X, n0, ih0 + 2, iw0 + xt, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
		float x3 = (lx3 ? get4d(X, n0, ih0 + 3, iw0 + xt, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)
		Ds[ty >> 2][tx << 1][xt] = x0 - x2;
		Ds[ty >> 2][tx << 1][xt + 4] = x1 + x2;
		Ds[ty >> 2][tx << 1][xt + 8] = x2 - x1;
		Ds[ty >> 2][tx << 1][xt + 12] = x1 - x3;

		bool lx4 = ((ih1) >= 0) && ((ih1) < IH) && ((iw1 + xt) >= 0) && ((iw1 + xt) < IW);
		bool lx5 = ((ih1 + 1) >= 0) && ((ih1 + 1) < IH) && ((iw1 + xt) >= 0) && ((iw1 + xt) < IW);
		bool lx6 = ((ih1 + 2) >= 0) && ((ih1 + 2) < IH) && ((iw1 + xt) >= 0) && ((iw1 + xt) < IW);
		bool lx7 = ((ih1 + 3) >= 0) && ((ih1 + 3) < IH) && ((iw1 + xt) >= 0) && ((iw1 + xt) < IW);
		float x4 = (lx4 ? get4d(X, n1, ih1, iw1 + xt, xic, IH, IW, IC) : 0);//(n, ih    , iw + t, ic)
		float x5 = (lx5 ? get4d(X, n1, ih1 + 1, iw1 + xt, xic, IH, IW, IC) : 0);//(n, ih + 1, iw + t, ic)
		float x6 = (lx6 ? get4d(X, n1, ih1 + 2, iw1 + xt, xic, IH, IW, IC) : 0);//(n, ih + 2, iw + t, ic)
		float x7 = (lx7 ? get4d(X, n1, ih1 + 3, iw1 + xt, xic, IH, IW, IC) : 0);//(n, ih + 3, iw + t, ic)
		Ds[ty >> 2][(tx << 1) + 1][xt] = x4 - x6;
		Ds[ty >> 2][(tx << 1) + 1][xt + 4] = x5 + x6;
		Ds[ty >> 2][(tx << 1) + 1][xt + 8] = x6 - x5;
		Ds[ty >> 2][(tx << 1) + 1][xt + 12] = x5 - x7;
		__syncthreads();

		float4 b0 = *(float4*)(&Ds[ty >> 2][(tx << 1)][xt << 2]);
		*(float4*)(&Ds[ty >> 2][(tx << 1)][xt << 2]) = float4{
			b0.x - b0.z,
			b0.y + b0.z,
			b0.z - b0.y,
			b0.y - b0.w
		};

		float4 b1 = *(float4*)(&Ds[ty >> 2][(tx << 1) + 1][xt << 2]);
		*(float4*)(&Ds[ty >> 2][(tx << 1) + 1][xt << 2]) = float4{
			b1.x - b1.z,
			b1.y + b1.z,
			b1.z - b1.y,
			b1.y - b1.w
		};
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			//------------------------------------------------------------
			float4 g0 = Gs[ik][(ty << 1)][0], g1 = Gs[ik][(ty << 1) + 1][0];
			float4 d0 = *(float4*)(&Ds[ik][(tx << 1)][0]);
			float4 d1 = *(float4*)(&Ds[ik][(tx << 1) + 1][0]);
			a00.x += g0.x * d0.x; a00.y += g1.x * d0.x;
			a01.x += g0.y * d0.y; a01.y += g1.y * d0.y;
			a02.x += g0.z * d0.z; a02.y += g1.z * d0.z;
			a03.x += g0.w * d0.w; a03.y += g1.w * d0.w;

			a40.x += g0.x * d1.x; a40.y += g1.x * d1.x;
			a41.x += g0.y * d1.y; a41.y += g1.y * d1.y;
			a42.x += g0.z * d1.z; a42.y += g1.z * d1.z;
			a43.x += g0.w * d1.w; a43.y += g1.w * d1.w;

			//
			float4 g2 = Gs[ik][(ty << 1)][1], g3 = Gs[ik][(ty << 1) + 1][1];
			float4 d2 = *(float4*)(&Ds[ik][(tx << 1)][4]);
			float4 d3 = *(float4*)(&Ds[ik][(tx << 1) + 1][4]);
			a10.x += g2.x * d2.x; a10.y += g3.x * d2.x;
			a11.x += g2.y * d2.y; a11.y += g3.y * d2.y;
			a12.x += g2.z * d2.z; a12.y += g3.z * d2.z;
			a13.x += g2.w * d2.w; a13.y += g3.w * d2.w;

			a50.x += g2.x * d3.x; a50.y += g3.x * d3.x;
			a51.x += g2.y * d3.y; a51.y += g3.y * d3.y;
			a52.x += g2.z * d3.z; a52.y += g3.z * d3.z;
			a53.x += g2.w * d3.w; a53.y += g3.w * d3.w;

			//
			float4 g4 = Gs[ik][(ty << 1)][2], g5 = Gs[ik][(ty << 1) + 1][2];
			float4 d4 = *(float4*)(&Ds[ik][(tx << 1)][8]);
			float4 d5 = *(float4*)(&Ds[ik][(tx << 1) + 1][8]);
			a20.x += g4.x * d4.x; a20.y += g5.x * d4.x;
			a21.x += g4.y * d4.y; a21.y += g5.y * d4.y;
			a22.x += g4.z * d4.z; a22.y += g5.z * d4.z;
			a23.x += g4.w * d4.w; a23.y += g5.w * d4.w;

			a60.x += g4.x * d5.x; a60.y += g5.x * d5.x;
			a61.x += g4.y * d5.y; a61.y += g5.y * d5.y;
			a62.x += g4.z * d5.z; a62.y += g5.z * d5.z;
			a63.x += g4.w * d5.w; a63.y += g5.w * d5.w;

			//
			float4 g6 = Gs[ik][(ty << 1)][3], g7 = Gs[ik][(ty << 1) + 1][3];
			float4 d6 = *(float4*)(&Ds[ik][(tx << 1)][12]);
			float4 d7 = *(float4*)(&Ds[ik][(tx << 1) + 1][12]);
			a30.x += g6.x * d6.x; a30.y += g7.x * d6.x;
			a31.x += g6.y * d6.y; a31.y += g7.y * d6.y;
			a32.x += g6.z * d6.z; a32.y += g7.z * d6.z;
			a33.x += g6.w * d6.w; a33.y += g7.w * d6.w;

			a70.x += g6.x * d7.x; a70.y += g7.x * d7.x;
			a71.x += g6.y * d7.y; a71.y += g7.y * d7.y;
			a72.x += g6.z * d7.z; a72.y += g7.z * d7.z;
			a73.x += g6.w * d7.w; a73.y += g7.w * d7.w;
		}
		__syncthreads();
	}

	//group0-----------------------------------------------------------------
	float2 b00, b01, b02, b03;
	b00.x = a00.x + a10.x + a20.x; b00.y = a00.y + a10.y + a20.y;
	b01.x = a01.x + a11.x + a21.x; b01.y = a01.y + a11.y + a21.y;
	b02.x = a02.x + a12.x + a22.x; b02.y = a02.y + a12.y + a22.y;
	b03.x = a03.x + a13.x + a23.x; b03.y = a03.y + a13.y + a23.y;

	float2 y00, y01;
	y00.x = b00.x + b01.x + b02.x, y01.x = b01.x - b02.x - b03.x;
	y00.y = b00.y + b01.y + b02.y, y01.y = b01.y - b02.y - b03.y;
	*(float2*)(&get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC)) = y00;
	*(float2*)(&get4d(Y, n0, oh0, ow0 + 1, oc0, OH, OW, OC)) = y01;

	//-----------------------------------------------------------------
	float2 b10, b11, b12, b13;
	b10.x = a10.x - a20.x - a30.x; b10.y = a10.y - a20.y - a30.y;
	b11.x = a11.x - a21.x - a31.x; b11.y = a11.y - a21.y - a31.y;
	b12.x = a12.x - a22.x - a32.x; b12.y = a12.y - a22.y - a32.y;
	b13.x = a13.x - a23.x - a33.x; b13.y = a13.y - a23.y - a33.y;

	float2 y10, y11;
	y10.x = b10.x + b11.x + b12.x, y11.x = b11.x - b12.x - b13.x;
	y10.y = b10.y + b11.y + b12.y, y11.y = b11.y - b12.y - b13.y;
	*(float2*)(&get4d(Y, n0, oh0 + 1, ow0, oc0, OH, OW, OC)) = y10;
	*(float2*)(&get4d(Y, n0, oh0 + 1, ow0 + 1, oc0, OH, OW, OC)) = y11;

	//group1-----------------------------------------------------------------
	float2 b40, b41, b42, b43;
	b40.x = a40.x + a50.x + a60.x; b40.y = a40.y + a50.y + a60.y;
	b41.x = a41.x + a51.x + a61.x; b41.y = a41.y + a51.y + a61.y;
	b42.x = a42.x + a52.x + a62.x; b42.y = a42.y + a52.y + a62.y;
	b43.x = a43.x + a53.x + a63.x; b43.y = a43.y + a53.y + a63.y;

	float2 y40, y41;
	y40.x = b40.x + b41.x + b42.x, y41.x = b41.x - b42.x - b43.x;
	y40.y = b40.y + b41.y + b42.y, y41.y = b41.y - b42.y - b43.y;
	*(float2*)(&get4d(Y, n0, oh1, ow1, oc0, OH, OW, OC)) = y40;
	*(float2*)(&get4d(Y, n0, oh1, ow1 + 1, oc0, OH, OW, OC)) = y41;

	//-----------------------------------------------------------------
	float2 b50, b51, b52, b53;
	b50.x = a50.x - a60.x - a70.x; b50.y = a50.y - a60.y - a70.y;
	b51.x = a51.x - a61.x - a71.x; b51.y = a51.y - a61.y - a71.y;
	b52.x = a52.x - a62.x - a72.x; b52.y = a52.y - a62.y - a72.y;
	b53.x = a53.x - a63.x - a73.x; b53.y = a53.y - a63.y - a73.y;

	float2 y50, y51;
	y50.x = b50.x + b51.x + b52.x, y51.x = b51.x - b52.x - b53.x;
	y50.y = b50.y + b51.y + b52.y, y51.y = b51.y - b52.y - b53.y;
	*(float2*)(&get4d(Y, n0, oh1 + 1, ow1, oc0, OH, OW, OC)) = y50;
	*(float2*)(&get4d(Y, n0, oh1 + 1, ow1 + 1, oc0, OH, OW, OC)) = y51;
}
#endif

#endif

