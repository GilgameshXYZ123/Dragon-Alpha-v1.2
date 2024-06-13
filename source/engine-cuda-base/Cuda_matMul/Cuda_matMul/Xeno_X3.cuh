

//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef NEW_T1_KERNEL_8_8_KERNEL1
#define NEW_T1_KERNEL_8_8_KERNEL1

#define	new88T1_kernel1(LB, stream, A, B,  C, N, M, K, SA, SB) \
	NEW_88T1_kernel1<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(2<<LB, 1<<LB>>1), 0, stream >>>\
			(A, B, C, SA, SB, K)

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 1.656 msec, Performace = 10374.3  GFlop/s
//LB = 3: Size = 8, Time = 1.867 msec, Performace =  9201.86 GFlop/s
//for [4096 * 4096 * 4096]:
//LB = 4: Size = 64, Time = 13.083 msec, Performace = 10505.2  GFlop/s
//LB = 3: Size = 64, Time = 14.822 msec, Performace =  9272.63 GFlop/s
template<int LB, int STEP>
__global__ void NEW_88T1_kernel1(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	      float*  __restrict__ C,
	int SA, int SB, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;
	const int log_tile = 1;
	const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
	const int by = ((bidx / (gridDim.x << log_tile)) << log_tile) + (bidx & ((1 << log_tile) - 1));
	const int bx = (bidx % (gridDim.x << log_tile)) >> log_tile;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB >> 1][(2 << LB) + 1];

	//compute 8*8 elements
	float4  c0 = F32_4_0,  c1 = F32_4_0,  c2 = F32_4_0,  c3 = F32_4_0;
	float4  c4 = F32_4_0,  c5 = F32_4_0,  c6 = F32_4_0,  c7 = F32_4_0;
	float4  c8 = F32_4_0,  c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	//prepare for A[K, N], B[K, M]
	const int bY = (by << LB << 3);
	const int bX = (bx << LB << 3);
	const int tY = bY + (tx << 2); A += tY;
	const int tX = bX + (tx << 2); B += tX;

	//prepare for C[N, M]
	const int idx = (ty << 1 << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = ((vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1)) << 1;
	const int uy = ((vy >> 2) * 8 + ((vx & 15) >> 1)) << 1;

	const int cY = bY + (uy << 2);
	const int cX = bX + (ux << 2);
	const int C0 = cY * SB + cX;

	//compute area-------------------------------------------------------
	//load 4 elements from A, B(transposed)
	const int k = ty;
	const int aoffset = k * SA, boffset = k * SB;
	float4 av = *(float4*)(A + aoffset);
	float4 bv = *(float4*)(B + boffset);
	As[0][ty][tx] = av; Bs[0][ty][tx] = bv;
	__syncthreads();

	for (int ok = STEP; ok < K; ok += STEP) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik][ux + 1];
			float4 a0 = As[buf][ik][uy], a1 = As[buf][ik][uy + 1];

			simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
			simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
			simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
			simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
			simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from A, B(transposed)
		const int k = ok + ty;
		const int aoffset = k * SA, boffset = k * SB;
		float4 av = *(float4*)(A + aoffset);
		float4 bv = *(float4*)(B + boffset);
		As[buf][ty][tx] = av; Bs[buf][ty][tx] = bv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik][ux + 1];
		float4 a0 = As[buf][ik][uy], a1 = As[buf][ik][uy + 1];

		simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
		simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
		simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
		simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
		simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}

	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;
	const int C4 = C3 + SB, C5 = C4 + SB, C6 = C5 + SB, C7 = C6 + SB;

	*(float4*)(C + C0) = c0;  *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2;  *(float4*)(C + C1 + 4) = c3;
	*(float4*)(C + C2) = c4;  *(float4*)(C + C2 + 4) = c5;
	*(float4*)(C + C3) = c6;  *(float4*)(C + C3 + 4) = c7;
	*(float4*)(C + C4) = c8;  *(float4*)(C + C4 + 4) = c9;
	*(float4*)(C + C5) = c10; *(float4*)(C + C5 + 4) = c11;
	*(float4*)(C + C6) = c12; *(float4*)(C + C6 + 4) = c13;
	*(float4*)(C + C7) = c14; *(float4*)(C + C7 + 4) = c15;
}

#endif