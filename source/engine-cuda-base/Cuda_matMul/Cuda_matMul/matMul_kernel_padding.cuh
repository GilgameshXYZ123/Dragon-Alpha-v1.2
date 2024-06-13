#pragma once

#ifndef MATMUL_KERNEL_PADDING_H
#define MATMUL_KERNEL_PADDING_H

//We have:
//(1) N % 4 == 0, N >= 4
//(2) M % 4 == 0, M >= 4
//(3) K % 4 == 0, K >= 4
//(4) SB: stride of Maxtrix B: original == M, changed with GemmBranch
#ifndef MATMUL_KERNEL_PADDING_CALL
#define MATMUL_KERNEL_PADDING_CALL

//LB = log2(BLOCK_SIZE)
#define	k88_pm8_mgk(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_8_padding_M8_mgk<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3((M +(1<<3<<LB)-1)>>3>>LB, (N+(1<<3<<LB)-1)>>3>>LB),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, M, SB, K)

#define	k88_p_mgk(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_8_padding_mgk<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3((M +(1<<3<<LB)-1)>>3>>LB, (N+(1<<3<<LB)-1)>>3>>LB),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, M, SB, K)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0, M % 8 == 0
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_8_8_PADDING_M8_MGK
#define MATMUL_KERNEL_8_8_PADDING_M8_MGK

//for [2000 * 2000 * 2048]: 
//LB = 4: Size = 7.62939, Time = 1.596 msec, Performace = 10265.7 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_8_8_padding_M8_mgk(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int N, int M, int SB, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(4 << LB) + 4];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SB + X;//C[Y, X]
	const int tY = (Y + ((tx >= STEP) << 2) + (tx & 3)); A += tY * K;
	const int tX = (X + ((ty >= STEP) << 2)); B += tX;

	//compute area-------------------------------------------------------
	int B0 = (ty & STEP_m1) * SB;
	int A0 = (tx & STEP_m1) >> 2 << 2;
	const int As_x = A0 + ((tx >= STEP) << LB >> 1);
	const int As_y = (ty << 2) + (tx & 3);

	bool lA = (tY < N); A = IF_int(lA, A, FZERO4); A0 = IF_int(lA, A0, 0);
	bool lB = (tX < M); B = IF_int(lB, B, FZERO4); B0 = IF_int(lB, B0, 0);

	float4 av = *(float4*)(A + A0);//transpose A
	As[0][As_x    ][As_y] = av.x;
	As[0][As_x + 1][As_y] = av.y;
	As[0][As_x + 2][As_y] = av.z;
	As[0][As_x + 3][As_y] = av.w;

	Bs[0][ty][tx] = *(float4*)(B + B0);
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&As[buf][ik][ty << 2]);
			float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP * lA; B += (SB << LB >> 1) * lB;//K += STEP

		float4 av = *(float4*)(A + A0);//transpose A
		As[buf][As_x][As_y] = av.x;
		As[buf][As_x + 1][As_y] = av.y;
		As[buf][As_x + 2][As_y] = av.z;
		As[buf][As_x + 3][As_y] = av.w;

		Bs[buf][ty][tx] = *(float4*)(B + B0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&As[buf][ik][ty << 2]);
		float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}

	bool lx7 = (X + 7 < M);
	bool ly0 = (Y     < N), ly1 = (Y + 1 < N);
	bool ly2 = (Y + 2 < N), ly3 = (Y + 3 < N), ly4 = (Y + 4 < N);
	bool ly5 = (Y + 5 < N), ly6 = (Y + 6 < N), ly7 = (Y + 7 < N);

	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;
	const int C4 = C3 + SB, C5 = C4 + SB, C6 = C5 + SB, C7 = C6 + SB;

	IFW8(C, C0, ly0 && lx7, c0, c1);
	IFW8(C, C1, ly1 && lx7, c2, c3);
	IFW8(C, C2, ly2 && lx7, c4, c5);
	IFW8(C, C3, ly3 && lx7, c6, c7);
	IFW8(C, C4, ly4 && lx7, c8, c9);
	IFW8(C, C5, ly5 && lx7, c10, c11);
	IFW8(C, C6, ly6 && lx7, c12, c13);
	IFW8(C, C7, ly7 && lx7, c14, c15);
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_8_8_PADDING_MGK
#define MATMUL_KERNEL_8_8_PADDING_MGK

//for [2040 * 2040 * 2048]: 
//LB = 4: Size = 7.93762, Time = 1.669 msec, Performace = 10213.2 GFlop/s
//LB = 3: Size = 7.93762, Time = 2.03  msec, Performace = 8397 GFlop/s
//for [2000 * 2000 * 2048]: 
//LB = 4: Size = 7.62939, Time = 1.669 msec, Performace = 9816.66 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_8_8_padding_mgk(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int N, int M, int SB, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(4 << LB) + 4];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SB + X;//C[Y, X]
	const int tY = (Y + ((tx >= STEP) << 2) + (tx & 3)); A += tY * K;
	const int tX = (X + ((ty >= STEP) << 2)); B += tX;

	//compute area-------------------------------------------------------
	int B0 = (ty & STEP_m1) * SB;
	int A0 = (tx & STEP_m1) >> 2 << 2;
	const int As_x = A0 + ((tx >= STEP) << LB >> 1);
	const int As_y = (ty << 2) + (tx & 3);

	bool lA = (tY < N); A = IF_int(lA, A, FZERO4); A0 = IF_int(lA, A0, 0);
	bool lB = (tX < M); B = IF_int(lB, B, FZERO4); B0 = IF_int(lB, B0, 0);

	float4 av = *(float4*)(A + A0);//transpose A
	As[0][As_x    ][As_y] = av.x;
	As[0][As_x + 1][As_y] = av.y;
	As[0][As_x + 2][As_y] = av.z;
	As[0][As_x + 3][As_y] = av.w;

	Bs[0][ty][tx] = *(float4*)(B + B0);
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&As[buf][ik       ][ty << 2]);
			float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP * lA; B += (SB << LB >> 1) * lB;//K += STEP

		float4 av = *(float4*)(A + A0);//transpose A
		As[buf][As_x][As_y] = av.x;
		As[buf][As_x + 1][As_y] = av.y;
		As[buf][As_x + 2][As_y] = av.z;
		As[buf][As_x + 3][As_y] = av.w;

		Bs[buf][ty][tx] = *(float4*)(B + B0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&As[buf][ik       ][ty << 2]);
		float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}

	bool lx3 = (X + 3 < M), lx7 = (X + 7 < M);
	bool ly0 = (Y     < N), ly1 = (Y + 1 < N);
	bool ly2 = (Y + 2 < N), ly3 = (Y + 3 < N), ly4 = (Y + 4 < N);
	bool ly5 = (Y + 5 < N), ly6 = (Y + 6 < N), ly7 = (Y + 7 < N);
	
	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;
	const int C4 = C3 + SB, C5 = C4 + SB, C6 = C5 + SB, C7 = C6 + SB;

	IFW4(C, C0, ly0 && lx3, c0);  IFW4(C, C0 + 4, ly0 && lx7, c1);
	IFW4(C, C1, ly1 && lx3, c2);  IFW4(C, C1 + 4, ly1 && lx7, c3);
	IFW4(C, C2, ly2 && lx3, c4);  IFW4(C, C2 + 4, ly2 && lx7, c5);
	IFW4(C, C3, ly3 && lx3, c6);  IFW4(C, C3 + 4, ly3 && lx7, c7);
	IFW4(C, C4, ly4 && lx3, c8);  IFW4(C, C4 + 4, ly4 && lx7, c9);
	IFW4(C, C5, ly5 && lx3, c10); IFW4(C, C5 + 4, ly5 && lx7, c11);
	IFW4(C, C6, ly6 && lx3, c12); IFW4(C, C6 + 4, ly6 && lx7, c13);
	IFW4(C, C7, ly7 && lx3, c14); IFW4(C, C7 + 4, ly7 && lx7, c15);
}

#endif

#endif