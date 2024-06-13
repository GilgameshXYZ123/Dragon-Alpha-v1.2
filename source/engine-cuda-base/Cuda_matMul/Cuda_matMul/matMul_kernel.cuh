#pragma once

#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H

//We have:
//(1) N % 4 == 0, N >= 4
//(2) M % 4 == 0, M >= 4
//(3) K % 4 == 0, K >= 4
//(4) SB: stride of Maxtrix B: original == M, changed with GemmBranch
#ifndef MATMUL_KERNEL_CALL
#define MATMUL_KERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]======================================================
#define	k88_mgk(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_8_mgk<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define	k88(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_8<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k84(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k48(LB, stream, A, B, C, N, M, K, SB) \
	kernel_4_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k82(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k28(LB, stream, A, B, C, N, M, K, SB) \
	kernel_2_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)
#define k44(LB, stream, A, B, C, N, M, K, SB) \
	kernel_4_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k42(LB, stream, A, B, C, N, M, K, SB) \
	kernel_4_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k24(LB, stream, A, B, C, N, M, K, SB) \
	kernel_2_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//======[Small]=======================================================
#define k22(LB, stream, A, B, C, N, M, K, SB) \
	kernel_2_2<LB, (1<<LB)>\
		<<< dim3(M>>1>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k81(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k41(LB, stream, A, B, C, N, M, K, SB) \
	kernel_4_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k21(LB, stream, A, B, C, N, M, K, SB) \
	kernel_2_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k18(LB, stream, A, B, C, N, M, K, SB) \
	kernel_1_8<LB, (1<<LB)>\
		<<< dim3(N>>LB, M>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k14(LB, stream, A, B, C, N, M, K, SB) \
	kernel_1_4<LB, (1<<LB)>\
		<<< dim3(N>>LB, M>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k12(LB, stream, A, B, C, N, M, K, SB) \
	kernel_1_2<LB, (1<<LB)>\
		<<< dim3(N>>LB, M>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

//--------------------------------------------------------------------
#define k11(LB, stream, A, B, C, N, M, K, SB) \
	kernel_1_1<LB, (1<<LB)>\
		<<< dim3(N>>LB, M>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define knaive(GRID_SIZE, stream, A, B, C, N, M, K, SB) \
	kernel_naive\
		<<< dim3(GRID_SIZE, GRID_SIZE), dim3(M/GRID_SIZE, N/GRID_SIZE), 0, stream>>>\
			(A, B, C, SB, K)

#endif


//======[Common]======================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_8_8_MGK
#define MATMUL_KERNEL_8_8_MGK

//for: (N, M, K) = (2048, 2048, 2048)
//LB = 4: Size = 8, Time = 1.691 msec, Performace = 10159.6  GFlop/s
//LB = 3: Size = 8, Time = 1.849 msec, Performace =  9291.44 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_8_8_mgk(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4  c0 = F32_4_0,  c1 = F32_4_0,  c2 = F32_4_0,  c3 = F32_4_0;
	float4  c4 = F32_4_0,  c5 = F32_4_0,  c6 = F32_4_0,  c7 = F32_4_0;
	float4  c8 = F32_4_0,  c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	//prepare for C[N, M]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = (vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1);
	const int uy = (vy >> 2) * 8 + ((vx & 15) >> 1);

	const int bY = (by << LB << 3);
	const int bX = (bx << LB << 3);

	const int cY = bY + (uy << 3);
	const int cX = bX + (ux << 3);
	const int C0 = cY * SB + cX;//C[Y, X]

	//compute area-------------------------------------------------------
	const int A0 = (bY + (ty << 3) + ((tx >= STEP) << 2)) * K + (tx & STEP_m1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;
	const int B0 = (bX + (tx << 3) + ((ty >= STEP) << 2)) + (ty & STEP_m1) * SB;

	float4 a0 = float4{ A[A0] ,A[A1], A[A2], A[A3] };//transpose A
	float4 b0 = *(float4*)(B + B0);
	As[0][tx][ty] = a0;
	Bs[0][ty][tx] = b0;
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik + STEP][ux];
			float4 a0 = As[buf][ik][uy], a1 = As[buf][ik + STEP][uy];

			simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
			simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
			simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
			simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
			simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		float4 a0 = float4{ A[A0] ,A[A1], A[A2], A[A3] };//transpose A
		float4 b0 = *(float4*)(B + B0);
		As[buf][tx][ty] = a0;
		Bs[buf][ty][tx] = b0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik + STEP][ux];
		float4 a0 = As[buf][ik][uy], a1 = As[buf][ik + STEP][uy];

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_8_8
#define MATMUL_KERNEL_8_8

//for [2048 * 2048 * 2048]: 
//LB = 4: Size = 8.01562, Time = 1.721 msec, Performace = 10002 GFlop/s
//LB = 3: Size = 8.01562, Time = 1.792 msec, Performace = 9605.7 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_8_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4  c0 = F32_4_0,  c1 = F32_4_0,  c2 = F32_4_0,  c3 = F32_4_0;
	float4  c4 = F32_4_0,  c5 = F32_4_0,  c6 = F32_4_0,  c7 = F32_4_0;
	float4  c8 = F32_4_0,  c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	//prepare for C[N, M]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = (vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1);
	const int uy = (vy >> 2) * 8 + ((vx & 15) >> 1);

	const int bY = (by << LB << 3);
	const int bX = (bx << LB << 3);

	const int cY = bY + (uy << 3);
	const int cX = bX + (ux << 3);
	const int C0 = cY * SB + cX;//C[Y, X]

	//compute area-------------------------------------------------------
	const int A0 = (bY + (ty << 3) + ((tx >= STEP) << 2)) * K + (tx & STEP_m1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;
	const int B0 = (bX + (tx << 3) + ((ty >= STEP) << 2)) + (ty & STEP_m1) * SB;

	float4 a0 = float4{ A[A0] ,A[A1], A[A2], A[A3] };//transpose A
	float4 b0 = *(float4*)(B + B0);
	As[0][tx][ty] = a0;
	Bs[0][ty][tx] = b0;
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik + STEP][ux];
			float4 a0 = As[buf][ik][uy], a1 = As[buf][ik + STEP][uy];

			simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
			simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
			simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
			simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
			simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		float4 a0 = float4{ A[A0] ,A[A1], A[A2], A[A3] };//transpose A
		float4 b0 = *(float4*)(B + B0);
		As[buf][tx][ty] = a0;
		Bs[buf][ty][tx] = b0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik + STEP][ux];
		float4 a0 = As[buf][ik][uy], a1 = As[buf][ik + STEP][uy];

		simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
		simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
		simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
		simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
		simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}
	buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

	const int RK = (K & STEP_m1); if (RK) {//process remainder
		bool lA = (tx & STEP_m1) < RK;
		bool lB = (ty & STEP_m1) < RK;
		float4 a0 = (lA ? float4{ A[A0] ,A[A1], A[A2], A[A3] } : F32_4_0);
		float4 b0 = (lB ? *(float4*)(B + B0) : F32_4_0);
		As[buf][tx][ty] = a0;
		Bs[buf][ty][tx] = b0;
		__syncthreads();

		for (int ik = 0; ik < RK; ik++) {
			float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik + STEP][ux];
			float4 a0 = As[buf][ik][uy], a1 = As[buf][ik + STEP][uy];

			simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
			simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
			simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
			simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
			simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_8_4
#define MATMUL_KERNEL_8_4

//for [2048 * 2048 * 2048]: 
//LB = 4: Size = 8, Time = 1.988 msec, Performace = 8641.78 GFlop/s
//LB = 3: Size = 8, Time = 2.668 msec, Performace = 6439.23 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_4(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//compute 4*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;

	const int Y = ((blockIdx.y << LB) + ty) << 3; A += Y * K;//A[Y, 0]
	const int X = ((blockIdx.x << LB) + tx) << 2; B += X;//B[0, X]
	const int C0 = Y * SB + X;//C[Y, X]
	
	//compute area-------------------------------------------------------
	const int Ax = tx - ((tx >= STEP) << LB >> 1);
	const int A0 = ((tx >= STEP) << 2) * K + Ax;//[Ay, Ax]
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int By = ty >> 1, Bx = ((ty & 1) << 1);
	const int B0 = By * SB + Bx;//[By, Bx]
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	
	As[buf][tx][ty] = float4{ A[A0] ,A[A1], A[A2], A[A3] };//transpose A
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM4(c0, a0.x, b0);
			simdMM4(c1, a0.y, b0);
			simdMM4(c2, a0.z, b0);
			simdMM4(c3, a0.w, b0);
			simdMM4(c4, a1.x, b0);
			simdMM4(c5, a1.y, b0);
			simdMM4(c6, a1.z, b0);
			simdMM4(c7, a1.w, b0);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//B += SB*STEP

		As[buf][tx][ty] = float4{ A[A0] ,A[A1], A[A2], A[A3] };//transpose A
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM4(c0, a0.x, b0);
		simdMM4(c1, a0.y, b0);
		simdMM4(c2, a0.z, b0);
		simdMM4(c3, a0.w, b0);
		simdMM4(c4, a1.x, b0);
		simdMM4(c5, a1.y, b0);
		simdMM4(c6, a1.z, b0);
		simdMM4(c7, a1.w, b0);
	}
	A += STEP; B += (SB << LB >> 1);//B += SB*STEP

	//when K % STEP != 0-----------------------------------
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0, a1;
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);
		a1.x = get(A, 4, ik, K);
		a1.y = get(A, 5, ik, K);
		a1.z = get(A, 6, ik, K);
		a1.w = get(A, 7, ik, K);

		float4 b0 = *(float4*)(B + ik * SB);

		simdMM4(c0, a0.x, b0);
		simdMM4(c1, a0.y, b0);
		simdMM4(c2, a0.z, b0);
		simdMM4(c3, a0.w, b0);
		simdMM4(c4, a1.x, b0);
		simdMM4(c5, a1.y, b0);
		simdMM4(c6, a1.z, b0);
		simdMM4(c7, a1.w, b0);
	}
	//when K % STEP != 0-----------------------------------

	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;
	const int C4 = C3 + SB, C5 = C4 + SB, C6 = C5 + SB, C7 = C6 + SB;

	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c1;
	*(float4*)(C + C2) = c2;
	*(float4*)(C + C3) = c3;
	*(float4*)(C + C4) = c4;
	*(float4*)(C + C5) = c5;
	*(float4*)(C + C6) = c6;
	*(float4*)(C + C7) = c7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_4_8
#define MATMUL_KERNEL_4_8

//for [2048 * 2048 * 2048]:  
//LB = 4: Size = 8, Time = 2.115 msec, Performace = 8122.87 GFlop/s
//LB = 3: Size = 8, Time = 2.663 msec, Performace = 6451.32 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88

	//compute 4*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;

	const int Y = ((blockIdx.y << LB) + ty) << 2; A = A + Y * K;//A[Y, 0]
	const int X = ((blockIdx.x << LB) + tx) << 3; B = B + X;//B[0, X]
	const int C0 = Y * SB + X;//C[Y, X]

	//compute area-------------------------------------------------------
	const int Ax = (tx >> 1), Ay = ((tx & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = Ay * K + Ax, A1 = A0 + K;

	const int Bx = ((ty >= STEP) << 2);
	const int By = ty - ((ty >= STEP) << LB >> 1);
	const int B0 = By * SB + Bx;

	As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
	Bs[buf][ty][tx] = *(float4*)(B + B0);
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
		Bs[buf][ty][tx] = *(float4*)(B + B0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}
	A += STEP; B += (SB << LB >> 1);//K += STEP

	//when K % STEP != 0-----------------------------------
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0;
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);

		float4 b0 = *(float4*)(&get(B, ik, 0, SB));
		float4 b1 = *(float4*)(&get(B, ik, 4, SB));

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}
	//when K % STEP != 0-----------------------------------

	const int C1 = C0 + SB;
	const int C2 = C1 + SB;
	const int C3 = C2 + SB;

	*(float4*)(C + C0) = c0;  *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2;  *(float4*)(C + C1 + 4) = c3;
	*(float4*)(C + C2) = c4;  *(float4*)(C + C2 + 4) = c5;
	*(float4*)(C + C3) = c6;  *(float4*)(C + C3 + 4) = c7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_4_4
#define MATMUL_KERNEL_4_4

//for [2048 * 2048 * 2048]: 
//LB = 4: Size = 8, Time = 2.506 msec, Performace = 6855.49 GFlop/s
//LB = 3: Size = 8, Time = 3.57  msec, Performace = 4812.29 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_4(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//compute 4*4 elements
	float4 c0 = F32_4_0, c1 = F32_4_0;
	float4 c2 = F32_4_0, c3 = F32_4_0;

	const int Y = (blockIdx.y << 2 << LB); A += Y * K;//A[Y, 0]
	const int X = (blockIdx.x << 2 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	//compute area---------------------------------------------
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = Ay * K + Ax;//[Ay, Ax]
	const int A1 = A0 + K;

	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int B0 = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A;
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
			simdMM4(c2, a.z, b);
			simdMM4(c3, a.w, b);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A;
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}
	A += STEP; B += (SB << LB >> 1);//K += STEP

	//when K % STEP != 0--------------------------------------
	ty <<= 2; tx <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a;
		a.x = get(A, ty, ik, K);
		a.y = get(A, ty + 1, ik, K);
		a.z = get(A, ty + 2, ik, K);
		a.w = get(A, ty + 3, ik, K);

		float4 b = *(float4*)(&get(B, ik, tx, SB));

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB;
	const int C2 = C1 + SB;
	const int C3 = C2 + SB;

	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c1;
	*(float4*)(C + C2) = c2;
	*(float4*)(C + C3) = c3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_8_2
#define MATMUL_KERNEL_8_2

//for [2048 * 2048 * 2048]: 
//LB = 4: Size = 8, Time = 2.702 msec, Performace = 6358.2 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_2(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//compute 8*2 elements
	float2 c0 = F32_2_0, c1 = F32_2_0, c2 = F32_2_0, c3 = F32_2_0;
	float2 c4 = F32_2_0, c5 = F32_2_0, c6 = F32_2_0, c7 = F32_2_0;

	int Y = (blockIdx.y << 3 << LB); A += Y * K;
	int X = (blockIdx.x << 1 << LB); B += X;
	C = &get(C, Y, X, SB);//C[Y, X]
	
	//compute area---------------------------------------------------
	const int Ay = (ty << 3) + ((tx >= STEP) << 2);
	const int Ax = (tx - ((tx >= STEP) << LB >> 1));
	const int A0 = Ay * K + Ax;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int B0 = By * SB + Bx;//[By, Bx]

	Bs[buf][Bs_y][Bs_x] = B[B0];
	As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transpose A
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float2 b0 = *(float2*)(&Bs[buf][ik][tx << 1]);

			simdMM2(c0, a0.x, b0); simdMM2(c1, a0.y, b0);
			simdMM2(c2, a0.z, b0); simdMM2(c3, a0.w, b0);
			simdMM2(c4, a1.x, b0); simdMM2(c5, a1.y, b0);
			simdMM2(c6, a1.z, b0); simdMM2(c7, a1.w, b0);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		Bs[buf][Bs_y][Bs_x] = B[B0];
		As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transpose A
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float2 b0 = *(float2*)(&Bs[buf][ik][tx << 1]);

		simdMM2(c0, a0.x, b0); simdMM2(c1, a0.y, b0);
		simdMM2(c2, a0.z, b0); simdMM2(c3, a0.w, b0);
		simdMM2(c4, a1.x, b0); simdMM2(c5, a1.y, b0);
		simdMM2(c6, a1.z, b0); simdMM2(c7, a1.w, b0);
	}
	A += STEP; B += (SB << LB >> 1);//K += STEP

	//when K % STEP != 0-----------------------------------
	ty <<= 3; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0, a1;
		a0.x = get(A, ty    , ik, K);
		a0.y = get(A, ty + 1, ik, K);
		a0.z = get(A, ty + 2, ik, K);
		a0.w = get(A, ty + 3, ik, K);
		a1.x = get(A, ty + 4, ik, K);
		a1.y = get(A, ty + 5, ik, K);
		a1.z = get(A, ty + 6, ik, K);
		a1.w = get(A, ty + 7, ik, K);

		float2 b0 = *(float2*)(&get(B, ik, tx, SB));

		simdMM2( c0, a0.x, b0);
		simdMM2( c1, a0.y, b0);
		simdMM2( c2, a0.z, b0);
		simdMM2( c3, a0.w, b0);
		simdMM2( c4, a1.x, b0);
		simdMM2(c5, a1.y, b0);
		simdMM2(c6, a1.z, b0);
		simdMM2(c7, a1.w, b0);
	}
	//when K % STEP != 0-----------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;
	const int C4 = C3 + SB, C5 = C4 + SB, C6 = C5 + SB, C7 = C6 + SB;

	*(float2*)(C + C0) = c0;
	*(float2*)(C + C1) = c1;
	*(float2*)(C + C2) = c2;
	*(float2*)(C + C3) = c3;
	*(float2*)(C + C4) = c4;
	*(float2*)(C + C5) = c5;
	*(float2*)(C + C6) = c6;
	*(float2*)(C + C7) = c7;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_2_8
#define MATMUL_KERNEL_2_8

//for [2048 * 2048 * 2048]: 
//LB = 4: Size = 8, Time = 3.118 msec, Performace = 5509.9 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88

	//compute 2*8 = 16 elements
	float4 c0 = F32_4_0, c1 = F32_4_0;
	float4 c2 = F32_4_0, c3 = F32_4_0;

	int Y = (blockIdx.y << 1 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << 3 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	//compute area-------------------------------------------------------
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = Ay * K + Ax;//[Ay, Ax]

	const int Bx = (tx << 3) + ((ty >= STEP) << 2);
	const int By = ty - ((ty >= STEP) << LB >> 1);
	const int B0 = By * SB + Bx;//[By, Bx]

	Bs[buf][ty][tx] = *(float4*)(B + B0);
	As[buf][As_x][As_y] = A[A0];//transpose A
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);
			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		Bs[buf][ty][tx] = *(float4*)(B + B0);
		As[buf][As_x][As_y] = A[A0];//transpose A
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);
		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}
	A += STEP; B += (SB << LB >> 1);//K += STEP

	//when K % STEP != 0----------------------------------------
	ty <<= 1; tx <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a0;
		a0.x = get(A, ty    , ik, K);
		a0.y = get(A, ty + 1, ik, K);

		float4 b0 = *(float4*)(&get(B, ik, tx    , SB));
		float4 b1 = *(float4*)(&get(B, ik, tx + 4, SB));

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}
	//when K % STEP != 0----------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB;
	*(float4*)(C + C0) = c0; *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2; *(float4*)(C + C1 + 4) = c3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_4_2
#define MATMUL_KERNEL_4_2

//for [2048 * 2048 * 2048]: 
//LB = 4: Size = 8, Time = 3.378 msec, Performace = 5085.81 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_2(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];
	
	//compute 4*2 = 8 elements
	float2 c0 = F32_2_0, c1 = F32_2_0;
	float2 c2 = F32_2_0, c3 = F32_2_0;

	int Y = (blockIdx.y << 2 << LB); A = A + Y * K;//A[Y, 0]
	int X = (blockIdx.x << 1 << LB); B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]
	
	//compute area------------------------------------------
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = Ay * K + Ax, A1 = A0 + K;

	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int B0 = By * SB + Bx;

	As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
	Bs[buf][Bs_y][Bs_x] = B[B0];
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);
			float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);

			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
			simdMM2(c2, a.z, b);
			simdMM2(c3, a.w, b);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
		Bs[buf][Bs_y][Bs_x] = B[B0];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);
		float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}
	A += STEP; B += (SB << LB >> 1);//K += STEP

	//when K % STEP != 0--------------------------------------
	ty <<= 2; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a;
		a.x = get(A, ty    , ik, K);
		a.y = get(A, ty + 1, ik, K);
		a.z = get(A, ty + 2, ik, K);
		a.w = get(A, ty + 3, ik, K);

		float2 b = *(float2*)(&get(B, ik, tx, SB));

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;

	*(float2*)(C + C0) = c0;
	*(float2*)(C + C1) = c1;
	*(float2*)(C + C2) = c2;
	*(float2*)(C + C3) = c3;
}

#endif 


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_2_4
#define MATMUL_KERNEL_2_4

//for [2048 * 2048 * 2048]: 
//LB = 4: Size = 8, Time = 3.844 msec, Performace = 4469.27 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_4(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	float4 c0 = F32_4_0, c1 = F32_4_0;//compute 2*4 = 8 elements

	int Y = (blockIdx.y << 1 << LB); A = A + Y * K;//A[Y, 0]
	int X = (blockIdx.x << 2 << LB); B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	//compute area--------------------------------------------
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = Ay * K + Ax;//[Ay, Ax]

	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1)); 
	const int B0 = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = A[A0];//transpose A
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = *(float2*)(&As[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		As[buf][As_x][As_y] = A[A0]; //transpose A
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = *(float2*)(&As[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
	}
	A += STEP; B += (SB << LB >> 1);//K += STEP

	//when K % STEP != 0--------------------------------------
	ty <<= 1; tx <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a;
		a.x = get(A, ty    , ik, K);
		a.y = get(A, ty + 1, ik, K);

		float4 b = *(float4*)(&get(B, ik, tx, SB));
		
		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB;
	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c1;
}

#endif


//======[Small]=====================================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_2_2
#define MATMUL_KERNEL_2_2

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 4.974 msec, Performace = 3453.93 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_2(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	//compute 2*2 elements
	float2 c0 = F32_2_0, c1 = F32_2_0;
	
	int Y = (blockIdx.y << 1 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << 1 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	//compute area------------------------------------------
	const int A0 = (ty << 1) * K + tx, A1 = A0 + K;
	const int B0 = ty * SB + (tx << 1);

	As[buf][tx][ty] = float2{ A[A0], A[A1] };//transpose A
	Bs[buf][ty][tx] = *(float2*)(B + B0);
	__syncthreads();
	
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Bs[buf][ik][tx];
			float2 a = As[buf][ik][ty];
			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
		}
		buf ^= 1; A += STEP; B += (SB << LB);//K += STEP

		As[buf][tx][ty] = float2{ A[A0], A[A1] };//transpose A
		Bs[buf][ty][tx] = *(float2*)(B + B0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Bs[buf][ik][tx];
		float2 a = As[buf][ik][ty];
		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}
	A += STEP; B += (SB << LB);//K += STEP

	//when K % STEP != 0--------------------------------------
	ty <<= 1; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a;//transposed A
		a.x = get(A, ty    , ik, K);
		a.y = get(A, ty + 1, ik, K);

		float2 b = *(float2*)(&get(B, ik, tx, SB));

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB;
	*(float2*)(C + C0) = c0;
	*(float2*)(C + C1) = c1;
}

#endif


//------------------------------------------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_8_1
#define MATMUL_KENERL_8_1

//[M = 16]: LB = 4: Size = 0.25, Time = 1.254 msec, Performace = 428.127 GFlop/s
//[M = 16]: LB = 3: Size = 0.25, Time = 4.602 msec, Performace = 116.66 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_1(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 3 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);

	const int A0 = (ty << 3) * K + tx;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;
	const int A4 = A3 + K, A5 = A4 + K, A6 = A5 + K, A7 = A6 + K;
	const int B0 = ty * SB + tx;//[ty, tx]

	//transposed A
	As[buf][tx][(ty << 1)    ] = float4{ A[A0], A[A1], A[A2], A[A3] };
	As[buf][tx][(ty << 1) + 1] = float4{ A[A4], A[A5], A[A6], A[A7] };

	Bs[buf][ty][tx] = B[B0];
	A += STEP; B += (SB << LB);//K +=  STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][(ty << 1)];
			float4 a1 = As[buf][ik][(ty << 1) + 1];
			float b = Bs[buf][ik][tx];

			simdMM4(c0, b, a0);
			simdMM4(c1, b, a1);
		}
		buf ^= 1;

		//transposed A
		As[buf][tx][(ty << 1)    ] = float4{ A[A0], A[A1], A[A2], A[A3] };
		As[buf][tx][(ty << 1) + 1] = float4{ A[A4], A[A5], A[A6], A[A7] };

		Bs[buf][ty][tx] = B[B0];
		A += STEP; B += (SB << LB);//K += STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][(ty << 1)];
		float4 a1 = As[buf][ik][(ty << 1) + 1];
		float b = Bs[buf][ik][tx];

		simdMM4(c0, b, a0);
		simdMM4(c1, b, a1);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0, a1;
		a0.x = get(A, ty, ik, K);
		a0.y = get(A, ty + 1, ik, K);
		a0.z = get(A, ty + 2, ik, K);
		a0.w = get(A, ty + 3, ik, K);
		a1.x = get(A, ty + 4, ik, K);
		a1.y = get(A, ty + 5, ik, K);
		a1.z = get(A, ty + 6, ik, K);
		a1.w = get(A, ty + 7, ik, K);

		float b = get(B, ik, tx, SB);

		simdMM4(c0, b, a0);
		simdMM4(c1, b, a1);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;
	const int C4 = C3 + SB, C5 = C4 + SB, C6 = C5 + SB, C7 = C6 + SB;

	C[C0] = c0.x;
	C[C1] = c0.y;
	C[C2] = c0.z;
	C[C3] = c0.w;
	C[C4] = c1.x;
	C[C5] = c1.y;
	C[C6] = c1.z;
	C[C7] = c1.w;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_4_1
#define MATMUL_KENERL_4_1

//[M = 16]: LB = 4: Size = 0.25, Time = 1.196 msec, Performace = 448.889 GFlop/s
//[M =  8]: LB = 3: Size = 0.25, Time = 4.594 msec, Performace = 116.863 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_1(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 2 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);

	const int A0 = (ty << 2) * K + tx;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;
	const int B0 = ty * SB + tx;//[ty, tx]

	As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transposed A
	Bs[buf][ty][tx] = B[B0];
	A += STEP; B += (SB << LB);//K +=  STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 a = As[buf][ik][ty];
			float  b = Bs[buf][ik][tx];
			simdMM4(c0, b, a);
		}
		buf ^= 1;

		As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transposed A
		Bs[buf][ty][tx] = B[B0];
		A += STEP; B += (SB << LB);//K += STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 a = As[buf][ik][ty];
		float  b = Bs[buf][ik][tx];
		simdMM4(c0, b, a);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a;//transposed A
		a.x = get(A, ty, ik, K);
		a.y = get(A, ty + 1, ik, K);
		a.z = get(A, ty + 2, ik, K);
		a.w = get(A, ty + 3, ik, K);

		float b = get(B, ik, tx, SB);

		simdMM4(c0, b, a);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;

	C[C0] = c0.x;
	C[C1] = c0.y;
	C[C2] = c0.z;
	C[C3] = c0.w;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_2_1
#define MATMUL_KENERL_2_1

//for [1024 * 1024 * 1024]: 
//LB = 4: Size = 1, Time = 1.133 msec, Performace = 1895.4 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_1(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	float2 c0 = F32_2_0;//compute 2*1 elements

	int Y = (blockIdx.y << 1 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);

	//compute area------------------------------------------
	const int A0 = (ty << 1) * K + tx, A1 = A0 + K;
	const int B0 = ty * SB + tx;//[ty, tx]

	As[buf][tx][ty] = float2{ A[A0], A[A1] };//transposed A
	Bs[buf][ty][tx] = B[B0];
	__syncthreads();
	
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = As[buf][ik][ty];
			float  b = Bs[buf][ik][tx];
			simdMM2(c0, b, a);
		}
		buf ^= 1; A += STEP; B += (SB << LB);//K += STEP

		As[buf][tx][ty] = float2{ A[A0], A[A1] };//transposed A
		Bs[buf][ty][tx] = B[B0];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = As[buf][ik][ty];
		float  b = Bs[buf][ik][tx];
		simdMM2(c0, b, a);
	}
	A += STEP; B += (SB << LB);//K += STEP

	//when K % STEP != 0--------------------------------------
	ty <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a;//transposed A
		a.x = get(A, ty, ik, K);
		a.y = get(A, ty + 1, ik, K);

		float b = get(B, ik, tx, SB);

		simdMM2(c0, b, a);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB;
	C[C0] = c0.x;
	C[C1] = c0.y;
}

#endif


//-------------------------------------------------------------------
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*8), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_1_8
#define MATMUL_KERNEL_1_8

//[N = 16]: LB = 4: Size = 0.25, Time = 1.012 msec, Performace = 530.505 GFlop/s
//[N =  8]: LB = 3: Size = 0.25, Time = 1.44  msec, Performace = 372.827 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(2 << LB) + 1];

	int Y = (blockIdx.x << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.y << 3 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	const int A0 = tx * K + ty;//[ty, tx]
	const int B0 = tx * SB + (ty << 3);//[ty, tx<<3]
	
	Bs[buf][tx][(ty << 1)] = *(float4*)(B + B0);
	Bs[buf][tx][(ty << 1) + 1] = *(float4*)(B + B0 + 4);
	As[buf][ty][tx] = A[A0];
	A += STEP; B += (SB << LB);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
		{
			float4 b0 = Bs[buf][ik][(ty << 1)];
			float4 b1 = Bs[buf][ik][(ty << 1) + 1];
			float a = As[buf][ik][tx];

			simdMM4(c0, a, b0); 
			simdMM4(c1, a, b1);
		}
		buf ^= 1;

		Bs[buf][tx][(ty << 1)] = *(float4*)(B + B0);
		Bs[buf][tx][(ty << 1) + 1] = *(float4*)(B + B0 + 4);
		As[buf][ty][tx] = A[A0];
		A += STEP; B += (SB << LB);//B += SB * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float4 b0 = Bs[buf][ik][(ty << 1)];
		float4 b1 = Bs[buf][ik][(ty << 1) + 1];
		float a = As[buf][ik][tx];

		simdMM4(c0, a, b0);
		simdMM4(c1, a, b1);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++) {
		float a = get(A, tx, ik, K);
		float4 b0 = *(float4*)(&get(B, ik, ty, SB));
		float4 b1 = *(float4*)(&get(B, ik, ty + 4, SB));
		simdMM4(c0, a, b0); simdMM4(c1, a, b1);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = tx * SB + ty;
	*(float4*)(C + C0) = c0; *(float4*)(C + C0 + 4) = c1;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*4), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_1_4
#define MATMUL_KERNEL_1_4

//[N = 16]: LB = 4: Size = 0.25, Time = 1.062 msec, Performace = 505.528 GFlop/s
//[N =  8]: LB = 3: Size = 0.25, Time = 1.988 msec, Performace = 270.056 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_4(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.x << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.y << 2 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	const int A0 = tx * K + ty;//[ty, tx]
	const int B0 = tx * SB + (ty << 2);//[ty, tx<<3]

	Bs[buf][tx][ty] = *(float4*)(B + B0);
	As[buf][ty][tx] = A[A0];
	A += STEP; B += (SB << LB);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = Bs[buf][ik][ty];
			float  a = As[buf][ik][tx];
			simdMM4(c0, a, b);
		}
		buf ^= 1;

		Bs[buf][tx][ty] = *(float4*)(B + B0);
		As[buf][ty][tx] = A[A0];
		A += STEP; B += (SB << LB);//B += SB * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = Bs[buf][ik][ty];
		float  a = As[buf][ik][tx];
		simdMM4(c0, a, b);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++) {
		float  a = get(A, tx, ik, K);
		float4 b = *(float4*)(&get(B, ik, ty, SB));
		simdMM4(c0, a, b);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = tx * SB + ty;
	*(float4*)(C + C0) = c0; 
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_1_2
#define MATMUL_KERNEL_1_2

//for [1024 * 1024 * 1024]: 
//LB = 4: Size = 1, Time = 1.657 msec, Performace = 1296.01 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_2(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	float2 c0 = F32_2_0;//compute 1*2 = 2 elements

	int Y = (blockIdx.x << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.y << 1 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	//compute area------------------------------------------
	const int A0 = tx * K + ty;
	const int B0 = tx * SB + (ty << 1);

	Bs[buf][tx][ty] = *(float2*)(B + B0);
	As[buf][ty][tx] = A[A0];
	__syncthreads();

	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Bs[buf][ik][ty];
			float  a = As[buf][ik][tx];
			simdMM2(c0, a, b);
		}
		buf ^= 1; A += STEP; B += (SB << LB);//K += STEP

		Bs[buf][tx][ty] = *(float2*)(B + B0);
		As[buf][ty][tx] = A[A0];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Bs[buf][ik][ty];
		float  a = As[buf][ik][tx];
		simdMM2(c0, a, b);
	}
	A += STEP; B += (SB << LB);//K += STEP

	//when K % STEP != 0--------------------------------------
	ty <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++) {
		float  a = get(A, tx, ik, K);
		float2 b = *(float2*)(&get(B, ik, ty, SB));
		simdMM2(c0, a, b);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = tx * SB + ty;
	*(float2*)(C + C0) = c0;
}

#endif


//------------------------------------------------------------
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_1_1
#define MATMUL_KENERL_1_1

//for [1024 * 1024 * 1024]: 
//LB = 4: Size = 1, Time = 2.204 msec, Performace = 974.357 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_1(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SB, int K)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float As[2][1 << LB][(1 << LB) + 1];
	__shared__ float Bs[2][1 << LB][(1 << LB) + 1];

	float c0 = 0.0f;//compute 1*1 elements

	int Y = (blockIdx.y << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);

	//compute area------------------------------------------
	const int A0 = ty * K + tx;
	const int B0 = ty * SB + tx;

	As[buf][tx][ty] = A[A0];//transposed A
	Bs[buf][ty][tx] = B[B0];
	__syncthreads();

	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float a = As[buf][ik][ty];
			float b = Bs[buf][ik][tx];
			c0 += b * a;
		}
		buf ^= 1; A += STEP; B += (SB << LB);//K += STEP

		As[buf][tx][ty] = A[A0];//transposed A
		Bs[buf][ty][tx] = B[B0];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float a = As[buf][ik][ty];
		float b = Bs[buf][ik][tx];
		c0 += b * a;
	}
	A += STEP; B += (SB << LB);//K += STEP

	//when K % STEP != 0--------------------------------------
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++) {
		float a = get(A, ty, ik, K);//transposed A
		float b = get(B, ik, tx, SB);
		c0 += b * a;
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	C[C0] = c0;
}

#endif


//(Y: N, X: M) X*Y<=1024
#ifndef MATMUL_KERNEL_NAIVE
#define MATMUL_KERNEL_NAIVE

// for 32*32*32: Performance= 11.36 GFlop/s, Time= 0.006 msec
// for 31*31*31: Performance=  9.74 GFlop/s, Time= 0.006 msec
__global__ void kernel_naive(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x + blockIdx.x*blockDim.x;
	int ty = threadIdx.y + blockIdx.y*blockDim.y;
	float v = 0;
	for (int k = 0; k < K; k++)
		v += get(A, ty, k, K) * get(B, k, tx, SB);
	get(C, ty, tx, SB) = v;
}
#endif

#endif