#pragma once

#ifndef MUTMUL_T1_SK_KERNEL_H
#define MUTMUL_T1_SK_KERNEL_H

//Split K to improve parallelism:
//(1) N % 4 == 0, N >= 4
//(2) M % 4 == 0, M >= 4
//(3) K % 4 == 0, K >= 4
//<1> A   belongs to Mat[K, N]
//<2> A^T belongs to Mat[N, K]
//<3> get(A^T, i, k, K) = get(A, k, i, N)
//<4> SB: stride of Matrix B: original == M, changed with GemmBranch
//<5> SA: stride of Matrix A: original == N, changed with GemmBranch
#ifndef MUTMUL_T1_SK_KERNEL_CALL
#define MUTMUL_T1_SK_KERNEL_CALL

//LB = log2(BLOCK_SIZE)

#define	k88T1SK_mgk(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_8_8_mgk<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB, GZ), dim3(2<<LB, 1<<LB>>1), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k88T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_8_8<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB, GZ), dim3(2<<LB, 1<<LB>>1), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k84T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_8_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>3>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k48T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_4_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>2>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k44T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_4_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>2>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k82T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_8_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>3>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k28T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_2_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>1>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k42T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_4_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>2>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k24T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_2_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>1>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k22T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_2_2<LB, (1<<LB)>\
		<<< dim3(M>>1>>LB, N>>1>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k21T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_2_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>1>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k12T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_1_2<LB, (1<<LB)>\
		<<< dim3(M>>1>>LB, N>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#define	k11T1SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB) \
	kernel_t1_SK_1_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SA, SB, K, K_slice)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
#ifndef MUTMUL_T1_SK_KERNEL_8_8_MGK
#define MUTMUL_T1_SK_KERNEL_8_8_MGK

//for[512 * 2048 * 8192]
//LB = 4: Size = 8, Time = 1.819 msec, Performace = 9444.68 GFlop/s
//LB = 3: Size = 8.00391, Time = 1.73 msec, Performace = 9935.41 GFlop/s

template<int LB, int STEP>
__global__ void kernel_t1_SK_8_8_mgk(
	const float* __restrict__ A,//[K, N: SA]
	const float* __restrict__ B,//[K, M: SB]
	      float* __restrict__ C,//for bz == 0
	     float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB >> 1][(2 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	const int bz = blockIdx.z;
	const int K_start = K_slice * bz;
	const int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	//compute 8*8 elements
	float4  c0 = F32_4_0,  c1 = F32_4_0,  c2 = F32_4_0,  c3 = F32_4_0;
	float4  c4 = F32_4_0,  c5 = F32_4_0,  c6 = F32_4_0,  c7 = F32_4_0;
	float4  c8 = F32_4_0,  c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;
	
	//prepare for C[N, M]
	const int idx = (ty << 1 << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = ((vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1)) << 1;
	const int uy = ((vy >> 2) * 8 + ((vx & 15) >> 1)) << 1;

	const int bY = (by << LB << 3);
	const int bX = (bx << LB << 3);
	
	const int cY = bY + (uy << 2);
	const int cX = bX + (ux << 2);
	const int C0 = cY * SB + cX;//C[Y, X]

	//compute area-------------------------------------------------------
	int A0 = (ty + K_start) * SA + bY + (tx << 2);
	int B0 = (ty + K_start) * SB + bX + (tx << 2);

	float4 a0 = *(float4*)(A + A0);
	float4 b0 = *(float4*)(B + B0);
	As[0][ty][tx] = a0;  
	Bs[0][ty][tx] = b0;
	__syncthreads();

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++) {
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
		buf ^= 1; A0 += STEP * SA; B0 += STEP * SB;

		float4 a0 = *(float4*)(A + A0);
		float4 b0 = *(float4*)(B + B0);
		As[buf][ty][tx] = a0;
		Bs[buf][ty][tx] = b0;
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
#ifndef MUTMUL_T1_SK_KERNEL_8_8
#define MUTMUL_T1_SK_KERNEL_8_8

//for[512 * 2048 * 8192]
//LB = 4: Size = 8, Time = 1.693 msec, Performace = 10147.6 GFlop/s
//LB = 3: Size = 8.00391, Time = 1.756 msec, Performace = 9788.3 GFlop/s

template<int LB, int STEP, int STEP_m1>
__global__ void kernel_t1_SK_8_8(
	const float* __restrict__ A,//[K, N:SA]
	const float* __restrict__ B,//[K, M:SB]
	      float* __restrict__ C,//for bz == 0
	      float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB >> 1][(2 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	const int bz = blockIdx.z;
	const int K_start = K_slice * bz;
	const int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	//compute 8*8 elements
	float4  c0 = F32_4_0,  c1 = F32_4_0,  c2 = F32_4_0,  c3 = F32_4_0;
	float4  c4 = F32_4_0,  c5 = F32_4_0,  c6 = F32_4_0,  c7 = F32_4_0;
	float4  c8 = F32_4_0,  c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;
	
	//prepare for C[N, M]
	const int idx = (ty << 1 << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = ((vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1)) << 1;
	const int uy = ((vy >> 2) * 8 + ((vx & 15) >> 1)) << 1;

	const int bY = (by << LB << 3);
	const int bX = (bx << LB << 3);
	
	const int cY = bY + (uy << 2);
	const int cX = bX + (ux << 2);
	const int C0 = cY * SB + cX;//C[Y, X]

	//compute area-------------------------------------------------------
	int A0 = (ty + K_start) * SA + bY + (tx << 2);
	int B0 = (ty + K_start) * SB + bX + (tx << 2);

	float4 a0 = *(float4*)(A + A0);
	float4 b0 = *(float4*)(B + B0);
	As[0][ty][tx] = a0;  
	Bs[0][ty][tx] = b0;
	__syncthreads();

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++) {
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
		buf ^= 1; A0 += STEP * SA; B0 += STEP * SB;

		float4 a0 = *(float4*)(A + A0);
		float4 b0 = *(float4*)(B + B0);
		As[buf][ty][tx] = a0;
		Bs[buf][ty][tx] = b0;
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
	buf ^= 1; A0 += STEP * SA; B0 += STEP * SB;

	const int RK = (K_slice & STEP_m1); if(RK) {//process remainder
		bool lk = (ty < RK);
		float4 a0 = (lk ? *(float4*)(A + A0) : F32_4_0);
		float4 b0 = (lk ? *(float4*)(B + B0) : F32_4_0);
		As[buf][ty][tx] = a0;
		Bs[buf][ty][tx] = b0;
		__syncthreads();

		for (int ik = 0; ik <RK; ik++) {
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
#ifndef MUTMUL_T1_SK_KERNEL_8_4
#define MUTMUL_T1_SK_KERNEL_8_4

//LB = 4: Size = 1, Time = 1.708 msec, Performace = 1257.31 GFlop/s
//LB = 3: Size = 1, Time = 1.832 msec, Performace = 1172.21 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_8_4(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88T1
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T1

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE;
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
	As[buf][tx][ty] = *(float4*)(A + Axy);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area-------------------------------------------------------
	float4  c0 = make_float4(0, 0, 0, 0);
	float4  c2 = make_float4(0, 0, 0, 0);
	float4  c4 = make_float4(0, 0, 0, 0);
	float4  c6 = make_float4(0, 0, 0, 0);
	float4  c8 = make_float4(0, 0, 0, 0);
	float4 c10 = make_float4(0, 0, 0, 0);
	float4 c12 = make_float4(0, 0, 0, 0);
	float4 c14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM4(c0, a0.x, b0);
			simdMM4(c2, a0.y, b0);
			simdMM4(c4, a0.z, b0);
			simdMM4(c6, a0.w, b0);
			simdMM4(c8, a1.x, b0);
			simdMM4(c10, a1.y, b0);
			simdMM4(c12, a1.z, b0);
			simdMM4(c14, a1.w, b0);
		}
		buf ^= 1;

		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
		As[buf][tx][ty] = *(float4*)(A + Axy);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM4(c0, a0.x, b0);
		simdMM4(c2, a0.y, b0);
		simdMM4(c4, a0.z, b0);
		simdMM4(c6, a0.w, b0);
		simdMM4(c8, a1.x, b0);
		simdMM4(c10, a1.y, b0);
		simdMM4(c12, a1.z, b0);
		simdMM4(c14, a1.w, b0);
	}

	//when K % STEP != 0------------------------------------
	ty <<= 3; tx <<= 2;
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float4 a0 = *(float4*)(&get(A, ik, ty, SA));
		float4 a1 = *(float4*)(&get(A, ik, ty + 4, SA));
		float4 b0 = *(float4*)(&get(B, ik, tx, SB));

		simdMM4(c0, a0.x, b0);
		simdMM4(c2, a0.y, b0);
		simdMM4(c4, a0.z, b0);
		simdMM4(c6, a0.w, b0);
		simdMM4(c8, a1.x, b0);
		simdMM4(c10, a1.y, b0);
		simdMM4(c12, a1.z, b0);
		simdMM4(c14, a1.w, b0);
	}
	//when K % STEP != 0------------------------------------

	*(float4*)(&get(C, ty    , tx, SB)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c2;
	*(float4*)(&get(C, ty + 2, tx, SB)) = c4;
	*(float4*)(&get(C, ty + 3, tx, SB)) = c6;
	*(float4*)(&get(C, ty + 4, tx, SB)) = c8;
	*(float4*)(&get(C, ty + 5, tx, SB)) = c10;
	*(float4*)(&get(C, ty + 6, tx, SB)) = c12;
	*(float4*)(&get(C, ty + 7, tx, SB)) = c14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
#ifndef MUTMUL_T1_SK_KERNEL_4_8
#define MUTMUL_T1_SK_KERNEL_4_8

//LB = 4: Size = 1, Time = 2.086 msec, Performace = 1029.47  GFlop/s
//LB = 4: Size = 1, Time = 2.242 msec, Performace =  957.843 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_4_8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T1
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88T1

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	Bs[buf][ty][tx] = *(float4*)(B + Byx);
	As[buf][As_x][As_y] = *(float2*)(A + Axy);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0), c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0), c3 = make_float4(0, 0, 0, 0);
	float4 c4 = make_float4(0, 0, 0, 0), c5 = make_float4(0, 0, 0, 0);
	float4 c6 = make_float4(0, 0, 0, 0), c7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
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
		buf ^= 1;

		Bs[buf][ty][tx] = *(float4*)(B + Byx);
		As[buf][As_x][As_y] = *(float2*)(A + Axy);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
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

	//when K % STEP != 0--------------------------------------
	ty <<= 2; tx <<= 3;
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float4 a0 = *(float4*)(&get(A, ik, ty, SA));
		float4 b0 = *(float4*)(&get(B, ik, tx, SB));
		float4 b1 = *(float4*)(&get(B, ik, tx + 4, SB));

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}
	//when K % STEP != 0--------------------------------------

	*(float4*)(&get(C, ty    , tx, SB)) = c0; *(float4*)(&get(C, ty    , tx + 4, SB)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c2; *(float4*)(&get(C, ty + 1, tx + 4, SB)) = c3;
	*(float4*)(&get(C, ty + 2, tx, SB)) = c4; *(float4*)(&get(C, ty + 2, tx + 4, SB)) = c5;
	*(float4*)(&get(C, ty + 3, tx, SB)) = c6; *(float4*)(&get(C, ty + 3, tx + 4, SB)) = c7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
#ifndef MUTMUL_T1_SK_KERNEL_4_4
#define MUTMUL_T1_SK_KERNEL_4_4

//LB = 4: Size = 1, Time = 2.12  msec, Performace = 1012.96  GFlop/s
//LB = 3: Size = 1, Time = 2.292 msec, Performace =  936.947 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_4_4(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE;
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
	As[buf][As_x][As_y] = *(float2*)(A + Axy);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area----------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0);
	float4 c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
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
		buf ^= 1;

		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
		As[buf][As_x][As_y] = *(float2*)(A + Axy);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
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

	//when K % STEP != 0---------------------------------------------
	ty <<= 2; tx <<= 2;
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float4 a = *(float4*)(&get(A, ik, ty, SA));
		float4 b = *(float4*)(&get(B, ik, tx, SB));

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}
	//when K % STEP != 0---------------------------------------------

	*(float4*)(&get(C, ty   , tx, SB)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c1;
	*(float4*)(&get(C, ty + 2, tx, SB)) = c2;
	*(float4*)(&get(C, ty + 3, tx, SB)) = c3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
#ifndef MUTMUL_T1_SK_KERNEL_8_2
#define MUTMUL_T1_SK_KERNEL_8_2

//LB = 4: Size = 1, Time = 2.19  msec, Performace = 980.586 GFlop/s
//LB = 3: Size = 1, Time = 2.312 msec, Performace = 928.842 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_8_2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88T1
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T1

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE;
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = (ty >> 1), Bx = ((tx << 1) + (ty & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	Bs[buf][Bs_y][Bs_x] = B[Byx];
	As[buf][tx][ty] = *(float4*)(A + Axy);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area----------------------------------------------------
	float2  c0 = make_float2(0, 0);
	float2  c2 = make_float2(0, 0);
	float2  c4 = make_float2(0, 0);
	float2  c6 = make_float2(0, 0);
	float2  c8 = make_float2(0, 0);
	float2 c10 = make_float2(0, 0);
	float2 c12 = make_float2(0, 0);
	float2 c14 = make_float2(0, 0);

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 b0 = *(float2*)(&Bs[buf][ik][tx << 1]);
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM2(c0, a0.x, b0);
			simdMM2(c2, a0.y, b0);
			simdMM2(c4, a0.z, b0);
			simdMM2(c6, a0.w, b0);
			simdMM2(c8, a1.x, b0);
			simdMM2(c10, a1.y, b0);
			simdMM2(c12, a1.z, b0);
			simdMM2(c14, a1.w, b0);
		}
		buf ^= 1;

		Bs[buf][Bs_y][Bs_x] = B[Byx];
		As[buf][tx][ty] = *(float4*)(A + Axy);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 b0 = *(float2*)(&Bs[buf][ik][tx << 1]);
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM2(c0, a0.x, b0);
		simdMM2(c2, a0.y, b0);
		simdMM2(c4, a0.z, b0);
		simdMM2(c6, a0.w, b0);
		simdMM2(c8, a1.x, b0);
		simdMM2(c10, a1.y, b0);
		simdMM2(c12, a1.z, b0);
		simdMM2(c14, a1.w, b0);
	}

	//when K % STEP != 0-------------------------------------
	ty <<= 3; tx <<= 1;
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float4 a0 = *(float4*)(&get(A, ik, ty, SA));
		float4 a1 = *(float4*)(&get(A, ik, ty + 4, SA));
		float2 b0 = *(float2*)(&get(B, ik, tx, SB));

		simdMM2(c0, a0.x, b0);
		simdMM2(c2, a0.y, b0);
		simdMM2(c4, a0.z, b0);
		simdMM2(c6, a0.w, b0);
		simdMM2(c8, a1.x, b0);
		simdMM2(c10, a1.y, b0);
		simdMM2(c12, a1.z, b0);
		simdMM2(c14, a1.w, b0);
	}
	//when K % STEP != 0-------------------------------------

	*(float2*)(&get(C, ty    , tx, SB)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SB)) = c2;
	*(float2*)(&get(C, ty + 2, tx, SB)) = c4;
	*(float2*)(&get(C, ty + 3, tx, SB)) = c6;
	*(float2*)(&get(C, ty + 4, tx, SB)) = c8;
	*(float2*)(&get(C, ty + 5, tx, SB)) = c10;
	*(float2*)(&get(C, ty + 6, tx, SB)) = c12;
	*(float2*)(&get(C, ty + 7, tx, SB)) = c14;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
#ifndef MUTMUL_T1_SK_KERNEL_2_8
#define MUTMUL_T1_SK_KERNEL_2_8

//LB = 4: Size = 1, Time = 2.764 msec, Performace = 776.948 GFlop/s
//LB = 3: Size = 1, Time = 2.942 msec, Performace = 729.94  GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_2_8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T1
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88T1

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	Bs[buf][ty][tx] = *(float4*)(B + Byx);
	As[buf][As_x][As_y] = A[Axy];
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area----------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0), c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0), c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);
			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		}
		buf ^= 1;

		Bs[buf][ty][tx] = *(float4*)(B + Byx);
		As[buf][As_x][As_y] = A[Axy];
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);
		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}

	//when K % STEP != 0-------------------------------------
	ty <<= 1; tx <<= 3;
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float2 a0 = *(float2*)(&get(A, ik, ty, SA));
		float4 b0 = *(float4*)(&get(B, ik, tx, SB));
		float4 b1 = *(float4*)(&get(B, ik, tx + 4, SB));

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}
	//when K % STEP != 0-------------------------------------

	*(float4*)(&get(C, ty, tx, SB)) = c0; *(float4*)(&get(C, ty, tx + 4, SB)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c2; *(float4*)(&get(C, ty + 1, tx + 4, SB)) = c3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
#ifndef MUTMUL_T1_SK_KERNEL_4_2
#define MUTMUL_T1_SK_KERNEL_4_2

//LB = 4: Size = 1, Time = 2.642 msec, Performace = 812.825 GFlop/s
//LB = 4: Size = 1, Time = 3.278 msec, Performace = 655.12  GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_4_2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE;
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));

	Bs[buf][Bs_y][Bs_x] = B[Byx];
	As[buf][As_x][As_y] = *(float2*)(A + Axy);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area----------------------------------------------------
	float2 c0 = make_float2(0, 0);
	float2 c1 = make_float2(0, 0);
	float2 c2 = make_float2(0, 0);
	float2 c3 = make_float2(0, 0);
	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
			simdMM2(c2, a.z, b);
			simdMM2(c3, a.w, b);
		}
		buf ^= 1;

		Bs[buf][Bs_y][Bs_x] = B[Byx];
		As[buf][As_x][As_y] = *(float2*)(A + Axy);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}

	//when K % STEP != 0-----------------------------------------
	ty <<= 2; tx <<= 1;
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float4 a = *(float4*)(&get(A, ik, ty, SA));
		float2 b = *(float2*)(&get(B, ik, tx, SB));

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}
	//when K % step != 0-----------------------------------------

	*(float2*)(&get(C, ty    , tx, SB)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SB)) = c1;
	*(float2*)(&get(C, ty + 2, tx, SB)) = c2;
	*(float2*)(&get(C, ty + 3, tx, SB)) = c3;
}

#endif 


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
#ifndef MUTMUL_T1_SK_KERNEL_2_4
#define MUTMUL_T1_SK_KERNEL_2_4

//LB = 4: Size = 1, Time = 3.292 msec, Performace = 652.334 GFlop/s
//LB = 3: Size = 1, Time = 3.548 msec, Performace = 605.266 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_2_4(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE;
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
	As[buf][As_x][As_y] = A[Axy];
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area----------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			float2 a = *(float2*)(&As[buf][ik][ty << 1]);
			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
		}
		buf ^= 1;

		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
		As[buf][As_x][As_y] = A[Axy];
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		float2 a = *(float2*)(&As[buf][ik][ty << 1]);
		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
	}

	//when K % STEP != 0---------------------------------------
	ty <<= 1; tx <<= 2;
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++) {
		float4 b = *(float4*)(&get(B, ik, tx, SB));
		float2 a = *(float2*)(&get(A, ik, ty, SA));
		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
	}
	//when K % STEP != 0---------------------------------------

	*(float4*)(&get(C, ty    , tx, SB)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), K >= BLOCK_SIZE
#ifndef MUTMUL_T1_SK_KERNEL_2_2
#define MUTMUL_T1_SK_KERNEL_2_2

//LB = 4: Size = 1, Time = 3.516 msec, Performace = 610.775 GFlop/s
//LB = 3: Size = 1, Time = 4.458 msec, Performace = 481.715 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_2_2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Axy = tx * SA + (ty << 1);
	const int Byx = ty * SB + (tx << 1);

	Bs[buf][ty][tx] = *(float2*)(B + Byx);
	As[buf][tx][ty] = *(float2*)(A + Axy);
	A += (SA << LB); B += (SB << LB);//A += SA * STEP
	__syncthreads();

	//compute area--------------------------------------------------
	float2 c0 = make_float2(0, 0);
	float2 c1 = make_float2(0, 0);
	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Bs[buf][ik][tx];
			float2 a = As[buf][ik][ty];
			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
		}
		buf ^= 1;

		Bs[buf][ty][tx] = *(float2*)(B + Byx);
		As[buf][tx][ty] = *(float2*)(A + Axy);
		A += (SA << LB); B += (SB << LB);//A += SA * STEP
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Bs[buf][ik][tx];
		float2 a = As[buf][ik][ty];
		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}

	//when K % STEP != 0 --------------------------------------
	ty <<= 1; tx <<= 1;
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++) {
		float2 b = *(float2*)(&get(B, ik, tx, SB));
		float2 a = *(float2*)(&get(A, ik, ty, SA));
		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}
	//when K % STEP != 0 --------------------------------------

	*(float2*)(&get(C, ty    , tx, SB)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SB)) = c1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
#ifndef MUTMUL_T1_SK_KERNEL_2_1
#define MUTMUL_T1_SK_KERNEL_2_1

//LB = 4: Size = 1, Time = 5.184 msec, Performace = 414.252 GFlop/s
//LB = 3: Size = 1, Time = 6.2   msec, Performace = 346.368 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_2_1(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << LB);//X = blockIdx.x * BLOCK_SIZE
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Axy = tx * SA + (ty << 1);
	const int Byx = ty * SB + tx;

	Bs[buf][ty][tx] = B[Byx];
	As[buf][tx][ty] = *(float2*)(A + Axy);
	A += (SA << LB); B += (SB << LB);//A += SA * STEP
	__syncthreads();

	//compute area--------------------------------------------------
	float2 c0 = make_float2(0, 0);
	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Bs[buf][ik][tx];
			float2 a = As[buf][ik][ty];
			simdMM2(c0, b, a);
		}
		buf ^= 1;

		Bs[buf][ty][tx] = B[Byx];
		As[buf][tx][ty] = *(float2*)(A + Axy);
		A += (SA << LB); B += (SB << LB);//A += SA * STEP
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float  b = Bs[buf][ik][tx];
		float2 a = As[buf][ik][ty];
		simdMM2(c0, b, a);
	}

	//when K % STEP != 0 --------------------------------------
	ty <<= 1;
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++) {
		float  b = get(B, ik, tx, SB);
		float2 a = *(float2*)(&get(A, ik, ty, SA));
		simdMM2(c0, b, a);
	}
	//when K % STEP != 0 --------------------------------------

	get(C, ty    , tx, SB) = c0.x;
	get(C, ty + 1, tx, SB) = c0.y;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2), K >= BLOCK_SIZE
#ifndef MUTMUL_T1_SK_KERNEL_1_2
#define MUTMUL_T1_SK_KERNEL_1_2

//LB = 4: Size = 1, Time = 6.746 msec, Performace = 318.334 GFlop/s
//LB = 3: Size = 1, Time = 7.874 msec, Performace = 272.731 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_1_2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Axy = tx * SA + ty;
	const int Byx = ty * SB + (tx << 1);

	Bs[buf][ty][tx] = *(float2*)(B + Byx);
	As[buf][tx][ty] = A[Axy];
	A += (SA << LB); B += (SB << LB);//A += SA * STEP
	__syncthreads();

	//compute area--------------------------------------------------
	float2 c0 = make_float2(0, 0);
	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Bs[buf][ik][tx];
			float  a = As[buf][ik][ty];
			simdMM2(c0, a, b);
		}
		buf ^= 1;

		Bs[buf][ty][tx] = *(float2*)(B + Byx);
		As[buf][tx][ty] = A[Axy];
		A += (SA << LB); B += (SB << LB);//A += SA * STEP
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Bs[buf][ik][tx];
		float  a = As[buf][ik][ty];
		simdMM2(c0, a, b);
	}

	//when K % STEP != 0 --------------------------------------
	tx <<= 1;
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++) {
		float2 b = *(float2*)(&get(B, ik, tx, SB));
		float  a = get(A, ik, ty, SA);
		simdMM2(c0, a, b);
	}
	//when K % STEP != 0 --------------------------------------

	*(float2*)(&get(C, ty, tx, SB)) = c0;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
#ifndef MUTMUL_T1_SK_KERNEL_1_1
#define MUTMUL_T1_SK_KERNEL_1_1

//LB = 4: Size = 1, Time =   9.46 msec, Performace = 227.007 GFlop/s
//LB = 3: Size = 1, Time = 11.302 msec, Performace = 190.009 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_SK_1_1(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float* __restrict__ C,//for bz == 0
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SA, int SB, int K, int K_slice)
{
	bool buf = 0;
	__shared__ float As[2][1 << LB][(1 << LB) + 1];
	__shared__ float Bs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int K_start = K_slice * bz;
	int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * SA * SB;//stride = SA * SB = N * M
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	int Y = (blockIdx.y << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = &get(A, K_start, Y, SA);//A[K_start, Y]
	B = &get(B, K_start, X, SB);//B[K_start, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Axy = tx * SA + ty;
	const int Byx = ty * SB + tx;

	Bs[buf][ty][tx] = B[Byx];
	As[buf][tx][ty] = A[Axy];
	A += (SA << LB); B += (SB << LB);//A += SA * STEP
	__syncthreads();

	//compute area--------------------------------------------------
	float c = 0;
	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float b = Bs[buf][ik][tx];
			float a = As[buf][ik][ty];
			c += a * b;
		}
		buf ^= 1;

		Bs[buf][ty][tx] = B[Byx];
		As[buf][tx][ty] = A[Axy];
		A += (SA << LB); B += (SB << LB);//A += SA * STEP
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float b = Bs[buf][ik][tx];
		float a = As[buf][ik][ty];
		c += a * b;
	}

	//when K % STEP != 0 --------------------------------------
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++) {
		float b = get(B, ik, tx, SB);
		float a = get(A, ik, ty, SA);
		c += a * b;
	}
	//when K % STEP != 0 --------------------------------------

	get(C, ty, tx, SB) = c;
}

#endif

#endif