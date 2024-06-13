#pragma once

#ifndef MATMUL_T2_SK_UERNEL_H
#define MATMUL_T2_SK_UERNEL_H

//Split K to improve parallelism:
//(1) N % 4 == 0, N >= 4
//(2) M % 4 == 0, M >= 4
//(3) K % 4 == 0, K >= 4
#ifndef MATMUL_T2_SK_UERNEL_CALL
#define MATMUL_T2_SK_UERNEL_CALL

//LB = log2(BLOCK_SIZE

#define	u88T2SK_mgk(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	uernel_t2_SK_8_8_mgk<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>3>>LB, GZ), dim3(1<<LB>>1, 2<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define	u84T2SK_mgk(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	uernel_t2_SK_8_4_mgk<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(M>>2>>LB, N>>3>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define	u48T2SK_mgk(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	uernel_t2_SK_4_8_mgk<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>2>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#ifndef MATMUL_T2_SK_UERNEL_8_8_MGK
#define MATMUL_T2_SK_UERNEL_8_8_MGK

//for [512 * 2048 * 8192]:
//LB = 4: Size = 8, Time = 1.995 msec, Performace = 8611.46 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_t2_SK_8_8_mgk(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	      float*  __restrict__ C,
	      float*  __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(2 << LB) + 1];

	//compute 8*8 elements
	float4  c0 = F32_4_0,  c1 = F32_4_0,  c2 = F32_4_0,  c3 = F32_4_0;
	float4  c4 = F32_4_0,  c5 = F32_4_0,  c6 = F32_4_0,  c7 = F32_4_0;
	float4  c8 = F32_4_0,  c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	//======================================================================
	const int bz = blockIdx.z;
	const int K_start = K_slice * bz;
	const int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * Cstride;
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	//prepare for C[N, M]
	const int idx = (ty << LB >> 1) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = ((vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1)) << 1;
	const int uy = ((vy >> 2) * 8 + ((vx & 15) >> 1)) << 1;

	const int bY = (by << LB << 3);
	const int bX = (bx << LB << 3);

	const int cY = bY + (uy << 2);
	const int cX = bX + (ux << 2); 
	const int C0 = cY * SC + cX;//C[Y, X]

	//compute area-------------------------------------------------------
	const int tY = bY + (ty << 2); A += tY * K + K_start;
	const int tX = bX + (ty << 2); B += tX * K + K_start;

	//load 4 elem from A(transposed), 4 elem from B
	const int k0 = (tx << 1), k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;

	float2 a0 = *(float2*)(A + k0);
	float2 a1 = *(float2*)(A + k1);
	float2 a2 = *(float2*)(A + k2);
	float2 a3 = *(float2*)(A + k3);

	float2 b0 = *(float2*)(B + k0);
	float2 b1 = *(float2*)(B + k1);
	float2 b2 = *(float2*)(B + k2);
	float2 b3 = *(float2*)(B + k3);

	//write to shared memory
	As[0][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[0][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };
	Bs[0][(tx << 1)    ][ty] = float4{ b0.x, b1.x, b2.x, b3.x };
	Bs[0][(tx << 1) + 1][ty] = float4{ b0.y, b1.y, b2.y, b3.y };
	__syncthreads();

	for (int ok = STEP2; ok < K_slice; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
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
		buf ^= 1; A += STEP2; B += STEP2;//K += STEP2

		//load 4 elem from A(transposed), 4 elem from B
		const int k0 = (tx << 1), k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;

		float2 a0 = *(float2*)(A + k0);
		float2 a1 = *(float2*)(A + k1);
		float2 a2 = *(float2*)(A + k2);
		float2 a3 = *(float2*)(A + k3);

		float2 b0 = *(float2*)(B + k0);
		float2 b1 = *(float2*)(B + k1);
		float2 b2 = *(float2*)(B + k2);
		float2 b3 = *(float2*)(B + k3);

		As[buf][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };
		Bs[buf][(tx << 1)    ][ty] = float4{ b0.x, b1.x, b2.x, b3.x };
		Bs[buf][(tx << 1) + 1][ty] = float4{ b0.y, b1.y, b2.y, b3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik][ux + 1];
		float4 a0 = As[buf][ik][uy], a1 = As[buf][ik][uy + 1];

		simdMM4( c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4( c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4( c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4( c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4( c8, a1.x, b0); simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}

	const int C1 = C0 + SC, C2 = C1 + SC, C3 = C2 + SC;
	const int C4 = C3 + SC, C5 = C4 + SC, C6 = C5 + SC, C7 = C6 + SC;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#ifndef MATMUL_T2_SK_UERNEL_8_4_MGK
#define MATMUL_T2_SK_UERNEL_8_4_MGK

//for[128 * 4096 * 8192]:
//LB = 4: Size = 4, Time = 1.416 msec, Performace = 6066.34 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uernel_t2_SK_8_4_mgk(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K, 
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];//follow k44

	//compute 8*4 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;

	//======================================================================
	const int bz = blockIdx.z;
	const int K_start = K_slice * bz;
	const int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * Cstride;
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	const int Y = ((blockIdx.y << LB) + ty) << 3; A += Y * K + K_start;
	const int X = ((blockIdx.x << LB) + tx) << 2; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area-------------------------------------------------------
	const int A0 = ((tx >= STEP) << 2)*K + ((tx & STEP_m1) << 1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int By = (ty >> 1) << 1;
	const int B0 = ((ty & 1) << 1) * K + By, B1 = B0 + K;
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[0][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[0][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 2 elem from B
	float2 b0 = *(float2*)(B + B0);
	float2 b1 = *(float2*)(B + B1);
	Bs[0][Bs_y][Bs_x] = float2{ b0.x, b1.x };
	Bs[0][Bs_y + 1][Bs_x] = float2{ b0.y, b1.y };
	__syncthreads();

	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
			float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.y, b0);
			simdMM4(c2, a0.z, b0); simdMM4(c3, a0.w, b0);
			simdMM4(c4, a1.x, b0); simdMM4(c5, a1.y, b0);
			simdMM4(c6, a1.z, b0); simdMM4(c7, a1.w, b0);
		}
		buf ^= 1; A += STEP2; B += STEP2;//K += STEP

		//load 4 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 2 elem from B
		float2 b0 = *(float2*)(B + B0);
		float2 b1 = *(float2*)(B + B1);
		Bs[buf][Bs_y][Bs_x] = float2{ b0.x, b1.x };
		Bs[buf][Bs_y + 1][Bs_x] = float2{ b0.y, b1.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
		float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.y, b0);
		simdMM4(c2, a0.z, b0); simdMM4(c3, a0.w, b0);
		simdMM4(c4, a1.x, b0); simdMM4(c5, a1.y, b0);
		simdMM4(c6, a1.z, b0); simdMM4(c7, a1.w, b0);
	}

	const int C1 = C0 + SC, C2 = C1 + SC, C3 = C2 + SC;
	const int C4 = C3 + SC, C5 = C4 + SC, C6 = C5 + SC, C7 = C6 + SC;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#ifndef MATMUL_T2_SK_UERNEL_4_8_MGK
#define MATMUL_T2_SK_UERNEL_4_8_MGK

//for [1248 * 4096 * 8192]:
//LB = 4: Size = 4, Time = 1.714 msec, Performace = 5011.63 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void uernel_t2_SK_4_8_mgk(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];//follow k44
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];//follow k88

	//compute 4*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0;
	float4 c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0;
	float4 c6 = F32_4_0, c7 = F32_4_0;

	//======================================================================
	const int bz = blockIdx.z;
	const int K_start = K_slice * bz;
	const int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * Cstride;
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	const int Y = ((blockIdx.y << LB) + ty) << 2; A += Y * K + K_start;
	const int X = ((blockIdx.x << LB) + tx) << 3; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area-------------------------------------------------------
	const int A0 = ((tx & 1) << 1)*K + ((tx >> 1) << 1), A1 = A0 + K;
	const int As_x = ((tx >> 1) << 1), As_y = ((ty << 1) + (tx & 1));

	const int B0 = ((ty >= STEP) << 2)*K + ((ty & STEP_m1) << 1);
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	//load 4 elem from B (transposed)
	float2 b0 = *(float2*)(B + B0);
	float2 b1 = *(float2*)(B + B1);
	float2 b2 = *(float2*)(B + B2);
	float2 b3 = *(float2*)(B + B3);
	Bs[0][(ty << 1)][tx] = float4{ b0.x, b1.x, b2.x, b3.x };
	Bs[0][(ty << 1) + 1][tx] = float4{ b0.y, b1.y, b2.y, b3.y };

	//load 2 elem from A (transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	As[0][As_x][As_y] = float2{ a0.x, a1.x };
	As[0][As_x + 1][As_y] = float2{ a0.y, a1.y };
	__syncthreads();

	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
			float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
		}
		buf ^= 1; A += STEP2; B += STEP2;//K += STEP

		//load 4 elem from B (transposed)s
		float2 b0 = *(float2*)(B + B0);
		float2 b1 = *(float2*)(B + B1);
		float2 b2 = *(float2*)(B + B2);
		float2 b3 = *(float2*)(B + B3);
		Bs[buf][(ty << 1)][tx] = float4{ b0.x, b1.x, b2.x, b3.x };
		Bs[buf][(ty << 1) + 1][tx] = float4{ b0.y, b1.y, b2.y, b3.y };

		//load 2 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		As[buf][As_x][As_y] = float2{ a0.x, a1.x };
		As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
		float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
	}

	const int C1 = C0 + SC;
	const int C2 = C1 + SC;
	const int C3 = C2 + SC;

	*(float4*)(C + C0) = c0;  *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2;  *(float4*)(C + C1 + 4) = c3;
	*(float4*)(C + C2) = c4;  *(float4*)(C + C2 + 4) = c5;
	*(float4*)(C + C3) = c6;  *(float4*)(C + C3 + 4) = c7;
}

#endif

#endif