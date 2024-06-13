#pragma once

#ifndef MATMUL_T2_SK_KERNEL_H
#define MATMUL_T2_SK_KERNEL_H

//Split K to improve parallelism:
//(1) N % 4 == 0, N >= 4
//(2) M % 4 == 0, M >= 4
//(3) K % 4 == 0, K >= 4
#ifndef MATMUL_T2_SK_KERNEL_CALL
#define MATMUL_T2_SK_KERNEL_CALL

//======[Extra]=================================================
#define	k88T2SK_v_mgk(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_8_8V_mgk<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define	k88T2SK_v(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_8_8V<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

//======[Common]================================================
#define	k88T2SK_mgk(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_8_8_mgk<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB, GZ), dim3(1<<LB>>1, 2<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define	k88T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_8_8<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB, GZ), dim3(1<<LB>>1, 2<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define	k84T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_8_4<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>2>>LB, N>>3>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define	k48T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_4_8<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>2>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define	k44T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_4_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>2>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define	k82T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_8_2<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>1>>LB, N>>3>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define	k28T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_2_8<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>1>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define	k42T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_4_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>2>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

//======[Small]===========================================
#define k22T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_2_2<LB, (1<<LB)>\
		<<< dim3(M>>1>>LB, N>>1>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define k81T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_8_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>3>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define k18T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_1_8<LB, (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#define k11T2SK(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SC) \
	kernel_t2_SK_1_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SC, K, K_slice, Cstride)

#endif


//======[Extra]=================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
//LB = 4: K >= 8
//LB = 3: don't use it
#ifndef MATMUL_T2_SK_KERNEL_8_8V_MGK
#define MATMUL_T2_SK_KERNEL_8_8V_MGK

//for [128 * 4096 * 8192]:
//LB = 4: Size = 4, Time = 1.166 msec, Performace = 7367.01 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_t2_SK_8_8V_mgk(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K, 
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float As[2][1 << LB][(4 << LB) + 4];
	__shared__ float Bs[2][1 << LB][(4 << LB) + 4];

	//compute 8*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
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

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SC + X;//C[Y, X]
	A += (Y + ((tx >= STEP) << 2) + (tx & 3)) * K + K_start;
	B += (X + ((ty >= STEP) << 2) + (ty & 3)) * K + K_start;

	//compute area-------------------------------------------------------
	const int A0 = (tx & STEP_m1) >> 2 << 2;
	const int As_x = A0 + ((tx >= STEP) << LB >> 1);
	const int As_y = (ty << 2) + (tx & 3);

	const int B0 = (ty & STEP_m1) >> 2 << 2;
	const int Bs_x = B0 + ((ty >= STEP) << LB >> 1);
	const int Bs_y = (tx << 2) + (ty & 3);

	float4 av = *(float4*)(A + A0);//transpose A
	As[0][As_x][As_y] = av.x;
	As[0][As_x + 1][As_y] = av.y;
	As[0][As_x + 2][As_y] = av.z;
	As[0][As_x + 3][As_y] = av.w;

	float4 bv = *(float4*)(B + B0);//transpose B
	Bs[0][Bs_x][Bs_y] = bv.x;
	Bs[0][Bs_x + 1][Bs_y] = bv.y;
	Bs[0][Bs_x + 2][Bs_y] = bv.z;
	Bs[0][Bs_x + 3][Bs_y] = bv.w;
	__syncthreads();

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&As[buf][ik][ty << 2]);
			float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
			float4 b0 = *(float4*)(&Bs[buf][ik][tx << 2]);
			float4 b1 = *(float4*)(&Bs[buf][ik + STEP][tx << 2]);

			simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		float4 av = *(float4*)(A + A0);//transpose A
		As[buf][As_x    ][As_y] = av.x;
		As[buf][As_x + 1][As_y] = av.y;
		As[buf][As_x + 2][As_y] = av.z;
		As[buf][As_x + 3][As_y] = av.w;

		float4 bv = *(float4*)(B + B0);//transpose B
		Bs[buf][Bs_x    ][Bs_y] = bv.x;
		Bs[buf][Bs_x + 1][Bs_y] = bv.y;
		Bs[buf][Bs_x + 2][Bs_y] = bv.z;
		Bs[buf][Bs_x + 3][Bs_y] = bv.w;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&As[buf][ik][ty << 2]);
		float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
		float4 b0 = *(float4*)(&Bs[buf][ik][tx << 2]);
		float4 b1 = *(float4*)(&Bs[buf][ik + STEP][tx << 2]);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
//LB = 4: K >= 8
//LB = 3: don't use it
#ifndef MATMUL_T2_SK_KERNEL_8_8V
#define MATMUL_T2_SK_KERNEL_8_8V

//for [128 * 4096 * 8192]:
//LB = 4: Size = 4, Time = 1.169 msec, Performace = 7348.1 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_t2_SK_8_8V(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float As[2][1 << LB][(4 << LB) + 4];
	__shared__ float Bs[2][1 << LB][(4 << LB) + 4];

	//compute 8*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
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

	const int Y = ((blockIdx.y << LB) + ty) << 3; A += Y * K + K_start;
	const int X = ((blockIdx.x << LB) + tx) << 3; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area-------------------------------------------------------
	int A0 = (tx & STEP_m1) >> 2 << 2;
	const int As_x = A0 + ((tx >= STEP) << LB >> 1);
	const int As_y = (ty << 2) + (tx & 3);
	A0 += (((tx >= STEP) << 2) + (tx & 3)) * K;

	int B0 = (ty & STEP_m1) >> 2 << 2;
	const int Bs_x = B0 + ((ty >= STEP) << LB >> 1);
	const int Bs_y = (tx << 2) + (ty & 3);
	B0 += (((ty >= STEP) << 2) + (ty & 3)) * K;

	float4 av = *(float4*)(A + A0);//transpose A
	As[0][As_x][As_y] = av.x;
	As[0][As_x + 1][As_y] = av.y;
	As[0][As_x + 2][As_y] = av.z;
	As[0][As_x + 3][As_y] = av.w;

	float4 bv = *(float4*)(B + B0);//transpose B
	Bs[0][Bs_x][Bs_y] = bv.x;
	Bs[0][Bs_x + 1][Bs_y] = bv.y;
	Bs[0][Bs_x + 2][Bs_y] = bv.z;
	Bs[0][Bs_x + 3][Bs_y] = bv.w;
	__syncthreads();

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&As[buf][ik][ty << 2]);
			float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
			float4 b0 = *(float4*)(&Bs[buf][ik][tx << 2]);
			float4 b1 = *(float4*)(&Bs[buf][ik + STEP][tx << 2]);

			simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		float4 av = *(float4*)(A + A0);//transpose A
		As[buf][As_x][As_y] = av.x;
		As[buf][As_x + 1][As_y] = av.y;
		As[buf][As_x + 2][As_y] = av.z;
		As[buf][As_x + 3][As_y] = av.w;

		float4 bv = *(float4*)(B + B0);//transpose B
		Bs[buf][Bs_x][Bs_y] = bv.x;
		Bs[buf][Bs_x + 1][Bs_y] = bv.y;
		Bs[buf][Bs_x + 2][Bs_y] = bv.z;
		Bs[buf][Bs_x + 3][Bs_y] = bv.w;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&As[buf][ik][ty << 2]);
		float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
		float4 b0 = *(float4*)(&Bs[buf][ik][tx << 2]);
		float4 b1 = *(float4*)(&Bs[buf][ik + STEP][tx << 2]);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}
	A += STEP; B += STEP;//K += STEP

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & STEP_m1; ik < RK; ik++)
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

		float4 b0, b1;
		b0.x = get(B, 0, ik, K);
		b0.y = get(B, 1, ik, K);
		b0.z = get(B, 2, ik, K);
		b0.w = get(B, 3, ik, K);
		b1.x = get(B, 4, ik, K);
		b1.y = get(B, 5, ik, K);
		b1.z = get(B, 6, ik, K);
		b1.w = get(B, 7, ik, K);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}
	//when K_slice % STEP != 0-------------------------------------

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


//======[Common]================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
//LB = 4: K >= 8
#ifndef MATMUL_T2_SK_KERNEL_8_8_MGK
#define MATMUL_T2_SK_KERNEL_8_8_MGK

//for [512 * 2048 * 8192]:
//LB = 4: Size = 8, Time = 1.952 msec, Performace = 8801.16 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_SK_8_8_mgk(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K, 
	int K_slice, int Cstride)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB >> 1][(2 << LB) + 1];

	//compute 8*8 = 64 elements
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
	const int k0 = tx, k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;
	float4 a0 = float4{ A[k0], A[k1], A[k2], A[k3] };//transposed A
	float4 b0 = float4{ B[k0], B[k1], B[k2], B[k3] };//transposed B
	As[0][tx][ty] = a0;
	Bs[0][tx][ty] = b0;
	__syncthreads();

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ++ok) {
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
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		//load 4 elem from A(transposed), 4 elem from B
		const int k0 = tx, k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;
		float4 a0 = float4{ A[k0], A[k1], A[k2], A[k3] };//transposed A
		float4 b0 = float4{ B[k0], B[k1], B[k2], B[k3] };//transposed B
		As[buf][tx][ty] = a0; 
		Bs[buf][tx][ty] = b0;
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_T2_SK_KERNEL_8_8
#define MATMUL_T2_SK_KERNEL_8_8

//for [512 * 2048 * 8192]:
//LB = 4: Size = 8.00391, Time = 1.971 msec, Performace = 8720.58 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_t2_SK_8_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K, 
	int K_slice, int Cstride)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB >> 1][(2 << LB) + 1];

	//compute 8*8 = 64 elements
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
	
	//compute area----------------------------------------------------------
	const int tY = bY + (ty << 2); A += tY * K + K_start;
	const int tX = bX + (ty << 2); B += tX * K + K_start;

	//load 4 elem from A(transposed), 4 elem from B
	const int k0 = tx, k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;
	float4 a0 = float4{ A[k0], A[k1], A[k2], A[k3] };//transposed A
	float4 b0 = float4{ B[k0], B[k1], B[k2], B[k3] };//transposed B
	As[0][tx][ty] = a0;
	Bs[0][tx][ty] = b0;
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
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		//load 4 elem from A(transposed), 4 elem from B
		const int k0 = tx, k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;
		float4 a0 = float4{ A[k0], A[k1], A[k2], A[k3] };//transposed A
		float4 b0 = float4{ B[k0], B[k1], B[k2], B[k3] };//transposed B
		As[buf][tx][ty] = a0;
		Bs[buf][tx][ty] = b0;
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
	A += ((uy - ty) << 2) * K + STEP;//K += STEP
	B += ((ux - ty) << 2) * K + STEP;

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & STEP_m1; ik < RK; ik++) {
		float4 a0, a1;
		a0.x = get(A, 0, ik, K); a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K); a0.w = get(A, 3, ik, K);
		a1.x = get(A, 4, ik, K); a1.y = get(A, 5, ik, K);
		a1.z = get(A, 6, ik, K); a1.w = get(A, 7, ik, K);

		float4 b0, b1;
		b0.x = get(B, 0, ik, K); b0.y = get(B, 1, ik, K);
		b0.z = get(B, 2, ik, K); b0.w = get(B, 3, ik, K);
		b1.x = get(B, 4, ik, K); b1.y = get(B, 5, ik, K);
		b1.z = get(B, 6, ik, K); b1.w = get(B, 7, ik, K);

		simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
		simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
		simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
		simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
		simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}
	//when K_slice % STEP != 0-------------------------------------

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_T2_SK_KERNEL_8_4
#define MATMUL_T2_SK_KERNEL_8_4

//for [128 * 4096 * 8192]:
//LB = 4: Size = 4, Time = 1.617 msec, Performace = 5312.27 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_t2_SK_8_4(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88T2
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2

	//======================================================================
	const int bz = blockIdx.z;
	const int K_start = K_slice * bz;
	const int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * Cstride;
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	//compute 8*4 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;

	const int Y = ((blockIdx.y << LB) + ty) << 3; A += Y * K + K_start;
	const int X = ((blockIdx.x << LB) + tx) << 2; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area-------------------------------------------------------
	const int A0 = ((tx >= STEP) << 2) * K + (tx & STEP_m1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int Bs_y = (ty >> 1), Bs_x = ((tx << 1) + (ty & 1));
	const int B0 = ((ty & 1) << 1) * K + (ty >> 1), B1 = B0 + K;

	As[0][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transpose A
	Bs[0][Bs_y][Bs_x] = float2{ B[B0], B[B1] };//transpose B
	__syncthreads();

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.y, b0);
			simdMM4(c2, a0.z, b0); simdMM4(c3, a0.w, b0);
			simdMM4(c4, a1.x, b0); simdMM4(c5, a1.y, b0);
			simdMM4(c6, a1.z, b0); simdMM4(c7, a1.w, b0);
		}
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transpose A
		Bs[buf][Bs_y][Bs_x] = float2{ B[B0], B[B1] };//transpose B
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.y, b0);
		simdMM4(c2, a0.z, b0); simdMM4(c3, a0.w, b0);
		simdMM4(c4, a1.x, b0); simdMM4(c5, a1.y, b0);
		simdMM4(c6, a1.z, b0); simdMM4(c7, a1.w, b0);
	}
	A += STEP; B += STEP;//K += STEP

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & STEP_m1; ik < RK; ik++)
	{
		float4 b0;//transposed B
		b0.x = get(B, 0, ik, K);
		b0.y = get(B, 1, ik, K);
		b0.z = get(B, 2, ik, K);
		b0.w = get(B, 3, ik, K);

		float4 a0, a1;//transposed A
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);
		a1.x = get(A, 4, ik, K);
		a1.y = get(A, 5, ik, K);
		a1.z = get(A, 6, ik, K);
		a1.w = get(A, 7, ik, K);

		simdMM4(c0, a0.x, b0);
		simdMM4(c1, a0.y, b0);
		simdMM4(c2, a0.z, b0);
		simdMM4(c3, a0.w, b0);
		simdMM4(c4, a1.x, b0);
		simdMM4(c5, a1.y, b0);
		simdMM4(c6, a1.z, b0);
		simdMM4(c7, a1.w, b0);
	}
	//when K_slice % STEP != 0-------------------------------------

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_T2_SK_KERNEL_4_8
#define MATMUL_T2_SK_KERNEL_4_8

//for [128 * 4096 * 8192]:
//LB = 4: Size = 4, Time = 1.882 msec, Performace = 4564.26 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_t2_SK_4_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88T2

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

	const int Y = ((blockIdx.y << LB) + ty) << 2; A += Y * K + K_start;
	const int X = ((blockIdx.x << LB) + tx) << 3; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area-------------------------------------------------------
	const int Ax = (tx >> 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = ((tx & 1) << 1) * K + Ax, A1 = A0 + K;

	const int Bx = ((ty >= STEP) << 2);
	const int B0 = Bx * K + (ty & STEP_m1);
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	As[0][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
	Bs[0][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };
	__syncthreads();

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		}
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
		Bs[buf][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}
	A += STEP; B += STEP;//K += STEP

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float4 a0;//transposed A
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);

		float4 b0, b1;//transposed B
		b0.x = get(B, 0, ik, K);
		b0.y = get(B, 1, ik, K);
		b0.z = get(B, 2, ik, K);
		b0.w = get(B, 3, ik, K);
		b1.x = get(B, 4, ik, K);
		b1.y = get(B, 5, ik, K);
		b1.z = get(B, 6, ik, K);
		b1.w = get(B, 7, ik, K);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}
	//when K_slice % STEP != 0-------------------------------------

	const int C1 = C0 + SC;
	const int C2 = C1 + SC;
	const int C3 = C2 + SC;

	*(float4*)(C + C0) = c0; *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2; *(float4*)(C + C1 + 4) = c3;
	*(float4*)(C + C2) = c4; *(float4*)(C + C2 + 4) = c5;
	*(float4*)(C + C3) = c6; *(float4*)(C + C3 + 4) = c7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_T2_SK_KERNEL_4_4
#define MATMUL_T2_SK_KERNEL_4_4

//for[2048 * 2048 * 2048]:
//LB = 4: Size = 4, Time = 2.354 msec, Performace = 3649.08 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_SK_4_4(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//compute 4*4 elements
	float4 c0 = F32_4_0, c1 = F32_4_0;
	float4 c2 = F32_4_0, c3 = F32_4_0;

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
	const int X = ((blockIdx.x << LB) + tx) << 2; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area-------------------------------------------------------
	const int A0 = ((tx & 1) << 1) * K + (tx >> 1), A1 = A0 + K;
	const int As_x = (tx >> 1), As_y = ((ty << 1) + (tx & 1));

	const int B0 = ((ty & 1) << 1) * K + (ty >> 1), B1 = B0 + K;
	const int Bs_y = (ty >> 1), Bs_x = ((tx << 1) + (ty & 1));

	As[0][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
	Bs[0][Bs_y][Bs_x] = float2{ B[B0], B[B1] };//transpose B
	__syncthreads();

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
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
		Bs[buf][Bs_y][Bs_x] = float2{ B[B0], B[B1] };//transpose B
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
	A += STEP; B += STEP;//K += STEP

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float4 a;//transpose A
		a.x = get(A, 0, ik, K);
		a.y = get(A, 1, ik, K);
		a.z = get(A, 2, ik, K);
		a.w = get(A, 3, ik, K);

		float4 b;//transpose B
		b.x = get(B, 0, ik, K);
		b.y = get(B, 1, ik, K);
		b.z = get(B, 2, ik, K);
		b.w = get(B, 3, ik, K);

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}
	//when K_slice % STEP != 0-------------------------------------

	const int C1 = C0 + SC;
	const int C2 = C1 + SC;
	const int C3 = C2 + SC;

	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c1;
	*(float4*)(C + C2) = c2;
	*(float4*)(C + C3) = c3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_T2_SK_KERNEL_8_2
#define MATMUL_T2_SK_KERNEL_8_2

//for[2048 * 2048 * 2048]:
//LB = 4: Size = 4, Time = 2.134 msec, Performace = 4025.27 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_t2_SK_8_2(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88T2
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2

	//compute 8*2 elements
	float2 c0 = F32_2_0, c1 = F32_2_0, c2 = F32_2_0, c3 = F32_2_0;
	float2 c4 = F32_2_0, c5 = F32_2_0, c6 = F32_2_0, c7 = F32_2_0;

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
	const int X = ((blockIdx.x << LB) + tx) << 1; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area-------------------------------------------------------
	const int A0 = ((tx >= STEP) << 2)*K + (tx & STEP_m1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int B0 = (ty & 1)*K + (ty >> 1);
	const int Bs_y = (ty >> 1), Bs_x = ((tx << 1) + (ty & 1));

	As[0][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transpose A
	Bs[0][Bs_y][Bs_x] = B[B0];//transpose B
	__syncthreads();

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
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
		buf ^= 1; A += STEP; B += STEP;

		As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transpose A
		Bs[buf][Bs_y][Bs_x] = B[B0];//transpose B
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
	A += STEP; B += STEP;

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & STEP_m1; ik < RK; ik++)
	{
		float4 a0, a1;//transpose A
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);
		a1.x = get(A, 4, ik, K);
		a1.y = get(A, 5, ik, K);
		a1.z = get(A, 6, ik, K);
		a1.w = get(A, 7, ik, K);

		float2 b0;//transpose B
		b0.x = get(B, 0, ik, K);
		b0.y = get(B, 1, ik, K);

		simdMM2(c0, a0.x, b0);
		simdMM2(c1, a0.y, b0);
		simdMM2(c2, a0.z, b0);
		simdMM2(c3, a0.w, b0);
		simdMM2(c4, a1.x, b0);
		simdMM2(c5, a1.y, b0);
		simdMM2(c6, a1.z, b0);
		simdMM2(c7, a1.w, b0);
	}
	//when K_slice % STEP != 0-------------------------------------

	const int C1 = C0 + SC, C2 = C1 + SC, C3 = C2 + SC;
	const int C4 = C3 + SC, C5 = C4 + SC, C6 = C5 + SC, C7 = C6 + SC;

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
#ifndef MATMUL_T2_SK_KERNEL_2_8
#define MATMUL_T2_SK_KERNEL_2_8

//for[2048 * 2048 * 2048]:
//LB = 4: Size = 4, Time = 2.885 msec, Performace = 2977.45 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void kernel_t2_SK_2_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88T2

	//compute 2*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0;
	float4 c2 = F32_4_0, c3 = F32_4_0;

	//======================================================================
	const int bz = blockIdx.z;
	const int K_start = K_slice * bz;
	const int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * Cstride;
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	const int Y = ((blockIdx.y << LB) + ty) << 1; A += Y * K + K_start;
	const int X = ((blockIdx.x << LB) + tx) << 3; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area-----------------------------------------------
	const int A0 = (tx & 1) * K + (tx >> 1);
	const int As_x = (tx >> 1), As_y = ((ty << 1) + (tx & 1));

	const int B0 = ((ty >= STEP) << 2) * K + (ty & STEP_m1);
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	As[0][As_x][As_y] = A[A0];//transpose A
	Bs[0][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };//transpose B
	__syncthreads();

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		}
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		As[buf][As_x][As_y] = A[A0];//transpose A
		Bs[buf][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };//transpose B
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}
	A += STEP; B += STEP;//K += STEP

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & STEP_m1; ik < RK; ik++)
	{
		float2 a0;//transpose A
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);

		float4 b0, b1;//transpose B
		b0.x = get(B, 0, ik, K);
		b0.y = get(B, 1, ik, K);
		b0.z = get(B, 2, ik, K);
		b0.w = get(B, 3, ik, K);
		b1.x = get(B, 4, ik, K);
		b1.y = get(B, 5, ik, K);
		b1.z = get(B, 6, ik, K);
		b1.w = get(B, 7, ik, K);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}
	//when K_slice % STEP != 0-------------------------------------

	const int C1 = C0 + SC;
	*(float4*)(C + C0) = c0; *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2; *(float4*)(C + C1 + 4) = c3;
}

#endif

	
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
#ifndef MATMUL_T2_SK_KERNEL_4_2
#define MATMUL_T2_SK_KERNEL_4_2

//for[2048 * 1024 * 1024]£º
//LB = 4: Size = 4, Time = 2.846 msec, Performace = 3018.25 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_SK_4_2(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
		  float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//compute 4*2 elements
	float2 c0 = F32_2_0, c1 = F32_2_0;
	float2 c2 = F32_2_0, c3 = F32_2_0;

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
	const int X = ((blockIdx.x << LB) + tx) << 1; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area-------------------------------------------------------
	const int A0 = ((tx & 1) << 1) * K + (tx >> 1), A1 = A0 + K;
	const int As_x = (tx >> 1), As_y = ((ty << 1) + (tx & 1));

	const int B0 = (ty & 1) * K + (ty >> 1);
	const int Bs_y = (ty >> 1), Bs_x = ((tx << 1) + (ty & 1));

	As[0][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
	Bs[0][Bs_y][Bs_x] = B[B0];//transpose B
	__syncthreads();

	for (int ok = 1, OK = (K_slice << 1 >> LB); ok < OK; ok++)
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
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
		Bs[buf][Bs_y][Bs_x] = B[B0];//transpose B
		__syncthreads();
	}
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);
		float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}
	A += STEP; B += STEP;//K += STEP

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float4 a;//transpose A
		a.x = get(A, 0, ik, K);
		a.y = get(A, 1, ik, K);
		a.z = get(A, 2, ik, K);
		a.w = get(A, 3, ik, K);

		float2 b;//transpose B
		b.x = get(B, 0, ik, K);
		b.y = get(B, 1, ik, K);

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}
	//when K_slice % STEP != 0-------------------------------------

	const int C1 = C0 + SC;
	const int C2 = C1 + SC;
	const int C3 = C2 + SC;

	*(float2*)(C + C0) = c0;
	*(float2*)(C + C1) = c1;
	*(float2*)(C + C2) = c2;
	*(float2*)(C + C3) = c3;
}

#endif 


//======[Small]===========================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), K >= BLOCK_SIZE
#ifndef MATMUL_T2_SK_KERNEL_2_2
#define MATMUL_T2_SK_KERNEL_2_2

//for [2048 * 1024 * 1024]
//LB = 4: Size = 4, Time = 3.339 msec, Performace = 2572.61 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_SK_2_2(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 2];

	float2 c0 = F32_2_0, c1 = F32_2_0;//compute 2*2 elements

	//======================================================================
	const int bz = blockIdx.z;
	const int K_start = K_slice * bz;
	const int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * Cstride;
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	const int Y = ((blockIdx.y << LB) + ty) << 1; A += Y * K + K_start;
	const int X = ((blockIdx.x << LB) + tx) << 1; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]	

	//compute area----------------------------------------------------------
	const int A0 = tx, A1 = A0 + K;
	const int B0 = ty, B1 = B0 + K;

	As[0][tx][ty] = float2{ A[A0], A[A1] };//transpose A
	Bs[0][ty][tx] = float2{ B[B0], B[B1] };//transpose B
	__syncthreads();

	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = As[buf][ik][ty];
			float2 b = Bs[buf][ik][tx];
			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
		}
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		As[buf][tx][ty] = float2{ A[A0], A[A1] };//transpose A
		Bs[buf][ty][tx] = float2{ B[B0], B[B1] };//transpose B
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = As[buf][ik][ty];
		float2 b = Bs[buf][ik][tx];
		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}
	A += STEP; B += STEP;//K += STEP

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float2 a;//transpose A
		a.x = get(A, 0, ik, K);
		a.y = get(A, 1, ik, K);

		float2 b;//transpose B
		b.x = get(B, 0, ik, K);
		b.y = get(B, 1, ik, K);

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}
	//when K_slice % STEP != 0-------------------------------------

	const int C1 = C0 + SC;
	*(float2*)(C + C0) = c0;
	*(float2*)(C + C1) = c1;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 8
#ifndef MATMUL_T2_SK_KERNEL_8_1
#define MATMUL_T2_SK_KERNEL_8_1

//for[2048 * 1024 * 1024]:
//LB = 4: Size = 4, Time = 2.524 msec, Performace = 3403.3 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_SK_8_1(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	float4 c0 = F32_4_0, c1 = F32_4_0;//compute 8*1 elements

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
	const int X = ((blockIdx.x << LB) + tx); B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area-------------------------------------------------------
	const int A0 = tx, A1 = tx + K, A2 = A1 + K, A3 = A2 + K;
	const int A4 = A3 + K, A5 = A4 + K, A6 = A5 + K, A7 = A6 + K;

	As[0][tx][(ty << 1)    ] = float4{ A[A0], A[A1], A[A2], A[A3] };//transpose A
	As[0][tx][(ty << 1) + 1] = float4{ A[A4], A[A5], A[A6], A[A7] };//transpose A
	Bs[0][ty][tx] = B[ty];//transpose B
	__syncthreads();

	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
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
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		As[buf][tx][(ty << 1)    ] = float4{ A[A0], A[A1], A[A2], A[A3] };//transpose A
		As[buf][tx][(ty << 1) + 1] = float4{ A[A4], A[A5], A[A6], A[A7] };//transpose A
		Bs[buf][ty][tx] = B[ty];//transpose B
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
	A += STEP; B += STEP;//K += STEP

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float4 a0, a1;//transpose A
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);
		a1.x = get(A, 4, ik, K);
		a1.y = get(A, 5, ik, K);
		a1.z = get(A, 6, ik, K);
		a1.w = get(A, 7, ik, K);

		float b = get(B, 0, ik, K);//transpose B

		simdMM4(c0, b, a0);
		simdMM4(c1, b, a1);
	}
	//when K_slice % STEP != 0-------------------------------------

	const int C1 = C0 + SC, C2 = C1 + SC, C3 = C2 + SC;
	const int C4 = C3 + SC, C5 = C4 + SC, C6 = C5 + SC, C7 = C6 + SC;

	C[C0] = c0.x; C[C1] = c0.y; C[C2] = c0.z; C[C3] = c0.w;
	C[C4] = c1.x; C[C5] = c1.y; C[C6] = c1.z; C[C7] = c1.w;
}

#endif


//(Y: BLOCK_SIZE  , X: BLOCK_SIZE*8), K >= BLOCK_SIZE
//LB = 4: K >= 8
#ifndef MATMUL_T2_SK_KERNEL_1_8
#define MATMUL_T2_SK_KERNEL_1_8

//for[2048 * 1024 * 1024]:
//LB = 4: Size = 4, Time = 6.644 msec, Performace = 1292.89 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_SK_1_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	      float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(2 << LB) + 1];

	float4 c0 = F32_4_0, c1 = F32_4_0;//compute 1*8 elements

	//======================================================================
	const int bz = blockIdx.z;
	const int K_start = K_slice * bz;
	const int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * Cstride;
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	const int Y = ((blockIdx.y << LB) + ty); A += Y * K + K_start;
	const int X = ((blockIdx.x << LB) + tx) << 3; B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute area----------------------------------------------------------
	const int B0 = ty, B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;
	const int B4 = B3 + K, B5 = B4 + K, B6 = B5 + K, B7 = B6 + K;

	Bs[0][ty][(tx << 1)    ] = float4{ B[B0], B[B1], B[B2], B[B3] };//transpose B
	Bs[0][ty][(tx << 1) + 1] = float4{ B[B4], B[B5], B[B6], B[B7] };
	As[0][tx][ty] = A[tx];//transpose A
	__syncthreads();

	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Bs[buf][ik][(tx << 1)];
			float4 b1 = Bs[buf][ik][(tx << 1) + 1];
			float a = As[buf][ik][ty];

			simdMM4(c0, a, b0);
			simdMM4(c1, a, b1);
		}
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		Bs[buf][ty][(tx << 1)    ] = float4{ B[B0], B[B1], B[B2], B[B3] };//transpose B
		Bs[buf][ty][(tx << 1) + 1] = float4{ B[B4], B[B5], B[B6], B[B7] };
		As[buf][tx][ty] = A[tx];//transpose A
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Bs[buf][ik][(tx << 1)];
		float4 b1 = Bs[buf][ik][(tx << 1) + 1];
		float a = As[buf][ik][ty];

		simdMM4(c0, a, b0);
		simdMM4(c1, a, b1);
	}
	A += STEP; B += STEP;//K += STEP

	//when K % STEP_slice != 0-------------------------------------
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++)
	{
		float4 b0, b1;//transpose B
		b0.x = get(B, 0, ik, K);
		b0.y = get(B, 1, ik, K);
		b0.z = get(B, 2, ik, K);
		b0.w = get(B, 3, ik, K);
		b1.x = get(B, 4, ik, K);
		b1.y = get(B, 5, ik, K);
		b1.z = get(B, 6, ik, K);
		b1.w = get(B, 7, ik, K);

		float a = A[ik];//transpose A

		simdMM4(c1, a, b1);
		simdMM4(c0, a, b0);
	}
	//when K % STEP_slice != 0-------------------------------------

	*(float4*)(C + C0) = c0; *(float4*)(C + C0 + 4) = c1;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 8
#ifndef MATMUL_T2_SK_KERNEL_1_1
#define MATMUL_T2_SK_KERNEL_1_1

//for [2048 * 1024 * 1024]
//LB = 4: Size = 4, Time = 9.94 msec, Performace = 864.178 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_SK_1_1(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
		  float* __restrict__ Cbuf,
	int SC, int K,
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float As[2][1 << LB][(1 << LB) + 1];
	__shared__ float Bs[2][1 << LB][(1 << LB) + 1];

	float c0 = 0.0f;//compute 1*1 element

	//======================================================================
	const int bz = blockIdx.z;
	const int K_start = K_slice * bz;
	const int K_end = IF_int((bz != (gridDim.z - 1)), (K_start + K_slice), K);
	K_slice = K_end - K_start;

	//dst[bz] = Cbuf[bz - 1, sizeW], bz >= 1, dst[0] = C
	Cbuf += (bz - 1) * Cstride;
	C = IF_int((bz != 0), Cbuf, C);//deltaW -> dst
	//======================================================================

	const int Y = ((blockIdx.y << LB) + ty); A += Y * K + K_start;
	const int X = ((blockIdx.x << LB) + tx); B += X * K + K_start;
	const int C0 = Y * SC + X;//C[Y, X]	

	//compute area-------------------------------------------------------
	As[0][tx][ty] = A[tx];//transpose A
	Bs[0][ty][tx] = B[ty];//transpose B
	__syncthreads();

	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float a = As[buf][ik][ty];
			float b = Bs[buf][ik][tx];
			c0 += a * b;
		}
		buf ^= 1; A += STEP; B += STEP;//K += STEP

		As[buf][tx][ty] = A[tx];//transpose A
		Bs[buf][ty][tx] = B[ty];//transpose B
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float a = As[buf][ik][ty];
		float b = Bs[buf][ik][tx];
		c0 += a * b;
	}
	A += STEP; B += STEP;//K += STEP

	//when K_slice % STEP != 0-------------------------------------
	for (int ik = 0, RK = K_slice & (STEP - 1); ik < RK; ik++) {
		float a = A[ik];//transpose A
		float b = B[ik];//transpose B
		c0 += a * b;
	}
	//when K % STEP_slice != 0-------------------------------------

	C[C0] = c0;
}

#endif

#endif
