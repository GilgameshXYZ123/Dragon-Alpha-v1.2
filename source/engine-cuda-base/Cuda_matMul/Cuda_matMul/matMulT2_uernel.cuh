#pragma once

#ifndef MATMUL_T2_UERNEL_H
#define MATMUL_T2_UERNEL_H

//B   belongs to Mat[M, K]
//B^T belongs to Mat[K, M]
//get(B^T, k, j, M) = get(B, j, k, K)
//for the first stack of function:
//SA = SB = K
//SC = M
#ifndef MATMUL_T2_UERNEL_CALL
#define MATMUL_T2_UERNEL_CALL

//LB = log2(BLOCK_SIZE)

#define	u88T2_mgk(LB, stream, A, B, C, N, M, K, SC) \
	uernel_t2_8_8_mgk<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB>>1, 2<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#define	u84T2_mgk(LB, stream, A, B, C, N, M, K, SC) \
	uernel_t2_8_4_mgk<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>2>>LB, N>>3>>LB), dim3(1<<LB>>1, 2<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#define	u48T2_mgk(LB, stream, A, B, C, N, M, K, SC) \
	uernel_t2_4_8_mgk<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(N>>2>>LB, M>>3>>LB), dim3(1<<LB>>1, 2<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#define	u44T2_mgk(LB, stream, A, B, C, N, M, K, SC) \
	uernel_t2_4_4_mgk<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(N>>2>>LB, M>>2>>LB), dim3(1<<LB>>1, 2<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0, Good
//LB = 3: K %  8 == 0, Bad
#ifndef MATMUL_T2_UERNEL_8_8_MGK
#define MATMUL_T2_UERNEL_8_8_MGK

//for (N, M, K) = (2048, 2048, 2048):
//LB = 4: Size = 8, Time = 1.598 msec, Performace = 10750.9 GFlop/s
//LB = 3: Size = 8, Time = 2.281 msec, Performace = 7531.73 GFlop/s
//for (N, M, K) = (512, 4096, 4096):
//LB = 4: Size = 8, Time = 1.881 msec, Performace = 9133.37 GFlop/s
//LB = 3: Size = 8, Time = 4.474 msec, Performace = 3839.93 GFlop/s
//for (N, M, K) = (4096, 4096, 4096):
//LB = 4: Size = 64, Time = 13.248 msec, Performace = 10374.3 GFlop/s
//for (N, M, K) = (512, 8192, 4096):
//LB = 4: Size = 16, Time = 3.561 msec, Performace = 9648.9 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_t2_8_8_mgk(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	      float*  __restrict__ C,
	int SC, int K)
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

	const int bY = (by << LB << 3);
	const int bX = (bx << LB << 3);
	
	//prepare for C[N, M]
	const int idx = (ty << LB >> 1) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = ((vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1)) << 1;
	const int uy = ((vy >> 2) * 8 + ((vx & 15) >> 1))        << 1;

	const int cY = bY + (uy << 2);
	const int cX = bX + (ux << 2);
	const int C0 = cY * SC + cX;
	
	//compute area-------------------------------------------------------
	const int tY = bY + (ty << 2); A += tY * K;
	const int tX = bX + (ty << 2); B += tX * K;

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

	for (int ok = STEP2; ok < K; ok += STEP2) {
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
		buf ^= 1; A += STEP2; B += STEP2;//K += STEP

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#ifndef MATMUL_T2_UERNEL_8_4_MGK
#define MATMUL_T2_UERNEL_8_4_MGK

//for (N, M, K) = (2048, 2048, 2048):
//LB = 4: Size = 8, Time = 2.059 msec, Performace = 8343.79 GFlop/s
//LB = 4: Size = 8, Time = 4.241 msec, Performace = 4050.9  GFlop/s
//for (N, M, K) = (2048, 2048, 2048):
//LB = 4: Size = 8, Time = 2.081 msec, Performace = 8255.58 GFlop/s
//for (N, M, K) = (512, 8192, 4096):
//LB = 4: Size = 16, Time = 4.26 msec, Performace = 8065.67 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_t2_8_4_mgk(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SC, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];

	//compute 8*4 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;

	const int bY = (blockIdx.y << LB << 3);
	const int bX = (blockIdx.x << LB << 2);
	const int tY = bY + (ty << 2); A += tY * K;
	const int tX = bX + (ty << 1); B += tX * K;
	
	//prepare for C[N, M]
	const int idx = (ty << LB >> 1) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = ((vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1)) << 1;
	const int uy = ((vy >> 2) * 8 + ((vx & 15) >> 1))        << 1;

	const int cY = bY + (uy << 2);
	const int cX = bX + (ux << 1);
	const int C0 = cY * SC + cX;
	
	//compute area-------------------------------------------------------
	//load 4 elem from A(transposed), 2 elem from B
	const int k0 = (tx << 1);
	const int k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;
	float2 a0 = *(float2*)(A + k0);
	float2 a1 = *(float2*)(A + k1);
	float2 a2 = *(float2*)(A + k2);
	float2 a3 = *(float2*)(A + k3);

	float2 b0 = *(float2*)(B + k0);
	float2 b1 = *(float2*)(B + k1);

	//write to shared memory
	As[0][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[0][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };
	Bs[0][(tx << 1)    ][ty] = float2{ b0.x, b1.x };
	Bs[0][(tx << 1) + 1][ty] = float2{ b0.y, b1.y };
	__syncthreads();

	for (int ok = STEP2; ok < K; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 b0 = *(float4*)(&Bs[buf][ik][ux]);
			float4 a0 = As[buf][ik][uy], a1 = As[buf][ik][uy + 1];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.y, b0);
			simdMM4(c2, a0.z, b0); simdMM4(c3, a0.w, b0);
			simdMM4(c4, a1.x, b0); simdMM4(c5, a1.y, b0);
			simdMM4(c6, a1.z, b0); simdMM4(c7, a1.w, b0);
		}
		buf ^= 1;

		//load 4 elem from A(transposed), 4 elem from B
		const int k0 = (tx << 1) + ok; 
		const int k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;
		float2 a0 = *(float2*)(A + k0);
		float2 a1 = *(float2*)(A + k1); 
		float2 a2 = *(float2*)(A + k2); 
		float2 a3 = *(float2*)(A + k3);

		float2 b0 = *(float2*)(B + k0);
		float2 b1 = *(float2*)(B + k1);

		//write to shared memory
		As[buf][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };
		Bs[buf][(tx << 1)    ][ty] = float2{ b0.x, b1.x };
		Bs[buf][(tx << 1) + 1][ty] = float2{ b0.y, b1.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 b0 = *(float4*)(&Bs[buf][ik][ux]);
		float4 a0 = As[buf][ik][uy], a1 = As[buf][ik][uy + 1];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.y, b0);
		simdMM4(c2, a0.z, b0); simdMM4(c3, a0.w, b0);
		simdMM4(c4, a1.x, b0); simdMM4(c5, a1.y, b0);
		simdMM4(c6, a1.z, b0); simdMM4(c7, a1.w, b0);
	}

	const int C1 = C0 + SC, C2 = C1 + SC, C3 = C2 + SC;
	const int C4 = C3 + SC, C5 = C4 + SC, C6 = C5 + SC, C7 = C6 + SC;

	*(float4*)(C + C0) = c0; *(float4*)(C + C1) = c1;
	*(float4*)(C + C2) = c2; *(float4*)(C + C3) = c3;
	*(float4*)(C + C4) = c4; *(float4*)(C + C5) = c5;
	*(float4*)(C + C6) = c6; *(float4*)(C + C7) = c7;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#ifndef MATMUL_T2_UERNEL_4_8_MGK
#define MATMUL_T2_UERNEL_4_8_MGK

//for (N, M, K) = (2048, 2048, 2048):
//LB = 4: Size = 8, Time = 2.026 msec, Performace = 8479.7 GFlop/s
//for (N, M, K) = (512, 4096, 4096):
//LB = 4: Size = 8, Time = 1.993 msec, Performace = 8620.1 GFlop/s
//for (N, M, K) = (512, 8192, 4096):
//LB = 4: Size = 16, Time = 4.077 msec, Performace = 8427.7 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_t2_4_8_mgk(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SC, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];
	__shared__ float4 Bs[2][1 << LB][(2 << LB) + 1];

	//compute 4*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;

	const int bY = (blockIdx.x << LB << 2);
	const int bX = (blockIdx.y << LB << 3);
	const int tY = bY + (ty << 1); A += tY * K;
	const int tX = bX + (ty << 2); B += tX * K;
	
	//prepare for C[N, M]
	const int idx = (ty << LB >> 1) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = ((vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1)) << 1;
	const int uy = ((vy >> 2) * 8 + ((vx & 15) >> 1))        << 1;

	const int cY = bY + (uy << 1);
	const int cX = bX + (ux << 2);
	const int C0 = cY * SC + cX;
	
	//compute area-------------------------------------------------------
	//load 4 elem from A(transposed), 4 elem from B
	const int k0 = (tx << 1);
	const int k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;
	float2 b0 = *(float2*)(B + k0);
	float2 b1 = *(float2*)(B + k1);
	float2 b2 = *(float2*)(B + k2);
	float2 b3 = *(float2*)(B + k3);

	float2 a0 = *(float2*)(A + k0);
	float2 a1 = *(float2*)(A + k1);

	//write to shared memory
	Bs[0][(tx << 1)    ][ty] = float4{ b0.x, b1.x, b2.x, b3.x };
	Bs[0][(tx << 1) + 1][ty] = float4{ b0.y, b1.y, b2.y, b3.y };
	As[0][(tx << 1)    ][ty] = float2{ a0.x, a1.x };
	As[0][(tx << 1) + 1][ty] = float2{ a0.y, a1.y };
	__syncthreads();

	for (int ok = STEP2; ok < K; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 a0 = *(float4*)(&As[buf][ik][uy]);
			float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik][ux + 1];

			simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
		}
		buf ^= 1;

		//load 4 elem from A(transposed), 4 elem from B
		const int k0 = (tx << 1) + ok; 
		const int k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;
		float2 b0 = *(float2*)(B + k0);
		float2 b1 = *(float2*)(B + k1);
		float2 b2 = *(float2*)(B + k2);
		float2 b3 = *(float2*)(B + k3);

		float2 a0 = *(float2*)(A + k0);
		float2 a1 = *(float2*)(A + k1); 

		//write to shared memory
		Bs[buf][(tx << 1)    ][ty] = float4{ b0.x, b1.x, b2.x, b3.x };
		Bs[buf][(tx << 1) + 1][ty] = float4{ b0.y, b1.y, b2.y, b3.y };
		As[buf][(tx << 1)][ty] = float2{ a0.x, a1.x };
		As[buf][(tx << 1) + 1][ty] = float2{ a0.y, a1.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik][ux + 1];
		float4 a0 = *(float4*)(&As[buf][ik][uy]);

		simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
	}

	const int C1 = C0 + SC, C2 = C1 + SC, C3 = C2 + SC;
	*(float4*)(C + C0) = c0; *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2; *(float4*)(C + C1 + 4) = c3;
	*(float4*)(C + C2) = c4; *(float4*)(C + C2 + 4) = c5;
	*(float4*)(C + C3) = c6; *(float4*)(C + C3 + 4) = c7;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#ifndef MATMUL_T2_UERNEL_4_4_MGK
#define MATMUL_T2_UERNEL_4_4_MGK

//for (N, M, K) = (2048, 2048, 2048):
//LB = 4: Size = 8, Time = 2.19 msec, Performace = 7844.69 GFlop/s
//for (N, M, K) = (512, 4096, 4096):
//LB = 4: Size = 8, Time = 2.108 msec, Performace = 8149.84 GFlop/s
//for (N, M, K) = (512, 8192, 4096):
//LB = 4: Size = 16, Time = 4.233 msec, Performace = 8117.11 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_t2_4_4_mgk(
	const float* __restrict__ A,
	const float* __restrict__ B,
	      float* __restrict__ C,
	int SC, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];

	//compute 4*4 elements
	float4 c0 = F32_4_0, c1 = F32_4_0;
	float4 c2 = F32_4_0, c3 = F32_4_0;

	const int bY = (blockIdx.x << LB << 2);
	const int bX = (blockIdx.y << LB << 2);
	const int tY = bY + (ty << 1); A += tY * K;
	const int tX = bX + (ty << 1); B += tX * K;
	
	//prepare for C[N, M]
	const int idx = (ty << LB >> 1) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = ((vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1)) << 1;
	const int uy = ((vy >> 2) * 8 + ((vx & 15) >> 1))        << 1;

	const int cY = bY + (uy << 1);
	const int cX = bX + (ux << 1);
	const int C0 = cY * SC + cX;
	
	//compute area-------------------------------------------------------
	//load 2 elem from A(transposed), 2 elem from B
	const int k0 = (tx << 1), k1 = k0 + K;
	float2 b0 = *(float2*)(B + k0);
	float2 b1 = *(float2*)(B + k1);
	float2 a0 = *(float2*)(A + k0);
	float2 a1 = *(float2*)(A + k1);

	//write to shared memory
	Bs[0][(tx << 1)    ][ty] = float2{ b0.x, b1.x };
	Bs[0][(tx << 1) + 1][ty] = float2{ b0.y, b1.y };
	As[0][(tx << 1)    ][ty] = float2{ a0.x, a1.x };
	As[0][(tx << 1) + 1][ty] = float2{ a0.y, a1.y };
	__syncthreads();

	for (int ok = STEP2; ok < K; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 b = *(float4*)(&Bs[buf][ik][ux]);
			float4 a = *(float4*)(&As[buf][ik][uy]);

			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
			simdMM4(c2, a.z, b);
			simdMM4(c3, a.w, b);
		}
		buf ^= 1;

		//load 2 elem from A(transposed), 2 elem from B
		const int k0 = (tx << 1) + ok, k1 = k0 + K;
		float2 b0 = *(float2*)(B + k0);
		float2 b1 = *(float2*)(B + k1);
		float2 a0 = *(float2*)(A + k0);
		float2 a1 = *(float2*)(A + k1); 

		//write to shared memory
		Bs[buf][(tx << 1)    ][ty] = float2{ b0.x, b1.x };
		Bs[buf][(tx << 1) + 1][ty] = float2{ b0.y, b1.y };
		As[buf][(tx << 1)    ][ty] = float2{ a0.x, a1.x };
		As[buf][(tx << 1) + 1][ty] = float2{ a0.y, a1.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 b = *(float4*)(&Bs[buf][ik][ux]);
		float4 a = *(float4*)(&As[buf][ik][uy]);

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}

	const int C1 = C0 + SC, C2 = C1 + SC, C3 = C2 + SC;
	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c1; 
	*(float4*)(C + C2) = c2;
	*(float4*)(C + C3) = c3;
}

#endif

#endif