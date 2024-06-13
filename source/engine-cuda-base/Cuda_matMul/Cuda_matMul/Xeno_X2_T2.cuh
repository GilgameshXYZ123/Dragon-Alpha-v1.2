


#ifndef NEWMAT_T2_UERNEL_8_8_MGK1
#define NEWMAT_T2_UERNEL_8_8_MGK1

#define	new88T2_mgk_k1(LB, stream, A, B, C, N, M, K, SC) \
	new_t2_8_8_mgk_kernel1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 2.002 msec, Performace = 8581.35 GFlop/s
//LB = 3: Size = 8, Time = 2.281 msec, Performace = 7531.73 GFlop/s
//for: [4096 * 4096 * 4096]
//LB = 4: Size = 64, Time = 16.881 msec, Performace = 8141.63 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void new_t2_8_8_mgk_kernel1(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	      float*  __restrict__ C,
	int SC, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4  c0 = F32_4_0,  c1 = F32_4_0,  c2 = F32_4_0,  c3 = F32_4_0;
	float4  c4 = F32_4_0,  c5 = F32_4_0,  c6 = F32_4_0,  c7 = F32_4_0;
	float4  c8 = F32_4_0,  c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	const int bY = (blockIdx.y << LB << 3);
	const int bX = (blockIdx.x << LB << 3);
	const int tY = bY + (ty << 3) + ((tx >= STEP) << 2);
	const int tX = bX + (tx << 3) + ((ty >= STEP) << 2);

	A += tY * K;
	B += tX * K;
	
	//prepare for C[N, M]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int uy = ((idx & 1) + ((idx >> (LB + 1)) << 1));//((idx & 1) + ((idx / 32) * 2))
	const int ux = ((idx & ((1 << (LB + 1)) - 1)) >> 1);//(idx % 32) / 2

	const int cY = bY + (uy << 3);
	const int cX = bX + (ux << 3);
	const int C0 = cY * SC + cX;
	
	//compute area-------------------------------------------------------
	const int A0 = ((tx & STEP_m1) << 1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int B0 =  ((ty & STEP_m1) << 1);
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[0][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[0][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elem from B
	float2 b0 = *(float2*)(B + B0);
	float2 b1 = *(float2*)(B + B1);
	float2 b2 = *(float2*)(B + B2);
	float2 b3 = *(float2*)(B + B3);
	Bs[buf][(ty << 1)    ][tx] = float4{ b0.x, b1.x, b2.x, b3.x };
	Bs[buf][(ty << 1) + 1][tx] = float4{ b0.y, b1.y, b2.y, b3.y };
	__syncthreads();

	for (int ok = STEP2; ok < K; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 a0 = As[buf][ik][uy], a1 = As[buf][ik + STEP2][uy];
			float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik + STEP2][ux];

			simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP2; B += STEP2;//K += STEP

		//load 4 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 4 elem from B
		float2 b0 = *(float2*)(B + B0);
		float2 b1 = *(float2*)(B + B1);
		float2 b2 = *(float2*)(B + B2);
		float2 b3 = *(float2*)(B + B3);
		Bs[buf][(ty << 1)][tx] = float4{ b0.x, b1.x, b2.x, b3.x };
		Bs[buf][(ty << 1) + 1][tx] = float4{ b0.y, b1.y, b2.y, b3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 a0 = As[buf][ik][uy], a1 = As[buf][ik + STEP2][uy];
		float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik + STEP2][ux];

		simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
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



#ifndef NEWMAT_T2_UERNEL_8_8_MGK2
#define NEWMAT_T2_UERNEL_8_8_MGK2

#define	new88T2_mgk_k2(LB, stream, A, B, C, N, M, K, SC) \
	new_t2_8_8_mgk_kernel2<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB>>1, 2<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 1.654 msec, Performace = 10386.9 GFlop/s
//LB = 3: Size = 8, Time = 2.281 msec, Performace = 7531.73 GFlop/s
//for: [4096 * 4096 * 4096]
//LB = 4: Size = 64, Time = 13.173 msec, Performace = 10433.4 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void new_t2_8_8_mgk_kernel2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	      float*  __restrict__ C,
	int SC, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(2 << LB) + 1];

	//compute 8*8 elements
	float4  c0 = F32_4_0,  c1 = F32_4_0,  c2 = F32_4_0,  c3 = F32_4_0;
	float4  c4 = F32_4_0,  c5 = F32_4_0,  c6 = F32_4_0,  c7 = F32_4_0;
	float4  c8 = F32_4_0,  c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	const int bY = (blockIdx.y << LB << 3);
	const int bX = (blockIdx.x << LB << 3);
	const int tY = bY + (ty << 2);
	const int tX = bX + (ty << 2);
	A += tY * K;
	B += tX * K;
	
	//prepare for C[N, M]
	const int idx = (ty << LB >> 1) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = ((vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1)) << 1;
	const int uy = ((vy >> 2) * 8 + ((vx & 15) >> 1))        << 1;

	const int cY = bY + (uy << 2);
	const int cX = bX + (ux << 2);
	const int C0 = cY * SC + cX;
	
	//compute area-------------------------------------------------------
	const int A0 = tx << 1;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int B0 = tx << 1;
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);//tx = 8, ty = 32
	As[0][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[0][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elem from B
	float2 b0 = *(float2*)(B + B0);
	float2 b1 = *(float2*)(B + B1);
	float2 b2 = *(float2*)(B + B2);
	float2 b3 = *(float2*)(B + B3);
	Bs[buf][(tx << 1)    ][ty] = float4{ b0.x, b1.x, b2.x, b3.x };
	Bs[buf][(tx << 1) + 1][ty] = float4{ b0.y, b1.y, b2.y, b3.y };
	__syncthreads();

	for (int ok = STEP2; ok < K; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 a0 = As[buf][ik][uy], a1 = As[buf][ik][uy + 1];
			float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik][ux + 1];

			simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP2; B += STEP2;//K += STEP

		//load 4 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 4 elem from B
		float2 b0 = *(float2*)(B + B0);
		float2 b1 = *(float2*)(B + B1);
		float2 b2 = *(float2*)(B + B2);
		float2 b3 = *(float2*)(B + B3);
		Bs[buf][(tx << 1)    ][ty] = float4{ b0.x, b1.x, b2.x, b3.x };
		Bs[buf][(tx << 1) + 1][ty] = float4{ b0.y, b1.y, b2.y, b3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 a0 = As[buf][ik][uy], a1 = As[buf][ik][uy + 1];
		float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik][ux + 1];

		simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
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



#ifndef NEWMAT_T2_UERNEL_8_8_MGK3
#define NEWMAT_T2_UERNEL_8_8_MGK3

#define	new88T2_mgk_k3(LB, stream, A, B, C, N, M, K, SC) \
	new_t2_8_8_mgk_kernel3<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB>>1, 2<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 1.654 msec, Performace = 10386.9 GFlop/s
//LB = 3: Size = 8, Time = 2.383 msec, Performace = 7209.34 GFlop/s
//for: [4096 * 4096 * 4096]
//LB = 4: Size = 8, Time = 1.644 msec, Performace = 10450 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void new_t2_8_8_mgk_kernel3(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	      float*  __restrict__ C,
	int SC, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(2 << LB) + 1];

	//compute 8*8 elements
	float4  c0 = F32_4_0,  c1 = F32_4_0,  c2 = F32_4_0,  c3 = F32_4_0;
	float4  c4 = F32_4_0,  c5 = F32_4_0,  c6 = F32_4_0,  c7 = F32_4_0;
	float4  c8 = F32_4_0,  c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	const int bY = (blockIdx.y << LB << 3);
	const int bX = (blockIdx.x << LB << 3);
	const int tY = bY + (ty << 2); A += tY * K;
	const int tX = bX + (ty << 2); B += tX * K;
	
	//prepare for C[N, M]
	const int idx = (ty << LB >> 1) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = ((vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1)) << 1;
	const int uy = ((vy >> 2) * 8 + ((vx & 15) >> 1))        << 1;

	const int cY = bY + (uy << 2);
	const int cX = bX + (ux << 2);
	const int C0 = cY * SC + cX;
	
	//compute area-------------------------------------------------------
	//load 4 elem from A(transposed), 4 elem from B
	const int k0 = (tx << 1);
	const int k1 = k0 + K, k2 = k0 + (K << 1), k3 = k0 + K * 3;
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

			simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
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
		float4 a0 = As[buf][ik][uy], a1 = As[buf][ik][uy + 1];
		float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik][ux + 1];

		simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
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
