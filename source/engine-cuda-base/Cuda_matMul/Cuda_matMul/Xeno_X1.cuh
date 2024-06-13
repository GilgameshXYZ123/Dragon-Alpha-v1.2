
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#ifndef GVY2_KERNEL1
#define GVY2_KERNEL1

#define	gvy_k1(LB, stream, A, B, C, N, M, K, SC) \
	GVY_kernel1<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)


//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 1.965 msec, Performace = 8742.94 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void GVY_kernel1(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute 8*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	//compute area-------------------------------------------------------
	const int Ax = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	const int A0 = (Y + ((tx >= STEP) << 2))*K + Ax;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int By = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	const int B0 = (X + ((ty >= STEP) << 2))*K + By;
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[0][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[0][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elem from B
	float2 b0 = *(float2*)(B + B0);
	float2 b1 = *(float2*)(B + B1);
	float2 b2 = *(float2*)(B + B2);
	float2 b3 = *(float2*)(B + B3);
	Bs[buf][(ty << 1)][tx] = float4{ b0.x, b1.x, b2.x, b3.x };
	Bs[buf][(ty << 1) + 1][tx] = float4{ b0.y, b1.y, b2.y, b3.y };
	__syncthreads();

	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];

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
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#ifndef GVY2_KERNEL2
#define GVY2_KERNEL2

#define	gvy_k2(LB, stream, A, B, C, N, M, K, SC) \
	GVY_kernel2<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 1.965 msec, Performace = 8742.94 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void GVY_kernel2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute 8*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	//compute area-------------------------------------------------------
	const int A0 = (Y + ((tx / STEP) << 2))*K + ((tx & STEP_m1) << 1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int B0 = (X + ((ty / STEP) << 2))*K + ((ty & STEP_m1) << 1);
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[0][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[0][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elem from B
	float2 b0 = *(float2*)(B + B0);
	float2 b1 = *(float2*)(B + B1);
	float2 b2 = *(float2*)(B + B2);
	float2 b3 = *(float2*)(B + B3);
	Bs[buf][(ty << 1)][tx] = float4{ b0.x, b1.x, b2.x, b3.x };
	Bs[buf][(ty << 1) + 1][tx] = float4{ b0.y, b1.y, b2.y, b3.y };
	__syncthreads();

	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];

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
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#define GVY2_KERNEL3
#ifndef GVY2_KERNEL3
#define GVY2_KERNEL3

#define	gvy_k3(LB, stream, A, B, C, N, M, K, SC) \
	GVY_kernel3<LB, (1<<LB>>2), (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 1.965 msec, Performace = 8742.94 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP2_m1>
__global__ void GVY_kernel3(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SC + X;//C[Y, X]

	//compute 8*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	//compute area-------------------------------------------------------
	const int vtx = tx & STEP2_m1;
	const int vty = ty & STEP2_m1;
	const int Ak = ((vtx << 1) + (vtx >> 1)) << 1;
	const int Bk = ((vty >> 1) + (vty >> 1)) << 1;

	const int A0 = (Y + ((tx / STEP) << 1))*K + Ak;
	const int A1 = A0 + K;

	const int B0 = (X + ((ty / STEP) << 1))*K + Bk;
	const int B1 = B0 + K;

	const int As_x = (tx >> 1) << 2;
	const int Bs_y = (ty >> 1) << 2;

	//load 4 elem from A(transposed)
	float4 a0 = *(float4*)(A + A0);
	float4 a1 = *(float4*)(A + A1);
	As[0][Asx + 0][ty] = float2{ a0.x, a1.x };
	As[0][Asx + 1][ty] = float4{ a0.y, a1.y };
	As[0][Asx + 2][ty] = float2{ a0.z, a1.z };
	As[0][Asx + 3][ty] = float4{ a0.w, a1.w };

	//load 4 elem from B
	float4 b0 = *(float4*)(B + B0);
	float4 b1 = *(float4*)(B + B1);
	Bs[buf][(ty << 1) + 0][tx] = float4{ b0.x, b1.x, b2.x, b3.x };
	Bs[buf][(ty << 1) + 1][tx] = float4{ b0.y, b1.y, b2.y, b3.y };
	Bs[buf][(ty << 1) + 0][tx] = float4{ b0.z, b1.z, b2.z, b3.z };
	Bs[buf][(ty << 1) + 1][tx] = float4{ b0.w, b1.w, b2.w, b3.w };
	__syncthreads();

	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];

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
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#ifndef GVY2_KERNEL4
#define GVY2_KERNEL4

#define	gvy_k4(LB, stream, A, B, C, N, M, K, SC) \
	GVY_kernel4<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 2.082 msec, Performace = 8251.62 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void GVY_kernel4(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	      float*  __restrict__ C,
	int SC, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	const int Y = ((blockIdx.y << LB) + ty) << 3; 
	const int X = ((blockIdx.x << LB) + tx) << 3; 
	const int C0 = Y * SC + X;//C[Y, X]
	A += (Y + ((tx >= STEP) << 2) + (tx & 3)) * K;
	B += (X + ((ty >= STEP) << 2)) * K;

	//compute area-------------------------------------------------------
	const int A0 = (tx - ((tx >= STEP) << LB >> 1)) >> 2 << 2;
	const int As_x = A0 + ((tx >= STEP) << LB >> 1);

	const int B0 = ty - ((ty >= STEP) << LB >> 1);
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	float4 av = *(float4*)(A + A0);//transpose A
	*((float*)(&As[buf][As_x + 0][ty]) + (tx & 3)) = av.x;
	*((float*)(&As[buf][As_x + 1][ty]) + (tx & 3)) = av.y;
	*((float*)(&As[buf][As_x + 2][ty]) + (tx & 3)) = av.z;
	*((float*)(&As[buf][As_x + 3][ty]) + (tx & 3)) = av.w;

	Bs[buf][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };//transposed B
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

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
		*((float*)(&As[buf][As_x + 0][ty]) + (tx & 3)) = av.x;
		*((float*)(&As[buf][As_x + 1][ty]) + (tx & 3)) = av.y;
		*((float*)(&As[buf][As_x + 2][ty]) + (tx & 3)) = av.z;
		*((float*)(&As[buf][As_x + 3][ty]) + (tx & 3)) = av.w;

		Bs[buf][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };//transposed B
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
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


#ifndef GVY2_KERNEL5
#define GVY2_KERNEL5

#define	gvy_k5(LB, stream, A, B, C, N, M, K, SC) \
	GVY_kernel5<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 1.979 msec, Performace = 8681.08 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void GVY_kernel5(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	      float*  __restrict__ C,
	int SC, int K)
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
	const int C0 = Y * SC + X;//C[Y, X]
	A += (Y + ((tx >= STEP) << 2) + (tx & 3)) * K;
	B += (X + ((ty >= STEP) << 2)) * K;

	//compute area-------------------------------------------------------
	const int A0 = (tx - ((tx >= STEP) << LB >> 1)) >> 2 << 2;
	const int As_x = A0 + ((tx >= STEP) << LB >> 1);
	const int As_y = (ty << 2) + (tx & 3);

	const int B0 = ty - ((ty >= STEP) << LB >> 1);
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	float4 av = *(float4*)(A + A0);//transpose A
	As[buf][As_x    ][As_y] = av.x;
	As[buf][As_x + 1][As_y] = av.y;
	As[buf][As_x + 2][As_y] = av.z;
	As[buf][As_x + 3][As_y] = av.w;

	Bs[buf][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };//transposed B
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&As[buf][ik][ty << 2]);
			float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

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

		Bs[buf][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };//transposed B
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


#ifndef GVY2_KERNEL6
#define GVY2_KERNEL6

#define	gvy_k6(LB, stream, A, B, C, N, M, K, SC) \
	GVY_kernel6<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 1.785 msec, Performace = 9624.58 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void GVY_kernel6(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	      float*  __restrict__ C,
	int SC, int K)
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

	const int Y = ((blockIdx.y << LB) + ty) << 3; 
	const int X = ((blockIdx.x << LB) + tx) << 3; 
	const int C0 = Y * SC + X;//C[Y, X]
	A += (Y + ((tx >= STEP) << 2) + (tx & 3)) * K;
	B += (X + ((ty >= STEP) << 2) + (ty & 3)) * K;

	//compute area-------------------------------------------------------
	const int A0 = (tx - ((tx >= STEP) << LB >> 1)) >> 2 << 2;
	const int As_x = A0 + ((tx >= STEP) << LB >> 1);
	const int As_y = (ty << 2) + (tx & 3);

	const int B0 = (ty - ((ty >= STEP) << LB >> 1)) >> 2 << 2;
	const int Bs_x = B0 + ((ty >= STEP) << LB >> 1);
	const int Bs_y = (tx << 2) + (ty & 3);

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

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&As[buf][ik       ][ty << 2]);
			float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
			float4 b0 = *(float4*)(&Bs[buf][ik       ][tx << 2]);
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
		float4 a0 = *(float4*)(&As[buf][ik       ][ty << 2]);
		float4 a1 = *(float4*)(&As[buf][ik + STEP][ty << 2]);
		float4 b0 = *(float4*)(&Bs[buf][ik       ][tx << 2]);
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


#ifndef GVY2_KERNEL7
#define GVY2_KERNEL7

#define	gvy_k7(LB, stream, A, B, C, N, M, K, SC) \
	GVY_kernel7<LB, (1<<LB>>1), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//for [2048 * 2048 * 2048]:
//LB = 4: Size = 8, Time = 1.785 msec, Performace = 9624.58 GFlop/s
template<int LB, int STEP, int STEP_m1>
__global__ void GVY_kernel7(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
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

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SC + X;//C[Y, X]
	A += (Y + ((tx >= STEP) << 2) + (tx & 3)) * K;
	B += (X + ((ty >= STEP) << 2) + (ty & 3)) * K;

	//compute area-------------------------------------------------------
	const int A0 = (tx & STEP_m1) >> 2 << 2;
	const int As_x = A0 + ((tx >= STEP) << LB >> 1);
	const int As_y = (ty << 2) + (tx & 3);

	const int B0 = (ty & STEP_m1) >> 2 << 2;
	const int Bs_x = B0 + ((ty >= STEP) << LB >> 1);
	const int Bs_y = (tx << 2) + (ty & 3);

	float4 av = *(float4*)(A + A0);//transpose A
	As[0][As_x    ][As_y] = av.x;
	As[0][As_x + 1][As_y] = av.y;
	As[0][As_x + 2][As_y] = av.z;
	As[0][As_x + 3][As_y] = av.w;

	float4 bv = *(float4*)(B + B0);//transpose B
	Bs[0][Bs_x    ][Bs_y] = bv.x;
	Bs[0][Bs_x + 1][Bs_y] = bv.y;
	Bs[0][Bs_x + 2][Bs_y] = bv.z;
	Bs[0][Bs_x + 3][Bs_y] = bv.w;
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
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