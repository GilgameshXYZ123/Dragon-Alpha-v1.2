

//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef XXU_KERNEL1
#define XXU_KERNEL1

#define	xxu_kernel1(LB, stream, A, B, C, N, M, K, SB) \
	XXU_kernel1<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//for: 2048*2048*2048
//LB = 4: Size = 8, Time = 1.564 msec, Performace = 10984.6 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void XXU_kernel1(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//compute 8* elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SB + X;
	A += (Y + ((tx >= STEP) << 2)) * K;
	B += (X + ((ty >= STEP) << 2));

	const int B0 = ((ty - ((ty >= STEP) << LB >> 1)) << 1) * SB;
	const int B1 = B0 + SB;

	const int A0 = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[buf][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elem from B
	Bs[buf][(ty << 1)    ][tx] = *(float4*)(B + B0);
	Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);

	A += STEP2; B += (SB << LB);//K += STEP2
	__syncthreads();

	//compute area-------------------------------------------------------
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
		buf ^= 1;

		//load 4 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 4 elem from B
		Bs[buf][(ty << 1)    ][tx] = *(float4*)(B + B0);
		Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);

		A += STEP2; B += (SB << LB);//K += STEP2
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef XXU_KERNEL2
#define XXU_KERNEL2

#define	xxu_kernel2(LB, stream, A, B, C, N, M, K, SB) \
	XXU_kernel2<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//for: 2048*2048*2048
//LB = 4: Size = 8, Time = 1.564 msec, Performace = 10984.6 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void XXU_kernel2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//compute 8* elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	A += (Y + ((tx >= STEP) << 2)) * K;
	B += (X + ((ty >= STEP) << 2));
	const int C0 = Y * SB + X;

	//compute area-------------------------------------------------------
	const int B0 = ((ty & STEP_m1) << 1)*SB, B1 = B0 + SB;
	const int A0 = ((tx & STEP_m1) << 1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elem from B
	Bs[buf][(ty << 1)][tx] = *(float4*)(B + B0);
	Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);

	A += STEP2; B += (SB << LB);//K += STEP2
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
		buf ^= 1;

		//load 4 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 4 elem from B
		Bs[buf][(ty << 1)][tx] = *(float4*)(B + B0);
		Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);

		A += STEP2; B += (SB << LB);//K += STEP2
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef XXU_KERNEL3
#define XXU_KERNEL3

#define	xxu_kernel3(LB, stream, A, B, C, N, M, K, SB) \
	XXU_kernel3<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//for: 2048*2048*2048
//LB = 4: Size = 8, Time = 1.558 msec, Performace = 11026.9 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void XXU_kernel3(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//compute 8* elements
	float4 c0 = F32_4_0, c1 = F32_4_0, c2 = F32_4_0, c3 = F32_4_0;
	float4 c4 = F32_4_0, c5 = F32_4_0, c6 = F32_4_0, c7 = F32_4_0;
	float4 c8 = F32_4_0, c9 = F32_4_0, c10 = F32_4_0, c11 = F32_4_0;
	float4 c12 = F32_4_0, c13 = F32_4_0, c14 = F32_4_0, c15 = F32_4_0;

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SB + X;
	A += (Y + ((tx >= STEP) << 2)) * K;
	B += (X + ((ty >= STEP) << 2));

	const int B0 = ((ty - ((ty >= STEP) << LB >> 1)) << 1) * SB;
	const int B1 = B0 + SB;

	const int A0 = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elem from B
	Bs[buf][(ty << 1)][tx] = *(float4*)(B + B0);
	Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);
	__syncthreads();

	//compute area-------------------------------------------------------
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
		buf ^= 1;
		A += STEP2; B += (SB << LB);//K += STEP2

		//load 4 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 4 elem from B
		Bs[buf][(ty << 1)][tx] = *(float4*)(B + B0);
		Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);
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


//Split K
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef XXU_KERNEL4
#define XXU_KERNEL4

#define	xxu_kernel4(LB, GZ, stream, A, B, C, Cbuf, N, M, K, SB) \
	XXU_kernel4<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1>\
		<<< dim3(M>>3>>LB, N>>3>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, Cbuf, SB, K, K_slice, Cstride)

//for: 2048*2048*2048
//LB = 4: Size = 8, Time = 1.558 msec, Performace = 11026.9 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1>
__global__ void XXU_kernel4(
	const float*  __restrict__ A,//[N, K]
	const float*  __restrict__ B,//[K, M]
	float* __restrict__ C,//[N, M]
	float* __restrict__ Cbuf,//Cbuf[part, N, M]
	int SB, int K, 
	int K_slice, int Cstride)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//compute 8* elements
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
	const int C0 = Y * SB + X;
	A += (Y + ((tx >= STEP) << 2)) * K + K_start;
	B += (X + ((ty >= STEP) << 2)) + K_start * SB;

	const int B0 = ((ty - ((ty >= STEP) << LB >> 1)) << 1) * SB;
	const int B1 = B0 + SB;

	const int A0 = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elem from B
	Bs[buf][(ty << 1)][tx] = *(float4*)(B + B0);
	Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);
	__syncthreads();

	//compute area-------------------------------------------------------
	for (int ok = 1, OK = (K_slice >> LB); ok < OK; ok++)
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
		buf ^= 1; A += STEP2; B += (SB << LB);//K += STEP2

		//load 4 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 4 elem from B
		Bs[buf][(ty << 1)][tx] = *(float4*)(B + B0);
		Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);
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
#ifndef XXU_KERNEL5
#define XXU_KERNEL5

#define	xxu_kernel5(LB, stream, A, B, C, N, M, K, SB) \
	XXU_Kernel5<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//Size = 8, Time = 1.616 msec, Performace = 10631.1 GFlop/s
template<int LB, int STEP>
__global__ void XXU_Kernel5(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
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

	const int Y = ((blockIdx.y << LB) + ty) << 3; A += Y * K;
	const int X = ((blockIdx.x << LB) + tx) << 3; B += X;
	const int C0 = Y * SB + X;//C[Y, X]

	//compute area-------------------------------------------------------
	const int Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = ty - ((ty >= STEP) << LB >> 1);

	const int B0 = ((ty >= STEP) << 2) + By * SB;
	const int A0 = ((tx >= STEP) << 2) * K + Ax;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	float4 av;//transpose A
	av.x = A[A0];
	av.y = A[A1];
	av.z = A[A2];
	av.w = A[A3];
	As[buf][tx][ty] = av;

	Bs[buf][ty][tx] = *(float4*)(B + B0);
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		float4 av;//transpose A
		av.x = A[A0];
		av.y = A[A1];
		av.z = A[A2];
		av.w = A[A3];
		As[buf][tx][ty] = av;

		Bs[buf][ty][tx] = *(float4*)(B + B0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}
	A += STEP; B += (SB << LB >> 1);//K += STEP

	//when K % STEP != 0-----------------------------------
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 b0 = *(float4*)(&get(B, ik, 0, SB));
		float4 b1 = *(float4*)(&get(B, ik, 4, SB));

		float4 a0, a1;
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);
		a1.x = get(A, 4, ik, K);
		a1.y = get(A, 5, ik, K);
		a1.z = get(A, 6, ik, K);
		a1.w = get(A, 7, ik, K);
		simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}
	//when K % STEP != 0-----------------------------------

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


#ifndef XXU_KERNEL6
#define XXU_KERNEL6

#define	xxu_kernel6(LB, stream, A, B, C, N, M, K, SB) \
	XXU_Kernel6<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//Size = 8, Time = 1.599 msec, Performace = 10744.1 GFlop/s
template<int LB, int STEP>
__global__ void XXU_Kernel6(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
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
	const int C0 = Y * SB + X;//C[Y, X]
	A += (Y + ((tx >= STEP) << 2)) * K;
	B += (X + ((ty >= STEP) << 2));

	//compute area-------------------------------------------------------
	const int B0 = (ty - ((ty >= STEP) << LB >> 1)) * SB;
	const int A0 = (tx - ((tx >= STEP) << LB >> 1));
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	As[buf][tx][ty] = float4{ A[A0] ,A[A1], A[A2], A[A3] };//transpose A
	Bs[buf][ty][tx] = *(float4*)(B + B0);
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		As[buf][tx][ty] = float4{ A[A0] ,A[A1], A[A2], A[A3] };//transpose A
		Bs[buf][ty][tx] = *(float4*)(B + B0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
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


#ifndef XXU_KERNEL7
#define XXU_KERNEL7

#define	xxu_kernel7(LB, stream, A, B, C, N, M, K, SB) \
	XXU_Kernel7<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//Size = 8, Time = 1.603 msec, Performace = 10717.3 GFlop/s
template<int LB, int STEP>
__global__ void XXU_Kernel7(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
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
	const int C0 = Y * SB + X;//C[Y, X]
	A += (Y + ((tx >= STEP) << 2) + (tx & 3)) * K;
	B += (X + ((ty >= STEP) << 2));

	//compute area-------------------------------------------------------
	const int B0 = (ty - ((ty >= STEP) << LB >> 1)) * SB;
	const int A0 = (tx - ((tx >= STEP) << LB >> 1)) >> 2 << 2;
	const int As_x = A0 + ((tx >= STEP) << LB >> 1);
	
	float4 av = *(float4*)(A + A0);//transpose A
	*((float*)(&As[buf][As_x + 0][ty]) + (tx & 3)) = av.x;
	*((float*)(&As[buf][As_x + 1][ty]) + (tx & 3)) = av.y;
	*((float*)(&As[buf][As_x + 2][ty]) + (tx & 3)) = av.z;
	*((float*)(&As[buf][As_x + 3][ty]) + (tx & 3)) = av.w;

	Bs[buf][ty][tx] = *(float4*)(B + B0);
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		float4 av = *(float4*)(A + A0);//transpose A
		*((float*)(&As[buf][As_x + 0][ty]) + (tx & 3)) = av.x;
		*((float*)(&As[buf][As_x + 1][ty]) + (tx & 3)) = av.y;
		*((float*)(&As[buf][As_x + 2][ty]) + (tx & 3)) = av.z;
		*((float*)(&As[buf][As_x + 3][ty]) + (tx & 3)) = av.w;

		Bs[buf][ty][tx] = *(float4*)(B + B0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
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


#ifndef XXU_KERNEL8
#define XXU_KERNEL8

#define	xxu_kernel8(LB, stream, A, B, C, N, M, K, SB) \
	XXU_Kernel8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//Size = 8, Time = 1.603 msec, Performace = 10717.3 GFlop/s
template<int LB, int STEP>
__global__ void XXU_Kernel8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
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
	A += (Y + ((tx >= STEP) << 2) + (tx & 3)) * K;
	B += (X + ((ty >= STEP) << 2));

	//compute area-------------------------------------------------------
	const int B0 = (ty - ((ty >= STEP) << LB >> 1)) * SB;
	const int A0 = (tx - ((tx >= STEP) << LB >> 1)) >> 2 << 2;
	const int As_x = A0 + ((tx >= STEP) << LB >> 1);
	const int As_y = (ty << 2) + (tx & 3);

	float4 av = *(float4*)(A + A0);//transpose A
	As[buf][As_x    ][As_y] = av.x;
	As[buf][As_x + 1][As_y] = av.y;
	As[buf][As_x + 2][As_y] = av.z;
	As[buf][As_x + 3][As_y] = av.w;

	Bs[buf][ty][tx] = *(float4*)(B + B0);
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
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		float4 av = *(float4*)(A + A0);//transpose A
		As[buf][As_x    ][As_y] = av.x;
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


//ACC
#ifndef XXU_KERNEL9
#define XXU_KERNEL9

#define	xxu_kernel9(LB, stream, A, B, C, N, M, K, SB) \
	XXU_Kernel9<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//Size = 8, Time = 1.603 msec, Performace = 10717.3 GFlop/s
template<int LB, int STEP>
__global__ void XXU_Kernel9(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SB, int K)
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
	const int C0 = Y * SB + X;//C[Y, X]
	A += (Y + ((tx >= STEP) << 2)) * K;
	B += (X + ((ty >= STEP) << 2));

	//compute area-------------------------------------------------------
	const int B0 = (ty - ((ty >= STEP) << LB >> 1)) * SB;
	const int A0 = (tx - ((tx >= STEP) << LB >> 1));
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	As[buf][tx][ty] = float4{ A[A0] ,A[A1], A[A2], A[A3] };//transpose A
	Bs[buf][ty][tx] = *(float4*)(B + B0);
	__syncthreads();

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1; A += STEP; B += (SB << LB >> 1);//K += STEP

		As[buf][tx][ty] = float4{ A[A0] ,A[A1], A[A2], A[A3] };//transpose A
		Bs[buf][ty][tx] = *(float4*)(B + B0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
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

