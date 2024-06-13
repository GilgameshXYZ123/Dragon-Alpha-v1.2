

//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef VP_KERNEL1
#define VP_KERNEL1

#define vp_kernel1(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	VP_Kernel1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

//Size = 4, Time = 0.83 msec, Performace = 10349.3 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int MOVE_A, int MOVE_B>
__global__ void VP_Kernel1(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
	float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int Y0 = (Y + ((tx >= STEP) << 2)) * AK;
	const int Y1 = Y0 + AK, Y2 = Y1 + AK, Y3 = Y2 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	int batch = blockIdx.z;
	A += ((batch * N * AK) & (-MOVE_A));//A[batch * MOVE_A]
	B += ((batch * BK * CM) & (-MOVE_B)) + tX;//B[batch * MOVE_B, 0, tX]
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]

	//compute area-----------------------------------------------------
	//load 4 elements from A[batch]
	const int Ak = (tx & STEP_m1) << 1;
	const int A0 = Ak + Y0;
	const int A1 = Ak + Y1;
	const int A2 = Ak + Y2;
	const int A3 = Ak + Y3;
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elements from B[batch]
	const int Bk = (ty & STEP_m1) << 1;
	const int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
	Bs[buf][(ty << 1)][tx] = *(float4*)(B + boffset0);
	Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + boffset1);
	__syncthreads();

	for (int ok = STEP2; ok < BK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];

			simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from A[batch]
		const int Ak = ok + ((tx & STEP_m1) << 1);
		const int A0 = Ak + Y0;
		const int A1 = Ak + Y1;
		const int A2 = Ak + Y2;
		const int A3 = Ak + Y3;
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 4 elements from B[batch]
		const int Bk = ok + ((ty & STEP_m1) << 1);
		const int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
		Bs[buf][(ty << 1)][tx] = *(float4*)(B + boffset0);
		Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + boffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	const int C4 = C3 + CM, C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	*(float4*)(C + C0) = v0;  *(float4*)(C + C0 + 4) = v1;
	*(float4*)(C + C1) = v2;  *(float4*)(C + C1 + 4) = v3;
	*(float4*)(C + C2) = v4;  *(float4*)(C + C2 + 4) = v5;
	*(float4*)(C + C3) = v6;  *(float4*)(C + C3 + 4) = v7;
	*(float4*)(C + C4) = v8;  *(float4*)(C + C4 + 4) = v9;
	*(float4*)(C + C5) = v10; *(float4*)(C + C5 + 4) = v11;
	*(float4*)(C + C6) = v12; *(float4*)(C + C6 + 4) = v13;
	*(float4*)(C + C7) = v14; *(float4*)(C + C7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef VP_KERNEL2
#define VP_KERNEL2

#define vp_kernel2(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	VP_Kernel2<LB, (1<<LB>>1), (1<<LB>>1)-1, MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

//Size = 4, Time = 0.83 msec, Performace = 10349.3 GFlop/s
template<int LB, int STEP, int STEP_m1, int MOVE_A, int MOVE_B>
__global__ void VP_Kernel2(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
	float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int Y0 = (Y + (tx & 3) + ((tx >= STEP) << 2)) * AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	int batch = blockIdx.z;
	A += ((batch * N * AK) & (-MOVE_A));//A[batch * MOVE_A]
	B += ((batch * BK * CM) & (-MOVE_B)) + tX;//B[batch * MOVE_B, 0, tX]
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]

	//compute area-----------------------------------------------------
	//load 4 elements from B[batch]
	const int boffset = (ty & STEP_m1) * CM;
	Bs[buf][ty][tx] = *(float4*)(B + boffset);

	//load 4 elements from A[batch]
	const int aoffset = Y0 + ((tx & STEP_m1) >> 2 << 2);
	float4 av = *(float4*)(A + aoffset);
	const int As_x = ((tx & STEP_m1) >> 2 << 2) + ((tx >= STEP)*STEP);
	*((float*)(&As[buf][As_x + 0][ty]) + (tx & 3)) = av.x;
	*((float*)(&As[buf][As_x + 1][ty]) + (tx & 3)) = av.y;
	*((float*)(&As[buf][As_x + 2][ty]) + (tx & 3)) = av.z;
	*((float*)(&As[buf][As_x + 3][ty]) + (tx & 3)) = av.w;
	__syncthreads();

	for (int ok = STEP; ok < BK; ok += STEP)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from B[batch]
		const int boffset = (ok + (ty & STEP_m1)) * CM;
		Bs[buf][ty][tx] = *(float4*)(B + boffset);

		//load 4 elements from A[batch]
		const int aoffset = Y0 + ok + ((tx & STEP_m1) >> 2 << 2);;
		float4 av = *(float4*)(A + aoffset);
		*((float*)(&As[buf][As_x + 0][ty]) + (tx & 3)) = av.x;
		*((float*)(&As[buf][As_x + 1][ty]) + (tx & 3)) = av.y;
		*((float*)(&As[buf][As_x + 2][ty]) + (tx & 3)) = av.z;
		*((float*)(&As[buf][As_x + 3][ty]) + (tx & 3)) = av.w;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	const int C4 = C3 + CM, C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	*(float4*)(C + C0) = v0;  *(float4*)(C + C0 + 4) = v1;
	*(float4*)(C + C1) = v2;  *(float4*)(C + C1 + 4) = v3;
	*(float4*)(C + C2) = v4;  *(float4*)(C + C2 + 4) = v5;
	*(float4*)(C + C3) = v6;  *(float4*)(C + C3 + 4) = v7;
	*(float4*)(C + C4) = v8;  *(float4*)(C + C4 + 4) = v9;
	*(float4*)(C + C5) = v10; *(float4*)(C + C5 + 4) = v11;
	*(float4*)(C + C6) = v12; *(float4*)(C + C6 + 4) = v13;
	*(float4*)(C + C7) = v14; *(float4*)(C + C7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE * 8, X : BLOCK_SIZE * 8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef VP_KERNEL3
#define VP_KERNEL3

#define vp_kernel3(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	VP_Kernel3<LB, (1<<LB>>1), (1<<LB>>1)-1, MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

//Size = 8, Time = 1.62 msec, Performace = 10604.9 GFlop/s
template<int LB, int STEP, int STEP_m1, int MOVE_A, int MOVE_B>
__global__ void VP_Kernel3(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
	float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepared for: A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int tY = Y + (tx & 3) + ((tx >= STEP) << 2);

	//prepared for: B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute area-----------------------------------------------------
	int batch = blockIdx.z;
	A += ((batch * N  * AK) & (-MOVE_A)) + tY * AK;//A[batch * MOVE_A, tY, 0]
	B += ((batch * BK * CM) & (-MOVE_B)) + tX;//B[batch * MOVE_B, 0, tX]
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]

	//load 4 elements from A[batch]
	const int aoffset = ((tx & STEP_m1) >> 2 << 2);
	float4 av = *(float4*)(A + aoffset);
	const int As_x = ((tx & STEP_m1) >> 2 << 2) + ((tx >= STEP)*STEP);
	*((float*)(&As[buf][As_x + 0][ty]) + (tx & 3)) = av.x;
	*((float*)(&As[buf][As_x + 1][ty]) + (tx & 3)) = av.y;
	*((float*)(&As[buf][As_x + 2][ty]) + (tx & 3)) = av.z;
	*((float*)(&As[buf][As_x + 3][ty]) + (tx & 3)) = av.w;

	//load 4 elements from B[batch]
	const int boffset = (ty & STEP_m1) * CM;
	Bs[buf][ty][tx] = *(float4*)(B + boffset);
	__syncthreads();

	for (int ok = STEP; ok < BK; ok += STEP)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from A[batch]
		const int aoffset = ok + ((tx & STEP_m1) >> 2 << 2);
		float4 av = *(float4*)(A + aoffset);
		*((float*)(&As[buf][As_x + 0][ty]) + (tx & 3)) = av.x;
		*((float*)(&As[buf][As_x + 1][ty]) + (tx & 3)) = av.y;
		*((float*)(&As[buf][As_x + 2][ty]) + (tx & 3)) = av.z;
		*((float*)(&As[buf][As_x + 3][ty]) + (tx & 3)) = av.w;

		//load 4 elements from B[batch]
		const int boffset = (ok + (ty & STEP_m1)) * CM;
		Bs[buf][ty][tx] = *(float4*)(B + boffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	const int C4 = C3 + CM, C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	*(float4*)(C + C0) = v0;  *(float4*)(C + C0 + 4) = v1;
	*(float4*)(C + C1) = v2;  *(float4*)(C + C1 + 4) = v3;
	*(float4*)(C + C2) = v4;  *(float4*)(C + C2 + 4) = v5;
	*(float4*)(C + C3) = v6;  *(float4*)(C + C3 + 4) = v7;
	*(float4*)(C + C4) = v8;  *(float4*)(C + C4 + 4) = v9;
	*(float4*)(C + C5) = v10; *(float4*)(C + C5 + 4) = v11;
	*(float4*)(C + C6) = v12; *(float4*)(C + C6 + 4) = v13;
	*(float4*)(C + C7) = v14; *(float4*)(C + C7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef VP_KERNEL4
#define VP_KERNEL4

#define vp_kernel4(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	VP_Kernel4<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

//Size = 4, Time = 0.83 msec, Performace = 10349.3 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int MOVE_A, int MOVE_B>
__global__ void VP_Kernel4(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
	float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int Y0 = (Y + ((tx >= STEP) << 2)) * AK;
	const int Y1 = Y0 + AK, Y2 = Y1 + AK, Y3 = Y2 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute area-----------------------------------------------------
	const int batch = blockIdx.z;
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]
	A += ((batch * N * AK) & (-MOVE_A));//A[batch * MOVE_A]
	B += ((batch * BK * CM) & (-MOVE_B)) + tX;//B[batch * MOVE_B, 0, tX]

	//load 4 elements from A[batch]
	const int Ak = (tx & STEP_m1) << 1;
	const int A0 = Ak + Y0, A1 = Ak + Y1;
	const int A2 = Ak + Y2, A3 = Ak + Y3;
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elements from B[batch]
	const int Bk = (ty & STEP_m1) << 1;
	const int boffset0 = Bk * CM;
	const int boffset1 = boffset0 + CM;
	Bs[buf][(ty << 1)][tx] = *(float4*)(B + boffset0);
	Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + boffset1);
	__syncthreads();

	for (int ok = STEP2; ok < BK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];

			simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from A[batch]
		const int Ak = ok + ((tx & STEP_m1) << 1);
		const int A0 = Ak + Y0, A1 = Ak + Y1;
		const int A2 = Ak + Y2, A3 = Ak + Y3;
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 4 elements from B[batch]
		const int Bk = ok + ((ty & STEP_m1) << 1);
		const int boffset0 = Bk * CM;
		const int boffset1 = boffset0 + CM;
		Bs[buf][(ty << 1)][tx] = *(float4*)(B + boffset0);
		Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + boffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	const int C4 = C3 + CM, C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	*(float4*)(C + C0) = v0;  *(float4*)(C + C0 + 4) = v1;
	*(float4*)(C + C1) = v2;  *(float4*)(C + C1 + 4) = v3;
	*(float4*)(C + C2) = v4;  *(float4*)(C + C2 + 4) = v5;
	*(float4*)(C + C3) = v6;  *(float4*)(C + C3 + 4) = v7;
	*(float4*)(C + C4) = v8;  *(float4*)(C + C4 + 4) = v9;
	*(float4*)(C + C5) = v10; *(float4*)(C + C5 + 4) = v11;
	*(float4*)(C + C6) = v12; *(float4*)(C + C6 + 4) = v13;
	*(float4*)(C + C7) = v14; *(float4*)(C + C7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef VP_KERNEL5
#define VP_KERNEL5

#define vp_kernel5(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	VP_Kernel5<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

//Size = 4, Time = 0.83 msec, Performace = 10349.3 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int MOVE_A, int MOVE_B>
__global__ void VP_Kernel5(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
	float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	const int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 v8 = F32_4_0, v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int Y0 = (Y + ((tx >= STEP) << 2)) * AK;
	const int Y1 = Y0 + AK, Y2 = Y1 + AK, Y3 = Y2 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += ((batch * N  * AK) & (-MOVE_A));//A[batch * MOVE_A]
	B += ((batch * BK * CM) & (-MOVE_B)) + tX;//B[batch * MOVE_B, 0, tX]
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]

	//compute area-----------------------------------------------------
	A += ((tx & STEP_m1) << 1);
	B += ((ty & STEP_m1) << 1) * CM;
	
	//load 4 elements from B[batch]
	Bs[0][(ty << 1)][tx] = *(float4*)(B);
	Bs[0][(ty << 1) + 1][tx] = *(float4*)(B + CM);

	//load 4 elements from A[batch]
	float2 a0 = *(float2*)(A + Y0);
	float2 a1 = *(float2*)(A + Y1);
	float2 a2 = *(float2*)(A + Y2);
	float2 a3 = *(float2*)(A + Y3);
	As[0][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[0][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };
	__syncthreads();

	for (int ok = STEP2; ok < BK; ok += STEP2)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];

			simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;


		//load 4 elements from B[batch]
		const int boffset0 = ok * CM, boffset1 = boffset0 + CM;
		Bs[buf][(ty << 1)    ][tx] = *(float4*)(B + boffset0);
		Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + boffset1);

		//load 4 elements from A[batch]
		float2 a0 = *(float2*)(A + ok + Y0);
		float2 a1 = *(float2*)(A + ok + Y1);
		float2 a2 = *(float2*)(A + ok + Y2);
		float2 a3 = *(float2*)(A + ok + Y3);
		As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	const int C4 = C3 + CM, C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	*(float4*)(C + C0) = v0;  *(float4*)(C + C0 + 4) = v1;
	*(float4*)(C + C1) = v2;  *(float4*)(C + C1 + 4) = v3;
	*(float4*)(C + C2) = v4;  *(float4*)(C + C2 + 4) = v5;
	*(float4*)(C + C3) = v6;  *(float4*)(C + C3 + 4) = v7;
	*(float4*)(C + C4) = v8;  *(float4*)(C + C4 + 4) = v9;
	*(float4*)(C + C5) = v10; *(float4*)(C + C5 + 4) = v11;
	*(float4*)(C + C6) = v12; *(float4*)(C + C6 + 4) = v13;
	*(float4*)(C + C7) = v14; *(float4*)(C + C7 + 4) = v15;
}

#endif

