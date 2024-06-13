#pragma once

#ifndef BATCH_MATMUL_UERNEL_H
#define BATCH_MATMUL_UERNEL_H

//A[Batch,  N, AK] 
//B[Batch, BK,  M]
//C[Batch,  N,  M]
//(1) M % 4 == 0
//(2) N % 4 != 0
//(3) K = BK: AK % 4 == 0, BK % 4 != 0, AK >= BK, AK = (BK + 3) >> 2 << 2
#ifndef BATCH_MATMUL_UERNEL_CALL
#define BATCH_MATMUL_UERNEL_CALL

//LB = log2(BLOCK_SIZE)
//MOVE_A == 0: A is a 2D tensor[N, K], logically expand A to[Batch, N, K]
//MOVE_B == 0: B is a 2D tensor[K, M], logically expand B to[Barch, K, M]

#define bmm_u88_mk(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_uernel_8_8_MK<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1, MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_u44_mk(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_uernel_4_4_MK<LB, (1<<LB>>1), (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_u44_mk_p(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_uernel_4_4_MK_padding<LB, (1<<LB>>1), (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (4<<LB)-1)>>LB>>2), ((GN + (4<<LB)-1)>>LB>>2), Batch),\
			 dim3(1<<LB, 1<<LB), 0, stream>>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef BATCH_MATMUL_UERNEL_8_8_MK
#define BATCH_MATMUL_UERNEL_8_8_MK

//for [B, N, M, K] = [64, 512, 512, 512]:
//LB = 4: Size = 8, Time = 1.581 msec, Performace = 10866.5  GFlop/s
//LB = 3: Size = 8, Time = 1.768 msec, Performace =  9717.12 GFlop/s

template<int LB, int STEP, int STEP2, int STEP_m1, int MOVE_A, int MOVE_B>
__global__ void bmm_uernel_8_8_MK(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
	      float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int yindex, int xindex)
{
	const int by = blockIdx.y, bx = blockIdx.x;
	const int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//compute 8*8 elements
	float4  v0 = F32_4_0,  v1 = F32_4_0,  v2 = F32_4_0,  v3 = F32_4_0;
	float4  v4 = F32_4_0,  v5 = F32_4_0,  v6 = F32_4_0,  v7 = F32_4_0;
	float4  v8 = F32_4_0,  v9 = F32_4_0, v10 = F32_4_0, v11 = F32_4_0;
	float4 v12 = F32_4_0, v13 = F32_4_0, v14 = F32_4_0, v15 = F32_4_0;

	//prepare for C[N, M]
	const int idx = (ty << LB) + tx;//threadIdx: 256 = 32 * 8 = 16 * 16
	const int vy = idx >> 5, vx = idx & 31;
	const int ux = (vy & 3) * 4 + (vx >> 4) * 2 + (vx & 1);
	const int uy = (vy >> 2) * 8 + ((vx & 15) >> 1);

	const int bY = (by << LB << 3) + yindex;
	const int bX = (bx << LB << 3) + xindex;
	
	const int batch = blockIdx.z;//compute batch offset of A, B, C
	A += ((batch * N  * AK) & (-MOVE_A));
	B += ((batch * BK * CM) & (-MOVE_B));

	const int cY = bY + (uy << 3);
	const int cX = bX + (ux << 3);
	const int C0 = (batch*N + cY)*CM + cX;//C[batch, Y, X]

	//compute area---------------------------------------------------------------------
	const int A0 = (bY + (ty << 3) + ((tx >= STEP) << 2)) * AK + ((tx & STEP_m1) << 1);
	const int A1 = A0 + AK, A2 = A1 + AK, A3 = A2 + AK;

	const int B0 = bX + (tx << 3) + ((ty >= STEP) << 2) + ((ty & STEP_m1) << 1) * CM;
	const int B1 = B0 + CM;

	//load 4 elements from A[batch]
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);

	//load 4 elements from B[batch]
	float4 b0 = *(float4*)(B + B0);
	float4 b1 = *(float4*)(B + B1);

	//write to shared memory
	As[0][(tx << 1)    ][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[0][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };
	Bs[0][(ty << 1)    ][tx] = b0;
	Bs[0][(ty << 1) + 1][tx] = b1;
	__syncthreads();

	for (int ok = STEP2; ok < BK; ok += STEP2) {
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++) {
			float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik + STEP2][ux];
			float4 a0 = As[buf][ik][uy], a1 = As[buf][ik + STEP2][uy];

			simdMM4( v0, a0.x, b0); simdMM4( v1, a0.x, b1);
			simdMM4( v2, a0.y, b0); simdMM4( v3, a0.y, b1);
			simdMM4( v4, a0.z, b0); simdMM4( v5, a0.z, b1);
			simdMM4( v6, a0.w, b0); simdMM4( v7, a0.w, b1);
			simdMM4( v8, a1.x, b0); simdMM4( v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1; A += STEP2; B += (CM << LB);//K += STEP2

		//load 4 elements from A[batch]
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		
		//load 4 elements from B[batch]
		float4 b0 = *(float4*)(B + B0);
		float4 b1 = *(float4*)(B + B1);

		//write to shared memory
		As[buf][(tx << 1)    ][ty] = { a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = { a0.y, a1.y, a2.y, a3.y };
		Bs[buf][(ty << 1)    ][tx] = b0;
		Bs[buf][(ty << 1) + 1][tx] = b1;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++) {
		float4 b0 = Bs[buf][ik][ux], b1 = Bs[buf][ik + STEP2][ux];
		float4 a0 = As[buf][ik][uy], a1 = As[buf][ik + STEP2][uy];

		simdMM4( v0, a0.x, b0); simdMM4( v1, a0.x, b1);
		simdMM4( v2, a0.y, b0); simdMM4( v3, a0.y, b1);
		simdMM4( v4, a0.z, b0); simdMM4( v5, a0.z, b1);
		simdMM4( v6, a0.w, b0); simdMM4( v7, a0.w, b1);
		simdMM4( v8, a1.x, b0); simdMM4( v9, a1.x, b1);
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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef BATCH_MATMUL_UERNEL_4_4_MK
#define BATCH_MATMUL_UERNEL_4_4_MK

//LB = 4: Size = 1, Time = 1.998 msec, Performace = 1074.82 GFlop/s
//LB = 3: Size = 1, Time = 2.046 msec, Performace = 1049.6  GFlop/s
template<int LB, int STEP, int STEP2, int MOVE_A, int MOVE_B>
__global__ void bmm_uernel_4_4_MK(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
	float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int Y0 = (Y + ((tx & 1) << 1))*AK, Y1 = Y0 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int tX = X + ((ty & 1) << 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]

	const int As_x = (tx >> 1) << 1, As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1) << 1, Bs_x = (tx << 1) + (ty & 1);
	
	//load 2 elements from A[batch]
	int Ak = (tx >> 1) << 1;
	float2 a0 = *(float2*)(A + Ak + Y0);
	float2 a1 = *(float2*)(A + Ak + Y1);
	As[buf][As_x][As_y] = float2{ a0.x, a1.x };
	As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

	//load 2 elements from B[btch]
	int Bk = (ty >> 1) << 1;
	int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + boffset0);
	Bs[buf][Bs_y + 1][Bs_x] = *(float2*)(B + boffset1);
	__syncthreads();

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < (BK >> LB); ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from A[batch]
		int Ak = (((ok << LB) + tx) >> 1) << 1;
		float2 a0 = *(float2*)(A + Ak + Y0);
		float2 a1 = *(float2*)(A + Ak + Y1);
		As[buf][As_x][As_y] = float2{ a0.x, a1.x };
		As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

		//load 2 elements from B[batch]
		int Bk = (((ok << LB) + ty) >> 1) << 1;
		int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + boffset0);
		Bs[buf][Bs_y + 1][Bs_x] = *(float2*)(B + boffset1);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}

	const int C1 = C0 + CM;
	const int C2 = C1 + CM;
	const int C3 = C2 + CM;

	*(float4*)(C + C0) = v0;
	*(float4*)(C + C1) = v1;
	*(float4*)(C + C2) = v2;
	*(float4*)(C + C3) = v3;
}

#endif


//(Y: BLOCK_SIZE * 4, X: BLOCK_SIZE * 4), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 20
#ifndef BATCH_MATMUL_UERNEL_4_4_MK_PADDING
#define BATCH_MATMUL_UERNEL_4_4_MK_PADDING

//LB = 4: Size = 1, Time = 2.112 msec, Performace = 1016.8 GFlop/s
//LB = 3: Size = 1, Time = 2.102 msec, Performace = 1021.64 GFlop/s
template<int LB, int STEP, int STEP2, int MOVE_A, int MOVE_B>
__global__ void bmm_uernel_4_4_MK_padding(
	const float* __restrict__ A, //A[Batch, N, K]
	const float* __restrict__ B, //B[Batch, K, M]
	      float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int Y0 = (Y + ((tx & 1) << 1))*AK, Y1 = Y0 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int X0 = X + ((ty & 1) << 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + X0;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]
	
	const int As_x = (tx >> 1) << 1, As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1) << 1, Bs_x = (tx << 1) + (ty & 1);
	const int ye = (N - 1) * AK;
	const int xe = (CM - 2);//float2: X <= M - 2
	
	//load 2 elements from A[batch]
	int Ak = (tx >> 1) << 1;
	float2 a0 = (Y0 <= ye ? *(float2*)(A + Y0 + Ak) : F32_2_0);
	float2 a1 = (Y1 <= ye ? *(float2*)(A + Y1 + Ak) : F32_2_0);
	As[buf][As_x][As_y] = float2{ a0.x, a1.x };
	As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

	//load 2 elements from B[btch]
	int Bk = (ty >> 1) << 1;
	int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
	Bs[buf][Bs_y][Bs_x] = (X0 <= xe ? *(float2*)(B + boffset0) : F32_2_0);
	Bs[buf][Bs_y + 1][Bs_x] = (X0 <= xe ? *(float2*)(B + boffset1) : F32_2_0);
	__syncthreads();

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (BK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from A[batch]
		int Ak = (((ok << LB) + tx) >> 1) << 1;
		float2 a0 = (Y0 <= ye ? *(float2*)(A + Y0 + Ak) : F32_2_0);
		float2 a1 = (Y1 <= ye ? *(float2*)(A + Y1 + Ak) : F32_2_0);
		As[buf][As_x][As_y] = float2{ a0.x, a1.x };
		As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

		//load 2 elements from B[batch]
		int Bk = (((ok << LB) + ty) >> 1) << 1;
		int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
		Bs[buf][Bs_y][Bs_x] = (X0 <= xe ? *(float2*)(B + boffset0) : F32_2_0);
		Bs[buf][Bs_y + 1][Bs_x] = (X0 <= xe ? *(float2*)(B + boffset1) : F32_2_0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}

	const int C0 = Y*CM + X;//C[batch, Y, X]
	const int C1 = C0 + CM;
	const int C2 = C1 + CM;
	const int C3 = C2 + CM;

	//if X <= M - 4, (Y * M) + X <= Cend, we have: Y <= N -1
	int ce = (N * CM) - 4;//float4: (N - 1)*M + (M - 4) = N*M - 4
	if (X <= (CM - 4)) {//float4: X <= M - 4
		if (C0 <= ce) { *(float4*)(C + C0) = v0; }
		if (C1 <= ce) { *(float4*)(C + C1) = v1; }
		if (C2 <= ce) { *(float4*)(C + C2) = v2; }
		if (C3 <= ce) { *(float4*)(C + C3) = v3; }
	}
}

#endif

#endif