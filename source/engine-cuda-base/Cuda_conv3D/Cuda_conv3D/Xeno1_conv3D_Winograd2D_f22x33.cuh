#pragma once

#include "conv3D_Winograd2D_f22x33_util.cuh"

#ifndef CONV3D_WINOGRAD2D_F22X33_H
#define CONV3D_WINOGRAD2D_F22X33_H


#ifndef CONV3D_WINOGRAD2D_F22X33_CALL
#define CONV3D_WINOGRAD2D_F22X33_CALL


#endif


#ifndef WINOGRAD2D_KERNEL1
#define WINOGRAD2D_KERNEL1

//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int tiles_dim, int tiles_2d_dim)
{
	int tile_size = 4, tile_2d_s = 16;//4*4

	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float shared_mem[16 * BLOCK_C*BLOCK_N + 16 * BLOCK_C*BLOCK_K];
	float *Xs = (float*)shared_mem;             //16 * BLOCK_C*BLOCK_N
	float *Gs = (float*)&shared_mem[16 * BLOCK_C*BLOCK_N];//16 * BLOCK_C*BLOCK_K

	float4 accu[2][16] = { 0.0f };  // Accumulators 

	float X_tile[16]; // Prefetch input from GMEM
	float G_tile[32]; // Prefetch filter from GMEM

	float4 X_frag_mem[8], *X_frag = (float4*)X_frag_mem, *X_frag2 = (float4*)(X_frag + 4);
	float4 G_frag_mem[8], *G_frag = (float4*)G_frag_mem, *G_frag2 = (float4*)(G_frag + 4);
	float4 *swap;

	prefetch_X_tile<BLOCK_N>(X, X_tile, N, IH, IW, IC, tx, ty, bx, by, tiles_dim);
	prefetch_G_tile<BLOCK_K, BLOCK_N>(G, G_tile, OC, tx, ty, bz);

	int X_frag_offset = 2 * (BLOCK_C*BLOCK_N); // (2=8/4) SMEM input read offset
	int G_frag_offset = 2 * (BLOCK_C*BLOCK_K); // (2=8/4) SMEM filter read offset

	// Mainloop - iterates over the entire K dimension - not unrolled
	for (int oic = 0; oic < IC; oic += BLOCK_C) { // Current iteration

		float4 *A_frag = (float4*)(Xs + threadIdx.y*BLOCK_C*BLOCK_N);
		float4 *B_frag = (float4*)(Gs + threadIdx.y*BLOCK_C*BLOCK_K);

		load_and_transform_X_tile(X_tile, Xs, 
			N, IH, IW, IC,
			tiles_dim, tiles_2d_dim,
			tx, ty, bx, by);

		load_G_tile(G_tile, Gs, IC, OC,
			tx, ty);

		__syncthreads();

		prefetch_X_frag(X_frag, A_frag, X_frag_offset, tx, access_s[0][tx], access_s[1][tx]);
		prefetch_G_frag(G_frag, B_frag, G_frag_offset, tx, access_f_s[0][tx], access_f_s[1][tx]);

#pragma unroll
		for (int i = 0; i < BLOCK_C; i++) {

			if (i < (BLOCK_C - 1)) {
				A_frag += BLOCK_N / 4;
				B_frag += BLOCK_K / 4;

				prefetch_X_frag(X_frag2, A_frag, X_frag_offset, tx, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
				prefetch_G_frag(G_frag2, B_frag, G_frag_offset, tx, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);
			}

			outer_product(X_frag, G_frag, accu);

			swap = X_frag; X_frag = X_frag2; X_frag2 = swap;
			swap = G_frag; G_frag = G_frag2; G_frag2 = swap;
		}

		X += BLOCK_C;//[ic, ih, iw,  n] -> [n,  ih, iw, ic]
		G += BLOCK_C;//[ic, fh, fw, oc] -> [oc, fh, fw, ic]

		//X += N * BLOCK_C * IW * IH;
		//G += OC * BLOCK_C * 4 * 4;

		if (oic < (IC - BLOCK_C)) {
			prefetch_X_tile<BLOCK_N>(X, X_tile, N, IH, IW, IC, tx, ty, bx, by, tiles_dim);
			prefetch_G_tile<BLOCK_K, BLOCK_N>(G, G_tile, OC, tx, ty, bz);
		}

		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_Y_tile(accu, shared_mem, 
		tx, ty, 
		Y, 
		bx, by, bz, 
		OH, OW, OC,
		tiles_dim, N, 
		X_frag_mem, G_frag_mem);
}

#endif


#ifndef WINOGRAD2D_KERNEL2
#define WINOGRAD2D_KERNEL2

//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int tiles_dim, int tiles_2d_dim)
{
	int tile_size = 4, tile_2d_s = 16;//4*4

	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float shared_mem[16 * BLOCK_C*BLOCK_N + 16 * BLOCK_C*BLOCK_K];
	float *Xs = (float*)shared_mem;             //16 * BLOCK_C*BLOCK_N
	float *Gs = (float*)&shared_mem[16 * BLOCK_C*BLOCK_N];//16 * BLOCK_C*BLOCK_K

	float4 accu[2][16] = { 0.0f };  // Accumulators 

	float X_tile[16]; // Prefetch input from GMEM
	float G_tile[32]; // Prefetch filter from GMEM

	float4 X_frag_mem[8], *X_frag = (float4*)X_frag_mem, *X_frag2 = (float4*)(X_frag + 4);
	float4 G_frag_mem[8], *G_frag = (float4*)G_frag_mem, *G_frag2 = (float4*)(G_frag + 4);
	float4 *swap;

	prefetch_G_tile<BLOCK_K, BLOCK_N>(G, G_tile, OC, tx, ty, bz); 
	prefetch_X_tile<BLOCK_N>(X, X_tile, N, IH, IW, IC, tx, ty, bx, by, tiles_dim);
	
	int X_frag_offset = 2 * (BLOCK_C * BLOCK_N);
	int G_frag_offset = 2 * (BLOCK_C * BLOCK_K);

	for (int oic = 0; oic < IC; oic += BLOCK_C) {

		float4 *A_frag = (float4*)(Xs + ty*BLOCK_C*BLOCK_N);
		float4 *B_frag = (float4*)(Gs + ty*BLOCK_C*BLOCK_K);

		load_and_transform_X_tile(X_tile, Xs, 
			N, IH, IW, IC,
			tiles_dim, tiles_2d_dim,
			tx, ty, bx, by);

		load_G_tile(G_tile, Gs, IC, OC,
			tx, ty);

		__syncthreads();

		prefetch_X_frag(X_frag, A_frag, X_frag_offset, tx, access_s[0][tx], access_s[1][tx]);
		prefetch_G_frag(G_frag, B_frag, G_frag_offset, tx, access_f_s[0][tx], access_f_s[1][tx]);

#pragma unroll
		for (int i = 0; i < BLOCK_C; i++) {

			if (i < (BLOCK_C - 1)) {
				A_frag += BLOCK_N / 4;
				B_frag += BLOCK_K / 4;

				prefetch_X_frag(X_frag2, A_frag, X_frag_offset, tx, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
				prefetch_G_frag(G_frag2, B_frag, G_frag_offset, tx, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);
			}

			outer_product(X_frag, G_frag, accu);

			swap = X_frag; X_frag = X_frag2; X_frag2 = swap;
			swap = G_frag; G_frag = G_frag2; G_frag2 = swap;
		}

		//ic += BLOCK_C
		G += BLOCK_C * OC * 16;//[ic, fh, fw, oc] -> [ic, oc, fh, fw]
		X += BLOCK_C;//[ic, ih, iw, n] -> [n, ih, iw, ic]

		if (oic < (IC - BLOCK_C)) {
			prefetch_X_tile<BLOCK_N>(X, X_tile, N, IH, IW, IC, tx, ty, bx, by, tiles_dim);
			prefetch_G_tile<BLOCK_K, BLOCK_N>(G, G_tile, OC, tx, ty, bz);
		}
		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_Y_tile(accu, shared_mem,
		tx, ty,
		Y,
		bx, by, bz,
		OH, OW, OC,
		tiles_dim, N,
		X_frag_mem, G_frag_mem);
}

#endif


#ifndef WINOGRAD2D_KERNEL3
#define WINOGRAD2D_KERNEL3

//Size = 9, Time = 2.61437 msec, Performace = 7392.73 GFlop/s
//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int tiles_dim, int tiles_2d_dim)
{
	int ph = 1, pw = 1;

	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float shared_mem[16 * BLOCK_C*BLOCK_N + 16 * BLOCK_C*BLOCK_K];
	float *Xs = (float*)shared_mem;             //16 * BLOCK_C*BLOCK_N
	float *Gs = (float*)&shared_mem[16 * BLOCK_C*BLOCK_N];//16 * BLOCK_C*BLOCK_K

	float4 accu[2][16] = { 0.0f };  // Accumulators 

	//----------------------------------------
	const int oc = (bz * BLOCK_K + tx);
	const int n  = (bx * BLOCK_N) + tx;
	const int ih = ((by / tiles_dim) * 2) - ph;
	const int iw = ((by % tiles_dim) * 2) - pw;
	//----------------------------------------

	float X_tile[16]; // Prefetch input from GMEM
	float G_tile[32]; // Prefetch filter from GMEM

	float4 X_frag_mem[8], *X_frag = (float4*)X_frag_mem, *X_frag2 = (float4*)(X_frag + 4);
	float4 G_frag_mem[8], *G_frag = (float4*)G_frag_mem, *G_frag2 = (float4*)(G_frag + 4);
	float4 *swap;
	
	
	//======[prefetch_G_tile=]===================================================
	const int ic = ty;
	const int g0 = (ic * OC + (oc          )) * 16;          //G[ic     , oc, fh, fw]
	const int g1 = (ic * OC + (oc + BLOCK_N)) * 16;//G[ic + BN, oc, fh, fw]

	*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
	*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
	*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
	*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);
	*(float4*)(G_tile + 16) = *(float4*)(G + g1);
	*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
	*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
	*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);
	//======[prefetch_G_tile=]===================================================

	
	//======[prefetch_X_tile=]===================================================
#pragma unroll
	for (int i = 0; i < 4; i++) {
#pragma unroll
		for (int j = 0; j < 4; j++) {
			int iht = ih + i;
			int iwt = iw + j;

			int x0 = ((n*IH + iht)*IW + iwt)*IC + ic;
			bool lx = (iht >= 0) && (iht < IH) && (iwt >= 0) && (iwt < IW);

			X_tile[(i << 2) + j] = (lx ? X[x0] : 0);//[n, ih, iw, ic]
		}
	}
	//======[prefetch_X_tile=]===================================================


	int X_frag_offset = 2 * (BLOCK_C * BLOCK_N);
	int G_frag_offset = 2 * (BLOCK_C * BLOCK_K);

	for (int oic = 0; oic < IC; oic += BLOCK_C) 
	{
		float4 *A_frag = (float4*)(Xs + ty * BLOCK_C*BLOCK_N);
		float4 *B_frag = (float4*)(Gs + ty * BLOCK_C*BLOCK_K);

		//======[load_and_transform_X_tile]======================================================
		{
			float workspace[3];
#pragma unroll
			for (int j = 0; j < 4; j++) {
				workspace[0] = X_tile[j];
				workspace[1] = X_tile[j + 4];
				workspace[2] = X_tile[j + 8];

				X_tile[j] = workspace[0] - workspace[2];
				X_tile[j + 4] = workspace[1] + workspace[2];
				X_tile[j + 8] = workspace[2] - workspace[1];
				X_tile[j + 12] = workspace[1] - X_tile[j + 12];
			}

			int c_offset = BLOCK_N * BLOCK_C;
			int c_tensor = ty * BLOCK_N + tx;

#pragma unroll
			for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
				Xs[c_tensor + i * c_offset * 4] = d(X_tile, i, 0) - d(X_tile, i, 2);
				Xs[c_tensor + i * c_offset * 4 + c_offset] = d(X_tile, i, 1) + d(X_tile, i, 2);
				Xs[c_tensor + i * c_offset * 4 + 2 * c_offset] = d(X_tile, i, 2) - d(X_tile, i, 1);
				Xs[c_tensor + i * c_offset * 4 + 3 * c_offset] = d(X_tile, i, 1) - d(X_tile, i, 3);
			}
		}
		//======[load_and_transform_X_tile]======================================================

		load_G_tile(G_tile, Gs, IC, OC, tx, ty);

		__syncthreads();

		prefetch_X_frag(X_frag, A_frag, X_frag_offset, tx, access_s[0][tx], access_s[1][tx]);
		prefetch_G_frag(G_frag, B_frag, G_frag_offset, tx, access_f_s[0][tx], access_f_s[1][tx]);

#pragma unroll
		for (int i = 0; i < BLOCK_C; i++) {

			if (i < (BLOCK_C - 1)) {
				A_frag += BLOCK_N / 4;
				B_frag += BLOCK_K / 4;

				prefetch_X_frag(X_frag2, A_frag, X_frag_offset, tx, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
				prefetch_G_frag(G_frag2, B_frag, G_frag_offset, tx, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);
			}

			outer_product(X_frag, G_frag, accu);

			swap = X_frag; X_frag = X_frag2; X_frag2 = swap;
			swap = G_frag; G_frag = G_frag2; G_frag2 = swap;
		}

		//ic += BLOCK_C
		G += BLOCK_C * OC * 16;//[ic, fh, fw, oc] -> [ic, oc, fh, fw]
		X += BLOCK_C;//[ic, ih, iw, n] -> [n, ih, iw, ic]

		if (oic < (IC - BLOCK_C)) {
			//======[prefetch_G_tile=]===================================================
			const int ic = ty;
			const int g0 = (ic * OC + oc) * 16;//G[ic     , oc, fh, fw]
			const int g1 = (ic * OC + oc + BLOCK_N) * 16;//G[ic + BN, oc, fh, fw]

			*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
			*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
			*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
			*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

			*(float4*)(G_tile + 16) = *(float4*)(G + g1);
			*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
			*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
			*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);
			//======[prefetch_G_tile=]===================================================
			
			
			//======[prefetch_X_tile=]===================================================
#pragma unroll
			for (int i = 0; i < 4; i++) {
#pragma unroll
				for (int j = 0; j < 4; j++) {
					int iht = ih + i;
					int iwt = iw + j;

					int x0 = ((n*IH + iht)*IW + iwt)*IC + ic;
					bool lx = (iht >= 0) && (iht < IH) && (iwt >= 0) && (iwt < IW);

					X_tile[(i << 2) + j] = (lx ? X[x0] : 0);//[n, ih, iw, ic]
				}
			}
			//======[prefetch_X_tile=]===================================================
		}
		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_Y_tile(accu, shared_mem,
		tx, ty,
		Y,
		bx, by, bz,
		OH, OW, OC,
		tiles_dim, N,
		X_frag_mem, G_frag_mem);
}

#endif


#ifndef WINOGRAD2D_KERNEL4
#define WINOGRAD2D_KERNEL4

//Size = 9, Time = 2.4797 msec, Performace = 7794.23 GFlop/s
//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int tiles_dim, int tiles_2d_dim)
{
	int ph = 1, pw = 1;
	
	__shared__ float shared_mem[(16 * BLOCK_C*BLOCK_N) + (16 * BLOCK_C*BLOCK_K)];
	float *Xs = (float*)shared_mem;                       //Xs[16, BLOCK_C, BLOCK_N]
	float *Gs = (float*)&shared_mem[16 * BLOCK_C*BLOCK_N];//Gs[16, BLOCK_C, BLOCK_K]

	float4 accu[2][16] = { 0.0f };  // Accumulators 

	//----------------------------------------
	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;

	//<<< dim3(N / BN, tiles_2d_dim, OC / BK), dim3(BN, BC), 0 >>>
	const int oc = (bz * BLOCK_K) + tx;
	const int n  = (bx * BLOCK_N) + tx;
	const int ohs = by / tiles_dim, ows = by - ohs * tiles_dim;
	const int oh = (ohs << 1), ow = (ows << 1);
	const int ih = oh - ph, iw = ow - pw;
	//----------------------------------------

	float X_tile[16]; // Prefetch input from GMEM
	float G_tile[32]; // Prefetch filter from GMEM

	float4 X_frag_mem[8], *X_frag = (float4*)X_frag_mem, *X_frag2 = (float4*)(X_frag + 4);
	float4 G_frag_mem[8], *G_frag = (float4*)G_frag_mem, *G_frag2 = (float4*)(G_frag + 4);
	float4 *swap;

	const int ic = ty;

	//======[prefetch_G_tile=]===================================================
	{
		const int g0 = (ic * OC + (oc)) * 16;          //G[ic     , oc, fh, fw]
		const int g1 = (ic * OC + (oc + BLOCK_N)) * 16;//G[ic + BN, oc, fh, fw]

		*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
		*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
		*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
		*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

		*(float4*)(G_tile + 16) = *(float4*)(G + g1);
		*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
		*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
		*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);
	}
	
	//======[prefetch_X_tile=]===================================================
	{
#pragma unroll
		for (int i = 0; i < 4; i++) {
#pragma unroll
			for (int j = 0; j < 4; j++) {
				int iht = ih + i;
				int iwt = iw + j;

				int x0 = ((n*IH + iht)*IW + iwt)*IC + ic;
				bool lx = (iht >= 0) && (iht < IH) && (iwt >= 0) && (iwt < IW);

				X_tile[(i << 2) + j] = (lx ? X[x0] : 0);//[n, ih, iw, ic]
			}
		}
	}

	int X_frag_offset = 2 * (BLOCK_C * BLOCK_N);
	int G_frag_offset = 2 * (BLOCK_C * BLOCK_K);

	for (int oic = 0; oic < IC; oic += BLOCK_C)
	{
		float4 *A_frag = (float4*)(Xs + ty * BLOCK_C*BLOCK_N);
		float4 *B_frag = (float4*)(Gs + ty * BLOCK_C*BLOCK_K);

		//======[load_and_transform_X_tile]======================================================
		{
			float xb[3];

#pragma unroll
			for (int j = 0; j < 4; j++) {
				xb[0] = X_tile[j];
				xb[1] = X_tile[j + 4];
				xb[2] = X_tile[j + 8];

				X_tile[j]      = xb[0] - xb[2];//xj0 - xj2
				X_tile[j + 4]  = xb[1] + xb[2];//xj1 + xj2
				X_tile[j + 8]  = xb[2] - xb[1];//xj2 - xj1
				X_tile[j + 12] = xb[1] - X_tile[j + 12];//xj1 - xj3
			}

			//Xs[16, BLOCK_C, BLOCK_N]
			int c_offset = BLOCK_N * BLOCK_C;
			int c_tensor = ty * BLOCK_N + tx;//[ty, tx]

#pragma unroll
			for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
				Xs[c_tensor + i * c_offset * 4               ] = X_tile[(i << 2) + 0] - X_tile[(i << 2) + 2];//xi0 - xi2
				Xs[c_tensor + i * c_offset * 4 +     c_offset] = X_tile[(i << 2) + 1] + X_tile[(i << 2) + 2];//xi1 + xi2
				Xs[c_tensor + i * c_offset * 4 + 2 * c_offset] = X_tile[(i << 2) + 2] - X_tile[(i << 2) + 1];//xi2 - xi1
				Xs[c_tensor + i * c_offset * 4 + 3 * c_offset] = X_tile[(i << 2) + 1] - X_tile[(i << 2) + 3];//xi1 - xi3
			}
		}

		//======[load_G_tile]====================================================================
		{
			//Gs[16, BLOCK_C, BLOCK_K]
			int c_tensor_s = ty * BLOCK_K + tx; //[ty, tx]
			int c_offset_s = BLOCK_K * BLOCK_C;

			for (int k = 0; k < 2; k++) { // prefetch 2 filter tiles/thread
#pragma unroll
				for (int i = 0; i < 4; i++) {
					Gs[c_tensor_s + i * c_offset_s * 4 + 0 * c_offset_s] = G_tile[k * 16 + i * 4 + 0];
					Gs[c_tensor_s + i * c_offset_s * 4 + 1 * c_offset_s] = G_tile[k * 16 + i * 4 + 1];
					Gs[c_tensor_s + i * c_offset_s * 4 + 2 * c_offset_s] = G_tile[k * 16 + i * 4 + 2];
					Gs[c_tensor_s + i * c_offset_s * 4 + 3 * c_offset_s] = G_tile[k * 16 + i * 4 + 3];
				}

				c_tensor_s += BLOCK_N;
			}
		}
		__syncthreads();

		prefetch_X_frag(X_frag, A_frag, X_frag_offset, tx, access_s[0][tx], access_s[1][tx]);
		prefetch_G_frag(G_frag, B_frag, G_frag_offset, tx, access_f_s[0][tx], access_f_s[1][tx]);

#pragma unroll
		for (int i = 0; i < BLOCK_C; i++) {

			if (i < (BLOCK_C - 1)) {
				A_frag += BLOCK_N / 4;
				B_frag += BLOCK_K / 4;

				prefetch_X_frag(X_frag2, A_frag, X_frag_offset, tx, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
				prefetch_G_frag(G_frag2, B_frag, G_frag_offset, tx, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);
			}

			outer_product(X_frag, G_frag, accu);

			swap = X_frag; X_frag = X_frag2; X_frag2 = swap;
			swap = G_frag; G_frag = G_frag2; G_frag2 = swap;
		}

		//ic += BLOCK_C
		G += BLOCK_C * OC * 16;//[ic, fh, fw, oc] -> [ic, oc, fh, fw]
		X += BLOCK_C;//[ic, ih, iw, n] -> [n, ih, iw, ic]

		if (oic < (IC - BLOCK_C)) {
			const int ic = ty;
			//======[prefetch_G_tile=]===================================================
			{
				const int g0 = (ic * OC + oc) * 16;//G[ic     , oc, fh, fw]
				const int g1 = (ic * OC + oc + BLOCK_N) * 16;//G[ic + BN, oc, fh, fw]

				*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
				*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
				*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
				*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

				*(float4*)(G_tile + 16) = *(float4*)(G + g1);
				*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
				*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
				*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);
			}
			
			//======[prefetch_X_tile=]===================================================
			{
#pragma unroll
				for (int i = 0; i < 4; i++) {
#pragma unroll
					for (int j = 0; j < 4; j++) {
						int iht = ih + i;
						int iwt = iw + j;

						int x0 = ((n*IH + iht)*IW + iwt)*IC + ic;
						bool lx = (iht >= 0) && (iht < IH) && (iwt >= 0) && (iwt < IW);

						X_tile[(i << 2) + j] = (lx ? X[x0] : 0);//[n, ih, iw, ic]
					}
				}
			}
		}
		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_Y_tile(accu, shared_mem,
		tx, ty,
		Y,
		bx, by, bz,
		OH, OW, OC,
		tiles_dim, N,
		X_frag_mem, G_frag_mem);
}

#endif


#ifndef WINOGRAD2D_KERNEL5
#define WINOGRAD2D_KERNEL5

//Size = 9, Time = 2.39889 msec, Performace = 8056.81 GFlop/s
//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int tiles_dim, int tiles_2d_dim)
{
	int ph = 1, pw = 1;

	__shared__ float shared_mem[(16 * BLOCK_C*BLOCK_N) + (16 * BLOCK_C*BLOCK_K)];
	float *Xs = (float*)shared_mem;                       //Xs[16, BLOCK_C, BLOCK_N]
	float *Gs = (float*)&shared_mem[16 * BLOCK_C*BLOCK_N];//Gs[16, BLOCK_C, BLOCK_K]

	float4 accu[2][16] = { 0.0f };  // Accumulators 

	//----------------------------------------
	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;

	//<<< dim3(N / BN, tiles_2d_dim, OC / BK), dim3(BN, BC), 0 >>>
	const int oc = (bz * BLOCK_K) + tx;
	const int n = (bx * BLOCK_N) + tx;
	const int ohs = by / tiles_dim, ows = by - ohs * tiles_dim;
	const int oh = (ohs << 1), ow = (ows << 1);
	const int ih = oh - ph, iw = ow - pw;
	//----------------------------------------

	float X_tile[16]; // Prefetch input from GMEM
	float G_tile[32]; // Prefetch filter from GMEM

	float4 X_frag_mem[8], *X_frag = (float4*)X_frag_mem, *X_frag2 = (float4*)(X_frag + 4);
	float4 G_frag_mem[8], *G_frag = (float4*)G_frag_mem, *G_frag2 = (float4*)(G_frag + 4);
	float4 *swap;

	const int ic = ty;

	//======[prefetch_G_tile=]===================================================
	{
		const int g0 = (ic * OC + (oc)) * 16;          //G[ic     , oc, fh, fw]
		const int g1 = (ic * OC + (oc + BLOCK_N)) * 16;//G[ic + BN, oc, fh, fw]

		*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
		*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
		*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
		*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

		*(float4*)(G_tile + 16) = *(float4*)(G + g1);
		*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
		*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
		*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);
	}

	//======[prefetch_X_tile=]===================================================
	{
#pragma unroll
		for (int i = 0; i < 4; i++) {
#pragma unroll
			for (int j = 0; j < 4; j++) {
				int iht = ih + i;
				int iwt = iw + j;

				int x0 = ((n*IH + iht)*IW + iwt)*IC + ic;
				bool lx = (iht >= 0) && (iht < IH) && (iwt >= 0) && (iwt < IW);

				X_tile[(i << 2) + j] = (lx ? X[x0] : 0);//[n, ih, iw, ic]
			}
		}
	}

	int X_frag_offset = 2 * (BLOCK_C * BLOCK_N);
	int G_frag_offset = 2 * (BLOCK_C * BLOCK_K);

	for (int oic = 0; oic < IC; oic += BLOCK_C)
	{
		float4 *A_frag = (float4*)(Xs + ty * BLOCK_C*BLOCK_N);
		float4 *B_frag = (float4*)(Gs + ty * BLOCK_C*BLOCK_K);

		//======[load_and_transform_X_tile]======================================================
		{
			float xb[3];

#pragma unroll
			for (int j = 0; j < 4; j++) {
				xb[0] = X_tile[j];
				xb[1] = X_tile[j + 4];
				xb[2] = X_tile[j + 8];

				X_tile[j] = xb[0] - xb[2];//xj0 - xj2
				X_tile[j + 4] = xb[1] + xb[2];//xj1 + xj2
				X_tile[j + 8] = xb[2] - xb[1];//xj2 - xj1
				X_tile[j + 12] = xb[1] - X_tile[j + 12];//xj1 - xj3
			}

			//Xs[16, BLOCK_C, BLOCK_N]
			int c_offset = BLOCK_N * BLOCK_C;
			int c_tensor = ty * BLOCK_N + tx;//[ty, tx]

#pragma unroll
			for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
				Xs[c_tensor + i * c_offset * 4] = X_tile[(i << 2) + 0] - X_tile[(i << 2) + 2];//xi0 - xi2
				Xs[c_tensor + i * c_offset * 4 + c_offset] = X_tile[(i << 2) + 1] + X_tile[(i << 2) + 2];//xi1 + xi2
				Xs[c_tensor + i * c_offset * 4 + 2 * c_offset] = X_tile[(i << 2) + 2] - X_tile[(i << 2) + 1];//xi2 - xi1
				Xs[c_tensor + i * c_offset * 4 + 3 * c_offset] = X_tile[(i << 2) + 1] - X_tile[(i << 2) + 3];//xi1 - xi3
			}
		}

		//======[load_G_tile]====================================================================
		{
			//Gs[16, BLOCK_C, BLOCK_K]
			int c_tensor_s = ty * BLOCK_K + tx; //[ty, tx]
			int c_offset_s = BLOCK_K * BLOCK_C;

			for (int k = 0; k < 2; k++) { // prefetch 2 filter tiles/thread
#pragma unroll
				for (int i = 0; i < 4; i++) {
					Gs[c_tensor_s + i * c_offset_s * 4 + 0 * c_offset_s] = G_tile[k * 16 + i * 4 + 0];
					Gs[c_tensor_s + i * c_offset_s * 4 + 1 * c_offset_s] = G_tile[k * 16 + i * 4 + 1];
					Gs[c_tensor_s + i * c_offset_s * 4 + 2 * c_offset_s] = G_tile[k * 16 + i * 4 + 2];
					Gs[c_tensor_s + i * c_offset_s * 4 + 3 * c_offset_s] = G_tile[k * 16 + i * 4 + 3];
				}

				c_tensor_s += BLOCK_N;
			}
		}
		__syncthreads();

		//======[prefetch_X_frag]=================================================================
		{
			const int offset1 = access_s[0][tx];
			const int offset2 = access_s[1][tx];

			*((float4*)(X_frag    )) = *(A_frag + offset1); //ld_shared(A_frag + offset1);
			*((float4*)(X_frag + 1)) = *(A_frag + offset2);
			*((float4*)(X_frag + 2)) = *(A_frag + X_frag_offset + offset1);
			*((float4*)(X_frag + 3)) = *(A_frag + X_frag_offset + offset2); //3=2+1
		}

		//======[prefetch_G_frag]=================================================================
		{
			const int offset1 = access_f_s[0][tx];
			const int offset2 = access_f_s[1][tx];

			*((float4*)(G_frag    )) = *(B_frag + offset1);
			*((float4*)(G_frag + 1)) = *(B_frag + offset2);
			*((float4*)(G_frag + 2)) = *(B_frag + G_frag_offset + offset1);
			*((float4*)(G_frag + 3)) = *(B_frag + G_frag_offset + offset2);
		}

#pragma unroll
		for (int i = 0; i < BLOCK_C; i++) {

			if (i < (BLOCK_C - 1)) {
				A_frag += BLOCK_N / 4;
				B_frag += BLOCK_K / 4;

				//======[prefetch_X_frag]=================================================================
				{
					const int offset1 = access_s[0][tx];
					const int offset2 = access_s[1][tx];

					*((float4*)(X_frag2    )) = *(A_frag + offset1); //ld_shared(A_frag + offset1);
					*((float4*)(X_frag2 + 1)) = *(A_frag + offset2);
					*((float4*)(X_frag2 + 2)) = *(A_frag + X_frag_offset + offset1);
					*((float4*)(X_frag2 + 3)) = *(A_frag + X_frag_offset + offset2); //3=2+1
				}

				//======[prefetch_G_frag]=================================================================
				{
					const int offset1 = access_f_s[0][tx];
					const int offset2 = access_f_s[1][tx];

					*((float4*)(G_frag2)) = *(B_frag + offset1);
					*((float4*)(G_frag2 + 1)) = *(B_frag + offset2);
					*((float4*)(G_frag2 + 2)) = *(B_frag + G_frag_offset + offset1);
					*((float4*)(G_frag2 + 3)) = *(B_frag + G_frag_offset + offset2);
				}
			}

			outer_product(X_frag, G_frag, accu);

			swap = X_frag; X_frag = X_frag2; X_frag2 = swap;
			swap = G_frag; G_frag = G_frag2; G_frag2 = swap;
		}

		//ic += BLOCK_C
		G += BLOCK_C * OC * 16;//[ic, fh, fw, oc] -> [ic, oc, fh, fw]
		X += BLOCK_C;//[ic, ih, iw, n] -> [n, ih, iw, ic]

		if (oic < (IC - BLOCK_C)) {
			const int ic = ty;
			//======[prefetch_G_tile=]===================================================
			{
				const int g0 = (ic * OC + oc) * 16;//G[ic     , oc, fh, fw]
				const int g1 = (ic * OC + oc + BLOCK_N) * 16;//G[ic + BN, oc, fh, fw]

				*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
				*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
				*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
				*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

				*(float4*)(G_tile + 16) = *(float4*)(G + g1);
				*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
				*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
				*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);
			}

			//======[prefetch_X_tile=]===================================================
			{
#pragma unroll
				for (int i = 0; i < 4; i++) {
#pragma unroll
					for (int j = 0; j < 4; j++) {
						int iht = ih + i;
						int iwt = iw + j;

						int x0 = ((n*IH + iht)*IW + iwt)*IC + ic;
						bool lx = (iht >= 0) && (iht < IH) && (iwt >= 0) && (iwt < IW);

						X_tile[(i << 2) + j] = (lx ? X[x0] : 0);//[n, ih, iw, ic]
					}
				}
			}
		}
		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_Y_tile(accu, shared_mem,
		tx, ty,
		Y,
		bx, by, bz,
		OH, OW, OC,
		tiles_dim, N,
		X_frag_mem, G_frag_mem);
}

#endif


#ifndef WINOGRAD2D_KERNEL6
#define WINOGRAD2D_KERNEL6

//242 register
//Size = 9, Time = 2.39889 msec, Performace = 8056.81 GFlop/s
//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int tiles_dim, int tiles_2d_dim)
{
	int ph = 1, pw = 1;

	__shared__ float shared_mem[(16 * BLOCK_C*BLOCK_N) + (16 * BLOCK_C*BLOCK_K)];
	float *Xs = (float*)shared_mem;                       //Xs[16, BLOCK_C, BLOCK_N] [8, 16, 32]
	float *Gs = (float*)&shared_mem[16 * BLOCK_C*BLOCK_N];//Gs[16, BLOCK_C, BLOCK_K] [8, 16, 64]

	float4 accu[2][16] = { 0.0f };  // Accumulators 

	//----------------------------------------
	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;
	const int oc = (bz * BLOCK_K) + tx;

	//prepare for X[N, IH, IW, IC]
	const int n = (bx * BLOCK_N) + tx;
	const int ohs = by / tiles_dim, ows = by - ohs * tiles_dim;
	const int oh = (ohs << 1), ow = (ows << 1);
	const int ih = oh - ph, iw = ow - pw;
	X += ((n*IH + ih)*IW + iw + 1)*IC;//X[n, ih, iw + 1, 0]
	//----------------------------------------

	float X_tile[16]; // Prefetch input from GMEM
	float G_tile[32]; // Prefetch filter from GMEM

	float4 X_frag_mem[8], *X_frag = (float4*)X_frag_mem, *X_frag2 = (float4*)(X_frag + 4);
	float4 G_frag_mem[8], *G_frag = (float4*)G_frag_mem, *G_frag2 = (float4*)(G_frag + 4);
	float4 *swap;

	const int ic = ty;

	//======[prefetch_G_tile]===================================================
	{
		const int g0 = (ic * OC + (oc)) * 16;          //G[ic     , oc, fh, fw]
		const int g1 = (ic * OC + (oc + BLOCK_N)) * 16;//G[ic + BN, oc, fh, fw]

		*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
		*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
		*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
		*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

		*(float4*)(G_tile + 16) = *(float4*)(G + g1);
		*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
		*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
		*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);
	}

	//======[prefetch_X_tile]===================================================
	{
#pragma unroll
		for (int i = 0; i < 4; i++) {
			const int x1 = i * IW*IC + ic;
			const int x0 = x1 - IC;
			const int x2 = x1 + IC;
			const int x3 = x1 + (IC << 1);

			bool lx = (ih >= -i) && (ih < IH - i);
			bool lx0 = lx && (iw >= 0) && (iw < IW);
			bool lx1 = lx && (iw >= -1) && (iw < IW - 1);
			bool lx2 = lx && (iw >= -2) && (iw < IW - 2);
			bool lx3 = lx && (iw >= -3) && (iw < IW - 3);

			X_tile[(i << 2)] = (lx0 ? X[x0] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 1] = (lx1 ? X[x1] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 2] = (lx2 ? X[x2] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 3] = (lx3 ? X[x3] : 0);//[n, ih, iw, ic]
		}
	}

	int X_frag_offset = 2 * (BLOCK_C * BLOCK_N);
	int G_frag_offset = 2 * (BLOCK_C * BLOCK_K);

	const int xs_offset1 = access_s[0][tx];
	const int xs_offset2 = access_s[1][tx];
	const int gs_offset1 = access_f_s[0][tx];
	const int gs_offset2 = access_f_s[1][tx];

	for (int oic = 0; oic < IC; oic += BLOCK_C)
	{
		float4 *A_frag = (float4*)(Xs + ty * BLOCK_C*BLOCK_N);
		float4 *B_frag = (float4*)(Gs + ty * BLOCK_C*BLOCK_K);

		//======[load_and_transform_X_tile]======================================================
		{
			float xb[3];
#pragma unroll
			for (int j = 0; j < 4; j++) {
				xb[0] = X_tile[j];
				xb[1] = X_tile[j + 4];
				xb[2] = X_tile[j + 8];

				X_tile[j] = xb[0] - xb[2];//xj0 - xj2
				X_tile[j + 4] = xb[1] + xb[2];//xj1 + xj2
				X_tile[j + 8] = xb[2] - xb[1];//xj2 - xj1
				X_tile[j + 12] = xb[1] - X_tile[j + 12];//xj1 - xj3
			}

			//Xs[16, BLOCK_C, BLOCK_N]
			int c_offset = BLOCK_N * BLOCK_C;
			int c_tensor = ty * BLOCK_N + tx;//[ty, tx]

#pragma unroll
			for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
				Xs[c_tensor + i * c_offset * 4               ] = X_tile[(i << 2) + 0] - X_tile[(i << 2) + 2];//xi0 - xi2
				Xs[c_tensor + i * c_offset * 4 +     c_offset] = X_tile[(i << 2) + 1] + X_tile[(i << 2) + 2];//xi1 + xi2
				Xs[c_tensor + i * c_offset * 4 + 2 * c_offset] = X_tile[(i << 2) + 2] - X_tile[(i << 2) + 1];//xi2 - xi1
				Xs[c_tensor + i * c_offset * 4 + 3 * c_offset] = X_tile[(i << 2) + 1] - X_tile[(i << 2) + 3];//xi1 - xi3
			}
		}

		//======[load_G_tile]====================================================================
		{
			//Gs[16, BLOCK_C, BLOCK_K]
			int c_tensor_s = ty * BLOCK_K + tx; //[ty, tx]
			int c_offset_s = BLOCK_K * BLOCK_C;

			for (int k = 0; k < 2; k++) { // prefetch 2 filter tiles/thread
#pragma unroll
				for (int i = 0; i < 4; i++) {
					Gs[c_tensor_s + i * c_offset_s * 4 + 0 * c_offset_s] = G_tile[k * 16 + i * 4 + 0];
					Gs[c_tensor_s + i * c_offset_s * 4 + 1 * c_offset_s] = G_tile[k * 16 + i * 4 + 1];
					Gs[c_tensor_s + i * c_offset_s * 4 + 2 * c_offset_s] = G_tile[k * 16 + i * 4 + 2];
					Gs[c_tensor_s + i * c_offset_s * 4 + 3 * c_offset_s] = G_tile[k * 16 + i * 4 + 3];
				}

				c_tensor_s += BLOCK_N;
			}
		}
		__syncthreads();

		//======[prefetch {X, G} frag]=================================================================
		{
			//const int xs_offset1 = access_s[0][tx];
			//const int xs_offset2 = access_s[1][tx];
			//const int gs_offset1 = access_f_s[0][tx];
			//const int gs_offset2 = access_f_s[1][tx];

			//float4 *A_frag = (float4*)(Xs + ty * BLOCK_C*BLOCK_N);//[16, BC, BN]
			//float4 *B_frag = (float4*)(Gs + ty * BLOCK_C*BLOCK_K);//[16, BC, BK]

			//load 16 elements from Xs
			*((float4*)(X_frag    )) = *(A_frag + xs_offset1);                 //[ty, xs_offset1]
			*((float4*)(X_frag + 1)) = *(A_frag + xs_offset2);                 //[ty, xs_offset2]
			*((float4*)(X_frag + 2)) = *(A_frag + xs_offset1 + X_frag_offset);
			*((float4*)(X_frag + 3)) = *(A_frag + xs_offset2 + X_frag_offset);

			//load 16 elements from Gs
			*((float4*)(G_frag    )) = *(B_frag + gs_offset1);                //[ty, gs_offset2]
			*((float4*)(G_frag + 1)) = *(B_frag + gs_offset2);                //[ty, gs_offset2]
			*((float4*)(G_frag + 2)) = *(B_frag + gs_offset1 + G_frag_offset);
			*((float4*)(G_frag + 3)) = *(B_frag + gs_offset2 + G_frag_offset);
		}

#pragma unroll
		for (int i = 0; i < BLOCK_C; i++) {
			if (i < (BLOCK_C - 1)) {
				A_frag += BLOCK_N / 4;
				B_frag += BLOCK_K / 4;//[k]

				//======[prefetch {X, G} frag]=================================================================
				{
					//load 16 elements from Xs
					*((float4*)(X_frag2)) = *(A_frag + xs_offset1);
					*((float4*)(X_frag2 + 1)) = *(A_frag + xs_offset2);
					*((float4*)(X_frag2 + 2)) = *(A_frag + xs_offset1 + X_frag_offset);
					*((float4*)(X_frag2 + 3)) = *(A_frag + xs_offset2 + X_frag_offset);

					//load 16 elements from Gs
					*((float4*)(G_frag2)) = *(B_frag + gs_offset1);
					*((float4*)(G_frag2 + 1)) = *(B_frag + gs_offset2);
					*((float4*)(G_frag2 + 2)) = *(B_frag + gs_offset1 + G_frag_offset);
					*((float4*)(G_frag2 + 3)) = *(B_frag + gs_offset2 + G_frag_offset);
				}
			}

			outer_product(X_frag, G_frag, accu);

			swap = X_frag; X_frag = X_frag2; X_frag2 = swap;
			swap = G_frag; G_frag = G_frag2; G_frag2 = swap;
		}

		if (oic < (IC - BLOCK_C)) {
			//ic += BLOCK_C
			G += BLOCK_C * OC * 16;//[ic, fh, fw, oc] -> [ic, oc, fh, fw]
			X += BLOCK_C;//[ic, ih, iw, n] -> [n, ih, iw, ic]

			const int ic = ty;
			//======[prefetch_G_tile]===================================================
			{
				const int g0 = (ic * OC + oc) * 16;//G[ic     , oc, fh, fw]
				const int g1 = (ic * OC + oc + BLOCK_N) * 16;//G[ic + BN, oc, fh, fw]

				*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
				*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
				*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
				*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

				*(float4*)(G_tile + 16) = *(float4*)(G + g1);
				*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
				*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
				*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);
			}

			//======[prefetch_X_tile]===================================================
			{
#pragma unroll
				for (int i = 0; i < 4; i++) {
					const int x1 = i * IW  * IC + ic;
					const int x0 = x1 - IC;
					const int x2 = x1 + IC;
					const int x3 = x1 + (IC << 1);

					bool lx = (ih >= -i) && (ih < IH - i);
					bool lx0 = lx && (iw >=  0) && (iw < IW    );
					bool lx1 = lx && (iw >= -1) && (iw < IW - 1);
					bool lx2 = lx && (iw >= -2) && (iw < IW - 2);
					bool lx3 = lx && (iw >= -3) && (iw < IW - 3);

					X_tile[(i << 2)    ] = (lx0 ? X[x0] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 1] = (lx1 ? X[x1] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 2] = (lx2 ? X[x2] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 3] = (lx3 ? X[x3] : 0);//[n, ih, iw, ic]
				}
			}
		}
		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_Y_tile(accu, shared_mem,
		tx, ty,
		Y,
		bx, by, bz,
		OH, OW, OC,
		tiles_dim, N,
		X_frag_mem, G_frag_mem);
}

#endif


#ifndef WINOGRAD2D_KERNEL7
#define WINOGRAD2D_KERNEL7

//242 register
//Size = 9, Time = 2.39889 msec, Performace = 8056.81 GFlop/s
//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int tiles_dim, int tiles_2d_dim)
{
	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;

	int ph = 1, pw = 1;

	__shared__ float shared_mem[(16 * BLOCK_C*BLOCK_N) + (16 * BLOCK_C*BLOCK_K)];
	float *Xs = (float*)shared_mem;                       //Xs[16, BLOCK_C, BLOCK_N]
	float *Gs = (float*)&shared_mem[16 * BLOCK_C*BLOCK_N];//Gs[16, BLOCK_C, BLOCK_K]

	float4 X_frag_mem[8], *X_frag = (float4*)X_frag_mem, *X_frag2 = (float4*)(X_frag + 4);
	float4 G_frag_mem[8], *G_frag = (float4*)G_frag_mem, *G_frag2 = (float4*)(G_frag + 4);
	float4 *swap;

	float4 accu[2][16] = { 0.0f };  // Accumulators 

	//prepare for G[ic, oc, 4, 4]
	const int oc = (bz * BLOCK_K) + tx;
	G += oc << 4;//G[0, oc, 0, 0]

	//prepare for X[N, IH, IW, IC]
	const int n = (bx * BLOCK_N) + tx;
	const int ohs = by / tiles_dim, ows = by - ohs * tiles_dim;
	const int oh = (ohs << 1), ow = (ows << 1);
	const int ih = oh - ph, iw = ow - pw;
	X += ((n*IH + ih)*IW + iw + 1)*IC;//X[n, ih, iw + 1, 0]

	float X_tile[16], G_tile[32];

	//======[prefetch {G, X} tile]===================================================
	{
		const int ic = ty;
		const int g0 = (ic * OC) * 16;   //G[ic     , oc, fh, fw]
		const int g1 = g0 + BLOCK_N * 16;//G[ic + BN, oc, fh, fw]

		*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
		*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
		*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
		*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

		*(float4*)(G_tile + 16) = *(float4*)(G + g1);
		*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
		*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
		*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);

#pragma unroll
		for (int i = 0; i < 4; i++) {
			const int x1 = i * IW*IC + ic;
			const int x0 = x1 - IC;
			const int x2 = x1 + IC;
			const int x3 = x1 + (IC << 1);

			bool lx = (ih >= -i) && (ih < IH - i);
			bool lx0 = lx && (iw >= 0) && (iw < IW);
			bool lx1 = lx && (iw >= -1) && (iw < IW - 1);
			bool lx2 = lx && (iw >= -2) && (iw < IW - 2);
			bool lx3 = lx && (iw >= -3) && (iw < IW - 3);

			X_tile[(i << 2)    ] = (lx0 ? X[x0] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 1] = (lx1 ? X[x1] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 2] = (lx2 ? X[x2] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 3] = (lx3 ? X[x3] : 0);//[n, ih, iw, ic]
		}
	}

	int X_frag_offset = 2 * (BLOCK_C * BLOCK_N);
	int G_frag_offset = 2 * (BLOCK_C * BLOCK_K);

	const int xs_offset1 = access_s[0][tx];
	const int xs_offset2 = access_s[1][tx];
	const int gs_offset1 = access_f_s[0][tx];
	const int gs_offset2 = access_f_s[1][tx];

	for (int oic = 0; oic < IC; oic += BLOCK_C)
	{
		float4 *A_frag = (float4*)(Xs + ty * BLOCK_C*BLOCK_N);
		float4 *B_frag = (float4*)(Gs + ty * BLOCK_C*BLOCK_K);

		//======[load_and_transform_X_tile]======================================================
		{
			float xb[3];
#pragma unroll
			for (int j = 0; j < 4; j++) {
				xb[0] = X_tile[j];
				xb[1] = X_tile[j + 4];
				xb[2] = X_tile[j + 8];

				X_tile[j] = xb[0] - xb[2];//xj0 - xj2
				X_tile[j + 4] = xb[1] + xb[2];//xj1 + xj2
				X_tile[j + 8] = xb[2] - xb[1];//xj2 - xj1
				X_tile[j + 12] = xb[1] - X_tile[j + 12];//xj1 - xj3
			}

			//Xs[16, BLOCK_C, BLOCK_N]
			int c_offset = BLOCK_N * BLOCK_C;
			int c_tensor = ty * BLOCK_N + tx;//[ty, tx]

#pragma unroll
			for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
				Xs[c_tensor + i * c_offset * 4] = X_tile[(i << 2) + 0] - X_tile[(i << 2) + 2];//xi0 - xi2
				Xs[c_tensor + i * c_offset * 4 + c_offset] = X_tile[(i << 2) + 1] + X_tile[(i << 2) + 2];//xi1 + xi2
				Xs[c_tensor + i * c_offset * 4 + 2 * c_offset] = X_tile[(i << 2) + 2] - X_tile[(i << 2) + 1];//xi2 - xi1
				Xs[c_tensor + i * c_offset * 4 + 3 * c_offset] = X_tile[(i << 2) + 1] - X_tile[(i << 2) + 3];//xi1 - xi3
			}
		}

		//======[load_G_tile]====================================================================
		{
			//Gs[16, BLOCK_C, BLOCK_K]
			int c_tensor_s = ty * BLOCK_K + tx; //[ty, tx]
			int c_offset_s = BLOCK_K * BLOCK_C;

			for (int k = 0; k < 2; k++) { // prefetch 2 filter tiles/thread
#pragma unroll
				for (int i = 0; i < 4; i++) {
					Gs[c_tensor_s + i * c_offset_s * 4 + 0 * c_offset_s] = G_tile[k * 16 + i * 4 + 0];
					Gs[c_tensor_s + i * c_offset_s * 4 + 1 * c_offset_s] = G_tile[k * 16 + i * 4 + 1];
					Gs[c_tensor_s + i * c_offset_s * 4 + 2 * c_offset_s] = G_tile[k * 16 + i * 4 + 2];
					Gs[c_tensor_s + i * c_offset_s * 4 + 3 * c_offset_s] = G_tile[k * 16 + i * 4 + 3];
				}

				c_tensor_s += BLOCK_N;
			}
		}
		__syncthreads();

		//======[prefetch {X, G} frag]=================================================================
		{
			//load 16 elements from Xs
			*((float4*)(X_frag)) = *(A_frag + xs_offset1);
			*((float4*)(X_frag + 1)) = *(A_frag + xs_offset2);
			*((float4*)(X_frag + 2)) = *(A_frag + xs_offset1 + X_frag_offset);
			*((float4*)(X_frag + 3)) = *(A_frag + xs_offset2 + X_frag_offset);

			//load 16 elements from Gs
			*((float4*)(G_frag)) = *(B_frag + gs_offset1);
			*((float4*)(G_frag + 1)) = *(B_frag + gs_offset2);
			*((float4*)(G_frag + 2)) = *(B_frag + gs_offset1 + G_frag_offset);
			*((float4*)(G_frag + 3)) = *(B_frag + gs_offset2 + G_frag_offset);
		}

#pragma unroll
		for (int i = 0; i < BLOCK_C; i++) {
			if (i < (BLOCK_C - 1)) {
				A_frag += BLOCK_N / 4;
				B_frag += BLOCK_K / 4;//[k]

				//======[prefetch {X, G} frag]=================================================================
				{
					//load 16 elements from Xs
					*((float4*)(X_frag2)) = *(A_frag + xs_offset1);
					*((float4*)(X_frag2 + 1)) = *(A_frag + xs_offset2);
					*((float4*)(X_frag2 + 2)) = *(A_frag + xs_offset1 + X_frag_offset);
					*((float4*)(X_frag2 + 3)) = *(A_frag + xs_offset2 + X_frag_offset);

					//load 16 elements from Gs
					*((float4*)(G_frag2)) = *(B_frag + gs_offset1);
					*((float4*)(G_frag2 + 1)) = *(B_frag + gs_offset2);
					*((float4*)(G_frag2 + 2)) = *(B_frag + gs_offset1 + G_frag_offset);
					*((float4*)(G_frag2 + 3)) = *(B_frag + gs_offset2 + G_frag_offset);
				}
			}

			outer_product2(X_frag, G_frag, accu);

			swap = X_frag; X_frag = X_frag2; X_frag2 = swap;
			swap = G_frag; G_frag = G_frag2; G_frag2 = swap;
		}

		if (oic < (IC - BLOCK_C)) {
			//ic += BLOCK_C
			G += BLOCK_C * OC * 16;//[ic, fh, fw, oc] -> [ic, oc, fh, fw]
			X += BLOCK_C;//[ic, ih, iw, n] -> [n, ih, iw, ic]
			
			//======[prefetch {G, X} tile]===================================================
			{
				const int ic = ty;
				const int g0 = (ic * OC) * 16;    //G[ic     , oc, fh, fw]
				const int g1 = g0 + BLOCK_N * 16;//G[ic + BN, oc, fh, fw]

				*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
				*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
				*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
				*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

				*(float4*)(G_tile + 16) = *(float4*)(G + g1);
				*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
				*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
				*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);

				for (int i = 0; i < 4; i++) {
					const int x1 = i * IW  * IC + ic;
					const int x0 = x1 - IC;
					const int x2 = x1 + IC;
					const int x3 = x1 + (IC << 1);

					bool lx = (ih >= -i) && (ih < IH - i);
					bool lx0 = lx && (iw >= 0) && (iw < IW);
					bool lx1 = lx && (iw >= -1) && (iw < IW - 1);
					bool lx2 = lx && (iw >= -2) && (iw < IW - 2);
					bool lx3 = lx && (iw >= -3) && (iw < IW - 3);

					X_tile[(i << 2)] = (lx0 ? X[x0] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 1] = (lx1 ? X[x1] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 2] = (lx2 ? X[x2] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 3] = (lx3 ? X[x3] : 0);//[n, ih, iw, ic]
				}
			}
		}
		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_Y_tile2(
		accu, shared_mem,
		tx, ty,
		Y,
		bx, by, bz,
		OH, OW, OC,
		tiles_dim,
		X_frag_mem, G_frag_mem);
}

#endif


#ifndef WINOGRAD2D_KERNEL8
#define WINOGRAD2D_KERNEL8

//242 register
//Size = 9, Time = 2.39889 msec, Performace = 8056.81 GFlop/s
//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int tiles_dim, int tiles_2d_dim)
{
	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;

	int ph = 1, pw = 1;

	__shared__ float shared_mem[(16 * BLOCK_C*BLOCK_N) + (16 * BLOCK_C*BLOCK_K)];
	float *Xs = (float*)shared_mem;                       //Xs[16, BLOCK_C, BLOCK_N]
	float *Gs = (float*)&shared_mem[16 * BLOCK_C*BLOCK_N];//Gs[16, BLOCK_C, BLOCK_K]

	float4 X_frag_mem[8], *Xf = (float4*)X_frag_mem;
	float4 G_frag_mem[8], *Gf = (float4*)G_frag_mem;

	float4 a[2][16] = { 0.0f };  // Accumulators 

	//prepare for G[ic, oc, 4, 4]
	const int oc = (bz * BLOCK_K) + tx;
	G += oc << 4;//G[0, oc, 0, 0]

	//prepare for X[N, IH, IW, IC]
	const int n = (bx * BLOCK_N) + tx;
	const int ohs = by / tiles_dim, ows = by - ohs * tiles_dim;
	const int oh = (ohs << 1), ow = (ows << 1);
	const int ih = oh - ph, iw = ow - pw;
	X += ((n*IH + ih)*IW + iw + 1)*IC;//X[n, ih, iw + 1, 0]

	float X_tile[16], G_tile[32];

	//======[prefetch {G, X} tile]===================================================
	{
		const int ic = ty;
		const int g0 = (ic * OC) * 16;   //G[ic     , oc, fh, fw]
		const int g1 = g0 + BLOCK_N * 16;//G[ic + BN, oc, fh, fw]

		*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
		*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
		*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
		*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

		*(float4*)(G_tile + 16) = *(float4*)(G + g1);
		*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
		*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
		*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);

#pragma unroll
		for (int i = 0; i < 4; i++) {
			const int x1 = i * IW*IC + ic;
			const int x0 = x1 - IC;
			const int x2 = x1 + IC;
			const int x3 = x1 + (IC << 1);

			bool lx = (ih >= -i) && (ih < IH - i);
			bool lx0 = lx && (iw >= 0) && (iw < IW);
			bool lx1 = lx && (iw >= -1) && (iw < IW - 1);
			bool lx2 = lx && (iw >= -2) && (iw < IW - 2);
			bool lx3 = lx && (iw >= -3) && (iw < IW - 3);

			X_tile[(i << 2)] = (lx0 ? X[x0] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 1] = (lx1 ? X[x1] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 2] = (lx2 ? X[x2] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 3] = (lx3 ? X[x3] : 0);//[n, ih, iw, ic]
		}
	}

	const int X_frag_offset = 2 * (BLOCK_C * BLOCK_N);
	const int G_frag_offset = 2 * (BLOCK_C * BLOCK_K);

	const int xs_offset1 = access_s[0][tx], xs_offset2 = access_s[1][tx];
	const int gs_offset1 = access_f_s[0][tx], gs_offset2 = access_f_s[1][tx];

	for (int oic = 0; oic < IC; oic += BLOCK_C)
	{
		float4 *A_frag = (float4*)(Xs + ty * BLOCK_C*BLOCK_N);
		float4 *B_frag = (float4*)(Gs + ty * BLOCK_C*BLOCK_K);

		//======[load_and_transform_{X£¬ G}_tile]======================================================
		{
			//======[load_and_transform_X_tile]======================================================
			float xb[3];
#pragma unroll
			for (int j = 0; j < 4; j++) {
				xb[0] = X_tile[j];
				xb[1] = X_tile[j + 4];
				xb[2] = X_tile[j + 8];

				X_tile[j] = xb[0] - xb[2];//xj0 - xj2
				X_tile[j + 4] = xb[1] + xb[2];//xj1 + xj2
				X_tile[j + 8] = xb[2] - xb[1];//xj2 - xj1
				X_tile[j + 12] = xb[1] - X_tile[j + 12];//xj1 - xj3
			}

			//Xs[16, BLOCK_C, BLOCK_N]
			int c_offset = BLOCK_N * BLOCK_C;
			int c_tensor = ty * BLOCK_N + tx;//[ty, tx]

#pragma unroll
			for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
				Xs[c_tensor + i * c_offset * 4] = X_tile[(i << 2) + 0] - X_tile[(i << 2) + 2];//xi0 - xi2
				Xs[c_tensor + i * c_offset * 4 + c_offset] = X_tile[(i << 2) + 1] + X_tile[(i << 2) + 2];//xi1 + xi2
				Xs[c_tensor + i * c_offset * 4 + 2 * c_offset] = X_tile[(i << 2) + 2] - X_tile[(i << 2) + 1];//xi2 - xi1
				Xs[c_tensor + i * c_offset * 4 + 3 * c_offset] = X_tile[(i << 2) + 1] - X_tile[(i << 2) + 3];//xi1 - xi3
			}
	
			//======[load_G_tile]====================================================================
			//Gs[16, BLOCK_C, BLOCK_K]
			int c_tensor_s = ty * BLOCK_K + tx; //[ty, tx]
			int c_offset_s = BLOCK_K * BLOCK_C;

			for (int k = 0; k < 2; k++) { // prefetch 2 filter tiles/thread
#pragma unroll
				for (int i = 0; i < 4; i++) {
					Gs[c_tensor_s + i * c_offset_s * 4 + 0 * c_offset_s] = G_tile[k * 16 + i * 4 + 0];
					Gs[c_tensor_s + i * c_offset_s * 4 + 1 * c_offset_s] = G_tile[k * 16 + i * 4 + 1];
					Gs[c_tensor_s + i * c_offset_s * 4 + 2 * c_offset_s] = G_tile[k * 16 + i * 4 + 2];
					Gs[c_tensor_s + i * c_offset_s * 4 + 3 * c_offset_s] = G_tile[k * 16 + i * 4 + 3];
				}

				c_tensor_s += BLOCK_N;
			}
		}
		__syncthreads();

#pragma unroll
		for (int i = 0; i < BLOCK_C; i++) 
		{
			//computing group1
			Xf[0] = *(A_frag + xs_offset1); Xf[1] = *(A_frag + xs_offset2);
			Gf[0] = *(B_frag + gs_offset1); Gf[1] = *(B_frag + gs_offset2);

			simdMM4_xzyw(a[0][0], Gf[0].x, Xf[0]); simdMM4_xzyw(a[0][1], Gf[0].x, Xf[1]);
			simdMM4_xzyw(a[0][2], Gf[0].y, Xf[0]); simdMM4_xzyw(a[0][3], Gf[0].y, Xf[1]);
			simdMM4_xzyw(a[0][4], Gf[0].z, Xf[0]); simdMM4_xzyw(a[0][5], Gf[0].z, Xf[1]);
			simdMM4_xzyw(a[0][6], Gf[0].w, Xf[0]); simdMM4_xzyw(a[0][7], Gf[0].w, Xf[1]);

			simdMM4_xzyw(a[0][8],  Gf[1].x, Xf[0]); simdMM4_xzyw(a[0][9],  Gf[1].x, Xf[1]);
			simdMM4_xzyw(a[0][10], Gf[1].y, Xf[0]); simdMM4_xzyw(a[0][11], Gf[1].y, Xf[1]);
			simdMM4_xzyw(a[0][12], Gf[1].z, Xf[0]); simdMM4_xzyw(a[0][13], Gf[1].z, Xf[1]);
			simdMM4_xzyw(a[0][14], Gf[1].w, Xf[0]); simdMM4_xzyw(a[0][15], Gf[1].w, Xf[1]);

			//computing group2
			Xf[2] = *(A_frag + xs_offset1 + X_frag_offset); Xf[3] = *(A_frag + xs_offset2 + X_frag_offset);
			Gf[2] = *(B_frag + gs_offset1 + G_frag_offset); Gf[3] = *(B_frag + gs_offset2 + G_frag_offset);

			simdMM4_xzyw(a[1][0], Gf[2].x, Xf[2]); simdMM4_xzyw(a[1][1], Gf[2].x, Xf[3]);
			simdMM4_xzyw(a[1][2], Gf[2].y, Xf[2]); simdMM4_xzyw(a[1][3], Gf[2].y, Xf[3]);
			simdMM4_xzyw(a[1][4], Gf[2].z, Xf[2]); simdMM4_xzyw(a[1][5], Gf[2].z, Xf[3]);
			simdMM4_xzyw(a[1][6], Gf[2].w, Xf[2]); simdMM4_xzyw(a[1][7], Gf[2].w, Xf[3]);

			simdMM4_xzyw(a[1][8],  Gf[3].x, Xf[2]); simdMM4_xzyw(a[1][9],  Gf[3].x, Xf[3]);
			simdMM4_xzyw(a[1][10], Gf[3].y, Xf[2]); simdMM4_xzyw(a[1][11], Gf[3].y, Xf[3]);
			simdMM4_xzyw(a[1][12], Gf[3].z, Xf[2]); simdMM4_xzyw(a[1][13], Gf[3].z, Xf[3]);
			simdMM4_xzyw(a[1][14], Gf[3].w, Xf[2]); simdMM4_xzyw(a[1][15], Gf[3].w, Xf[3]);

			A_frag += BLOCK_N / 4; B_frag += BLOCK_K / 4;//[k]
		}

		if (oic < (IC - BLOCK_C)) {
			//ic += BLOCK_C
			G += BLOCK_C * OC * 16;//[ic, fh, fw, oc] -> [ic, oc, fh, fw]
			X += BLOCK_C;//[ic, ih, iw, n] -> [n, ih, iw, ic]

			//======[prefetch {G, X} tile]===================================================
			{
				const int ic = ty;
				const int g0 = (ic * OC) * 16;    //G[ic     , oc, fh, fw]
				const int g1 = g0 + BLOCK_N * 16;//G[ic + BN, oc, fh, fw]

				*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
				*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
				*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
				*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

				*(float4*)(G_tile + 16) = *(float4*)(G + g1);
				*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
				*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
				*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);

				for (int i = 0; i < 4; i++) {
					const int x1 = i * IW  * IC + ic;
					const int x0 = x1 - IC;
					const int x2 = x1 + IC;
					const int x3 = x1 + (IC << 1);

					bool lx = (ih >= -i) && (ih < IH - i);
					bool lx0 = lx && (iw >= 0) && (iw < IW);
					bool lx1 = lx && (iw >= -1) && (iw < IW - 1);
					bool lx2 = lx && (iw >= -2) && (iw < IW - 2);
					bool lx3 = lx && (iw >= -3) && (iw < IW - 3);

					X_tile[(i << 2)] = (lx0 ? X[x0] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 1] = (lx1 ? X[x1] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 2] = (lx2 ? X[x2] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 3] = (lx3 ? X[x3] : 0);//[n, ih, iw, ic]
				}
			}
		}
		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_Y_tile2(
		a, shared_mem,
		tx, ty,
		Y,
		bx, by, bz,
		OH, OW, OC,
		tiles_dim,
		X_frag_mem, G_frag_mem);
}

#endif


#ifndef WINOGRAD2D_KERNEL9
#define WINOGRAD2D_KERNEL9

//242 register
//Size = 9, Time = 2.39889 msec, Performace = 8056.81 GFlop/s
//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int tiles_dim, int tiles_2d_dim)
{
	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;

	int ph = 1, pw = 1;

	__shared__ float shared_mem[(16 * BLOCK_C*BLOCK_N) + (16 * BLOCK_C*BLOCK_K)];
	float *Xs = (float*)shared_mem;                       //Xs[16, BLOCK_C, BLOCK_N]
	float *Gs = (float*)&shared_mem[16 * BLOCK_C*BLOCK_N];//Gs[16, BLOCK_C, BLOCK_K]

	float4 X_frag_mem[8], *Xf = (float4*)X_frag_mem;
	float4 G_frag_mem[8], *Gf = (float4*)G_frag_mem;

	float4 a[2][16] = { 0.0f };  // Accumulators 

	//prepare for G[IC, OC, 4, 4]
	const int oc = (bz * BLOCK_K) + tx;
	G += oc << 4;//G[0, oc, 0, 0]

	//prepare for X[N, IH, IW, IC]
	const int n = (bx * BLOCK_N) + tx;
	const int ohs = by / tiles_dim, ows = by - ohs * tiles_dim;
	const int oh = (ohs << 1), ow = (ows << 1);
	const int ih = oh - ph, iw = ow - pw;
	X += ((n*IH + ih)*IW + iw + 1)*IC;//X[n, ih, iw + 1, 0]

	float X_tile[16], G_tile[32];

	//======[prefetch {G, X} tile]===================================================
	{
		const int ic = ty;
		const int g0 = (ic * OC) * 16;   //G[ic     , oc, fh, fw]
		const int g1 = g0 + BLOCK_N * 16;//G[ic + BN, oc, fh, fw]

		*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
		*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
		*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
		*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

		*(float4*)(G_tile + 16) = *(float4*)(G + g1);
		*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
		*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
		*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);

#pragma unroll
		for (int i = 0; i < 4; i++) {
			const int x1 = i * IW*IC + ic;
			const int x0 = x1 - IC;
			const int x2 = x1 + IC;
			const int x3 = x1 + (IC << 1);

			bool lx = (ih >= -i) && (ih < IH - i);
			bool lx0 = lx && (iw >= 0) && (iw < IW);
			bool lx1 = lx && (iw >= -1) && (iw < IW - 1);
			bool lx2 = lx && (iw >= -2) && (iw < IW - 2);
			bool lx3 = lx && (iw >= -3) && (iw < IW - 3);

			X_tile[(i << 2)] = (lx0 ? X[x0] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 1] = (lx1 ? X[x1] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 2] = (lx2 ? X[x2] : 0);//[n, ih, iw, ic]
			X_tile[(i << 2) + 3] = (lx3 ? X[x3] : 0);//[n, ih, iw, ic]
		}
	}

	const int X_frag_offset = 2 * (BLOCK_C * BLOCK_N);
	const int G_frag_offset = 2 * (BLOCK_C * BLOCK_K);

	const int xs_offset1 = access_s[0][tx], xs_offset2 = access_s[1][tx];
	const int gs_offset1 = access_f_s[0][tx], gs_offset2 = access_f_s[1][tx];

	for (int oic = 0; oic < IC; oic += BLOCK_C)
	{
		float4 *A_frag = (float4*)(Xs + ty * BLOCK_C*BLOCK_N);
		float4 *B_frag = (float4*)(Gs + ty * BLOCK_C*BLOCK_K);

		//======[load_and_transform_{X£¬ G}_tile]======================================================
		{
			//======[load_and_transform_X_tile]======================================================
			float xb[3];
#pragma unroll
			for (int j = 0; j < 4; j++) {
				xb[0] = X_tile[j];
				xb[1] = X_tile[j + 4];
				xb[2] = X_tile[j + 8];

				X_tile[j] = xb[0] - xb[2];//xj0 - xj2
				X_tile[j + 4] = xb[1] + xb[2];//xj1 + xj2
				X_tile[j + 8] = xb[2] - xb[1];//xj2 - xj1
				X_tile[j + 12] = xb[1] - X_tile[j + 12];//xj1 - xj3
			}

			//Xs[16, BLOCK_C, BLOCK_N]
			int c_offset = BLOCK_N * BLOCK_C;
			int c_tensor = ty * BLOCK_N + tx;//[ty, tx]

#pragma unroll
			for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
				Xs[c_tensor + i * c_offset * 4] = X_tile[(i << 2) + 0] - X_tile[(i << 2) + 2];//xi0 - xi2
				Xs[c_tensor + i * c_offset * 4 + c_offset] = X_tile[(i << 2) + 1] + X_tile[(i << 2) + 2];//xi1 + xi2
				Xs[c_tensor + i * c_offset * 4 + 2 * c_offset] = X_tile[(i << 2) + 2] - X_tile[(i << 2) + 1];//xi2 - xi1
				Xs[c_tensor + i * c_offset * 4 + 3 * c_offset] = X_tile[(i << 2) + 1] - X_tile[(i << 2) + 3];//xi1 - xi3
			}

			//======[load_G_tile]====================================================================
			//Gs[16, BLOCK_C, BLOCK_K]
			int c_tensor_s = ty * BLOCK_K + tx; //[ty, tx]
			int c_offset_s = BLOCK_K * BLOCK_C;

			for (int k = 0; k < 2; k++) { // prefetch 2 filter tiles/thread
#pragma unroll
				for (int i = 0; i < 4; i++) {
					Gs[c_tensor_s + i * c_offset_s * 4 + 0 * c_offset_s] = G_tile[k * 16 + i * 4 + 0];
					Gs[c_tensor_s + i * c_offset_s * 4 + 1 * c_offset_s] = G_tile[k * 16 + i * 4 + 1];
					Gs[c_tensor_s + i * c_offset_s * 4 + 2 * c_offset_s] = G_tile[k * 16 + i * 4 + 2];
					Gs[c_tensor_s + i * c_offset_s * 4 + 3 * c_offset_s] = G_tile[k * 16 + i * 4 + 3];
				}

				c_tensor_s += BLOCK_N;
			}
		}
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < BLOCK_C; ik++)
		{
			//======[computing group1]===================================================
			Xf[0] = *(A_frag + xs_offset1 + (BLOCK_N / 4) * ik);
			Xf[1] = *(A_frag + xs_offset2 + (BLOCK_N / 4) * ik);
			Gf[0] = *(B_frag + gs_offset1 + (BLOCK_K / 4) * ik);
			Gf[1] = *(B_frag + gs_offset2 + (BLOCK_K / 4) * ik);

			simdMM4_xzyw(a[0][0], Gf[0].x, Xf[0]); simdMM4_xzyw(a[0][1], Gf[0].x, Xf[1]);
			simdMM4_xzyw(a[0][2], Gf[0].y, Xf[0]); simdMM4_xzyw(a[0][3], Gf[0].y, Xf[1]);
			simdMM4_xzyw(a[0][4], Gf[0].z, Xf[0]); simdMM4_xzyw(a[0][5], Gf[0].z, Xf[1]);
			simdMM4_xzyw(a[0][6], Gf[0].w, Xf[0]); simdMM4_xzyw(a[0][7], Gf[0].w, Xf[1]);

			simdMM4_xzyw(a[0][8], Gf[1].x, Xf[0]); simdMM4_xzyw(a[0][9], Gf[1].x, Xf[1]);
			simdMM4_xzyw(a[0][10], Gf[1].y, Xf[0]); simdMM4_xzyw(a[0][11], Gf[1].y, Xf[1]);
			simdMM4_xzyw(a[0][12], Gf[1].z, Xf[0]); simdMM4_xzyw(a[0][13], Gf[1].z, Xf[1]);
			simdMM4_xzyw(a[0][14], Gf[1].w, Xf[0]); simdMM4_xzyw(a[0][15], Gf[1].w, Xf[1]);

			//======[computing group2]===================================================
			Xf[2] = *(A_frag + xs_offset1 + X_frag_offset + (BLOCK_N / 4) * ik);
			Xf[3] = *(A_frag + xs_offset2 + X_frag_offset + (BLOCK_N / 4) * ik);
			Gf[2] = *(B_frag + gs_offset1 + G_frag_offset + (BLOCK_K / 4) * ik);
			Gf[3] = *(B_frag + gs_offset2 + G_frag_offset + (BLOCK_K / 4) * ik);

			simdMM4_xzyw(a[1][0], Gf[2].x, Xf[2]); simdMM4_xzyw(a[1][1], Gf[2].x, Xf[3]);
			simdMM4_xzyw(a[1][2], Gf[2].y, Xf[2]); simdMM4_xzyw(a[1][3], Gf[2].y, Xf[3]);
			simdMM4_xzyw(a[1][4], Gf[2].z, Xf[2]); simdMM4_xzyw(a[1][5], Gf[2].z, Xf[3]);
			simdMM4_xzyw(a[1][6], Gf[2].w, Xf[2]); simdMM4_xzyw(a[1][7], Gf[2].w, Xf[3]);

			simdMM4_xzyw(a[1][8], Gf[3].x, Xf[2]); simdMM4_xzyw(a[1][9], Gf[3].x, Xf[3]);
			simdMM4_xzyw(a[1][10], Gf[3].y, Xf[2]); simdMM4_xzyw(a[1][11], Gf[3].y, Xf[3]);
			simdMM4_xzyw(a[1][12], Gf[3].z, Xf[2]); simdMM4_xzyw(a[1][13], Gf[3].z, Xf[3]);
			simdMM4_xzyw(a[1][14], Gf[3].w, Xf[2]); simdMM4_xzyw(a[1][15], Gf[3].w, Xf[3]);
		}

		if (oic < (IC - BLOCK_C)) 
		{
			//ic += BLOCK_C
			G += BLOCK_C * OC * 16;//[ic, fh, fw, oc] -> [ic, oc, fh, fw]
			X += BLOCK_C;//[ic, ih, iw, n] -> [n, ih, iw, ic]

			//======[prefetch {G, X} tile]===================================================
			{
				const int ic = ty;
				const int g0 = (ic * OC) * 16;    //G[ic     , oc, fh, fw]
				const int g1 = g0 + BLOCK_N * 16;//G[ic + BN, oc, fh, fw]

				*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
				*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
				*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
				*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

				*(float4*)(G_tile + 16) = *(float4*)(G + g1);
				*(float4*)(G_tile + 20) = *(float4*)(G + g1 + 4);
				*(float4*)(G_tile + 24) = *(float4*)(G + g1 + 8);
				*(float4*)(G_tile + 28) = *(float4*)(G + g1 + 12);

				for (int i = 0; i < 4; i++) {
					const int x1 = i * IW  * IC + ic;
					const int x0 = x1 - IC;
					const int x2 = x1 + IC;
					const int x3 = x1 + (IC << 1);

					bool lx = (ih >= -i) && (ih < IH - i);
					bool lx0 = lx && (iw >= 0) && (iw < IW);
					bool lx1 = lx && (iw >= -1) && (iw < IW - 1);
					bool lx2 = lx && (iw >= -2) && (iw < IW - 2);
					bool lx3 = lx && (iw >= -3) && (iw < IW - 3);

					X_tile[(i << 2)    ] = (lx0 ? X[x0] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 1] = (lx1 ? X[x1] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 2] = (lx2 ? X[x2] : 0);//[n, ih, iw, ic]
					X_tile[(i << 2) + 3] = (lx3 ? X[x3] : 0);//[n, ih, iw, ic]
				}
			}
		}
		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_Y_tile2(
		a, shared_mem,
		tx, ty,
		Y,
		bx, by, bz,
		OH, OW, OC,
		tiles_dim,
		X_frag_mem, G_frag_mem);
}

#endif


#ifndef WINOGRAD2D_KERNEL10
#define WINOGRAD2D_KERNEL10

//242 register
//Size = 9, Time = 2.39889 msec, Performace = 8056.81 GFlop/s
//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel10(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int tiles_dim, int tiles_2d_dim)
{
	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;

	//(16 * 8 * 32) + (16 * 8 * 64) = 16 * 8 * 96

	__shared__ float shared_mem[(16 * BLOCK_C*BLOCK_N) + (16 * BLOCK_C*BLOCK_K)];

	float *Xs = (float*)shared_mem;                       //Xs[16, BLOCK_C, BLOCK_N]
	float *Gs = (float*)&shared_mem[16 * BLOCK_C * BLOCK_N];//Gs[16, BLOCK_C, BLOCK_K]

	float4 X_frag_mem[8], *Xf = (float4*)X_frag_mem;
	float4 G_frag_mem[8], *Gf = (float4*)G_frag_mem;

	const int X_frag_offset = 2 * (BLOCK_C * BLOCK_N);
	const int G_frag_offset = 2 * (BLOCK_C * BLOCK_K);
	const int xs_offset1 = access_s[0][tx], xs_offset2 = access_s[1][tx];
	const int gs_offset1 = access_f_s[0][tx], gs_offset2 = access_f_s[1][tx];
	
	//dim3(N/BLOCK_N, (OH/2)*(OW/2), OC / BLOCK_K), dim3(BLOCK_N, BLOCK_C)
	//BLOCK: dim3(32, 8)
	//dim3(BN, BC)

	//G[ty, bz * BLOCK_K + tx, 0, 0]
	//X[bx * blockDim.x + tx, oh, ow, ty]

	//prepare for G[IC, OC, 4, 4]
	const int oc = (bz * BLOCK_K) + tx;
	G += oc << 4;//G[0, oc, 0, 0]

	//prepare for X[N, IH, IW, IC]
	const int n = (bx * BLOCK_N) + tx;
	const int ohs = by / tiles_dim, ows = by - ohs * tiles_dim;
	const int oh = (ohs << 1), ow = (ows << 1);
	const int ih = oh - ph, iw = ow - pw;
	X += ((n*IH + ih)*IW + iw + 1)*IC;//X[n, ih, iw + 1, 0]

	float4 a[2][16] = { 0.0f };  // Accumulators 
	for (int oic = 0; oic < IC; oic += BLOCK_C)
	{
		const int ic = ty;
		 
		//======[load G to shared memory]===================================================
		{
			const int g0 = (ic * OC) << 4, g1 = g0 + (BLOCK_N << 4);//G[ic + (BK / 2), oc, fh, fw]
			float gv[32];

			//k = 0
			*(float4*)(gv     ) = *(float4*)(G + g0     );//[ic, oc, 0, 0 - 3]
			*(float4*)(gv +  4) = *(float4*)(G + g0 +  4);//[ic, oc, 1, 0 - 3]
			*(float4*)(gv +  8) = *(float4*)(G + g0 +  8);//[ic, oc, 2, 0 - 3]
			*(float4*)(gv + 12) = *(float4*)(G + g0 + 12);//[ic, oc, 3, 0 - 3]


			//k = 1
			*(float4*)(gv + 16) = *(float4*)(G + g1     );//[ic + BN, oc, 0, 0 - 3]
			*(float4*)(gv + 20) = *(float4*)(G + g1 +  4);//[ic + BN, oc, 1, 0 - 3]
			*(float4*)(gv + 24) = *(float4*)(G + g1 +  8);//[ic + BN, oc, 2, 0 - 3]
			*(float4*)(gv + 28) = *(float4*)(G + g1 + 12);//[ic + BN, oc, 3, 0 - 3]

			int c_tensor_s = ty * BLOCK_K + tx;//[ty, tx]
			int c_offset_s = BLOCK_K * BLOCK_C;
			

#pragma unroll
			for (int k = 0; k < 2; k++) { // prefetch 2 filter tiles / thread
#pragma unroll
				for (int i = 0; i < 4; i++) {
					//Gs[16, BLOCK_C, BLOCK_K] = Gs[16, 8, 64] 
					//(ty, tx) = (8, 32)
					//k = 0                                     k = 1
					//Gs[(i, 0), ty, tx] = G_tile[k, i, 0]		Gs[(i, 0), ty, tx + 32] = G_tile[k, i, 0]
					//Gs[(i, 1), ty, tx] = G_tile[k, i, 1]		Gs[(i, 1), ty, tx + 32] = G_tile[k, i, 1]
					//Gs[(i, 2), ty, tx] = G_tile[k, i, 2]		Gs[(i, 2), ty, tx + 32] = G_tile[k, i, 2]
					//Gs[(i, 3), ty, tx] = G_tile[k, i, 3]		Gs[(i, 3), ty, tx + 32] = G_tile[k, i, 3]

					Gs[c_tensor_s + i * c_offset_s * 4 + 0 * c_offset_s] = gv[k * 16 + i * 4 + 0];
					Gs[c_tensor_s + i * c_offset_s * 4 + 1 * c_offset_s] = gv[k * 16 + i * 4 + 1];
					Gs[c_tensor_s + i * c_offset_s * 4 + 2 * c_offset_s] = gv[k * 16 + i * 4 + 2];
					Gs[c_tensor_s + i * c_offset_s * 4 + 3 * c_offset_s] = gv[k * 16 + i * 4 + 3];
				}
				c_tensor_s += BLOCK_N;
			}
		}

		//======[load X to shared memory]===================================================
		{
			float xv[16];

#pragma unroll
			for (int i = 0; i < 4; i++) {
				const int x1 = i * IW*IC + ic;
				const int x0 = x1 - IC;
				const int x2 = x1 + IC;
				const int x3 = x1 + (IC << 1);

				bool lx  = (ih >= -i) && (ih < IH - i);
				bool lx0 = lx && (iw >= 0) && (iw < IW);
				bool lx1 = lx && (iw >= -1) && (iw < IW - 1);
				bool lx2 = lx && (iw >= -2) && (iw < IW - 2);
				bool lx3 = lx && (iw >= -3) && (iw < IW - 3);

				xv[(i << 2)    ] = (lx0 ? X[x0] : 0);//[n, ih + i, iw    , ic]
				xv[(i << 2) + 1] = (lx1 ? X[x1] : 0);//[n, ih + i, iw + 1, ic]
				xv[(i << 2) + 2] = (lx2 ? X[x2] : 0);//[n, ih + i, iw + 2, ic]
				xv[(i << 2) + 3] = (lx3 ? X[x3] : 0);//[n, ih + i, iw + 3, ic]
			}

			//======[load_and_transform_X_tile]======================================================
			float xb[3];
#pragma unroll
			for (int j = 0; j < 4; j++) {
				xb[0] = xv[j];
				xb[1] = xv[j + 4];
				xb[2] = xv[j + 8];

				//xj0 - xj2
				//xj1 + xj2
				//xj2 - xj1
				//xj1 - xj3

				xv[j] = xb[0] - xb[2];
				xv[j + 4] = xb[1] + xb[2];
				xv[j + 8] = xb[2] - xb[1];
				xv[j + 12] = xb[1] - xv[j + 12];
			}

			int c_offset = BLOCK_N * BLOCK_C;
			int c_tensor = ty * BLOCK_N + tx;//[ty, tx]

#pragma unroll
			for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
				//Xs[16, BLOCK_C, BLOCK_N] = Xs[16, 8, 32]
				//(tx, ty) = (BN, BC) = (32, 8)

				//Xs[(i, 0), ty, tx] = xi0 - xi2
				//Xs[(i, 1), ty, tx] = xi1 + xi2
				//Xs[(i, 2), ty, tx] = xi2 - xi1
				//Xs[(i, 3), ty, tx] = xi1 - xi3

				Xs[c_tensor + i * c_offset * 4] = xv[(i << 2) + 0] - xv[(i << 2) + 2];
				Xs[c_tensor + i * c_offset * 4 + c_offset] = xv[(i << 2) + 1] + xv[(i << 2) + 2];
				Xs[c_tensor + i * c_offset * 4 + 2 * c_offset] = xv[(i << 2) + 2] - xv[(i << 2) + 1];
				Xs[c_tensor + i * c_offset * 4 + 3 * c_offset] = xv[(i << 2) + 1] - xv[(i << 2) + 3];
			}
		}

		G += BLOCK_C * OC * 16; X += BLOCK_C;//ic += BLOCK_C
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < BLOCK_C; ik++)
		{
			//(tx, ty) = (BN, BC) = (32, 8)

			//Xs[16, BLOCK_C, BLOCK_N] = Xs[16, 8, 32]
			//X_frag_offset = 2 * (BLOCK_C * BLOCK_N);
			//xs_offset1 = access_s[0][tx];
			//xs_offset2 = access_s[1][tx];
			//{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3},
			//{ 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7 }
			//xs_offset1 = (tx % 2);
			//xs_offset2 = (tx % 2) + 4;
			//Xs[0, ty, ik, xs_offset1 * 4]
			//Xs[0, ty, ik, xs_offset2 * 4]
			//Xs[8, ty, ik, xs_offset1 * 4]
			//Xs[8, ty, ik, xs_offset2 * 4]
			float4 *A_frag = (float4*)(Xs + ty * BLOCK_C*BLOCK_N);//Xs[0, ty]
			Xf[0] = *(A_frag + xs_offset1 + (BLOCK_N / 4) * ik);//8 * ik
			Xf[1] = *(A_frag + xs_offset2 + (BLOCK_N / 4) * ik);
			Xf[2] = *(A_frag + xs_offset1 + X_frag_offset + (BLOCK_N / 4) * ik);
			Xf[3] = *(A_frag + xs_offset2 + X_frag_offset + (BLOCK_N / 4) * ik);

			//Gs[16, BLOCK_C, BLOCK_K] = Gs[16, 8, 64]
			//G_frag_offset = 2 * (BLOCK_C * BLOCK_K);
			//gs_offset1 = access_f_s[0][tx];
			//gs_offset2 = access_f_s[1][tx];
			//Gs[0, ty, ik, gs_offset1 * 4]
			//Gs[0, ty, ik, gs_offset2 * 4]
			//Gs[8, ty, ik, gs_offset1 * 4]
			//Gs[8, ty, ik, gs_offset2 * 4]
			//{ <0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7>, <0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7> }
			//{ <8, 8, 9, 9,10,10,11,11,12,12,13,13,14,14,15,15>, <8, 8, 9,9, 10,10,11,11,12,12,13,13,14,14,15,15> }
			//gs_offset1 = (tx % 8)
			//gs_offset2 = (tx % 8) + 8
			float4 *B_frag = (float4*)(Gs + ty * BLOCK_C*BLOCK_K);//Gs[0, ty]
			Gf[0] = *(B_frag + gs_offset1 + (BLOCK_K / 4) * ik);
			Gf[1] = *(B_frag + gs_offset2 + (BLOCK_K / 4) * ik);
			Gf[2] = *(B_frag + gs_offset1 + G_frag_offset + (BLOCK_K / 4) * ik);
			Gf[3] = *(B_frag + gs_offset2 + G_frag_offset + (BLOCK_K / 4) * ik);

			//
			simdMM4_xzyw(a[0][0], Gf[0].x, Xf[0]); simdMM4_xzyw(a[0][1], Gf[0].x, Xf[1]);
			simdMM4_xzyw(a[0][2], Gf[0].y, Xf[0]); simdMM4_xzyw(a[0][3], Gf[0].y, Xf[1]);
			simdMM4_xzyw(a[0][4], Gf[0].z, Xf[0]); simdMM4_xzyw(a[0][5], Gf[0].z, Xf[1]);
			simdMM4_xzyw(a[0][6], Gf[0].w, Xf[0]); simdMM4_xzyw(a[0][7], Gf[0].w, Xf[1]);

			simdMM4_xzyw(a[0][8], Gf[1].x, Xf[0]); simdMM4_xzyw(a[0][9], Gf[1].x, Xf[1]);
			simdMM4_xzyw(a[0][10], Gf[1].y, Xf[0]); simdMM4_xzyw(a[0][11], Gf[1].y, Xf[1]);
			simdMM4_xzyw(a[0][12], Gf[1].z, Xf[0]); simdMM4_xzyw(a[0][13], Gf[1].z, Xf[1]);
			simdMM4_xzyw(a[0][14], Gf[1].w, Xf[0]); simdMM4_xzyw(a[0][15], Gf[1].w, Xf[1]);

			//
			simdMM4_xzyw(a[1][0], Gf[2].x, Xf[2]); simdMM4_xzyw(a[1][1], Gf[2].x, Xf[3]);
			simdMM4_xzyw(a[1][2], Gf[2].y, Xf[2]); simdMM4_xzyw(a[1][3], Gf[2].y, Xf[3]);
			simdMM4_xzyw(a[1][4], Gf[2].z, Xf[2]); simdMM4_xzyw(a[1][5], Gf[2].z, Xf[3]);
			simdMM4_xzyw(a[1][6], Gf[2].w, Xf[2]); simdMM4_xzyw(a[1][7], Gf[2].w, Xf[3]);

			simdMM4_xzyw(a[1][8], Gf[3].x, Xf[2]); simdMM4_xzyw(a[1][9], Gf[3].x, Xf[3]);
			simdMM4_xzyw(a[1][10], Gf[3].y, Xf[2]); simdMM4_xzyw(a[1][11], Gf[3].y, Xf[3]);
			simdMM4_xzyw(a[1][12], Gf[3].z, Xf[2]); simdMM4_xzyw(a[1][13], Gf[3].z, Xf[3]);
			simdMM4_xzyw(a[1][14], Gf[3].w, Xf[2]); simdMM4_xzyw(a[1][15], Gf[3].w, Xf[3]);
		}
		__syncthreads();
	} 

	// Transpose, transform and store accumulated result
	store_Y_tile3(
		a, shared_mem,
		tx, ty,
		Y,
		bx, by, bz,
		OH, OW, OC,
		tiles_dim);
}

#endif


#ifndef WINOGRAD2D_KERNEL11
#define WINOGRAD2D_KERNEL11

//242 register
//Size = 9, Time = 2.39889 msec, Performace = 8056.81 GFlop/s
//Size = 9, Time = 1.46493 msec, Performace = 13193.4 GFlop/s
//
template<int BLOCK_C = BC, int BLOCK_K = BK, int BLOCK_N = BN>
__global__ void Winograd2d_kernel11(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ G,
	      float* __restrict__ Y, int OH, int OW,
	int N, int IC, int OC,
	int ph, int pw,
	int TW, int THW)//IHW = TH * TW =(OH >> 1) * (OW >> 1)
{
	const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
	const int tx = threadIdx.x, ty = threadIdx.y;

	//(16 * 8 * 32) + (16 * 8 * 64) = 16 * 8 * 96
	__shared__ float Ys[(16 * BLOCK_C*BLOCK_N) + (16 * BLOCK_C*BLOCK_K)];

	float *Xs = (float*)Ys;                         //Xs[16, BLOCK_C, BLOCK_N]
	float *Gs = (float*)&Ys[16 * BLOCK_C * BLOCK_N];//Gs[16, BLOCK_C, BLOCK_K]

	const int xs_offset1 = access_s[0][tx] << 2;
	const int xs_offset2 = access_s[1][tx] << 2;
	const int gs_offset1 = access_f_s[0][tx] << 2;
	const int gs_offset2 = access_f_s[1][tx] << 2;

	//dim3(N/BLOCK_N, (OH/2)*(OW/2), OC/BLOCK_K), dim3(BLOCK_N, BLOCK_C)
	//dim3(N/32, (OH*OW)/4, OC/64), dim3(32, 8)

	//G[ty, bz * BLOCK_K + tx, 0, 0]
	//X[bx * blockDim.x + tx, oh, ow, ty]

	//prepare for G[IC, OC, 4, 4]
	const int oc = (bz * BLOCK_K) + tx;
	G += oc << 4;//G[0, oc, 0, 0]

	//prepare for X[N, IH, IW, IC]
	const int n = (bx * BLOCK_N) + tx;
	const int ts = by / TW, tw = by - ts * TW;
	const int oh = (ts << 1), ow = (tw << 1);
	const int ih = oh - ph, iw = ow - pw;
	X += ((n*IH + ih)*IW + iw + 1)*IC;//X[n, ih, iw + 1, 0]
	
	float4 a[2][16] = { 0.0f };  // Accumulators 
	for (int oic = 0; oic < IC; oic += BLOCK_C)
	{
		const int ic = ty;

		//======[load G to shared memory]===================================================
		{
			const int g0 = (ic * OC) << 4, g1 = g0 + (BLOCK_N << 4);//G[ic + (BK / 2), oc, fh, fw]
			float gv[32];

			//k = 0
			*(float4*)(gv) = *(float4*)(G + g0);//[ic, oc, 0, 0 - 3]
			*(float4*)(gv + 4) = *(float4*)(G + g0 + 4);//[ic, oc, 1, 0 - 3]
			*(float4*)(gv + 8) = *(float4*)(G + g0 + 8);//[ic, oc, 2, 0 - 3]
			*(float4*)(gv + 12) = *(float4*)(G + g0 + 12);//[ic, oc, 3, 0 - 3]

			//k = 1
			*(float4*)(gv + 16) = *(float4*)(G + g1);     //[ic + BN, oc, 0, 0 - 3]
			*(float4*)(gv + 20) = *(float4*)(G + g1 + 4); //[ic + BN, oc, 1, 0 - 3]
			*(float4*)(gv + 24) = *(float4*)(G + g1 + 8); //[ic + BN, oc, 2, 0 - 3]
			*(float4*)(gv + 28) = *(float4*)(G + g1 + 12);//[ic + BN, oc, 3, 0 - 3]

			int c_tensor_s = ty * BLOCK_K + tx;//[ty, tx]
			int c_offset_s = BLOCK_K * BLOCK_C;


#pragma unroll
			for (int k = 0; k < 2; k++) { // prefetch 2 filter tiles / thread
#pragma unroll
				for (int i = 0; i < 4; i++) {
					//Gs[16, BLOCK_C, BLOCK_K] = Gs[16, 8, 64] 
					//(ty, tx) = (8, 32)
					//k = 0                                     k = 1
					//Gs[(i, 0), ty, tx] = G_tile[k, i, 0]		Gs[(i, 0), ty, tx + 32] = G_tile[k, i, 0]
					//Gs[(i, 1), ty, tx] = G_tile[k, i, 1]		Gs[(i, 1), ty, tx + 32] = G_tile[k, i, 1]
					//Gs[(i, 2), ty, tx] = G_tile[k, i, 2]		Gs[(i, 2), ty, tx + 32] = G_tile[k, i, 2]
					//Gs[(i, 3), ty, tx] = G_tile[k, i, 3]		Gs[(i, 3), ty, tx + 32] = G_tile[k, i, 3]

					//Gs[(i, j), ty, tx] = G_tile[k, (i, j)]
					//Gs[(i, j), ty: ic, tx: oc]

					Gs[c_tensor_s + i * c_offset_s * 4 + 0 * c_offset_s] = gv[k * 16 + i * 4 + 0];
					Gs[c_tensor_s + i * c_offset_s * 4 + 1 * c_offset_s] = gv[k * 16 + i * 4 + 1];
					Gs[c_tensor_s + i * c_offset_s * 4 + 2 * c_offset_s] = gv[k * 16 + i * 4 + 2];
					Gs[c_tensor_s + i * c_offset_s * 4 + 3 * c_offset_s] = gv[k * 16 + i * 4 + 3];
				}
				c_tensor_s += BLOCK_N;
			}
		}

		//======[load X to shared memory]===================================================
		{
			float xv[16];

#pragma unroll
			for (int i = 0; i < 4; i++) {
				const int x1 = i * IW*IC + ic;
				const int x0 = x1 - IC;
				const int x2 = x1 + IC;
				const int x3 = x1 + (IC << 1);

				bool lx = (ih >= -i) && (ih < IH - i);
				bool lx0 = lx && (iw >= 0) && (iw < IW);
				bool lx1 = lx && (iw >= -1) && (iw < IW - 1);
				bool lx2 = lx && (iw >= -2) && (iw < IW - 2);
				bool lx3 = lx && (iw >= -3) && (iw < IW - 3);

				xv[(i << 2)    ] = (lx0 ? X[x0] : 0);//[n, ih + i, iw    , ic]
				xv[(i << 2) + 1] = (lx1 ? X[x1] : 0);//[n, ih + i, iw + 1, ic]
				xv[(i << 2) + 2] = (lx2 ? X[x2] : 0);//[n, ih + i, iw + 2, ic]
				xv[(i << 2) + 3] = (lx3 ? X[x3] : 0);//[n, ih + i, iw + 3, ic]
			}

			//======[load_and_transform_X_tile]======================================================
			float xb[3];
#pragma unroll
			for (int j = 0; j < 4; j++) {
				xb[0] = xv[j];
				xb[1] = xv[j + 4];
				xb[2] = xv[j + 8];

				//xj0 - xj2
				//xj1 + xj2
				//xj2 - xj1
				//xj1 - xj3

				xv[j] = xb[0] - xb[2];
				xv[j + 4] = xb[1] + xb[2];
				xv[j + 8] = xb[2] - xb[1];
				xv[j + 12] = xb[1] - xv[j + 12];
			}

			int c_offset = BLOCK_N * BLOCK_C;
			int c_tensor = ty * BLOCK_N + tx;//[ty, tx]

#pragma unroll
			for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
				//Xs[16, BLOCK_C, BLOCK_N] = Xs[16, 8, 32]
				//(tx, ty) = (BN, BC) = (32, 8)

				//Xs[(i, 0), ty, tx] = xi0 - xi2
				//Xs[(i, 1), ty, tx] = xi1 + xi2
				//Xs[(i, 2), ty, tx] = xi2 - xi1
				//Xs[(i, 3), ty, tx] = xi1 - xi3

				Xs[c_tensor + i * c_offset * 4               ] = xv[(i << 2) + 0] - xv[(i << 2) + 2];
				Xs[c_tensor + i * c_offset * 4 +     c_offset] = xv[(i << 2) + 1] + xv[(i << 2) + 2];
				Xs[c_tensor + i * c_offset * 4 + 2 * c_offset] = xv[(i << 2) + 2] - xv[(i << 2) + 1];
				Xs[c_tensor + i * c_offset * 4 + 3 * c_offset] = xv[(i << 2) + 1] - xv[(i << 2) + 3];
			}
		}

		G += BLOCK_C * OC * 16; X += BLOCK_C;//ic += BLOCK_C
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < BLOCK_C; ik++)
		{
			//Gs[16, BLOCK_C, BLOCK_K] = Gs[16, 8, 64]
			//G_frag_offset = 2 * (BLOCK_C * BLOCK_K);
			//gs_offset1 = access_f_s[0][tx]; -> aux
			//gs_offset2 = access_f_s[1][tx];
			//{ <0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7>, <0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7> }
			//{ <8, 8, 9, 9,10,10,11,11,12,12,13,13,14,14,15,15>, <8, 8, 9,9, 10,10,11,11,12,12,13,13,14,14,15,15> }
			//gs_offset1 = (tx % 8)
			//gs_offset2 = (tx % 8) + 8

			//Xs[16, BLOCK_C, BLOCK_N] = Xs[16, 8, 32]
			//X_frag_offset = 2 * (BLOCK_C * BLOCK_N); 
			//xs_offset1 = access_s[0][tx];
			//xs_offset2 = access_s[1][tx];
			//{ <0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>, <2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3> },
			//{ <4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5>, <6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7> }
			//xs_offset1 = (tx % 2);
			//xs_offset2 = (tx % 2) + 4;

			//g(ty) -> G[]

			//Gs[ty, ik, gs_offset1]
			//(1) ty -> (i, j) -> (ik)
			//(2) ik -> ty -> ic
			//(3) gs_offset1 -> tx -> (0: 32): 


			//Gs[(i, 0), ty, tx] = G_tile[k, i, 0]		Gs[(i, 0), ty, tx + 32] = G_tile[k, i, 0]
			//Gs[(i, 0), ty, tx] = G_tile[k, i, 0]		Gs[(i, 0), ty, tx + 32] = G_tile[k, i, 0]


			float4 G0 = *(float4*)(&get3d(Gs, ty, ik, gs_offset1, BLOCK_C, BLOCK_K));//Gs[ty, ik, gs_offset1]
			float4 G1 = *(float4*)(&get3d(Gs, ty, ik, gs_offset2, BLOCK_C, BLOCK_K));//Gs[ty, ik, gs_offset2]
			float4 X0 = *(float4*)(&get3d(Xs, ty, ik, xs_offset1, BLOCK_C, BLOCK_N));//Xs[ty, ik, xs_offset1]
			float4 X1 = *(float4*)(&get3d(Xs, ty, ik, xs_offset2, BLOCK_C, BLOCK_N));//Xs[ty, ik, xs_offset2]

			simdMM4_xzyw(a[0][0], G0.x, X0); simdMM4_xzyw(a[0][1], G0.x, X1);
			simdMM4_xzyw(a[0][2], G0.y, X0); simdMM4_xzyw(a[0][3], G0.y, X1);
			simdMM4_xzyw(a[0][4], G0.z, X0); simdMM4_xzyw(a[0][5], G0.z, X1);
			simdMM4_xzyw(a[0][6], G0.w, X0); simdMM4_xzyw(a[0][7], G0.w, X1);

			simdMM4_xzyw(a[0][8], G1.x, X0); simdMM4_xzyw(a[0][9], G1.x, X1);
			simdMM4_xzyw(a[0][10], G1.y, X0); simdMM4_xzyw(a[0][11], G1.y, X1);
			simdMM4_xzyw(a[0][12], G1.z, X0); simdMM4_xzyw(a[0][13], G1.z, X1);
			simdMM4_xzyw(a[0][14], G1.w, X0); simdMM4_xzyw(a[0][15], G1.w, X1);

			//
			float4 G2 = *(float4*)(&get3d(Gs, ty + 8, ik, gs_offset1, BLOCK_C, BLOCK_K));//Gs[ty + 8, ik, gs_offset1]
			float4 G3 = *(float4*)(&get3d(Gs, ty + 8, ik, gs_offset2, BLOCK_C, BLOCK_K));//Gs[ty + 8, ik, gs_offset2]
			float4 X2 = *(float4*)(&get3d(Xs, ty + 8, ik, xs_offset1, BLOCK_C, BLOCK_N));//Xs[ty + 8, ik, xs_offset1]
			float4 X3 = *(float4*)(&get3d(Xs, ty + 8, ik, xs_offset2, BLOCK_C, BLOCK_N));//Xs[ty + 8, ik, xs_offset2]

			simdMM4_xzyw(a[1][0], G2.x, X2); simdMM4_xzyw(a[1][1], G2.x, X3);
			simdMM4_xzyw(a[1][2], G2.y, X2); simdMM4_xzyw(a[1][3], G2.y, X3);
			simdMM4_xzyw(a[1][4], G2.z, X2); simdMM4_xzyw(a[1][5], G2.z, X3);
			simdMM4_xzyw(a[1][6], G2.w, X2); simdMM4_xzyw(a[1][7], G2.w, X3);

			simdMM4_xzyw(a[1][8], G3.x, X2); simdMM4_xzyw(a[1][9], G3.x, X3);
			simdMM4_xzyw(a[1][10], G3.y, X2); simdMM4_xzyw(a[1][11], G3.y, X3);
			simdMM4_xzyw(a[1][12], G3.z, X2); simdMM4_xzyw(a[1][13], G3.z, X3);
			simdMM4_xzyw(a[1][14], G3.w, X2); simdMM4_xzyw(a[1][15], G3.w, X3);
		}
		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_Y_tile3(
		a, Ys,
		tx, ty,
		Y,
		bx, by, bz,
		OH, OW, OC,
		TW);
}

#endif



cudaError_t winograd_32x64x8(
	float *X, cudaTextureObject_t texX, int IH, int IW,
	float *W, float *G,
	float *Y, int OH, int OW,
	int N, int IC, int OC)
{
	int filt_h = 3, int filt_w = 3;
	int ph = 1, pw = 1;

	int tile_size = 4;
	int tile_2d_s = tile_size * tile_size;

	int alpha = tile_size;
	  
	int tiles_dim = (OH >> 1);
	int tiles_2d_dim = (OH >> 1) * (OW >> 1);

	__winograd2D_f22x33_kremode(NULL, W, G, OC, IC);

	//FX <<<dim3(OC / BK, IC / BC), dim3(BN, BC) >>> (W, G, OC, IC, filt_h, filt_w, alpha);

	Winograd2d_kernel11 
		<<< dim3(N / BN, tiles_2d_dim, OC / BK), dim3(BN, BC), 0 >>>
		(X, IH, IW, G, Y,OH, OW, N, IC, OC, ph, pw, tiles_dim, tiles_2d_dim);

	return cudaGetLastError();
}


#endif


