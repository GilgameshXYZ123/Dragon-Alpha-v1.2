

#ifndef LOAD_TILE
#define LOAD_TILE

__device__ __forceinline__ void load_and_transform_X_tile(
	float *X_tile,
	float *Xs,
	int N, int IH, int IW, int IC,
	int tiles_dim, int tiles_2d_dim,
	int tx, int ty, int bx, int by)
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

	int c_offset = BN * BC;
	int c_tensor = ty * BN + tx;

#pragma unroll
	for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
		Xs[c_tensor + i * c_offset * 4] = d(X_tile, i, 0) - d(X_tile, i, 2);
		Xs[c_tensor + i * c_offset * 4 + c_offset] = d(X_tile, i, 1) + d(X_tile, i, 2);
		Xs[c_tensor + i * c_offset * 4 + 2 * c_offset] = d(X_tile, i, 2) - d(X_tile, i, 1);
		Xs[c_tensor + i * c_offset * 4 + 3 * c_offset] = d(X_tile, i, 1) - d(X_tile, i, 3);
	}
}


__device__ __forceinline__ void load_G_tile(
	float *G_tile, float *Gs,
	int IC, int OC,
	int tx, int ty)
{
	int c_tensor_s = ty * BK + tx;
	int c_offset_s = BK * BC;

	for (int k = 0; k < 2; k++) { // prefetch 2 filter tiles/thread
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				Gs[c_tensor_s + i * c_offset_s * 4 + j * c_offset_s] = G_tile[k * 16 + i * 4 + j];
			}
		}

		c_tensor_s += BN;
	}
}

#endif


#ifndef PREFETCH_G_TILE
#define PREFETCH_G_TILE

//template<int BLOCK_K. int BLOCK_N>
//__device__ __forceinline__ void prefetch_G_tile(
//	const float *pInputs, float *tiles,
//	int OC,
//	int tx, int ty, int bz)
//{
//	int c_tensor = bz * BK + (ty*OC << 4) + tx;
//
//	int acumm;
//#pragma unroll  
//	for (int i = 0; i < 4; i++) {
//		acumm = (i*OC << 2);
//		for (int j = 0; j < 4; j++) {
//			tiles[(i << 2) + j] = pInputs[acumm + j * OC + c_tensor];
//			tiles[16 + (i << 2) + j] = pInputs[acumm + j * OC + c_tensor + BN];
//		}
//	}
//}


template<int BLOCK_K, int BLOCK_N>
__device__ __forceinline__ void prefetch_G_tile(
	const float* __restrict__ G,
	float* __restrict__ G_tile,
	int OC, int tx, int ty, int bz)
{
	//<<<dim3(N / BN, tiles_2d_dim, OC / BK), dim3(BN, BC), 0 >>>
	//tx -> BN: ic
	//ty -> BC: oc
	//bz -> oc
	const int oc = (bz * BLOCK_K + tx);
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
}

#endif


#ifndef PREFETCH_X_TILE
#define PREFETCH_X_TILE

//template<int BLOCK_N>
//__device__ __forceinline__ void prefetch_X_tile(
//	const float* X, float* tile, 
//	int IH, int IW, int N, 
//	int tx, int ty,
//	int bx, int by,
//	int tiles_dim, short mask) 
//{
//	int c_tensor = (by%tiles_dim)*N * 2 + (by / tiles_dim)*N*IW * 2 + bx * BLOCK_N + ty * (N*IH*IW) + (tx / N) * 2 * N + (tx%N) - (N*IW + N);
//	int acumm, x;
//
//	if (mask == 0xFFFF) {
//
//		for (int i = 0; i < 4; i++) {
//			acumm = i * N*IW;
//#pragma unroll
//			for (int j = 0; j < 4; j++) {
//				tile[(i << 2) + j] = X[acumm + j * N + c_tensor];
//			}
//		}
//	}
//
//	else {
//		for (int i = 0; i < 4; i++) {
//			acumm = i * N*IW;
//#pragma unroll
//			for (int j = 0; j < 4; j++) {
//				x = (i << 2) + j;
//				tile[x] = 0;
//				if (mask&(1 << x))
//					tile[x] = X[acumm + j * N + c_tensor];
//			}
//		}
//	}
//}
//<<<dim3(N / BN, tiles_2d_dim, OC / BK), dim3(BN, BC), 0 >>>
//bx -> n
//by -> (oh, ow)
//tx -> BN: ic
//ty -> BC: oc

//[ic, ih, iw, n] -> [n, ih, iw, ic]
template<int BLOCK_N, int ph = 1, int pw = 1>
__device__ __forceinline__ void prefetch_X_tile(
	const float* __restrict__ X,
	float* __restrict__ X_tile,
	int N, int IH, int IW, int IC,
	int tx, int ty,
	int bx, int by,
	int tiles_dim)
{
	//<<<dim3(N / BN, tiles_2d_dim, OC / BK), dim3(BN, BC), 0 >>>
	//tx -> BN: ic
	//ty -> BC: oc
	//bz -> oc

	//int n = (bx * BLOCK_N) + ((tx / N) * 2) * N + (tx % N);
	int n = (bx * BLOCK_N) + tx;
	int ih = ((by / tiles_dim) * 2) - ph;
	int iw = ((by % tiles_dim) * 2) - pw;
	int ic = ty;

#pragma unroll
	for (int i = 0; i < 4; i++)
#pragma unroll
		for (int j = 0; j < 4; j++) {
			int iht = ih + i;
			int iwt = iw + j;

			int x0 = ((n*IH + iht)*IW + iwt)*IC + ic;
			bool lx = (iht >= 0) && (iht < IH) && (iwt >= 0) && (iwt < IW);

			X_tile[(i << 2) + j] = (lx ? X[x0] : 0);//[n, ih, iw, ic]
		}
}

#endif


#ifndef PREFETCH_FRAG
#define PREFETCH_FRAG

__device__  __forceinline__ void prefetch_G_frag(
	float4 *G_frag,
	float4 *B_frag, int G_frag_offset,
	int tx, int offset1, int offset2)
{
	*((float4*)(G_frag)) = *(B_frag + offset1);
	*((float4*)(G_frag + 1)) = *(B_frag + offset2);

	*((float4*)(G_frag + 2)) = *(B_frag + G_frag_offset + offset1);
	*((float4*)(G_frag + 3)) = *(B_frag + G_frag_offset + offset2);
}

__device__  __forceinline__ void prefetch_X_frag(
	float4* X_frag, float4 *A_frag,
	int X_frag_offset, int tx,
	int offset1, int offset2)
{
	*((float4*)(X_frag)) = *(A_frag + offset1); //ld_shared(A_frag + offset1);
	*((float4*)(X_frag + 1)) = *(A_frag + offset2);

	*((float4*)(X_frag + 2)) = *(A_frag + X_frag_offset + offset1);
	*((float4*)(X_frag + 3)) = *(A_frag + X_frag_offset + offset2); //3=2+1
}

#endif


#ifndef STORE_OUTER
#define STORE_OUTER

__device__ void  transform_Y_tile(
	float *Y, float2 *C_tile, float2 *At,
	int tiles_dim, int round, int c_tensor, int Cstride, short mask,
	int OW, int OC)
{
#pragma unroll
	for (int j = 0; j < 4; j++) {
		At[j].x = C_tile[j].x + C_tile[4 + j].x + C_tile[8 + j].x;
		At[j].y = C_tile[j].y + C_tile[4 + j].y + C_tile[8 + j].y;

		At[4 + j].x = C_tile[4 + j].x - C_tile[8 + j].x - C_tile[12 + j].x;
		At[4 + j].y = C_tile[4 + j].y - C_tile[8 + j].y - C_tile[12 + j].y;
	}

	int oc = (((round) / 2) * 32 + ((round) % 2) * 2);
	c_tensor += oc;

#pragma unroll
	for (int i = 0; i < 2; i++) {
		int x = i * 4;
		float y00 = At[x].x + At[x + 1].x + At[x + 2].x;
		float y10 = At[x].y + At[x + 1].y + At[x + 2].y;
		float y01 = At[x + 1].x - At[x + 2].x - At[x + 3].x;
		float y11 = At[x + 1].y - At[x + 2].y - At[x + 3].y;

		Y[(i * OC * OW + c_tensor)] = y00;
		Y[(i * OC * OW + c_tensor) + Cstride] = y10;
		Y[(i * OC * OW + c_tensor + OC)] = y01;
		Y[(i * OC * OW + c_tensor + OC) + Cstride] = y11;
	}
}


__device__ __inline__ void store_Y_tile(
	float4 acumm_smem[][16], float *shared_mem,
	int tx, int ty,
	float *C, 
	int bx, int by, int bz,
	int OH, int OW, int OC,
	int tiles_dim, int N,
	float4 *X_frag_mem, float4* G_frag_mem)
{
	float2 *out_smem = (float2*)shared_mem;
	float2 *accu = (float2 *)acumm_smem;

	float2 *Y_tile = (float2*)X_frag_mem;
	float2 *At = (float2*)G_frag_mem;

	int mask = 0x000F;
	if ((blockIdx.y / tiles_dim) == (tiles_dim - 1) && OW % 2) mask &= 0x0003;
	if (!((blockIdx.y + 1) % tiles_dim) && OW % 2)             mask &= 0X0005;

	// output transpose step
	int t = 0;
	int acumm1, acumm2;
	// For transposing
	//acumm1 = access_s_out[Inx]; //* 4
	acumm1 = ((tx % 8) / 2) * 34 + tx % 2 + (tx / 16) * 2 + ((tx / 8) % 2) * 8;
	acumm2 = acumm1 + 4;

	int acumm4 = BN_p * 16; //*4
	int idx = ty * BN_p;
	int idx2 = idx + BN_p * 8; //(BN_p*2 *8)/2

	// For transformating
	int offset = BN_p * 2; //*2/2
	int init = ((ty / 4)*BN_p * 16 + (ty % 4)*(32 + 2)) * 2 + tx;

	int oc = bz * BK + ((tx / 16) * 16 + (ty % 4) * 4 + ty / 4);
	int oh = (by / tiles_dim) * 2;
	int ow = (by % tiles_dim) * 2;
	int n = bx * BN + (tx % 16) * 2;
	int c_tensor = ((n*OH + oh)*OW + ow)*OC + oc;

#pragma unroll                                  
	for (int round = 0; round < 4; round++) {

		*((float2*)(out_smem + idx + acumm1)) = *(accu + t);
		*((float2*)(out_smem + idx + acumm1 + 16)) = *(accu + t + 1); // float 4, t
		*((float2*)(out_smem + idx + acumm2)) = *(accu + t + 2);
		*((float2*)(out_smem + idx + acumm2 + 16)) = *(accu + t + 3); // float 4, t+1

		*((float2*)(out_smem + idx2 + acumm1)) = *(accu + t + 32);
		*((float2*)(out_smem + idx2 + acumm1 + 16)) = *(accu + t + 33); // float 4, t+16
		*((float2*)(out_smem + idx2 + acumm2)) = *(accu + t + 34);
		*((float2*)(out_smem + idx2 + acumm2 + 16)) = *(accu + t + 35); // float 4, t+17

		*((float2*)(out_smem + idx + acumm4 + acumm1)) = *(accu + t + 4);
		*((float2*)(out_smem + idx + acumm4 + acumm1 + 16)) = *(accu + t + 5); // float 4, t+2
		*((float2*)(out_smem + idx + acumm4 + acumm2)) = *(accu + t + 6);
		*((float2*)(out_smem + idx + acumm4 + acumm2 + 16)) = *(accu + t + 7); // float 4, t+3

		*((float2*)(out_smem + idx2 + acumm4 + acumm1)) = *(accu + t + 36);
		*((float2*)(out_smem + idx2 + acumm4 + acumm1 + 16)) = *(accu + t + 37); // float 4, t+18
		*((float2*)(out_smem + idx2 + acumm4 + acumm2)) = *(accu + t + 38);
		*((float2*)(out_smem + idx2 + acumm4 + acumm2 + 16)) = *(accu + t + 39); // float 4, t+19

		t += 8;

		__syncthreads();


		for (int i = 0; i < 16; i++) {
			Y_tile[i].x = shared_mem[i*offset + init];
			Y_tile[i].y = shared_mem[i*offset + init + 32];
		}

		// transform output tiles
		transform_Y_tile(
			C, Y_tile, At,
			tiles_dim, round,
			c_tensor, OC * OH * OW,
			mask, OW, OC);
		__syncthreads();
	}
}

#endif


#ifndef STORE_OUTER2
#define STORE_OUTER2

__device__ void  transform_Y_tile2(
	float *Y, float2 *Y_tile, float2 *At,
	int tiles_dim, int round, int c_tensor, int Cstride,
	int OW, int OC)
{
#pragma unroll
	for (int j = 0; j < 4; j++) {
		At[j].x = Y_tile[j].x + Y_tile[4 + j].x + Y_tile[8 + j].x;
		At[j].y = Y_tile[j].y + Y_tile[4 + j].y + Y_tile[8 + j].y;

		At[4 + j].x = Y_tile[4 + j].x - Y_tile[8 + j].x - Y_tile[12 + j].x;
		At[4 + j].y = Y_tile[4 + j].y - Y_tile[8 + j].y - Y_tile[12 + j].y;
	}

	int oc = (((round) / 2) * 32 + ((round) % 2) * 2);
	c_tensor += oc;

#pragma unroll
	for (int i = 0; i < 2; i++) {
		int x = i * 4;
		float y00 = At[x].x + At[x + 1].x + At[x + 2].x;
		float y10 = At[x].y + At[x + 1].y + At[x + 2].y;
		float y01 = At[x + 1].x - At[x + 2].x - At[x + 3].x;
		float y11 = At[x + 1].y - At[x + 2].y - At[x + 3].y;

		Y[(i * OC * OW + c_tensor)] = y00;
		Y[(i * OC * OW + c_tensor) + Cstride] = y10;
		Y[(i * OC * OW + c_tensor + OC)] = y01;
		Y[(i * OC * OW + c_tensor + OC) + Cstride] = y11;
	}
}

__device__ __inline__ void store_Y_tile2(
	float4 acumm_smem[][16], float *shared_mem,
	int tx, int ty,
	float *Y,
	int bx, int by, int bz,
	int OH, int OW, int OC,
	int tiles_dim, 
	float4 *X_frag_mem, float4* G_frag_mem)
{
	float2 *out_smem = (float2*)shared_mem;
	float2 *accu = (float2 *)acumm_smem;

	float2 *Y_tile = (float2*)X_frag_mem;
	float2 *At     = (float2*)G_frag_mem;

	// output transpose step
	int t = 0;
	int acumm1 = ((tx % 8) / 2) * 34 + tx % 2 + (tx / 16) * 2 + ((tx / 8) % 2) * 8;
	int acumm2 = acumm1 + 4;
	int acumm4 = BN_p * 16; //*4

	int idx = ty * BN_p;
	int idx2 = idx + BN_p * 8; //(BN_p*2 *8)/2

	// For transformating
	int offset = BN_p * 2; //*2/2
	int init = ((ty / 4)*BN_p * 16 + (ty % 4)*(32 + 2)) * 2 + tx;

	int oc = bz * BK + ((tx / 16) * 16 + (ty % 4) * 4 + ty / 4);
	int oh = (by / tiles_dim) * 2;
	int ow = (by % tiles_dim) * 2;
	int n = bx * BN + (tx % 16) * 2;
	int c_tensor = ((n*OH + oh)*OW + ow)*OC + oc;
	int Cstride = OC * OH * OW;// n + 1

#pragma unroll                                  
	for (int round = 0; round < 4; round++) {
		*((float2*)(out_smem + idx + acumm1)) = *(accu + t);
		*((float2*)(out_smem + idx + acumm1 + 16)) = *(accu + t + 1); // float 4, t
		*((float2*)(out_smem + idx + acumm2)) = *(accu + t + 2);
		*((float2*)(out_smem + idx + acumm2 + 16)) = *(accu + t + 3); // float 4, t+1

		*((float2*)(out_smem + idx2 + acumm1)) = *(accu + t + 32);
		*((float2*)(out_smem + idx2 + acumm1 + 16)) = *(accu + t + 33); // float 4, t+16
		*((float2*)(out_smem + idx2 + acumm2)) = *(accu + t + 34);
		*((float2*)(out_smem + idx2 + acumm2 + 16)) = *(accu + t + 35); // float 4, t+17

		*((float2*)(out_smem + idx + acumm4 + acumm1)) = *(accu + t + 4);
		*((float2*)(out_smem + idx + acumm4 + acumm1 + 16)) = *(accu + t + 5); // float 4, t+2
		*((float2*)(out_smem + idx + acumm4 + acumm2)) = *(accu + t + 6);
		*((float2*)(out_smem + idx + acumm4 + acumm2 + 16)) = *(accu + t + 7); // float 4, t+3

		*((float2*)(out_smem + idx2 + acumm4 + acumm1)) = *(accu + t + 36);
		*((float2*)(out_smem + idx2 + acumm4 + acumm1 + 16)) = *(accu + t + 37); // float 4, t+18
		*((float2*)(out_smem + idx2 + acumm4 + acumm2)) = *(accu + t + 38);
		*((float2*)(out_smem + idx2 + acumm4 + acumm2 + 16)) = *(accu + t + 39); // float 4, t+19
		t += 8;
		__syncthreads();

#pragma unrol
		for (int i = 0; i < 16; i++) {
			Y_tile[i].x = shared_mem[i*offset + init];
			Y_tile[i].y = shared_mem[i*offset + init + 32];
		}

		//=======transform output tiles=========================
#pragma unroll
		for (int j = 0; j < 4; j++) {
			At[j].x = Y_tile[j].x + Y_tile[4 + j].x + Y_tile[8 + j].x;
			At[j].y = Y_tile[j].y + Y_tile[4 + j].y + Y_tile[8 + j].y;
			At[4 + j].x = Y_tile[4 + j].x - Y_tile[8 + j].x - Y_tile[12 + j].x;
			At[4 + j].y = Y_tile[4 + j].y - Y_tile[8 + j].y - Y_tile[12 + j].y;
		}

		const int oc = (((round) / 2) * 32 + ((round) % 2) * 2);
		const int yoffset = c_tensor + oc;

#pragma unroll
		for (int i = 0; i < 2; i++) {
			int x = i * 4;
			float y00 = At[x].x + At[x + 1].x + At[x + 2].x;
			float y10 = At[x].y + At[x + 1].y + At[x + 2].y;
			float y01 = At[x + 1].x - At[x + 2].x - At[x + 3].x;
			float y11 = At[x + 1].y - At[x + 2].y - At[x + 3].y;

			//OC * OH * OW
			Y[yoffset + (i * OW) * OC] = y00;
			Y[yoffset + (i * OW) * OC + Cstride] = y10;
			Y[yoffset + (i * OW + 1) * OC] = y01;
			Y[yoffset + (i * OW + 1) * OC + Cstride] = y11;
		}
		__syncthreads();
	}
}


#endif   


#ifndef STORE_OUTER3
#define STORE_OUTER3

	//(oc, n)
	//[ 0, 0], [ 0, 2], [ 0, 4], [ 0, 6], [ 0, 8], [ 0, 10], [ 0, 12], [0, 14], [0, 16], [0, 18], [0, 20], [0, 22], [0, 24], [0, 26], [0, 28], [0, 30], [16, 0], [16, 2], [16, 4], [16, 6], [16, 8], [16, 10], [16, 12], [16, 14], [16, 16], [16, 18], [16, 20], [16, 22], [16, 24], [16, 26], [16, 28], [16, 30],
	//[ 4, 0], [ 4, 2], [ 4, 4], [ 4, 6], [ 4, 8], [ 4, 10], [ 4, 12], [4, 14], [4, 16], [4, 18], [4, 20], [4, 22], [4, 24], [4, 26], [4, 28], [4, 30], [20, 0], [20, 2], [20, 4], [20, 6], [20, 8], [20, 10], [20, 12], [20, 14], [20, 16], [20, 18], [20, 20], [20, 22], [20, 24], [20, 26], [20, 28], [20, 30],
	//[ 8, 0], [ 8, 2], [ 8, 4], [ 8, 6], [ 8, 8], [ 8, 10], [ 8, 12], [8, 14], [8, 16], [8, 18], [8, 20], [8, 22], [8, 24], [8, 26], [8, 28], [8, 30], [24, 0], [24, 2], [24, 4], [24, 6], [24, 8], [24, 10], [24, 12], [24, 14], [24, 16], [24, 18], [24, 20], [24, 22], [24, 24], [24, 26], [24, 28], [24, 30],
	//[12, 0], [12, 2], [12, 4], [12, 6], [12, 8], [12, 10], [12, 12], [12, 14], [12, 16], [12, 18], [12, 20], [12, 22], [12, 24], [12, 26], [12, 28], [12, 30], [28, 0], [28, 2], [28, 4], [28, 6], [28, 8], [28, 10], [28, 12], [28, 14], [28, 16], [28, 18], [28, 20], [28, 22], [28, 24], [28, 26], [28, 28], [28, 30],

	//[ 1, 0], [ 1, 2], [ 1, 4], [ 1, 6], [ 1, 8], [ 1, 10], [ 1, 12], [1, 14], [1, 16], [1, 18], [1, 20], [1, 22], [1, 24], [1, 26], [1, 28], [1, 30], [17, 0], [17, 2], [17, 4], [17, 6], [17, 8], [17, 10], [17, 12], [17, 14], [17, 16], [17, 18], [17, 20], [17, 22], [17, 24], [17, 26], [17, 28], [17, 30],
	//[ 5, 0], [ 5, 2], [ 5, 4], [ 5, 6], [ 5, 8], [ 5, 10], [ 5, 12], [5, 14], [5, 16], [5, 18], [5, 20], [5, 22], [5, 24], [5, 26], [5, 28], [5, 30], [21, 0], [21, 2], [21, 4], [21, 6], [21, 8], [21, 10], [21, 12], [21, 14], [21, 16], [21, 18], [21, 20], [21, 22], [21, 24], [21, 26], [21, 28], [21, 30],
	//[ 9, 0], [ 9, 2], [ 9, 4], [ 9, 6], [ 9, 8], [ 9, 10], [ 9, 12], [9, 14], [9, 16], [9, 18], [9, 20], [9, 22], [9, 24], [9, 26], [9, 28], [9, 30], [25, 0], [25, 2], [25, 4], [25, 6], [25, 8], [25, 10], [25, 12], [25, 14], [25, 16], [25, 18], [25, 20], [25, 22], [25, 24], [25, 26], [25, 28], [25, 30],
	//[13, 0], [13, 2], [13, 4], [13, 6], [13, 8], [13, 10], [13, 12], [13, 14], [13, 16], [13, 18], [13, 20], [13, 22], [13, 24], [13, 26], [13, 28], [13, 30], [29, 0], [29, 2], [29, 4], [29, 6], [29, 8], [29, 10], [29, 12], [29, 14], [29, 16], [29, 18], [29, 20], [29, 22], [29, 24], [29, 26], [29, 28], [29, 30],

template<int BLOCK_N = BN, int BLOCK_K = BK>
__device__ __inline__ void store_Y_tile3(
	float4 ACCU[][16], float *Ys,
	int tx, int ty,
	float *Y,
	int bx, int by, int bz,
	int OH, int OW, int OC,
	int TW)
{
	//prepare for Y[N, OH, OW, OC]
	const int ts = by / TW, tw = by - ts * TW;
	const int oh = (ts << 1), ow = (tw << 1);

	//(tx, ty) = [32, 8]
	int oc = (bz * BLOCK_K) + ((tx / 16) * 16 + (ty % 4) * 4 + (ty / 4));
	int n  = (bx * BLOCK_N) + ((tx % 16) * 2);//2 element along n

	int c_tensor = ((n*OH + oh)*OW + ow)*OC + oc;
	int Cstride = OH * OW * OC;//n + 1
	//--------------------------------------------------------------------

	float2 *a = (float2 *)ACCU;
	float2 yv[16], At[16];//[{x, z}, {y, w} ]

	// output transpose step
	
	//int acumm1 = ((tx % 8) / 2) * 34 + tx % 2 + (tx / 16) * 2 + ((tx / 8) % 2) * 8;

	//int oc = ((tx / 16) * 16 + (ty % 4) * 4 + ty / 4);
	//int n =  ((tx % 16) * 2);//2 element along n

	int acumm1 = 
		((tx % 8) / 2) * (32 + 2) + 
		((tx / 8) % 2) * 8 +
		(tx % 2) + 
		(tx / 16) * 2;//n

	int init = (ty / 4)*BN_p * 16 + (ty % 4)*(32 + 2);
	
#pragma unroll                                  
	for (int round = 0, t = 0; round < 4; round++)//a[64] -> a[2][32]
	{
		// + 16 -> n1
		//0
		*(float2*)(Ys + ((ty     ) * BN_p + acumm1         ) * 2) = *(a + t    );//a[t, 0].{x, y} n0
		*(float2*)(Ys + ((ty     ) * BN_p + acumm1     + 16) * 2) = *(a + t + 1);//a[t, 0].{z, w} n1
		*(float2*)(Ys + ((ty     ) * BN_p + acumm1 + 4     ) * 2) = *(a + t + 2);//a[t, 1].{x, y}
		*(float2*)(Ys + ((ty     ) * BN_p + acumm1 + 4 + 16) * 2) = *(a + t + 3);//a[t, 1].{z, w}

		//8
		*(float2*)(Ys + ((ty +  8) * BN_p + acumm1         ) * 2) = *(a + t + 32);//a[t, 2].{x, y}
		*(float2*)(Ys + ((ty +  8) * BN_p + acumm1 +     16) * 2) = *(a + t + 33);//a[t, 2].{z, w}
		*(float2*)(Ys + ((ty +  8) * BN_p + acumm1 + 4     ) * 2) = *(a + t + 34);//a[t, 3].{x, y}
		*(float2*)(Ys + ((ty +  8) * BN_p + acumm1 + 4 + 16) * 2) = *(a + t + 35);//a[t, 3].{z, w}

		//16
		*(float2*)(Ys + ((ty + 16) * BN_p + acumm1         ) * 2) = *(a + t + 4);
		*(float2*)(Ys + ((ty + 16) * BN_p + acumm1     + 16) * 2) = *(a + t + 5);
		*(float2*)(Ys + ((ty + 16) * BN_p + acumm1 + 4     ) * 2) = *(a + t + 6);
		*(float2*)(Ys + ((ty + 16) * BN_p + acumm1 + 4 + 16) * 2) = *(a + t + 7);

		//24
		*(float2*)(Ys + ((ty + 24) * BN_p + acumm1         ) * 2) = *(a + t + 36);
		*(float2*)(Ys + ((ty + 24) * BN_p + acumm1 + 16    ) * 2) = *(a + t + 37); // float 4, t+18
		*(float2*)(Ys + ((ty + 24) * BN_p + acumm1 + 4     ) * 2) = *(a + t + 38);
		*(float2*)(Ys + ((ty + 24) * BN_p + acumm1 + 4 + 16) * 2) = *(a + t + 39); // float 4, t+19

		t += 8;
		__syncthreads();


		//(1) init = (ty / 4)*BN_p * 16 + (ty % 4)*(32 + 2);
		//(2) n = ((tx % 16) * 2)
#pragma unrol
		for (int i = 0; i < 16; i++) {
			//n0
			//Ys[(ty/4), i     , (ty%4), tx] 
			yv[i].x = Ys[(i*BN_p + init     ) * 2 + tx];

			//n1
			// [(ty/4), i + 16, (ty%4), tx] =>  [(ty/4) + 1, i, (ty%4), tx]
			yv[i].y = Ys[(i*BN_p + init + 16) * 2 + tx];
		}
		
		//(ty / 4)*BN_p * 16 + (ty % 4)*(32 + 2)

		//=======transform output tiles========================================
#pragma unroll
		for (int j = 0; j < 4; j++) {//n, n + 1
			//k0t = q0t + q1t + q2t
			//k1t = q1t - a2t - q3t

			//n0
			At[j    ].x = yv[j    ].x + yv[4 + j].x + yv[8 + j].x;
			At[4 + j].x = yv[4 + j].x - yv[8 + j].x - yv[12 + j].x;

			//n1 = n0 + 1
			At[j    ].y = yv[j    ].y + yv[4 + j].y + yv[8 + j].y;
			At[4 + j].y = yv[4 + j].y - yv[8 + j].y - yv[12 + j].y;
		}

		const int toc = ((round / 2) * 32 + (round % 2) * 2);
		int yoffset = c_tensor + toc;

		//n0
		Y[yoffset                ] = At[0].x + At[1].x + At[2].x;//y00 = k00 + k01 + k02
		Y[yoffset +            OC] = At[1].x - At[2].x - At[3].x;//y01 = k01 - k02 - k13
		Y[yoffset + (OW    ) * OC] = At[4].x + At[5].x + At[6].x;//y10 = k10 + k11 + k12
		Y[yoffset + (OW + 1) * OC] = At[5].x - At[6].x - At[7].x;//y11 = k11 - k12 - k13

		yoffset += OH * OW * OC;//n += 1;

		//n1 = n0 + 1
		Y[yoffset                ] = At[0].y + At[1].y + At[2].y;//y00 = k00 + k01 + k02
		Y[yoffset +            OC] = At[1].y - At[2].y - At[3].y;//y01 = k01 - k02 - k13
		Y[yoffset + (OW    ) * OC] = At[4].y + At[5].y + At[6].y;//y10 = k10 + k11 + k12
		Y[yoffset + (OW + 1) * OC] = At[5].y - At[6].y - At[7].y;//y11 = k11 - k12 - k13
		__syncthreads();
	}
}


#endif   


#ifndef OUTER_PRODUCT2
#define OUTER_PRODUCT2

__device__  __forceinline__ void outer_product2(
	float4* X_frag,
	float4* G_frag,
	float4 accu[][16])
{
	accu[0][0].x += X_frag[0].x*G_frag[0].x;
	accu[0][0].z += X_frag[0].y*G_frag[0].x;
	accu[0][0].y += X_frag[0].z*G_frag[0].x;
	accu[0][0].w += X_frag[0].w*G_frag[0].x;

	accu[0][1].x += X_frag[1].x*G_frag[0].x;
	accu[0][1].z += X_frag[1].y*G_frag[0].x;
	accu[0][1].y += X_frag[1].z*G_frag[0].x;
	accu[0][1].w += X_frag[1].w*G_frag[0].x;

	accu[0][2].x += X_frag[0].x*G_frag[0].y;
	accu[0][2].z += X_frag[0].y*G_frag[0].y;
	accu[0][2].y += X_frag[0].z*G_frag[0].y;
	accu[0][2].w += X_frag[0].w*G_frag[0].y;

	accu[0][3].x += X_frag[1].x*G_frag[0].y;
	accu[0][3].z += X_frag[1].y*G_frag[0].y;
	accu[0][3].y += X_frag[1].z*G_frag[0].y;
	accu[0][3].w += X_frag[1].w*G_frag[0].y;

	accu[0][4].x += X_frag[0].x*G_frag[0].z;
	accu[0][4].z += X_frag[0].y*G_frag[0].z;
	accu[0][4].y += X_frag[0].z*G_frag[0].z;
	accu[0][4].w += X_frag[0].w*G_frag[0].z;

	accu[0][5].x += X_frag[1].x*G_frag[0].z;
	accu[0][5].z += X_frag[1].y*G_frag[0].z;
	accu[0][5].y += X_frag[1].z*G_frag[0].z;
	accu[0][5].w += X_frag[1].w*G_frag[0].z;

	accu[0][6].x += X_frag[0].x*G_frag[0].w;
	accu[0][6].z += X_frag[0].y*G_frag[0].w;
	accu[0][6].y += X_frag[0].z*G_frag[0].w;
	accu[0][6].w += X_frag[0].w*G_frag[0].w;

	accu[0][7].x += X_frag[1].x*G_frag[0].w;
	accu[0][7].z += X_frag[1].y*G_frag[0].w;
	accu[0][7].y += X_frag[1].z*G_frag[0].w;
	accu[0][7].w += X_frag[1].w*G_frag[0].w;

	//
	accu[0][8].x += X_frag[0].x*G_frag[1].x;
	accu[0][8].z += X_frag[0].y*G_frag[1].x;
	accu[0][8].y += X_frag[0].z*G_frag[1].x;
	accu[0][8].w += X_frag[0].w*G_frag[1].x;

	accu[0][9].x += X_frag[1].x*G_frag[1].x;
	accu[0][9].z += X_frag[1].y*G_frag[1].x;
	accu[0][9].y += X_frag[1].z*G_frag[1].x;
	accu[0][9].w += X_frag[1].w*G_frag[1].x;

	accu[0][10].x += X_frag[0].x*G_frag[1].y;
	accu[0][10].z += X_frag[0].y*G_frag[1].y;
	accu[0][10].y += X_frag[0].z*G_frag[1].y;
	accu[0][10].w += X_frag[0].w*G_frag[1].y;

	accu[0][11].x += X_frag[1].x*G_frag[1].y;
	accu[0][11].z += X_frag[1].y*G_frag[1].y;
	accu[0][11].y += X_frag[1].z*G_frag[1].y;
	accu[0][11].w += X_frag[1].w*G_frag[1].y;

	accu[0][12].x += X_frag[0].x*G_frag[1].z;
	accu[0][12].z += X_frag[0].y*G_frag[1].z;
	accu[0][12].y += X_frag[0].z*G_frag[1].z;
	accu[0][12].w += X_frag[0].w*G_frag[1].z;

	accu[0][13].x += X_frag[1].x*G_frag[1].z;
	accu[0][13].z += X_frag[1].y*G_frag[1].z;
	accu[0][13].y += X_frag[1].z*G_frag[1].z;
	accu[0][13].w += X_frag[1].w*G_frag[1].z;

	accu[0][14].x += X_frag[0].x*G_frag[1].w;
	accu[0][14].z += X_frag[0].y*G_frag[1].w;
	accu[0][14].y += X_frag[0].z*G_frag[1].w;
	accu[0][14].w += X_frag[0].w*G_frag[1].w;

	accu[0][15].x += X_frag[1].x*G_frag[1].w;
	accu[0][15].z += X_frag[1].y*G_frag[1].w;
	accu[0][15].y += X_frag[1].z*G_frag[1].w;
	accu[0][15].w += X_frag[1].w*G_frag[1].w;




	//////
	accu[1][0].x += X_frag[2].x*G_frag[2].x;
	accu[1][0].z += X_frag[2].y*G_frag[2].x;
	accu[1][0].y += X_frag[2].z*G_frag[2].x;
	accu[1][0].w += X_frag[2].w*G_frag[2].x;

	accu[1][1].x += X_frag[3].x*G_frag[2].x;
	accu[1][1].z += X_frag[3].y*G_frag[2].x;
	accu[1][1].y += X_frag[3].z*G_frag[2].x;
	accu[1][1].w += X_frag[3].w*G_frag[2].x;

	accu[1][2].x += X_frag[2].x*G_frag[2].y;
	accu[1][2].z += X_frag[2].y*G_frag[2].y;
	accu[1][2].y += X_frag[2].z*G_frag[2].y;
	accu[1][2].w += X_frag[2].w*G_frag[2].y;

	accu[1][3].x += X_frag[3].x*G_frag[2].y;
	accu[1][3].z += X_frag[3].y*G_frag[2].y;
	accu[1][3].y += X_frag[3].z*G_frag[2].y;
	accu[1][3].w += X_frag[3].w*G_frag[2].y;

	accu[1][4].x += X_frag[2].x*G_frag[2].z;
	accu[1][4].z += X_frag[2].y*G_frag[2].z;
	accu[1][4].y += X_frag[2].z*G_frag[2].z;
	accu[1][4].w += X_frag[2].w*G_frag[2].z;

	accu[1][5].x += X_frag[3].x*G_frag[2].z;
	accu[1][5].z += X_frag[3].y*G_frag[2].z;
	accu[1][5].y += X_frag[3].z*G_frag[2].z;
	accu[1][5].w += X_frag[3].w*G_frag[2].z;

	accu[1][6].x += X_frag[2].x*G_frag[2].w;
	accu[1][6].z += X_frag[2].y*G_frag[2].w;
	accu[1][6].y += X_frag[2].z*G_frag[2].w;
	accu[1][6].w += X_frag[2].w*G_frag[2].w;

	accu[1][7].x += X_frag[3].x*G_frag[2].w;
	accu[1][7].z += X_frag[3].y*G_frag[2].w;
	accu[1][7].y += X_frag[3].z*G_frag[2].w;
	accu[1][7].w += X_frag[3].w*G_frag[2].w;

	//
	accu[1][8].x += X_frag[2].x*G_frag[3].x;
	accu[1][8].z += X_frag[2].y*G_frag[3].x;
	accu[1][8].y += X_frag[2].z*G_frag[3].x;
	accu[1][8].w += X_frag[2].w*G_frag[3].x;

	accu[1][9].x += X_frag[3].x*G_frag[3].x;
	accu[1][9].z += X_frag[3].y*G_frag[3].x;
	accu[1][9].y += X_frag[3].z*G_frag[3].x;
	accu[1][9].w += X_frag[3].w*G_frag[3].x;

	accu[1][10].x += X_frag[2].x*G_frag[3].y;
	accu[1][10].z += X_frag[2].y*G_frag[3].y;
	accu[1][10].y += X_frag[2].z*G_frag[3].y;
	accu[1][10].w += X_frag[2].w*G_frag[3].y;

	accu[1][11].x += X_frag[3].x*G_frag[3].y;
	accu[1][11].z += X_frag[3].y*G_frag[3].y;
	accu[1][11].y += X_frag[3].z*G_frag[3].y;
	accu[1][11].w += X_frag[3].w*G_frag[3].y;

	accu[1][12].x += X_frag[2].x*G_frag[3].z;
	accu[1][12].z += X_frag[2].y*G_frag[3].z;
	accu[1][12].y += X_frag[2].z*G_frag[3].z;
	accu[1][12].w += X_frag[2].w*G_frag[3].z;

	accu[1][13].x += X_frag[3].x*G_frag[3].z;
	accu[1][13].z += X_frag[3].y*G_frag[3].z;
	accu[1][13].y += X_frag[3].z*G_frag[3].z;
	accu[1][13].w += X_frag[3].w*G_frag[3].z;

	accu[1][14].x += X_frag[2].x*G_frag[3].w;
	accu[1][14].z += X_frag[2].y*G_frag[3].w;
	accu[1][14].y += X_frag[2].z*G_frag[3].w;
	accu[1][14].w += X_frag[2].w*G_frag[3].w;

	accu[1][15].x += X_frag[3].x*G_frag[3].w;
	accu[1][15].z += X_frag[3].y*G_frag[3].w;
	accu[1][15].y += X_frag[3].z*G_frag[3].w;
	accu[1][15].w += X_frag[3].w*G_frag[3].w;
}


#endif
