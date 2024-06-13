

#ifndef LOAD_TILE
#define LOAD_TILE

__device__ __forceinline__ void load_and_transform_X_tile(
	float *Btd, float *pOutputs, int in_h, int in_w,
	int tiles_dim, int in_c, int in_n, int tile_size,
	int tiles_2d_dim, int tile_2d_s,
	int Inx, int Iny, int TileX, int TileY)
{
	float workspace[3];

#pragma unroll
	for (int j = 0; j < 4; j++) {
		workspace[0] = Btd[j];
		workspace[1] = Btd[j + 4];
		workspace[2] = Btd[j + 8];

		Btd[j] = workspace[0] - workspace[2];
		Btd[j + 4] = workspace[1] + workspace[2];
		Btd[j + 8] = workspace[2] - workspace[1];
		Btd[j + 12] = workspace[1] - Btd[j + 12];
	}

	int c_offset = BN * BC;
	int c_tensor = Iny * BN + Inx;

#pragma unroll
	for (int i = 0; i < 4; i++) { // prefetch 1 input tile/thread
		pOutputs[c_tensor + i * c_offset * 4] = d(Btd, i, 0) - d(Btd, i, 2);
		pOutputs[c_tensor + i * c_offset * 4 + c_offset] = d(Btd, i, 1) + d(Btd, i, 2);
		pOutputs[c_tensor + i * c_offset * 4 + 2 * c_offset] = d(Btd, i, 2) - d(Btd, i, 1);
		pOutputs[c_tensor + i * c_offset * 4 + 3 * c_offset] = d(Btd, i, 1) - d(Btd, i, 3);
	}

}


__device__ __forceinline__ void load_G_tile(
	float *tiles, float *pOutputs,
	int filt_c, int filt_k,
	int Inx, int Iny)
{
	int c_tensor_s = Iny * BK + Inx;
	int c_offset_s = BK * BC;

	for (int k = 0; k < 2; k++) { // prefetch 2 filter tiles/thread
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				pOutputs[c_tensor_s + i * c_offset_s * 4 + j * c_offset_s] = tiles[k * 16 + i * 4 + j];
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


//<<<dim3(N / BN, tiles_2d_dim, OC / BK), dim3(BN, BC), 0 >>>
//tx -> BN: ic
//ty -> BC: oc
//bz -> oc

//G[ic, fh, fw, oc] -> G[oc, ic, fh, fw]
template<int BLOCK_K, int BLOCK_N>
__device__ __forceinline__ void prefetch_G_tile(
	const float* __restrict__ G,
	float* __restrict__ G_tile,
	int OC, int tx, int ty, int bz)
{
	const int oc = (bz * BLOCK_K + tx), ic = ty;
	const int g0 = (ic * OC + oc) * 16;
	*(float4*)(G_tile) = *(float4*)(G + g0);//[ic, oc, fh, fw]
	*(float4*)(G_tile + 4) = *(float4*)(G + g0 + 4);
	*(float4*)(G_tile + 8) = *(float4*)(G + g0 + 8);
	*(float4*)(G_tile + 12) = *(float4*)(G + g0 + 12);

	const int g1 = g0 + BLOCK_N * 16 * OC;
	*(float4*)(G_tile + 16) = *(float4*)(G + g1);//[ic + BN, oc, fh, fw]
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
	int n = bx * BLOCK_N + ((tx / N) * 2) * N + (tx % N);
	int ih = ((by / tiles_dim) * 2) - ph;
	int iw = ((by % tiles_dim) * 2) - pw;
	int ic = ty;

	const int xoffset = ((n*IH + ih)*IW + iw)*IC + ic;
#pragma unroll
	for (int i = 0; i < 4; i++)
#pragma unroll
		for (int j = 0; j < 4; j++) {
			const int x0 = xoffset + (i*IW + j)*IC;
			bool lx1 = (ih >= -i) && (ih < IH - i) && (iw >= -j) && (iw < IW - j);
			X_tile[(i << 2) + j] = (lx1 ? X[x0] : 0);//[n, ih, iw, ic]
		}
}

#endif


#ifndef PREFETCH_FRAG
#define PREFETCH_FRAG

__device__  __forceinline__ void prefetch_G_frag(
	float4 *filter_frag,
	float4 *B_frag, int f_frag_offset,
	int Inx, int offset1, int offset2) {

	*((float4*)(filter_frag)) = *(B_frag + offset1);
	*((float4*)(filter_frag + 1)) = *(B_frag + offset2);

	*((float4*)(filter_frag + 2)) = *(B_frag + f_frag_offset + offset1);
	*((float4*)(filter_frag + 3)) = *(B_frag + f_frag_offset + offset2);
}

__device__  __forceinline__ void prefetch_X_frag(
	float4* input_frag, float4 *A_frag, int frag_offset,
	int Inx, int offset1, int offset2) {

	*((float4*)(input_frag)) = *(A_frag + offset1); //ld_shared(A_frag + offset1);
	*((float4*)(input_frag + 1)) = *(A_frag + offset2);

	*((float4*)(input_frag + 2)) = *(A_frag + frag_offset + offset1);
	*((float4*)(input_frag + 3)) = *(A_frag + frag_offset + offset2); //3=2+1
}

#endif


#ifndef STORE_OUTER
#define STORE_OUTER

__device__ void  transform_Y_tile(float *Y, float2 *C_tile, float2 *At,
	int tiles_dim, int round, int N, int c_tensor, int Cstride, short mask,
	int OW, int OC)
{
	int oc = (((round) / 2) * 32 + ((round) % 2) * 2);
	c_tensor += oc;

#pragma unroll
	for (int j = 0; j < 4; j++) {
		At[j].x = C_tile[j].x + C_tile[4 + j].x + C_tile[8 + j].x;
		At[j].y = C_tile[j].y + C_tile[4 + j].y + C_tile[8 + j].y;

		At[4 + j].x = C_tile[4 + j].x - C_tile[8 + j].x - C_tile[12 + j].x;
		At[4 + j].y = C_tile[4 + j].y - C_tile[8 + j].y - C_tile[12 + j].y;
	}

	int x, x1;
#pragma unroll
	for (int i = 0; i < 2; i++) {
		x = i * 4;

		if (mask&(1 << (i * 2))) {
			Y[(i * OC * OW + c_tensor)          ] = At[x].x + At[x + 1].x + At[x + 2].x;
			Y[(i * OC * OW + c_tensor) + Cstride] = At[x].y + At[x + 1].y + At[x + 2].y;
		}

		if (mask&(1 << (i * 2 + 1))) {
			Y[(i * OC * OW + c_tensor + OC)          ] = At[x + 1].x - At[x + 2].x - At[x + 3].x;
			Y[(i * OC * OW + c_tensor + OC) + Cstride] = At[x + 1].y - At[x + 2].y - At[x + 3].y;
		}
	}
}


__device__ __inline__ void store_Y_tile(
	float4 acumm_smem[][16], float *shared_mem,
	int tx, int ty,
	float *C, int
	bx, int by, int bz,
	int OH, int OW, int OC,
	int tiles_dim, int N,
	float4 *X_frag_mem, float4* G_frag_mem)
{
	float2 *output_smem = (float2 *)shared_mem;
	float2 *accumulator = (float2 *)acumm_smem;

	float2 *C_tile = (float2*)X_frag_mem;
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

		*((float2*)(output_smem + idx + acumm1)) = *(accumulator + t);
		*((float2*)(output_smem + idx + acumm1 + 16)) = *(accumulator + t + 1); // float 4, t
		*((float2*)(output_smem + idx + acumm2)) = *(accumulator + t + 2);
		*((float2*)(output_smem + idx + acumm2 + 16)) = *(accumulator + t + 3); // float 4, t+1

		*((float2*)(output_smem + idx2 + acumm1)) = *(accumulator + t + 32);
		*((float2*)(output_smem + idx2 + acumm1 + 16)) = *(accumulator + t + 33); // float 4, t+16
		*((float2*)(output_smem + idx2 + acumm2)) = *(accumulator + t + 34);
		*((float2*)(output_smem + idx2 + acumm2 + 16)) = *(accumulator + t + 35); // float 4, t+17

		*((float2*)(output_smem + idx + acumm4 + acumm1)) = *(accumulator + t + 4);
		*((float2*)(output_smem + idx + acumm4 + acumm1 + 16)) = *(accumulator + t + 5); // float 4, t+2
		*((float2*)(output_smem + idx + acumm4 + acumm2)) = *(accumulator + t + 6);
		*((float2*)(output_smem + idx + acumm4 + acumm2 + 16)) = *(accumulator + t + 7); // float 4, t+3

		*((float2*)(output_smem + idx2 + acumm4 + acumm1)) = *(accumulator + t + 36);
		*((float2*)(output_smem + idx2 + acumm4 + acumm1 + 16)) = *(accumulator + t + 37); // float 4, t+18
		*((float2*)(output_smem + idx2 + acumm4 + acumm2)) = *(accumulator + t + 38);
		*((float2*)(output_smem + idx2 + acumm4 + acumm2 + 16)) = *(accumulator + t + 39); // float 4, t+19

		t += 8;

		__syncthreads();


		for (int i = 0; i < 16; i++) {
			C_tile[i].x = shared_mem[i*offset + init];
			C_tile[i].y = shared_mem[i*offset + init + 32];
		}

		// transform output tiles
		transform_Y_tile(
			C, C_tile, At,
			tiles_dim, round, N,
			c_tensor, OC * OH * OW,
			mask, OW, OC);
		__syncthreads();
	}
}


#endif    

