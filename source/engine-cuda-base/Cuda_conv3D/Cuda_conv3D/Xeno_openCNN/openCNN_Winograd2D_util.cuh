



__device__ __forceinline__ void load_and_transform_input_tile(float *Btd, float *pOutputs, int in_h, int in_w,
	int tiles_dim, int in_c, int in_n, int tile_size,
	int tiles_2d_dim, int tile_2d_s, int Inx, int Iny, int TileX, int TileY) 
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

__device__ __forceinline__ void load_filter_tile(float *tiles, float *pOutputs,
	int filt_c, int filt_k, int Inx, int Iny) {

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

__device__ __forceinline__ void prefetch_filter_tile(float *pInputs, float *tiles,
	int filt_k, int Inx, int Iny, int TileZ) {

	int c_tensor = TileZ * BK + (Iny*filt_k << 4) + Inx;

	int acumm;
#pragma unroll  
	for (int i = 0; i < 4; i++) {
		acumm = (i*filt_k << 2);
		for (int j = 0; j < 4; j++) {
			tiles[(i << 2) + j] = pInputs[acumm + j * filt_k + c_tensor];
			tiles[16 + (i << 2) + j] = pInputs[acumm + j * filt_k + c_tensor + BN];
		}
	}
}

__device__ __forceinline__ void prefetch_input_tile(float *pInputs, float *tile, int in_h, int in_w,
	int in_n, int Inx, int Iny, int TileX, int TileY, int tiles_dim, short mask) {

	int c_tensor = (TileY%tiles_dim)*in_n * 2 + (TileY / tiles_dim)*in_n*in_w * 2 + TileX * BN + Iny * (in_n*in_h*in_w) + (Inx / in_n) * 2 * in_n + (Inx%in_n) - (in_n*in_w + in_n);
	int acumm, x;

	if (mask == 0xFFFF) {

		for (int i = 0; i < 4; i++) {
			acumm = i * in_n*in_w;
#pragma unroll
			for (int j = 0; j < 4; j++) {
				tile[(i << 2) + j] = pInputs[acumm + j * in_n + c_tensor];
			}
		}

	}
	else {

		for (int i = 0; i < 4; i++) {
			acumm = i * in_n*in_w;
#pragma unroll
			for (int j = 0; j < 4; j++) {
				x = (i << 2) + j;
				tile[x] = 0;
				if (mask&(1 << x))
					tile[x] = pInputs[acumm + j * in_n + c_tensor];
			}
		}
	}
}

__device__  __forceinline__ void prefetch_filter_frag(float4 *filter_frag, float4 *B_frag, int f_frag_offset,
	int Inx, int offset1, int offset2) {

	*((float4*)(filter_frag)) = *(B_frag + offset1);
	*((float4*)(filter_frag + 1)) = *(B_frag + offset2);

	*((float4*)(filter_frag + 2)) = *(B_frag + f_frag_offset + offset1);
	*((float4*)(filter_frag + 3)) = *(B_frag + f_frag_offset + offset2);
}

__device__  __forceinline__ void prefetch_input_frag(float4* input_frag, float4 *A_frag, int frag_offset,
	int Inx, int offset1, int offset2) {

	*((float4*)(input_frag)) = *(A_frag + offset1); //ld_shared(A_frag + offset1);
	*((float4*)(input_frag + 1)) = *(A_frag + offset2);

	*((float4*)(input_frag + 2)) = *(A_frag + frag_offset + offset1);
	*((float4*)(input_frag + 3)) = *(A_frag + frag_offset + offset2); //3=2+1
}