#pragma once

#ifndef OPEN_CNN
#define OPEN_CNN

#include "Winograd2D_config.cuh"
#include "Winograd2D_util.cuh"
#include "outer_product.cuh"
#include "store_output_tile.cuh"
#include "FX_m2.cuh"


#define WINOGRAD_KERNEL1
#ifndef WINOGRAD_KERNEL1
#define WINOGRAD_KERNEL1

__global__ void open_CNN_Winograd_kernel1(float *A, float *B, float *C,
	int tiles_dim, int in_c, int in_n, int in_h, int in_w,
	int tile_size, int filt_k, int filt_c,
	int tiles_2d_dim, int out_c, int out_n,
	int tile_2d_s, int out_h, int out_w) 
{
	extern __shared__ float shared_mem[];
	float *input_smem = (float*)shared_mem;
	float *filter_smem = (float*)&shared_mem[16 * BC*BN];

	short m = 0xFFFF;
	if ((blockIdx.y / tiles_dim) == 0)   m &= 0xFFF0;
	if ((blockIdx.y / tiles_dim) == (tiles_dim - 1)) m &= (!(in_w % 2)) ? (0x0FFF) : (0x00FF);
	if (!((blockIdx.y + 1) % tiles_dim)) m &= (!(in_w % 2)) ? (0x7777) : (0x3333);
	if (!((blockIdx.y) % tiles_dim))   m &= 0xeeee;

	float img_tile[16]; // Prefetch input from GMEM
	float filter_tile[32]; // Prefetch filter from GMEM

	float4 input_frag_mem[8];  //2*2(2*8/4) Data to do Outer Product + prefetch f. SMEM (double_buffer)
	float4 filter_frag_mem[8]; //2*2 Data to do Outer Product + prefetch f. SMEM (double_buffer)
	float4 accumulator[2][16] = { 0.0f };  // Accumulators 

	float4 *A_frag; // Input data pointer
	int frag_offset = 2 * (BC*BN); // (2=8/4) SMEM input read offset

	float4 *B_frag; // Filter data pointer
	int f_frag_offset = 2 * (BC*BK); // (2=8/4) SMEM filter read offset

	float4 *input_frag = (float4*)input_frag_mem;
	float4 *filter_frag = (float4*)filter_frag_mem;

	float4 *swap;

	prefetch_input_tile(A, img_tile, in_h, in_w, in_n, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, tiles_dim, m);
	prefetch_filter_tile(B, filter_tile, filt_k, threadIdx.x, threadIdx.y, blockIdx.z);

	float4 *input_frag_buffer = (float4*)(input_frag + 4);
	float4 *filter_frag_buffer = (float4*)(filter_frag + 4);

	// Mainloop - iterates over the entire K dimension - not unrolled
	for (int iter = 0; iter < in_c; iter += BC) { // Current iteration

		A_frag = (float4*)(input_smem + threadIdx.y*BC*BN);
		B_frag = (float4*)(filter_smem + threadIdx.y*BC*BK);

		load_and_transform_input_tile(img_tile, input_smem, in_h, in_w,
			tiles_dim, in_c, in_n, tile_size,
			tiles_2d_dim, tile_2d_s, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
		load_filter_tile(filter_tile, filter_smem, filt_c, filt_k, threadIdx.x, threadIdx.y);

		__syncthreads();

		prefetch_input_frag(input_frag, A_frag, frag_offset, threadIdx.x, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
		prefetch_filter_frag(filter_frag, B_frag, f_frag_offset, threadIdx.x, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);

#pragma unroll
		for (int i = 0; i < BC; i++) {

			if (i < (BC - 1)) {
				A_frag += BN / 4;
				B_frag += BK / 4;

				prefetch_input_frag(input_frag_buffer, A_frag, frag_offset, threadIdx.x, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
				prefetch_filter_frag(filter_frag_buffer, B_frag, f_frag_offset, threadIdx.x, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);
			}

			outer_product(input_frag, filter_frag, accumulator);

			swap = input_frag;
			input_frag = input_frag_buffer;
			input_frag_buffer = swap;

			swap = filter_frag;
			filter_frag = filter_frag_buffer;
			filter_frag_buffer = swap;

		}

		A += in_n * BC*in_w*in_h;
		B += filt_k * BC * 4 * 4;

		if (iter < (in_c - BC)) {
			prefetch_input_tile(A, img_tile, in_h, in_w, in_n, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, tiles_dim, m);
			prefetch_filter_tile(B, filter_tile, filt_k, threadIdx.x, threadIdx.y, blockIdx.z);
		}

		__syncthreads();
	}

	// Transpose, transform and store accumulated result
	store_output_tile(accumulator, shared_mem, threadIdx.x, threadIdx.y, C, blockIdx.x, blockIdx.y, blockIdx.z,
		out_h, out_w,
		tiles_dim, out_n, input_frag_mem, filter_frag_mem, m);

}



#endif

//cudaError_t openCNN_winograd_32x64x8(
//	float *X, int IH, int IW,
//	float *W, float *G,
//	float *Y, int OH, int OW,
//	int N, int IC, int OC)
//{
//	int filt_h = 3, int filt_w = 3;
//	int m = 2;
//	int tile_size = 4;
//	int tiles_dim = ceil(ceil((double)(IH + 2) / 2) - 1);
//	int alpha = tile_size;
//
//	int tile_2d_s = tile_size * tile_size;
//	int tiles_2d_dim = tiles_dim * tiles_dim;
//
//	int smem_size = (16 * BC*BN + 16 * BC*BK) * 4;
//	FX <<<dim3(OC / BK, IC / BC), dim3(BN, BC) >>>
//		(W, G, OC, IC, filt_h, filt_w, alpha);
//
//#ifdef OPTSTS64_CMP
//	smem_size = 65536; // 64 KB
//	cudaFuncSetAttribute(Winograd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
//#endif
//
//	open_CNN_Winograd_kernel1 <<<dim3(N / BN, tiles_2d_dim, OC / BK), dim3(BN, 8), smem_size >>>
//		(X, G, Y, tiles_dim, IC, N, IH, IW, tile_size, OC, IC, tiles_2d_dim, OC, N, tile_2d_s, OH, OW);
//
//	return cudaGetLastError();
//}



#endif