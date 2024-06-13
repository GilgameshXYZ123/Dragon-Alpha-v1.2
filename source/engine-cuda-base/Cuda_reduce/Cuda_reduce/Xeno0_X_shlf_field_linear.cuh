


#ifndef X1_FIELD_LINEAR_KERNEL_4
#define X1_FIELD_LINEAR_KERNEL_4

//X -> N
//Y -> M
#define x1_field_linear4(stream, LBX, LBY, LTX, LTY, A, alpha, beta, N, M, V, width, stride) \
	x1_field_linear_kernel_4<LBX, LBY>\
		<<< dim3(N>>LBX>>LTX, M>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, alpha, beta, N, M, V, width, stride)

//Time = 0.17 mesc, Speed = 373.392GB/s

//[512*16*16, 128] -> [512*16*16, 1]: Time = 0.178 mesc, Speed = 356.61GB/s
template<int LBX, int LBY>
__global__ void x1_field_linear_kernel_4(
	const float* __restrict__ A,
	float alpha, float beta,
	int N, int M,
	float* __restrict__ V,
	int width, int stride)
{
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float2 As[1 << LBY][(2 << LBX) + 2];//[Y: M, X: N]

	//parallel field num = 4
	const int offsetY = (bx << LBX) + tx, stepY = (gridDim.x << LBX);
	const int offsetX = (by << LBY) + ty, stepX = (gridDim.y << LBY);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;
	for (int x4 = offsetX << 2; x4 < M4; x4 += stepX4)
	{
		float4 c = F32_4_0;//solve the errors
		float4 v = F32_4_0;//thread reduce: 4 local result

		for (int y = offsetY; y < N; y += stepY) {
			const int aoffset = y * M + x4;
			float4 a = *(float4*)(A + aoffset);//A[y, x4]
			simdLinear4(a, alpha, a, beta);//a = alpha*a + beta
			Kahan_simdAdd4(v, a, c);//v = v + a
		}
		*(float4*)(&As[ty][tx << 1]) = v;
		__syncthreads();

		if (LBX >= 6) {//block reduce: 4 global result
			if (tx < 64) {//128 -> 64
				const int yIdx = ((tx & 31) << 1) + (tx >> 5);
				float2 v1 = As[ty][yIdx], v2 = As[ty][yIdx + 64];
				simdAdd2(v1, v1, v2); As[ty][yIdx] = v1;
			}
			__syncthreads();
		}
		if (LBX >= 5) {
			if (tx < 32) {//64 -> 32
				const int yIdx = ((tx & 15) << 1) + (tx >> 4);
				float2 v1 = As[ty][yIdx], v2 = As[ty][yIdx + 32];
				simdAdd2(v1, v1, v2); As[ty][yIdx] = v1;
			}
			__syncthreads();
		}
		if (LBX >= 4) {
			if (tx < 16) {//32 -> 16
				const int yIdx = ((tx & 7) << 1) + (tx >> 3);
				float2 v1 = As[ty][yIdx], v2 = As[ty][yIdx + 16];
				simdAdd2(v1, v1, v2); As[ty][yIdx] = v1;
			}
			__syncthreads();
		}

		if (tx < 8) {//16 -> 8
			const int yIdx = ((tx & 3) << 1) + (tx >> 2);
			float2 v1 = As[ty][yIdx], v2 = As[ty][yIdx + 8];
			simdAdd2(v1, v1, v2); As[ty][yIdx] = v1;
		}
		__syncthreads();

		if (tx < 4) {//8 -> 4
			const int yIdx = ((tx & 1) << 1) + (tx >> 1);
			float2 v1 = As[ty][yIdx], v2 = As[ty][yIdx + 4];
			simdAdd2(v1, v1, v2); As[ty][yIdx] = v1;
		}
		__syncthreads();

		if (tx < 2) {//4 -> 2, save, 
			const int yIdx = ((tx & 0) << 1) + tx;
			float2 v1 = As[ty][yIdx], v2 = As[ty][yIdx + 2];
			simdAdd2(v1, v1, v2);

			const int xindex2 = x4 + (tx << 1);
			within_width2(v1, xindex2, stride, width);
			*(float2*)(&get(V, bx, xindex2, M)) = v1;
		}
		__syncthreads();
	}
}

#endif



