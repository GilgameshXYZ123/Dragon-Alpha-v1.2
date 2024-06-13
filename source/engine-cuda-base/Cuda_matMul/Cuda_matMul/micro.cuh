#pragma once

#ifndef MICRO_H
#define MICRO_H

inline int LOG2(int n) {
	int result = 0;
	if (n & 0xffff0000) { result += 16; n >>= 16; }
	if (n & 0x0000ff00) { result +=  8; n >>=  8; }
	if (n & 0x000000f0) { result +=  4; n >>=  4; }
	if (n & 0x0000000c) { result +=  2; n >>=  2; }
	if (n & 0x00000002) { result +=  1; n >>=  1; }
	return result;
}

#ifndef COMMON_MICRO
#define COMMON_MICRO


#define F32_2_0 float2{0, 0}
#define F32_4_0 float4{0, 0, 0, 0}

#define IF_int(flag, a, b) (((-(flag)) & ((a) - (b))) + (b))

#define IS_POWER2(x) ( ((x)>0) && ((x)&((x)-1))==0 )

#define get(A, i, j, stride) A[(i)*(stride) + (j)] //get2d
#define lget(A, i, j, lstride) A[((i)<<(lstride)) + (j)] //lget2d, lstride = log2(stride)

#define simdMM4(c, av, b) {(c).x += av * b.x; (c).y += av * b.y; (c).z += av * b.z; (c).w += av * b.w;}
#define simdMM2(c, av, b) {(c).x += av * b.x; (c).y += av * b.y;}

#define Mul4(c, av, b) {(c).x = (av) * (b).x; (c).y = (av) * (b).y; (c).z = (av) * (b).z; (c).w = (av) * (b).w;}
#define Mul2(c, av, b) {(c).x = (av) * (b).x; (c).y = (av) * (b).y; }

#define next_index(index, length) ((index) + 1)%(length)

#define next_cudaStream(stream, streams, index, length) \
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index]; { index = (index + 1) % length; }

#endif


#ifndef PADDING_MICRO
#define PADDING_MICRO

__device__ float HOLE[4200];
__device__ float FZERO4[4] = { 0, 0, 0, 0 };

__device__ __forceinline__ void IFW4(float* __restrict__ X, 
	int xoffset, bool flag, float4 v) 
{
	float* dst = IF_int(flag, X, HOLE);
	int offset = IF_int(flag, xoffset, (xoffset & 4095));
	*(float4*)(dst + offset) = v;
}

__device__ __forceinline__ void IFW8(float* __restrict__ X, 
	int xoffset, bool flag, float4 v0, float4 v1) 
{
	float* dst = IF_int(flag, X, HOLE);
	int offset = IF_int(flag, xoffset, (xoffset & 4095));
	*(float4*)(dst + offset    ) = v0;
	*(float4*)(dst + offset + 4) = v1;
}

#define QP_128(GN, GM)\
	(1.0f * ((GN + 127) >> 7 << 7) * ((GM + 127) >> 7 << 7) / (GN * GM))

#define QP_64(GN, GM)\
	(1.0f * ((GN + 63) >> 6 << 6) * ((GM + 63) >> 6 << 6) / (GN * GM))

#define VP_128(GN, GM)\
	(1.0f * (GN >> 7 << 7) * (GM >> 7 << 7) / (GN * GM))

#define VP_64(GN, GM)\
	(1.0f * (GN >> 6 << 6) * (GM >> 6 << 6) / (GN * GM))

#endif


#ifndef SPLIT_K_MICRO
#define SPLIT_K_MICRO

//GZ = gridDim.z
//GK = N * OH * OW, so: GK % 4 == 0
//GK_slice = (GK / gridDim.z) >> 3 << 3
//GK = GK_slice * gridDim.z + RGK
//As: GK % 8 == 0
//So: RGK % 4 == 0
//if: GK % 8 == 0, We have RGK % 8 == 0s
#define SK_K_slice(K, GZ)  ((K / GZ) >> 3 << 3)

inline int blockNum(int N, int M) {//{ N, M } % 4 == 0
	int bn = 0,bm = 0;
	for(;;) {
		if(N > 127) bn += N >> 7; N &= 127; if(N == 0) break;//2^7
		if(N >  63) bn += 1; N &= 63; if(N == 0) break;//2^6
		if(N >  31) bn += 1; N &= 31; if(N == 0) break;//2^5
		if(N >  15) bn += 1; N &= 15; if(N == 0) break;//2^4
		if(N >   7) bn += 1; N &=  7; if(N == 0) break;//2^3
		if(N >   3) bn += 1; break;
	}
    
	for(;;) {
		if(M > 127) bm += M >> 7; M &= 127; if(M == 0) break;//2^7
		if(M >  63) bm += 1; M &= 63; if(M == 0) break;//2^6, if M <= 63, M % 64 == M
		if(M >  31) bm += 1; M &= 31; if(M == 0) break;//2^5
		if(M >  15) bm += 1; M &= 15; if(M == 0) break;//2^4
		if(M >   7) bm += 1; M &=  7; if(M == 0) break;//2^3
		if(M >   3) bm += 1; break;
	}
	return bn * bm;
}

inline int matMul_gridZ(int N, int M, int K) {//[N, K] * [K, M] = [N, M]
	float fGridZ = 0.0f;
	int b0 = 1;
	if (K >= 512) {
		b0 = blockNum(N, M);
		if (b0 < 78) {//max_processor_num
			int b1 = blockNum(K, M);//[M, N] * [N, K] = [M, K]
			int b2 = blockNum(N, K);//[N, M] * [M, K] = [N, K]
			fGridZ = ((b1 + b2)) / (b0 * 2.0f);
		}
	}

	if (fGridZ < 1.8f) return 1;

	int GridZ = (int)fGridZ;
	
	int GridZ2 = (K >> 8);
	if (GridZ > GridZ2) GridZ = GridZ2;

	float coef = K / (K + 4096.0f);
	int GridZ3 = 156 / b0 * coef;
	if (GridZ > GridZ3) GridZ = GridZ3;

	if (GridZ < 2) GridZ = 2; else GridZ = (GridZ + 1) >> 1 << 1;
	if (GridZ > 40) GridZ = 40;
	return GridZ;
}

inline int matMulT2_gridZ(int N, int M, int K) {//[N, K] * [M, K] = [N, M]
	float fGridZ = 0.0f; int b2 = 1;
	if (K >= 512) {
		b2 = blockNum(N, M);
		if (b2 < 64) {//max_processor_num
			int b0 = blockNum(N, K);
			int b1 = blockNum(M, K);
			fGridZ = ((float)(b0 + b1)) / (b2 << 2);
		}
	}

	if (fGridZ < 1.8f) return 1;

	int GridZ = (int)fGridZ;

	int GridZ2 = (K >> 8);
	if (GridZ > GridZ2) GridZ = GridZ2;

	float coef = K / (K + 4096.0f);
	int GridZ3 = 156 / b2 * coef;
	if (GridZ > GridZ3) GridZ = GridZ3;

	if (GridZ < 2) GridZ = 2; else GridZ = (GridZ + 1) >> 1 << 1;
	if (GridZ > 40) GridZ = 40;
	return GridZ;
}

#endif

#endif