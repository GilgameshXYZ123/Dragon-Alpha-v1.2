#pragma once

#ifndef MATMUL_T2_H
#define MATMUL_T2_H

#include "matMulT2_kernel.cuh"
#include "matMulT2_uernel.cuh"
#include "matMulT2SK_kernel.cuh"
#include "matMulT2SK_uernel.cuh"

#ifdef COMPLIE//<<<<complie-area--------------------------------------------------

//Common
#ifndef MAT_MUL4X_T2
#define MAT_MUL4X_T2

#ifndef MAT_MUL4X_T2_MICRO
#define MAT_MUL4X_T2_MICRO

#define mm4xT2_Branch(SIZE_Y, SIZE_X) {\
	int N1 = (N & SIZE_Y), M1 = (M & SIZE_X);\
	if (N1 && M1) {\
		int N0 = N - N1, M0 = M - M1;\
		const float *A1 = &get(A, N0, 0, K), *B1 = &get(B, M0, 0, K);\
		float *C01 = &get(C, 0, M0, SC);\
		float *C10 = &get(C, N0, 0, SC), *C11 = &get(C, N0, M0, SC);\
		matMul4x_T2(streams, index, length, A , B1, C01, N0, M1, K, SC);\
		matMul4x_T2(streams, index, length, A1, B , C10, N1, M0, K, SC);\
		matMul4x_T2(streams, index, length, A1, B1, C11, N1, M1, K, SC);}\
	else if (N1){\
		int N0 = N - N1;\
		const float *A1 = &get(A, N0, 0, K);\
		float *C1 = &get(C, N0, 0, SC);\
		matMul4x_T2(streams, index, length, A1, B, C1, N1, M, K, SC);}\
	else if (M1){\
		int M0 = M - M1;\
		const float *B1 = &get(B, M0, 0, K);\
		float *C1 = &get(C, 0, M0, SC);\
		matMul4x_T2(streams, index, length, A, B1, C1, N, M1, K, SC);}}

#endif

//for the first stack of this function: SC = M
void matMul4x_T2(jlong *streams, int &index, int length,
	const float* A,
	const float* B,
	      float* C,
	int N, int M, int K, int SC)
{
	next_cudaStream(stream, streams, index, length);
	int size = N * M; if (size <= 256) { knaiveT2(2, stream, A, B, C, N, M, K, SC); return; }

	if ((N > 127) && (M > 127) && (K > 7)) {//[128, 128]
		if (!(K & 15)) u88T2_mgk(4, stream, A, B, C, N, M, K, SC);
		else k88T2(4, stream, A, B, C, N, M, K, SC); 
		mm4xT2_Branch(127, 127); return; 
	}

	if ((N > 63) && (M > 127) && (K > 7)) {//[64, 128]
		if (!(K & 15)) u48T2_mgk(4, stream, A, B, C, N, M, K, SC);
		else k48T2(4, stream, A, B, C, N, M, K, SC);
		mm4xT2_Branch(63, 127); return;
	}

	if ((N > 127) && (M > 63) && (K > 7)) {//[128, 64]
		if (!(K & 15)) u84T2_mgk(4, stream, A, B, C, N, M, K, SC);
		else k84T2(4, stream, A, B, C, N, M, K, SC);
		mm4xT2_Branch(127, 63); return;
	}

	if ((N > 63) && (M > 63)) {//[64, 64]
		if (!(K & 15)) u44T2_mgk(4, stream, A, B, C, N, M, K, SC);
		else k44T2(4, stream, A, B, C, N, M, K, SC);
		mm4xT2_Branch(63, 63); return; 
	}

	if ((N > 31) && (M > 63)) {//[32, 64]
		if (!(K & 7)) u48T2_mgk(3, stream, A, B, C, N, M, K, SC);
		else k48T2(3, stream, A, B, C, N, M, K, SC);
		mm4xT2_Branch(31, 63); return; 
	}
	if ((N > 63) && (M > 31)) {//[64, 32]
		if (!(K & 7)) u84T2_mgk(3, stream, A, B, C, N, M, K, SC);
		else k84T2(3, stream, A, B, C, N, M, K, SC); 
		mm4xT2_Branch(63, 31); return; 
	}
	
	if ((N > 63) && (M > 15)) { k82T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(63, 15); return; }//[64, 16]
	if ((N > 31) && (M > 31)) { k44T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(31, 31); return; }//[32, 32]
	if ((N > 15) && (M > 63)) { k28T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(15, 63); return; }//[16, 64]
	if ((N > 31) && (M > 15)) { k42T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(31, 15); return; }//[32, 16]

	if (K > 7) {
		if ((N > 63) && (M >  7)) { k81T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(63, 7); return; }//[64, 8]
		if ((N > 15) && (M > 15)) { k22T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(15, 15); return; }//[16, 16]
		if ((N >  7) && (M > 63)) { k18T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(7, 63); return; }//[8, 64]
	}

	if ((N > 31) && (M > 15)) { k84T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(31, 15); return; }
	if ((N > 15) && (M > 31)) { k48T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(15, 31); return; }
	if ((N > 15) && (M >  7)) { k42T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(15, 7); return; }

	if ((N > 31)) { k81T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(31, 3); return; }
	if ((N > 7) && (M > 7)) { k22T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(7, 7); return; }
	if ((M > 31)) { k18T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(3, 31); return; }
	k11T2(2, stream, A, B, C, N, M, K, SC);
}

#endif



//Split K to improve parallism
#ifndef MAT_MUL4X_T2_SK
#define MAT_MUL4X_T2_SK

#ifndef MAT_MUL4X_T2_SK_MICRO
#define MAT_MUL4X_T2_SK_MICRO

#define mmSK4xT2_Branch(SIZE_Y, SIZE_X) {\
	int N1 = (N & SIZE_Y), M1 = (M & SIZE_X);\
	if (N1 && M1) {\
		int N0 = N - N1, M0 = M - M1;\
		const float *A1 = &get(A, N0, 0, K), *B1 = &get(B, M0, 0, K);\
		float *C01    = &get(C   , 0, M0, SC);\
	    float *Cbuf01 = &get(Cbuf, 0, M0, SC);\
		float *C10    = &get(C   , N0, 0, SC), *C11    = &get(C   , N0, M0, SC);\
        float *Cbuf10 = &get(Cbuf, N0, 0, SC), *Cbuf11 = &get(Cbuf, N0, M0, SC);\
		matMul4x_T2_SK(streams, index, length, GZ, A , B1, C01, Cbuf01, N0, M1, K, SC, K_slice, Cstride);\
		matMul4x_T2_SK(streams, index, length, GZ, A1, B , C10, Cbuf10, N1, M0, K, SC, K_slice, Cstride);\
		matMul4x_T2_SK(streams, index, length, GZ, A1, B1, C11, Cbuf11, N1, M1, K, SC, K_slice, Cstride);}\
	else if (N1){\
		int N0 = N - N1;\
		const float *A1 = &get(A, N0, 0, K);\
		float *C1    = &get(C, N0, 0, SC);\
		float *Cbuf1 = &get(Cbuf, N0, 0, SC);\
		matMul4x_T2_SK(streams, index, length, GZ, A1, B, C1, Cbuf1, N1, M, K, SC, K_slice, Cstride);}\
	else if (M1){\
		int M0 = M - M1;\
		const float *B1 = &get(B, M0, 0, K);\
		float *C1    = &get(C   , 0, M0, SC);\
		float *Cbuf1 = &get(Cbuf, 0, M0, SC);\
		matMul4x_T2_SK(streams, index, length, GZ, A, B1, C1, Cbuf1, N, M1, K, SC, K_slice, Cstride);}}

#endif

//K >= 512: K % 4 == 0
//K_slice % 8 == 0
//for the first stack of this function: SC = M
void matMul4x_T2_SK(jlong *streams, int &index, int length, int GZ,
	const float* A,
	const float* B,
	      float* C, float* Cbuf,
	int N, int M, int K, int SC,
	int K_slice, int Cstride)
{
	next_cudaStream(stream, streams, index, length);

	if ((N > 127) && (M > 127)) {//[128, 128]
		if (!(K & 15) && !(K_slice & 15)) u88T2SK_mgk(4, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
		else if (!(K & 7)) k88T2SK_mgk(4, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
		else k88T2SK(4, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
		mmSK4xT2_Branch(127, 127); return;
	}
	if ((N > 63) && (M > 63)) {//[64, 64]
		if (!(K & 7)) u88T2SK_mgk(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
		else if (!(K & 3)) k88T2SK_mgk(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
		else k88T2SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
		mmSK4xT2_Branch(63, 63); return;
	}

	if ((N > 31) && (M > 63)) {//[32, 64]
		if (!(K & 7)) u48T2SK_mgk(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
		else k48T2SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
		mmSK4xT2_Branch(31, 63); return;
	}
	if ((N > 63) && (M > 31)) {//[64, 32]
		if (!(K & 7)) u84T2SK_mgk(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
		else k84T2SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
		mmSK4xT2_Branch(63, 31); return;
	}

	if ((N > 63) && (M > 15)) { k82T2SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC); mmSK4xT2_Branch(63, 15); return; }//[64, 16]
	if ((N > 31) && (M > 31)) { k44T2SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC); mmSK4xT2_Branch(31, 31); return; }//[32, 32]
	if ((N > 15) && (M > 63)) { k28T2SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC); mmSK4xT2_Branch(15, 63); return; }//[16, 64]
	if ((N > 31) && (M > 15)) { k42T2SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC); mmSK4xT2_Branch(31, 15); return; }//[32, 16]

	if ((N > 63) && (M >  7)) { k81T2SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC); mmSK4xT2_Branch(63,  7); return; }//[64, 8]
	if ((N > 15) && (M > 15)) { k22T2SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC); mmSK4xT2_Branch(15, 15); return; }//[16, 16]
	if ((N >  7) && (M > 63)) { k18T2SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SC); mmSK4xT2_Branch( 7, 63); return; }//[8, 64]

	if ((N > 31)) { k81T2SK(2, GZ, stream, A, B, C, Cbuf, N, M, K, SC); mmSK4xT2_Branch(31, 3); return; }
	if ((N > 7) && (M > 7)) { k22T2SK(2, GZ, stream, A, B, C, Cbuf, N, M, K, SC); mmSK4xT2_Branch(7, 7); return; }
	if ((M > 31)) { k18T2SK(2, GZ, stream, A, B, C, Cbuf, N, M, K, SC); mmSK4xT2_Branch(3, 31); return; }
	k11T2SK(2, GZ, stream, A, B, C, Cbuf, N, M, K, SC);
}

#endif

#endif//complie-area>>>>------------------------------------------------------------

#endif 
