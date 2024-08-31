#pragma once

#ifndef TEST_H
#define TEST_H

#ifndef COMPILE


#ifndef UTIL
#define UTIL

float *newPtr(int length)
{
	float *p = new float[length];
	memset(p, 0, sizeof(float)*length);
	return p;
}

float *newRandomPtr(int length)
{
	float *p = new float[length];
	for (int i = 0; i < length; i++) p[i] = 1.0f*(rand() % 1000) / 1000;
	return p;
}

float *newDevPtr(float *p, int length)
{
	float *dp = NULL;
	cudaError_t error = cudaMalloc((void**)&dp, length * sizeof(float)); printError(error);
	cudaMemcpy(dp, p, length * sizeof(float), cudaMemcpyHostToDevice); printError(error);
	return dp;
}

float *newDevPtr(int length)
{
	float *dp = NULL;
	cudaError_t error = cudaMalloc((void**)&dp, length * sizeof(float)); printError(error);
	error = cudaMemset(dp, 0, sizeof(float)*length); printError(error);
	return dp;
}

void println(float *p, int length)
{
	for (int i = 0; i < length; i++) cout << p[i] << ' ';
	cout << endl;
}

float samePercent(float *a, float *b, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		float dif = fabs(a[i] - b[i]) / fabs(a[i] + b[i]);
		if (dif < 1e-5) sum++;
		//else cout << dif<<" " << a[i] << ", " << b[i] << endl;
	}
	return 1.0f*sum / length;
}

void multiply(float *A, float *B, float *C, int N, int M, int K)
{
	for (int k = 0; k < K; k++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++)
				get(C, i, j, M) += get(A, i, k, K) * get(B, k, j, M);
}

void multiplyT1(float *A, float *B, float *C, int N, int M, int K)
{
	for (int k = 0; k < K; k++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++) 
				get(C, i, j, M) += get(A, k, i, N) * get(B, k, j, M);
}

void multiplyT2(float *A, float *B, float *C, int N, int M, int K)
{
	for (int k = 0; k < K; k++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++)
				get(C, i, j, M) += get(A, i, k, K) * get(B, j, k, K);
}

float zeroPercent(float *a, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
		if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}

#endif 


template<int LB>
void testCorrect(int N, int M, int K)
{
	printf("(N, M, K) = (%d, %d, %d)\n", N, M, K);
	
	float *A = newRandomPtr(N*K);
	float *B = newRandomPtr(K*M);
	float *C1 = newPtr(N*M);
	float *C2 = newPtr(N*M);

	float *dA = newDevPtr(A, N*K);
	float *dB = newDevPtr(B, K*M);
	float *dC = newDevPtr(N*M);
	cudaError_t error;

	//CPU------------------------------
	multiply(A, B, C1, N, M, K); int GZ = matMul_gridZ(N, M, K);
	//multiplyT1(A, B, C1, N, M, K);
	//multiplyT2(A, B, C1, N, M, K); int GZ = matMulT2_gridZ(N, M, K);

	cout << "CPU: "; println(C1, 10);

	//int GZ = K >> 9;//512
	
	int part = GZ - 1; 
	int Cstride = N * M;
	int size_Cbuf = part * N * M; 

	int K_slice = 0;
	float *dCbuf = NULL; 
	if (part > 0) {
		dCbuf = newDevPtr(size_Cbuf);
		K_slice = SK_K_slice(K, GZ);
	}

	cout << "GZ = " << GZ << endl;
	cout << "K_slice = " << K_slice << endl;
	cout << "sizeCbuf: " << size_Cbuf << endl;

	jlong streams[10] = { NULL }; int stream_length = 10;

	//GPU------------------------------
	//matMul===========================================
	{
		//k88_pm8_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//k88_p_mgk(LB, NULL, dA, dB, dC, N, M, K, M);

		//u88_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u84_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u48_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u44_mgk(LB, NULL, dA, dB, dC, N, M, K, M);

		//k88_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//k88(LB, NULL, dA, dB, dC, N, M, K, M);

		//k84(LB, NULL, dA, dB, dC, N, M, K, M);
		//k48(LB, NULL, dA, dB, dC, N, M, K, M);
		//k44(LB, NULL, dA, dB, dC, N, M, K, M);
		//k82(LB, NULL, dA, dB, dC, N, M, K, M);
		//k28(LB, NULL, dA, dB, dC, N, M, K, M);
		//k42(LB, NULL, dA, dB, dC, N, M, K, M);
		//k24(LB, NULL, dA, dB, dC, N, M, K, M);

		//k22(LB, NULL, dA, dB, dC, N, M, K, M);

		//k81(LB, NULL, dA, dB, dC, N, M, K, M);
		//k18(LB, NULL, dA, dB, dC, N, M, K, M);

		//k41(LB, NULL, dA, dB, dC, N, M, K, M);
		//k14(LB, NULL, dA, dB, dC, N, M, K, M);

		//k21(LB, NULL, dA, dB, dC, N, M, K, M);
		//k12(LB, NULL, dA, dB, dC, N, M, K, M);
		//k11(LB, NULL, dA, dB, dC, N, M, K, M);

		//s8x2_1(LB, NULL, dA, dB, dC, N, M, K, M);
		//s4x2_1(LB, NULL, dA, dB, dC, N, M, K, M);
		//s2x2_1(LB, NULL, dA, dB, dC, N, M, K, M);

		//int idx = 0; matMul4x(streams, idx, stream_length,  dA, dB, dC, N, M, K, M);
	}
	
	//matMul SK========================================
	{
		//k88SK_pm8_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k88SK_p_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

		//u88SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//u84SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//u48SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//u44SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

		//k88SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k88SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

		//k84SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k48SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k44SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k82SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k28SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k42SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k24SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

		//k22SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k21SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k12SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k11SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

		//int idx = 0; matMul_4x_SK(streams, idx, stream_length, GZ, dA, dB, dC, dCbuf, N, M, K, M, K_slice, Cstride);
		//if (GZ > 1) SKbuf_summary(NULL, dCbuf, dC, part, N * M);
	}

	//matMulT1=========================================
	{
		//matMulT1=========================================
		//k88T1_mgk(LB, NULL, dA, dB, dC, N, M, K, N, M);
		//k88T1(LB, NULL, dA, dB, dC, N, M, K, N, M);
		//k84T1(LB, NULL, dA, dB, dC, N, M, K, N, M);
		//k84T1(LB, NULL, dA, dB, dC, N, M, K, N, M);

		//k48T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k82T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k28T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k81T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k18T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k44T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k42T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k24T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		//k22T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
	}

	//matMulT1 SK======================================
	{
		//k88T1SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k88T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);

		//k84T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k48T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k44T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k82T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k28T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k42T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k24T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k22T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k21T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k12T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
		//k11T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);

		//if (GZ > 1) SKbuf_summary(NULL, dCbuf, dC, part, N * M);
	}

	//matMul T2========================================
	{
		//k88T2v_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//k88T2v(LB, NULL, dA, dB, dC, N, M, K, M);

		//u88T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u84T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u48T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//u44T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);

		//k88T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
		//k88T2(LB, NULL, dA, dB, dC, N, M, K, M);

		//k84T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k48T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k44T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k82T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k28T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k42T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k24T2(LB, NULL, dA, dB, dC, N, M, K, M);

		//k22T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k81T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k18T2(LB, NULL, dA, dB, dC, N, M, K, M);
		//k11T2(LB, NULL, dA, dB, dC, N, M, K, M);

		int index = 0; matMul4x_T2(streams, index, stream_length, dA, dB, dC, N, M, K, M);
	}

	//matMul T2 SK=====================================
	{
		//k88T2SK_v_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k88T2SK_v(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

		//u88T2SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//u84T2SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//u48T2SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

		//k88T2SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k88T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

		//k84T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k48T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

		//k44T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k82T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k28T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k42T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k24T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

		//k22T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k81T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k18T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		//k11T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
		
		//if (GZ > 1) SKbuf_summary(NULL, dCbuf, dC, part, N * M);
	}

	//compare--------------------------
	error = cudaStreamSynchronize(NULL);
	error = cudaMemcpy(C2, dC, sizeof(float)*N*M, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: ";  println(C2, 10);

	float sp = samePercent(C1, C2, M*N);
	cout << "Same Percent: " << sp << endl;

	float zp = zeroPercent(C2, M*N);
	cout << "Zero Percent: " << zp << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dB); printError(error);
	error = cudaFree(dC); printError(error);
	error = cudaGetLastError();
	cout << error << ":" << cudaGetErrorName(error) << endl
		<< cudaGetErrorString(error) << endl;


	delete A;
	delete B;
	delete C1;
	delete C2;

	if (sp < 0.999) exit(-2);

	if (error != cudaSuccess) {
		cout << endl << N << " " << M << " " << K << endl;
		exit(0);
	}
}

template<int LB>
void testSpeed(int nIter, int N, int M, int K)
{
	printf("(N, M, K) = (%d, %d, %d)\n", N, M, K);

	float *A = newRandomPtr(N*K);
	float *B = newRandomPtr(K*M);

	float *dA = newDevPtr(A, N*K);
	float *dB = newDevPtr(B, K*M);
	float *dC = newDevPtr(N*M);

	int GZ = matMul_gridZ(N, M, K);
	//int GZ = matMulT2_gridZ(N, M, K); 

	//int GZ = K >> 9;

	int part = GZ - 1;
	int Cstride = N * M;
	int size_Cbuf = part * N*M;
	
	int K_slice = 0;
	float *dCbuf = NULL;
	if (part > 0) {
		K_slice = SK_K_slice(K, GZ);
		dCbuf = newDevPtr(size_Cbuf);
	}

	cout << "GZ = " << GZ << endl;
	cout << "K_slice = " << K_slice << endl;
	cout << "sizeCbuf: " << size_Cbuf << endl;

	jlong streams[10]; int stream_length = 10;
	for (int i = 0; i < 10; i++) {
		cudaStream_t stm; cudaStreamCreate(&stm);
		streams[i] = (jlong)(intptr_t)stm;
	}

	cudaError_t error;
	
	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		//matMul===========================================
		{
			//k88_pm8_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//k88_p_mgk(LB, NULL, dA, dB, dC, N, M, K, M);

			u88_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u84_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u48_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u44_mgk(LB, NULL, dA, dB, dC, N, M, K, M);

			//k88_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//k88(LB, NULL, dA, dB, dC, N, M, K, M);

			//k84(LB, NULL, dA, dB, dC, N, M, K, M);
			//k48(LB, NULL, dA, dB, dC, N, M, K, M);
			//k44(LB, NULL, dA, dB, dC, N, M, K, M);
			//k82(LB, NULL, dA, dB, dC, N, M, K, M);
			//k28(LB, NULL, dA, dB, dC, N, M, K, M);
			//k42(LB, NULL, dA, dB, dC, N, M, K, M);
			//k24(LB, NULL, dA, dB, dC, N, M, K, M);

			//k22(LB, NULL, dA, dB, dC, N, M, K, M);
			
			//k81(LB, NULL, dA, dB, dC, N, M, K, M);
			//k18(LB, NULL, dA, dB, dC, N, M, K, M);

			//k41(LB, NULL, dA, dB, dC, N, M, K, M);
			//k14(LB, NULL, dA, dB, dC, N, M, K, M);

			//k21(LB, NULL, dA, dB, dC, N, M, K, M);
			//k12(LB, NULL, dA, dB, dC, N, M, K, M);
			//k11(LB, NULL, dA, dB, dC, N, M, K, M);

			//s8x2_1(LB, NULL, dA, dB, dC, N, M, K, M);
			//s4x2_1(LB, NULL, dA, dB, dC, N, M, K, M);
			//s2x2_1(LB, NULL, dA, dB, dC, N, M, K, M);

			//int idx = 0; matMul4x(streams, idx, stream_length, dA, dB, dC, N, M, K, M);
		}

		//matMul SK========================================
		{
			//k88SK_pm8_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k88SK_p_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//u88SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//u84SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//u48SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//u44SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//k88SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k88SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//k84SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k48SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k44SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//k82SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k28SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k42SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k24SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//k22SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k21SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k12SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k11SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//int idx = 0; matMul_4x_SK(streams, idx, stream_length, GZ, dA, dB, dC, dCbuf, N, M, K, M, K_slice, Cstride);
			//if (GZ > 1) SKbuf_summary(NULL, dCbuf, dC, part, N * M);
		}
		
		//matMulT1=========================================
		{
			//k88T1_mgk(LB, NULL, dA, dB, dC, N, M, K, N, M);
			//k88T1(LB, NULL, dA, dB, dC, N, M, K, N, M);

			//k84T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k48T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k82T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k28T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k81T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k18T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k44T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k42T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k24T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
			//k22T1(LB, NULL, 1.0f, dA, dB, dC, N, M, K, N, M);
		}

		//matMulT1 SK======================================
		{
			//k88T1SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k88T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);

			//k84T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k48T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k44T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k82T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k28T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k42T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k24T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k22T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k21T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k12T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);
			//k11T1SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, N, M);

			//if (GZ > 1) SKbuf_summary(NULL, dCbuf, dC, part, N * M);
		}
		
		//matMulT2=========================================
		{
			//k88T2v_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//k88T2v(LB, NULL, dA, dB, dC, N, M, K, M);

			//u88T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u84T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u48T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//u44T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);

			//k88T2_mgk(LB, NULL, dA, dB, dC, N, M, K, M);
			//k88T2(LB, NULL, dA, dB, dC, N, M, K, M);

			//k84T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k48T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k44T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k82T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k28T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k42T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k24T2(LB, NULL, dA, dB, dC, N, M, K, M);

			//k22T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k81T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k18T2(LB, NULL, dA, dB, dC, N, M, K, M);
			//k11T2(LB, NULL, dA, dB, dC, N, M, K, M);
		}

		//matMul T2 SK=====================================
		{
			//k88T2SK_v_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k88T2SK_v(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//u88T2SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//u84T2SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//u48T2SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//k88T2SK_mgk(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k88T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//k84T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k48T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//k44T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k82T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k28T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k42T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k24T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//k22T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k81T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k18T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);
			//k11T2SK(LB, GZ, NULL, dA, dB, dC, dCbuf, N, M, K, M);

			//if (GZ > 1) SKbuf_summary(NULL, dCbuf, dC, part, N * M);
		}
		error = cudaDeviceSynchronize(); printError(error);
	}

	clock_t end = clock();
	int div = end - start;
	float time = 1.0f*div / nIter;

	float size = 1.0f * N / 1024 * M / 1024 * K / 1024;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
	cout << "Size = " << size << ", Time = " << time << " msec, Performace = " << performance << " GFlop/s" << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dB); printError(error);
	error = cudaFree(dC); printError(error);
	error = cudaGetLastError(); printError(error);
}

main() 
{ 

	//=====[normal area]==============================================
	//int N = 2048, M = 2048, K = 2048 + 4;
	//int N = 2048, M = 2048, K = 4096;
	//int N = 512, M = 2048, K = 8192 ;
	//int N = 128, M = 128, K = 132;
	int N = 1024, M = 1024, K = 1024;

	//int N = 68, M = 124, K = 520;

	//testCorrect<4>(N, M, K); return 0;
	//testCorrect<4>(N/2, M/2, K/2);
	//testSpeed<4>(1, N, M, K);
	testSpeed<4>(1000, N * 2, M * 2, K * 2);
	//testSpeed<4>(1000, N, M, K);

	//=====[normal area]==============================================

	//int N = 256, M = 1000, K = 4096;

	//A: 4104.02 GFlop/s
	//B:

	//testSpeed<4>(1000, N, M, K);

	//int N = 1092, M = 1092, K = 1024;
	//int N = 512, M = 2048, K = 1024;
	//int N = 512, M = 128, K = 512;
	//int N = 256, M = 256, K = 1024 * 16;

	//int N = 1024 * 16, M = 16, K = 1024;
	//int N = 1024 * 16, M = 8, K = 2048;
	//int N = 1024 * 16, M = 4, K = 4096;

	//int N = 16, M = 1024 * 16, K = 1024;
	//int N = 8, M = 1024 * 16, K = 2048;
	//int N = 4, M = 1024 * 16, K = 2048;

	//testCorrect<4>(128, 4096, 2048);
	//testCorrect<4>(128, 4096, 4096 - 4);
	//testSpeed<4>(1000, 128, 4096, 8192);

	/*testCorrect<4>(256, 1000, 2048);
	testSpeed<4>(1000, 256, 1000, 2048); */

	/*testCorrect<4>(512, 1000, 2048);
	testSpeed<4>(1000, 512, 1000, 2048);*/

	//testCorrect<4>(256, 1008, 4096);//A: Size = 0.976562, Time = 0.305 msec, Performace = 6875.91 GFlop/s
	//testSpeed<4>(1000, 256, 1008, 4096);//

	/*testCorrect<3>(124, 4092, 2048);
	testSpeed<3>(1000, 124, 4092, 8192);*/

	//testCorrect<3>(124, 4088, 2048);
	//testSpeed<3>(1000, 124, 4088, 8192);

	//testCorrect<4>(N, M, K);
	//testCorrect<4>(N*2, M*2, K/4);
	//testCorrect<4>(N, M, K - 4);
	//testCorrect<4>(N, M, K - 4);
	//testCorrect<4>(N, M, K + 4);
	//testSpeed<4>(1000, N*2, M, K);
	//testSpeed<4>(1000, N * 2, M * 2, K * 2);
	//testSpeed<4>(1000, N*2, M*2, K*2);
	//testSpeed<4>(500, N*4, M*4, K*4);
 }

#endif

#endif