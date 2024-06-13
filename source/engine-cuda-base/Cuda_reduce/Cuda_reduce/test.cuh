#pragma once

#define TEST_H
#ifndef TEST_H
#define TEST_H


#ifndef UTIL
#define UTIL

float samePercent(float *a, float *b, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		if (a[i] == b[i]) { sum++; continue; }
		float dif = fabs((a[i] - b[i]) / (a[i] + b[i]));

		if (dif < 1e-3) sum++;
		//else cout << i << ":" << dif << ", a = " << a[i] << ", b = " << b[i] << endl;
	}
	return 1.0f*sum / length;
}

float* newRandomFloatVec(int length)
{
	float *p = new float[length];
	for (int i = 0; i < length; i++) p[i] = 1.0f*(rand() % 1000) / 1000 + 1;
	return p;
}

float next_float(float min, float max) {
	float v = (float)(rand() % 1000) / 1000;
	return v * (max - min) + min;
}

float* newRandomFloatVec(int lengthv, int width, int stride)//0-256
{
	int height = lengthv / stride;
	float *p = new float[lengthv], *tp = p;
	memset(p, 0, sizeof(float)*lengthv);
	for (int i = 0; i < height; i++)
{
		for (int j = 0; j < width; j++) 
			tp[j] = (float)(rand() % 1000) / 1000;
		tp += stride;
	}

	return p;
}

float *newDevFloatVec(int length)
{
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemset(dp, 0, size); printError(error);
	return dp;
}
float* newDevFloatVec(float *p, int length)
{
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemcpy(dp, p, size, cudaMemcpyHostToDevice); printError(error);
	return dp;
}
void println(float *p, int length)
{
	for (int i = 0; i < length; i++) cout << p[i] << ' ';
	cout << endl;
}

float zeroPercent(float *a, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
		if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}


void SumEachRow(float *A, float *V, int N, int M, int SA)
{
	memset(V, 0, sizeof(float)*N);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++) V[i] += get(A, i, j, SA);
}
void MsumOfEachRow(float **A, float *V, int N, int M)
{
	for (int i = 0; i < N; i++)
	{
		V[i] = 0;
		for (int j = 0; j < M; j++) V[i] += A[i][j];
	}
}

float sum(float* a, int length)
{
	float r = 0;
	for (int i = 0; i < length; i++) r += a[i];
	return r;
}

void SumEachField(float *A, float *V, int N, int M)
{
	memset(V, 0, sizeof(float)*M);
	for (int j = 0; j < M; j++)
		for (int i = 0; i < N; i++) V[j] += get(A, i, j, M);
}

#endif


#ifndef CPU_REDUCE_FUNCTIONS
#define CPU_REDUCE_FUNCTIONS

void cpu_field_max(float *A, int N, int M, float *V, int width, int stride)
{
	for (int j = 0; j < M; j++)
	{
		if (j%stride >= width) continue;
		float max = FLOAT_MIN;
		for (int i = 0; i < N; i++) {
			int index = i * M + j;
			if((index % stride) < width)
				max = (max >= A[index] ? max : A[index]);
		}
		if (j % stride < width) V[j] = max;
	}
}

void cpu_field_quadratic(float *A, float alpha, float beta, float gamma,
	float *V, int height, int width, int stride)
{
	memset(V, 0, sizeof(float)*width);
	for (int j = 0; j < stride; j++)
	{
		if ((j%stride) >= width) continue;
		for (int i = 0; i < height; i++)
		{
			float a = get(A, i, j, stride);
			V[j] += alpha * (a*a) + beta * a + gamma;
		}
	}
}

void cpu_row_quadratic(float *A, float alpha, float beta, float gamma,
	float *V, int N, int M, int width, int stride)
{
	memset(V, 0, sizeof(float)*N);
	for (int i = 0; i < N; i++) 
	for (int j = 0; j < M; j++)
	{
		float a = get(A, i, j, M);
		a = alpha * a*a + beta * a + gamma;
		V[i] += a * ((j % stride) < width);
	}
		
}

void cpu_field_linear(float *A, float alpha, float beta,
	float *V, int height, int width, int stride)
{
	memset(V, 0, sizeof(float)*stride);
	for (int j = 0; j < stride; j++)
	{
		if ((j%stride) >= width) continue;
		for (int i = 0; i < height; i++) {
			float a = get(A, i, j, stride);
			V[j] += alpha * a + beta;
		}
	}
}

void cpu_row_linear(float *A, float alpha, float beta,
	float *V, int N, int M, int width, int stride)
{
	memset(V, 0, sizeof(float)*N);
	for (int i = 0; i < N; i++)
	for (int j = 0; j < M; j++) 
	{
		float a = get(A, i, j, M);
		a = alpha * a + beta;
		V[i] += a * ((j % stride) < width);
	}
}

#endif


void testCorrectRow(int N, int M, int width)
{
	int lengthv = N * M;
	int stride = (width + 3) >> 2 << 2;
	printf("testCorrectRow:");
	printf("width, stride = (%d, %d)\n", width, stride);
	printf("N, M, lengthv = (%d, %d, %d)\n", N, M, lengthv);

	float *A = newRandomFloatVec(lengthv, width, stride);//[height, stride]
	float *dA = newDevFloatVec(A, lengthv);

	float *V1 = new float[N];//height

	int HV = row_nextM(M); cout << "HV = " << HV << endl;
	float *dV = newDevFloatVec(HV * N);//width -> HV

	float alpha = next_float(0, 1);
	float beta = next_float(0, 1);
	float gamma = next_float(0, 1);

	cudaError_t error;

	//CPU-------------------------------------------------------
	cpu_row_linear(A, alpha, beta, V1, N, M, width, stride);

	cout << "CPU: "; println(V1, 10);

	//GPU-------------------------------------------------------
	
	__row_linear(NULL, dA, alpha, beta, N, M, dV, dV, width, stride, 1);

	error = cudaDeviceSynchronize(); printError(error);
	float *Vr = new float[HV * N];
	error = cudaMemcpy(Vr, dV, sizeof(float)* HV * N, cudaMemcpyDeviceToHost); printError(error);
	error = cudaGetLastError(); printError(error);
	float* V2 = new float[N]; SumEachField(Vr, V2, 1, N);
	cout << "GPU: "; println(V2, 10);

	//compare------------------------------------------------
	float sp = samePercent(V1, V2, N); cout << "sp = " << sp << endl;
	float zp0 = zeroPercent(V1, N); cout << "zp0 = " << zp0 << endl;
	float zp1 = zeroPercent(V2, N); cout << "zp1 = " << zp1 << endl;
	
	error = cudaFree(dA); printError(error);
	error = cudaFree(dV); printError(error);

	delete A;
	delete V1;
	delete V2;

	if (sp != 1) {
		cout << "Error: N = " << N << ", M = " << M << endl;
		exit(2);
	}
}

void testSpeedRow(int nIter, int N, int M, int width)
{
	int lengthv = N * M;
	int stride = (width + 3) >> 2 << 2;
	printf("testSpeedRow:");
	printf("N, M, lengthv = (%d, %d, %d)\n", N, M, lengthv);

	float *A = newRandomFloatVec(lengthv, width, stride);
	float *dA = newDevFloatVec(A, lengthv);

	//GPU-----------------------------------------------
	cudaError_t error;

	int HV = row_nextM(M); cout << "HV = " << HV << endl;
	float *dV = newDevFloatVec(HV * M);

	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		__row_linear_stage(NULL, dA, 1, 1, N, M, dV, HV, width, stride);
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = (end - start);
	float time = 1.0f * div / nIter;
	int data_size = (N*M + HV * M) * sizeof(float);
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Time = " << time << " mesc, "
		<< "Speed = " << speed << "GB/s" << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dV); printError(error);
	delete A;
}


void testCorrectField(int height, int width)
{
	int stride = ((width + 3) >> 2) << 2;
	int lengthv = height * stride;
	printf("testCorrect_field: N, M, stride = (%d, %d, %d)\n", height, width, stride);

	float *A = newRandomFloatVec(lengthv, width, stride);
	float *dA = newDevFloatVec(A, lengthv);

	float *V1 = new float[stride];

	int HV = field_nextN(height, stride);
	cout << "HV = " << HV << endl;
	float *dV = newDevFloatVec(HV * stride);

	float alpha = next_float(0, 1);
	float beta = next_float(0, 1);
	float gamma = next_float(0, 1);

	cudaError_t error;
	//CPU----------------------------------------------------------
	cpu_field_linear(A, alpha, beta, V1, height, width, stride);

	cout << "CPU: "; println(V1, 10);

	//GPU----------------------------------------------------------
	__field_linear(NULL, dA, alpha, beta, height, stride, dV, dV, width, stride, 1);

	error = cudaDeviceSynchronize(); printError(error);

	float *Vr = new float[HV * stride];
	error = cudaMemcpy(Vr, dV, sizeof(float)* HV * stride, cudaMemcpyDeviceToHost); printError(error);
	error = cudaGetLastError(); printError(error);
	//float* V2 = new float[stride]; SumEachField(Vr, V2, HV, stride);
	float* V2 = new float[stride]; SumEachField(Vr, V2, 1, stride);

	cout << "GPU: "; println(V2, 10);

	//compare------------------------------------------------
	float sp = samePercent(V1, V2, stride); cout << "sp = " << sp << endl;
	float zp0 = zeroPercent(V1, stride); cout << "zp0 = " << zp0 << endl;
	float zp1 = zeroPercent(V2, stride); cout << "zp1 = " << zp1 << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dV); printError(error);

	delete A;
	delete V1;
	delete V2;

	if (sp != 1) {
		cout << "Error: N = " << height << ", M = " << width << endl;
		exit(2);
	}
}

void testSpeedField(int nIter, int height, int width)
{
	int stride = ((width + 3) >> 2) << 2;
	int lengthv = height * stride;
	printf("testCorrect_field: N, M, stride = (%d, %d, %d)\n", height, width, stride);

	float *A = newRandomFloatVec(lengthv, width, stride);
	float *dA = newDevFloatVec(A, lengthv);

	float *V1 = new float[stride];

	int HV = field_nextN(height, stride);
	cout << "HV = " << HV << endl;
	float *dV = newDevFloatVec(HV * stride);

	float alpha = next_float(0, 1);
	float beta = next_float(0, 1);
	float gamma = next_float(0, 1);
	
	cudaError_t error;

	//GPU-----------------------------------------------
	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		__field_linear(NULL, dA, alpha, beta, height, stride, dV, dV, width, stride, 1);
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = (end - start);
	float time = 1.0f * div / nIter;
	int data_size = (height*stride + HV * stride) * sizeof(float);
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Time = " << time << " mesc, "
		<< "Speed = " << speed << "GB/s" << endl;

	error = cudaFree(dA); printError(error);
	error = cudaFree(dV); printError(error);
	delete A;
}

void test()
{
	/*int N = 16, M = 16, width = 15;
	for (int n = 1; n <= 128; n++)
	for (int m = 4; m <= 32; m ++)
		testCorrectField(n, m);*/

	//64: 347:
	//128: 

	//int N = 512 * 16, M = 128;
	//int N = 512, M = 1024;
	//testCorrectField(N, M);
	//testSpeedField(500, N, M*32);

	//int N = 16, M = 16, width = 15;
	//for (int n = 1; n <= 128; n++)
	//for (int m = 4; m <= 32; m ++)
	//	testCorrectField(n, m);

	int N = 512 * 2, M = 1024;
	testCorrectRow(N, M, M);
	testSpeedRow(500, N * 32, M, M);
}
	


main() { test(); }
#endif