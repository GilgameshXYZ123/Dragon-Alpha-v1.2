#pragma once

#ifndef TEST_H
#define TEST_H

#ifndef COMPLIE//<<<<complie-area--------------------------------------------------

#ifndef UTIL
#define UTIL

char* newCharVec(int length) {
	char *p = new char[length];
	memset(p, 0, sizeof(char)*length);
	return p;
}

float* newFloatVec(int length) {
	float *p = new float[length];
	memset(p, 0, sizeof(float)*length);
	return p;
}

char* newRandomCharVec(int lengthv) {
	char *p = new char[lengthv], *tp = p;
	for(int i=0; i<lengthv; i++) tp[i] = (rand() % 255);
	return p;
}

float* newRandomFloatVec(int lengthv) {
	float *p = new float[lengthv], *tp = p;
	for (int i = 0; i < lengthv; i++) tp[i] = (rand() % 1000) / 1000.0f;
	return p;
}

char* newRandomCharVec(int lengthv, int width, int stride) {
	int height = lengthv / stride;
	char *p = new char[lengthv], *tp = p;
	memset(p, 0, sizeof(char)*lengthv);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) tp[j] = (rand() % 255);
		tp += stride;
	}
	return p;
}

char* newDevCharVec(int length) {
	char *dp = NULL;
	size_t size = sizeof(char)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemset(dp, 0, size); printError(error);
	return dp;
}

float* newDevFloatVec(int length) {
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemset(dp, 0, size); printError(error);
	return dp;
}

char* newDevCharVec(char *p, int length) {
	char *dp = NULL;
	size_t size = sizeof(char)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemcpy(dp, p, size, cudaMemcpyHostToDevice); printError(error);
	return dp;
}

float* newDevFloatVec(float *p, int length) {
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemcpy(dp, p, size, cudaMemcpyHostToDevice); printError(error);
	return dp;
}

void println(float *p, int length) {
	for (int i = 0; i < length; i++) cout << p[i] << ' ';
	cout << endl;
}

void println(char *p, int length) {
	for (int i = 0; i < length; i++) cout << (int)p[i] << ' ';
	cout << endl;
}

float samePercent(char *a, char *b, int length) {
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		//if (b[i] == 0) cout << "zero: " << i << endl;
		if (a[i] == b[i]) {sum++; continue; }
		/*else {
			cout << i << ":" << (int)a[i]  << ", " << (int)b[i] << endl;
		}*/
		//int dif = fabs(1.0f *(a[i] - b[i]) / (a[i] + b[i]));
		//if (dif < 0.03) { sum++;  }
		//else {
		//	//cout << i << ":" << dif  << ", " << (int)a[i]  << ", " << (int)b[i] << endl;
		//}
	}
	return 1.0f*sum / length;
}

float samePercent(float* a, float* b, int length) {
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		//if (b[i] == 0) cout << "zero: " << i << endl;
		if (a[i] == b[i]) { sum++; continue; }
		float dif = fabs((a[i] - b[i]) / (a[i] + b[i]));
		if (dif < 0.001f) { sum++;  }
		else {
			//cout << i << ":" << dif  << ", " << a[i]  << ", " << b[i] << endl;
		}
	}
	return 1.0f*sum / length;
}

float zeroPercent(char *a, int length) {
	int sum = 0;
	for (int i = 0; i < length; i++) if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}

float zeroPercent(float *a, int length) {
	int sum = 0;
	for (int i = 0; i < length; i++) if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}

float sum(char* a, int length) {
	float r = 0;
	for (int i = 0; i < length; i++) r += a[i];
	return r;
}

float next_float(int min, int max) {
	float base = (rand() % 1000) / 1000.0f;
	return base*(max - min) + min;
}

float max(float* a, int length) {
	int max = a[0];
	for (int i = 1; i < length; i++)
		if (a[i] > max) max = a[i];
	return max;
}

#endif

#include "test_function.cuh"
#include "test_affine.cuh"
#include "test_resize.cuh"
#include "test_pad_trim.cuh"
#include "test_extract_3channels.cuh"
#include "test_reduce.cuh"


#ifndef TRANSPOSE4D_CPU
#define TRANSPOSE4D_CPU

void img_tranpose4d_CPU(
	const char *X,
	      char *Y,
	int Xdim1, int Xdim2, int Xdim3,
	int Ydim1, int Ydim2, int Ydim3,
	int dimIdx1, int dimIdx2,
	int strideX, int strideY, int length)
{
	int Xdim23 = Xdim2 * Xdim3;
	int Xdim123 = Xdim1 * Xdim2 * Xdim3;

	int x[4];
	for (int i = 0; i < length; i++) {
		int xoffset = i;
		x[0] = xoffset / Xdim123; int xoffset_res = xoffset % Xdim123;
		x[1] = xoffset_res / Xdim23; xoffset_res %= Xdim23;
		x[2] = xoffset_res / Xdim3;
		x[3] = xoffset_res % Xdim3;

		int t = x[dimIdx1]; x[dimIdx1] = x[dimIdx2]; x[dimIdx2] = t;
		int yoffset = ((x[0] * Ydim1 + x[1])*Ydim2 + x[2])*Ydim3 + x[3];

		//consider the mem alignment
		xoffset = (xoffset / Xdim3)*strideX + (xoffset % Xdim3);
		yoffset = (yoffset / Ydim3)*strideY + (yoffset % Ydim3);

		Y[yoffset] = X[xoffset];
	}
}

#endif

void transpose_testCorrect(
	int dim0, int dim1, int dim2, int dim3,
	int dimIdx1, int dimIdx2)
{
	cout << "testCorrect:" << endl;
	cout << "Xdim: " << dim0 << ',' << dim1 << ',' << dim2 << ',' << dim3 << endl;

	int length = dim0 * dim1 * dim2 * dim3;
	int height = dim0 * dim1 * dim2;
	int strideX = (dim3 + 3) >> 2 << 2;
	int lengthXv = dim0 * dim1 * dim2 * strideX;

	char *X = newRandomCharVec(lengthXv, dim3, strideX);

	//exchange dims-------------------------------------------------------------
	int dim[4]{ dim0, dim1, dim2, dim3 };
	int t = dim[dimIdx1]; dim[dimIdx1] = dim[dimIdx2]; dim[dimIdx2] = t;

	int Ydim0 = dim[0], Ydim1 = dim[1], Ydim2 = dim[2], Ydim3 = dim[3];
	cout << "Ydim: " << Ydim0 << ',' << Ydim1 << ',' << Ydim2 << ',' << Ydim3 << endl;

	int strideY = (Ydim3 + 3) >> 2 << 2;
	int lengthYv = Ydim0 * Ydim1 * Ydim2 * strideY;

	//CPU---------------------------------------------------------
	char *Y1 = new char[lengthYv]; memset(Y1, 0, lengthYv * sizeof(char));

	img_tranpose4d_CPU(X, Y1,
		dim1, dim2, dim3,
		Ydim1, Ydim2, Ydim3,
		dimIdx1, dimIdx2,
		strideX, strideY,
		length);

	cout << "Y1 = "; println(Y1, 15);

	//GPU-----------------------------------------------------
	char *dX = newDevCharVec(X, lengthXv);
	char *dY = newDevCharVec(lengthYv);

	__img_transpose4d(NULL,
		dX, dY,
		dim1, dim2, dim3,
		Ydim1, Ydim2, Ydim3,
		dimIdx2, dimIdx1,
		strideX, strideY,
		length);

	cudaError_t error = cudaDeviceSynchronize(); printError(error);
	char *Y2 = new char[lengthYv]; memset(Y2, 0, lengthYv * sizeof(char));
	error = cudaMemcpy(Y2, dY, sizeof(char)*lengthYv, cudaMemcpyDeviceToHost); printError(error);
	cout << "Y2 = "; println(Y2, 30);

	//compare------------------------------------------------------
	float zpY1 = zeroPercent(Y1, lengthYv); cout << "zpY1 = " << zpY1 << endl;
	float zpY2 = zeroPercent(Y2, lengthYv); cout << "zpY2 = " << zpY2 << endl;
	float sp = samePercent(Y1, Y2, lengthYv); cout << "sp = " << sp << endl;

	delete X;
	delete Y1;
	delete Y2;

	error = cudaFree(dX); printError(error);
	error = cudaFree(dY); printError(error);

	if (sp != 1) exit(2);
	cout << endl;
}


void transpose_testSpeed(int nIter,
	int dim0, int dim1, int dim2, int dim3,
	int dimIndex1, int dimIndex2)
{
	cout << "testSpeed:" << endl;
	cout << "Xdim: " << dim0 << ',' << dim1 << ',' << dim2 << ',' << dim3 << endl;

	int length = dim0 * dim1 * dim2 * dim3;
	int height = dim0 * dim1 * dim2;
	int strideX = (dim3 + 3) >> 2 << 2;
	int lengthXv = dim0 * dim1 * dim2 * strideX;

	char *X = newRandomCharVec(lengthXv, dim3, strideX);

	//exchange dims-------------------------------------------------------------
	int dim[4]{ dim0, dim1, dim2, dim3 };
	int t = dim[dimIndex1]; dim[dimIndex1] = dim[dimIndex2]; dim[dimIndex2] = t;
	int Ydim0 = dim[0], Ydim1 = dim[1], Ydim2 = dim[2], Ydim3 = dim[3];
	cout << "Ydim: " << Ydim0 << ',' << Ydim1 << ',' << Ydim2 << ',' << Ydim3 << endl;
	int strideY = (Ydim3 + 3) >> 2 << 2;
	int lengthYv = Ydim0 * Ydim1 * Ydim2 * strideY;

	//---------------------------------------------------
	char *dX = newDevCharVec(X, lengthXv);
	char *dY = newDevCharVec(lengthYv);

	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		__img_transpose4d(NULL,
			dX, dY,
			dim1, dim2, dim3,
			Ydim1, Ydim2, Ydim3,
			dimIndex1, dimIndex2,
			strideX, strideY,
			length);
	}
	cudaError_t error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = (end - start);
	float time = 1.0f * div / nIter;
	int data_size = length * 2 * sizeof(char);
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Size = " << (1.0 * data_size / 1024 / 1024) << ", "
		<< "Time = " << time << " mesc, "
		<< "Speed = " << speed << " GB/s" << endl;

	error = cudaFree(dX); printError(error);
	error = cudaFreeHost(dX);

	delete X;
}


void test()
{
	//transpose------------------------------------
	//int N = 32, IH = 32, IW = 32, IC = 32;
	//int N = 32, IH = 32, IW = 32, IC = 8;
	int N = 32, IH = 32, IW = 32, IC = 8;

	transpose_testCorrect(N, IH, IW, IC, 2, 3);
	//transpose_testSpeed(1000, N, IH*4, IW*4, IC, 2, 3);
	transpose_testSpeed(1000, N, IH*4, IW * 4, IC, 2, 3);


	//extract 3 channels---------------------------
	/*int N = 32, IH = 64, IW = 64, IC = 16;
	int c0 = 3, c1 = 5, c2 = 11;
	
	for(int ih=4; ih<=32; ih++)
	for(int iw=4; iw<=112; iw++)
	for(int ic=12; ic<=32; ic++) 
		e3c_testCorrect(4,ih, iw, ic, c0, c1, c2);

	e3c_testCorrect(N, IH, IW, IC, c0, c1, c2);
	e3c_testSpeed(N*16, IH, IW, IC, c0, c1, c2);*/


	//elementwise functions------------------------
	//for (int h = 4; h <= 20; h += 2)
	//	for (int w = 4; w <= 32; w++)
	//		testCorrect(h, w);

	//for (int h = 14; h <= 120; h += 2)
	//	for (int w = 14; w <= 132; w++)
	//		testCorrect(h, w);

	////int height = 72, width = 35;
	//int height = 1024, width = 1023;
	//testCorrect(height, width);
	//testSpeed(height*4, width*2);
	
	//int height = 32 * 128 * 128, width = 31;
	//row_reduce_testCorrect(height, width);
	//row_reduce_testSpeed(height*4, 64);

	//affine functions------------------------------
	//int N = 128, C = 32;
	//for(int ih=4; ih<32; ih++)
	//for(int iw=4; iw<32; iw++)
	//	affine_testCorrect(N, ih, iw, C);

	//int N = 128, IH = 32, IW = 32, C = 32;
	//affine_testCorrect(N, IH, IW, C);
	//affine_testSpeed(N, IH, IW, C);


	//int N = 128, IH = 32, IW = 32, OH = 64, OW = 64, C = 32;
	//int N = 1, IH = 674, IW = 1200, OH = 100, OW = 100, C = 4;
	//resize_testCorrect(N, IH, IW, OH, OW, C);
	//resize_testSpeed(N, IH, IW, OH, OW, C);

	//int N = 128;
	//int IH = 32, IW = 32, IC = 32;
	//int OH = 36, OW = 36, OC = 36;
	//int ph0 = 3, pw0 = 3, pc0 = 0;

	/*int IH = 2, IW = 2, IC = 4;
	int OH = 4, OW = 4, OC = 4;
	int ph0 = 1, pw0 = 1, pc0 = 0;*/

	//pad_testCorrect(IH, IW, IC, OH, OW, OC, N, ph0, pw0, pc0);
	//pad_testSpeed(IH, IW, IC, OH, OW, OC, N, ph0, pw0, pc0);

	//pad_testCorrect(OH, OW, OC, IH, IW, IC, N, ph0, pw0, pc0);
	//pad_testSpeed(OH, OW, OC, IH, IW, IC, N, ph0, pw0, pc0);
}

main()
{
	test();
}

#endif//complie-area>>>>------------------------------------------------------------

#endif