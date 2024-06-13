#pragma once

#ifndef TEST_H
#define TEST_H

#ifndef COMPLIE//<<<<complie-area--------------------------------------------------

#ifndef UTIL
#define UTIL

float *newFloatVec(int length)
{
	float *p = new float[length];
	memset(p, 0, sizeof(float)*length);
	return p;
}

float* newRandomFloatVec(int lengthv, int width, int stride)//0-256
{
	int height = lengthv / stride;
	float *p = new float[lengthv] , *tp = p;
	memset(p, 0, sizeof(float)*lengthv);
	for (int i = 0; i < height; i++) 
	{
		for (int j = 0; j < width; j++) tp[j] = (float)(rand() % 1000) / 1000;
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
float samePercent(float *a, float *b, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		//if (b[i] == 0) cout << "zero: " << i << endl;
		if (a[i] == b[i])
		{
			sum++; continue;
		}
		float dif = fabs(a[i] - b[i]);
		//float dif = fabs((a[i] - b[i]) / (a[i] + b[i]));
		if (dif < 1e-3) sum++;
		//if (a[i] < b[i]) cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
		//else cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
	}
	return 1.0f*sum / length;
}

float zeroPercent(float *a, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
		if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}

float sum(float* a, int length)
{
	float r = 0;
	for (int i = 0; i < length; i++) r += a[i];
	return r;
}

#endif


#ifndef FUNCTION_CPU
#define FUNCTION_CPU

void linear(float alpha, float *X, float beta, float *Y,
	int height, int width, int stride)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++) Y[j] = X[j] * alpha + beta;
		X += stride; Y += stride;
	}
}

void binomial_row(
	float *X1, 
	float *Xrow2, int Xrow2_lengthv,
	float k11, float k12, float k22,
	float k1, float k2, float C,
	float* Y,
	int lengthv, int width, int stride)
{
	int field_size = lengthv / Xrow2_lengthv;
	for (int i = 0; i < field_size; i++)
	{
		for (int j = 0; j < Xrow2_lengthv; j++){
			if ((j%stride) < width)
			{
				float x1 = X1[j], x2 = Xrow2[j];
				Y[j] = k11 * (x1*x1) + k12 * (x1*x2) + k22 * (x2*x2) + k1 * x1 + k2 * x2 + C;
			}
		}
		X1 += Xrow2_lengthv;
		Y += Xrow2_lengthv;
	}
}


#endif

void testCorrect(int height, int width)
{
	int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
	int length = height * width;

	int Xrow2_height = 2;
	int Xrow2_lengthv = Xrow2_height * stride;

	printf("test correct:\n");
	printf("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
	printf("(lengthv, length) = (%d, %d)\n", lengthv, length);

	float *X = newRandomFloatVec(lengthv, width, stride);
	float *dX = newDevFloatVec(X, lengthv);
	float *Xrow2 = newRandomFloatVec(Xrow2_lengthv, width, stride);
	float *dXrow2 = newDevFloatVec(Xrow2, Xrow2_lengthv);

	float alpha = (rand() % 1000) / 1000.0f;
	float beta = (rand() % 1000) / 1000.0f;

	//CPU--------------------------------------------------
	float *Y1 = newFloatVec(lengthv);

	linear(alpha, X, beta, Y1, height, width, stride);
	cout << "CPU: "; println(Y1, 10);

	//GPU--------------------------------------------------
	cudaError_t error;
	
	float *dY = newDevFloatVec(lengthv);

	__linear2D(NULL, alpha, dX, beta, dY, lengthv, width, stride);

	error = cudaDeviceSynchronize(); printError(error);

	float *Y2 = new float[lengthv];
	error = cudaMemcpy(Y2, dY, sizeof(float)*lengthv, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: " ; println(Y2, 10);

	//compare----------------------------------------------
	float sp = samePercent(Y1, Y2, lengthv);
	cout << "sp: " << sp << endl;

	float zp0 = zeroPercent(Y1, lengthv);
	float zp1 = zeroPercent(Y2, lengthv);
	cout << "zp0: " << zp0 << endl;
	cout << "zp1: " << zp1 << endl;

	//clear------------------------------------------------
	error = cudaFree(dX); printError(error);
	error = cudaFree(dY); printError(error);

	delete X;
	delete Y1;
	delete Y2;

	if (sp != 1.0f) exit(2);
}

void testSpeed(int height, int width)
{
	int stride = (width + 3) >> 2 << 2;
	int length = height * width;
	int lengthv = height * stride;

	printf("\ntest speed:\n");
	printf("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
	printf("(lengthv, length) = (%d, %d)\n", lengthv, length);

	cudaError_t error;

	int nIter = 500;
	float *X = newRandomFloatVec(lengthv, width, stride);
	float *dX = newDevFloatVec(X, lengthv);

	float *dY = newDevFloatVec(lengthv);

	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		__linear2D(NULL, 1.0f, dX, 1.0f, dY, lengthv, width, stride);
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = end - start;
	float time = 1.0f * div / nIter;
	int data_size = (lengthv) * sizeof(float) * 2;
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Time = " << time << " mesc, "
		 << "Speed = " << speed << "GB/s" << endl;

	delete X; 
}

void test()
{
	/*for (int h = 4; h <= 20; h++)
		for (int w = 4; w <= 32; w++)
			testCorrect(h, w);*/

	/*for (int h = 4; h <= 20; h += 2)
		for (int w = 4; w <= 32; w++)
			testCorrect(h, w);*/

	//int height = 1024, width = 1024;
	int height = 8192, width = 8192;

	//testCorrect(height, width);
	//testCorrect(height, width + 1);
	//testCorrect(height, width - 1);
	testSpeed(height, width);
}

main() { test(); }

#endif//complie-area>>>>-----------------------------------------------------------

#endif