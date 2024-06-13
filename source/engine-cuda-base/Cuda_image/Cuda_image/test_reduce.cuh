#ifndef REDUCE_CPU
#define REDUCE_CPU

void img_row_linear(
	float alpha, const char *X, float beta,
	char *Y,
	int height, int width, int stride)
{
	unsigned char* uX = (unsigned char*)X;
	for (int i = 0; i < height; i++)
	{
		float fy = 0;
		for (int j = 0; j < width; j++) {
			fy += alpha * uX[j] + beta;
		}
		Y[i] = PIXEL_CLIP(fy);
		uX += stride;
	}
}


#endif

void row_reduce_testCorrect(int height, int width)
{
	int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
	int length = height * width;

	printf("test correct:\n");
	printf("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
	printf("(lengthv, length) = (%d, %d)\n", lengthv, length);

	char *X = newRandomCharVec(lengthv, width, stride);
	char *dX = newDevCharVec(X, lengthv);

	float alpha = (rand() % 1000) / 1000.0f;
	float beta = (rand() % 1000) / 1000.0f;
	float C = next_float(0, 1);

	//CPU--------------------------------------------------
	char *Y1 = newCharVec(height);

	img_row_linear(alpha, X, beta, Y1, height, width, stride);

	cout << "CPU: "; println(Y1, 10);

	//GPU--------------------------------------------------
	cudaError_t error;

	char *dY = newDevCharVec(height);

	//__img_row_linear2D(NULL, dX, alpha, beta, height, stride, dY, width, stride);

	error = cudaDeviceSynchronize(); printError(error);

	char *Y2 = new char[height];
	error = cudaMemcpy(Y2, dY, sizeof(char)*height, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: "; println(Y2, 10);

	//compare----------------------------------------------
	float sp = samePercent(Y1, Y2, height);
	cout << "sp: " << sp << endl;

	float zp0 = zeroPercent(Y1, height);
	float zp1 = zeroPercent(Y2, height);
	cout << "zp0: " << zp0 << endl;
	cout << "zp1: " << zp1 << endl;

	//clear------------------------------------------------
	error = cudaFree(dX); printError(error);
	error = cudaFree(dY); printError(error);

	delete X;
	delete Y1;
	delete Y2;

	if (sp < 0.99f) exit(2);
}

void row_reduce_testSpeed(int height, int width)
{
	int stride = (width + 3) >> 2 << 2;
	int length = height * width;
	int lengthv = height * stride;

	printf("\ntest speed:\n");
	printf("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
	printf("(lengthv, length) = (%d, %d)\n", lengthv, length);

	cudaError_t error;

	char *X = newRandomCharVec(lengthv, width, stride);
	char *dX = newDevCharVec(X, lengthv);

	char *X2 = newRandomCharVec(height);
	char *dX2 = newDevCharVec(X2, height);

	char *dY = newDevCharVec(height);

	int nIter = 1000;
	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		//__img_row_linear2D(NULL, dX, 1, 1, height, stride, dY, width, stride);
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = end - start;
	float time = 1.0f * div / nIter;

	int data_size = (lengthv + height) * sizeof(char);
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Size = " << 1.0f * data_size / 1024 / 1024 << ", "
		<< "Time = " << time << " mesc, "
		<< "Speed = " << speed << " GB/s" << endl;

	delete X;
}
