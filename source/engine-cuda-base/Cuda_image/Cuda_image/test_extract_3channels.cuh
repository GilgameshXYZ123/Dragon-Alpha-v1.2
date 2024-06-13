
#ifndef EXTRACT_3CHANNELS
#define EXTRACT_3CHANNELS

void extract_3channels_cpu(
	const char* X, int N, int IH, int IW, int IC,
	char* Y,
	int c0, int c1, int c2)
{
	for (int n = 0; n < N; n++)
	for (int ih = 0; ih < IH; ih++)
	for (int iw = 0; iw < IW; iw++)
	{
		int xindex = ((n*IH + ih)*IW + iw)*IC;
		char x0 = X[xindex + c0];
		char x1 = X[xindex + c1];
		char x2 = X[xindex + c2];

		int yindex = ((n*IH + ih)*IW + iw) << 2;
		Y[yindex    ] = x0;
		Y[yindex + 1] = x1;
		Y[yindex + 2] = x2;
	}
}

#endif

void e3c_testCorrect(int N, int IH, int IW, int IC, int c0, int c1, int c2)
{
	int Xsize = N * IH*IW * IC;
	int Ysize = N * IH*IW * 4;

	printf("test correct:\n");
	printf("(N, IH, IW, IC) = (%d, %d, %d, %d)\n", N, IH, IW, IC);
	printf("(Xsize, Ysize) = (%d, %d, %d, %d)\n", Xsize, Ysize);

	char *X = newRandomCharVec(Xsize);
	char *dX = newDevCharVec(X, Xsize);
	
	char *Y1 = newCharVec(Ysize);
	char *dY = newDevCharVec(Ysize);

	cudaError_t error;

	//CPU--------------------------------------------------
	extract_3channels_cpu(X, N, IH, IW, IC, Y1, c0, c1, c2);

	cout << "CPU: "; println(Y1, 10);

	//GPU--------------------------------------------------
	__img_extract_3channels(NULL, dX, IC, dY, c0, c1, c2, (N*IH*IW));

	error = cudaDeviceSynchronize(); printError(error);

	char *Y2 = new char[Ysize];
	error = cudaMemcpy(Y2, dY, sizeof(char)*Ysize, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: "; println(Y2, 10);

	//compare----------------------------------------------
	float sp = samePercent(Y1, Y2, Ysize);
	cout << "sp: " << sp << endl;

	float zp0 = zeroPercent(Y1, Ysize);
	float zp1 = zeroPercent(Y2, Ysize);
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

void e3c_testSpeed(int N, int IH, int IW, int IC, int c0, int c1, int c2)
{
	int Xsize = N * IH*IW * IC;
	int Ysize = N * IH*IW * 4;

	printf("test correct:\n");
	printf("(N, IH, IW, IC) = (%d, %d, %d, %d)\n", N, IH, IW, IC);
	printf("(Xsize, Ysize) = (%d, %d, %d, %d)\n", Xsize, Ysize);

	char *X = newRandomCharVec(Xsize);
	char *dX = newDevCharVec(X, Xsize);

	char *Y1 = newCharVec(Ysize);
	char *dY = newDevCharVec(Ysize);

	cudaError_t error;

	int nIter = 1000;
	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		__img_extract_3channels(NULL, dX, IC, dY, c0, c1, c2, (N*IH*IW));
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = end - start;
	float time = 1.0f * div / nIter;
	int data_size = (Ysize + Ysize) * sizeof(char);//trim
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	float unit_size = 1.0f * data_size / 1024 / 1024;

	cout << "Size = " << unit_size << ", "
		<< "Time = " << time << " mesc, "
		<< "Speed = " << speed << " GB/s" << endl;

	delete X;
}
