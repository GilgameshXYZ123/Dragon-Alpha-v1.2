#ifndef RESIZE_CPU
#define RESIZE_CPU

void img_resize_cpu(
	const char* X, int IH, int IW,
	char* Y, int OH, int OW,
	int N, int C)
{
	float fOH = OH, fOW = OW;
	int IH_m1 = IH - 1, IW_m1 = IW - 1;

	for(int n=0; n<N; n++)
	for (int oh = 0; oh < OH; oh++)
	for (int ow = 0; ow < OW; ow++)
	{
		int ih = lroundf(oh * IH / fOH);//find the nearset pixel
		int iw = lroundf(ow * IW / fOW);
		ih = IF_int((ih < IH_m1), ih, IH_m1);//ih <= IH_m1
		iw = IF_int((iw < IW_m1), iw, IW_m1);//iw <= IW_m1
		for (int c = 0; c < C; c++) {
			int yindex = ((n*OH + oh)*OW + ow)*C + c;
			int xindex = ((n*IH + ih)*IW + iw)*C + c;
			Y[yindex] = X[xindex];
		}
	}
}

#endif

void resize_testCorrect(int N, int IH, int IW, int OH, int OW, int C)
{
	int Xsize = N * IH*IW*C;
	int Ysize = N * OH*OW*C;

	printf("test correct affine:\n");
	printf("(N, IH, IW, C) = (%d, %d, %d, %d)\n", N, IH, IW, C);
	printf("(N, OH, OW, C) = (%d, %d, %d, %d)\n", N, OH, OW, C);
	printf("(Xsize, Ysize) = (%d, %d)\n", Xsize, Ysize);

	char *X = newRandomCharVec(Xsize);
	char *dX = newDevCharVec(X, Xsize);

	//CPU--------------------------------------------------
	char *Y1 = newCharVec(Ysize);

	img_resize_cpu(X, IH, IW, Y1, OH, OW, N, C);

	cout << "CPU: "; println(Y1, 10);

	//GPU--------------------------------------------------
	cudaError_t error;

	char *dY = newDevCharVec(Ysize);

	__img_resize(NULL, dX, IH, IW, dY, OH, OW, N, C);

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

void resize_testSpeed(int N, int IH, int IW, int OH, int OW, int C)
{
	int Xsize = N * IH*IW*C;
	int Ysize = N * OH*OW*C;

	printf("test correct affine:\n");
	printf("(N, IH, IW, C) = (%d, %d, %d, %d)\n", N, IH, IW, C);
	printf("(N, OH, OW, C) = (%d, %d, %d, %d)\n", N, OH, OW, C);
	printf("(Xsize, Ysize) = (%d, %d)\n", Xsize, Ysize);

	char *X = newRandomCharVec(Xsize);
	char *dX = newDevCharVec(X, Xsize);

	//GPU--------------------------------------------------
	cudaError_t error;

	char *dY = newDevCharVec(Ysize);

	int nIter = 1000;
	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		__img_resize(NULL, dX, IH, IW, dY, OH, OW, N, C);
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	error = cudaDeviceSynchronize(); printError(error);

	int div = end - start;
	float time = 1.0f * div / nIter;
	int data_size = (Ysize + Xsize) * sizeof(char);
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	float unit_size = 1.0f * data_size / 1024 / 1024;

	cout << "Size = " << unit_size << ", "
		<< "Time = " << time << " mesc, "
		<< "Speed = " << speed << "GB/s" << endl;

	delete X;
}
