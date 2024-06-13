
#ifndef CPU_PAD_TRIM
#define CPU_PAD_TRIM

void img_pad_cpu(
	const char *X, int IH, int IW, int IC,
		  char *Y, int OH, int OW, int OC,
	int N, int ph0, int pw0, int pc0)
{
	for (int n = 0; n < N; n++)
		for (int ih = 0; ih < IH; ih++)
		for (int iw = 0; iw < IW; iw++)
		for (int ic = 0; ic < IC; ic++)
		{
			const int xindex = ((n*IH + ih)*IW + iw)*IC + ic;
			int oh = ih + ph0;
			int ow = iw + pw0;
			int oc = ic + pc0;
			const int yindex = ((n*OH + oh)*OW + ow)*OC + oc;
			Y[yindex] = X[xindex];
		}
}

void img_trim_cpu(
	const char *X, int IH, int IW, int IC,
		  char *Y, int OH, int OW, int OC,
	int N, int ph0, int pw0, int pc0)
{
	for (int n = 0; n < N; n++)
		for (int oh = 0; oh < OH; oh++)
		for (int ow = 0; ow < OW; ow++)
		for (int oc = 0; oc < OC; oc++)
		{
			const int yindex = ((n*OH + oh)*OW + ow)*OC + oc;
			int ih = oh + ph0;
			int iw = ow + pw0;
			int ic = oc + pc0;
			const int xindex = ((n*IH + ih)*IW + iw)*IC + ic;
			Y[yindex] = X[xindex];
		}
}

#endif

void pad_testCorrect(int IH, int IW, int IC, 
	int OH, int OW, int OC,
	int N, int ph0, int pw0, int pc0)
{
	int Xsize = N * IH*IW*IC;
	int Ysize = N * OH*OW*OC;

	printf("test correct affine:\n");
	printf("(N, IH, IW, C) = (%d, %d, %d, %d)\n", N, IH, IW, IC);
	printf("(N, OH, OW, C) = (%d, %d, %d, %d)\n", N, OH, OW, OC);
	printf("(Xsize, Ysize) = (%d, %d)\n", Xsize, Ysize);

	char *X = newRandomCharVec(Xsize);
	char *dX = newDevCharVec(X, Xsize);

	//CPU--------------------------------------------------
	char *Y1 = newCharVec(Ysize);

	//img_pad_cpu(X, IH, IW, IC, Y1, OH, OW, OC, N, ph0, pw0, pc0);
	img_trim_cpu(X, IH, IW, IC, Y1, OH, OW, OC, N, ph0, pw0, pc0);

	cout << "CPU: "; println(Y1, 10);

	//GPU--------------------------------------------------
	cudaError_t error;

	char *dY = newDevCharVec(Ysize);

	//__img_pad(NULL, dX, IH, IW, IC, dY, OH, OW, OC, N, ph0, pw0, pc0);
	__img_trim(NULL, dX, IH, IW, IC, dY, OH, OW, OC, N, ph0, pw0, pc0);

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

void pad_testSpeed(int IH, int IW, int IC,
	int OH, int OW, int OC,
	int N, int ph0, int pw0, int pc0)
{
	int Xsize = N * IH*IW*IC;
	int Ysize = N * OH*OW*OC;

	printf("test correct affine:\n");
	printf("(N, IH, IW, C) = (%d, %d, %d, %d)\n", N, IH, IW, IC);
	printf("(N, OH, OW, C) = (%d, %d, %d, %d)\n", N, OH, OW, OC);
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
		//__img_pad(NULL, dX, IH, IW, IC, dY, OH, OW, OC, N, ph0, pw0, pc0);
		__img_trim(NULL, dX, IH, IW, IC, dY, OH, OW, OC, N, ph0, pw0, pc0);
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = end - start;
	float time = 1.0f * div / nIter;
	//int data_size = (Xsize * 2) * sizeof(char);//trim
	int data_size = (Ysize * 2) * sizeof(char);//trim
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	float unit_size = 1.0f * data_size / 1024 / 1024;

	cout << "Size = " << unit_size << ", "
		<< "Time = " << time << " mesc, "
		<< "Speed = " << speed << " GB/s" << endl;

	delete X;
}

