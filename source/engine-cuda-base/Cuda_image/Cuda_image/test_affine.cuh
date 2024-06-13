
#ifndef AFFINE_CPU
#define AFFINE_CPU

void revert(//find the reverse of M
	float m00, float m01, float m02,
	float m10, float m11, float m12,
	float &r00, float &r01, float &r02,
	float &r10, float &r11, float &r12)
{
	float det = m00 * m11 - m01 * m10;
	r00 =  m11 / det, r01 = -m01 / det, r02 = (m01*m12 - m11 * m02) / det;
	r10 = -m10 / det, r11 =  m00 / det, r12 = (m10*m02 - m00 * m12) / det;
}

void affine_output_size(int IH, int IW, int &OH, int &OW,
	float m00, float m01, float m02,
	float m10, float m11, float m12)
{
	//(0, 0) -> (oh0, ow0), (IH, IW) -> (oh3, ow3)
	float ow0 = m02, ow3 = m00 * IW + m01 * IH + m02;
	float oh0 = m12, oh3 = m10 * IW + m11 * IH + m12;

	//(0, IW) -> (oh1, ow1), (IH, 0) -> (oh2, ow2)
	float ow1 = m00 * IW + m02, ow2 = m01 * IH + m02;
	float oh1 = m10 * IW + m12, oh2 = m11 * IH + m12;

	float oh[4]{ oh0, oh1, oh2, oh3 };
	float ow[4]{ ow0, ow1, ow2, ow3 };

	OH = ceil(max(oh, 4));
	OW = ceil(max(ow, 4));
	cout << OH << ":" << OW << endl;
}

void img_affine_cpu(
	const char* X, int IH, int IW, 
	      char* Y, int OH, int OW,
	float r00, float r01, float r02,
	float r10, float r11, float r12,
	int N, int C)
{
	for(int n=0; n<N; n++)
		for (int oh = 0; oh < OH; oh++)//v = oh: OH -> y = ih
		for (int ow = 0; ow < OW; ow++)//u = ow: OW -> x = iw
		{
			//int iw = (int)(r00*ow + r01 * oh + r02);
			//int ih = (int)(r10*ow + r11 * oh + r12);
			int iw = lroundf(r00*ow + r01 * oh + r02);
			int ih = lroundf(r10*ow + r11 * oh + r12);

			bool in_range = (ih >= 0) && (ih < IH) && (iw >= 0) && (iw < IW);
			for (int c = 0; c < C; c++) {
				int yindex = ((n*OH + oh)*OW + ow)*C + c;
				int xindex = ((n*IH + ih)*IW + iw)*C + c;
				Y[yindex] = (in_range ? X[xindex] : 0);
			}
		}
}

#endif

void affine_testCorrect(int N, int IH, int IW, int C)
{
	float m00 = next_float(0, 1);
	float m01 = next_float(0, 1);
	float m02 = next_float(0, 1);
	float m10 = next_float(0, 1);
	float m11 = next_float(0, 1);
	float m12 = next_float(0, 1);

	int OH, OW;
	affine_output_size(IH, IW, OH, OW,
		m00, m01, m02,
		m10, m11, m12);

	float r00, r01, r02, r10, r11, r12;
	revert(//find the reverse of M
		m00, m01, m02, m10, m11, m12,
		r00, r01, r02, r10, r11, r12);

	int Xsize = N * IH*IW*C;
	int Ysize = N * OH*OW*C;
	
	printf("test correct affine:\n");
	printf("(N, IH, IW, C) = (%d, %d, %d, %d)\n", N, IH, IW, C);
	printf("(N, OH, OW, C) = (%d, %d, %d, %d)\n", N, OH, OW, C);
	printf("(Xsize, Ysize) = (%d, %d)\n", Xsize, Ysize);
	printf("(%f, %f, %f)\n", r00, r01, r02);
	printf("(%f, %f, %f)\n", r10, r11, r12);

	char *X = newRandomCharVec(Xsize);
	char *dX = newDevCharVec(X, Xsize);

	//CPU--------------------------------------------------
	char *Y1 = newCharVec(Ysize);

	img_affine_cpu(X, IH, IW, Y1, OH, OW, r00, r01, r02, r10, r11, r12, N, C);

	cout << "CPU: "; println(Y1, 10);

	//GPU--------------------------------------------------
	cudaError_t error;

	char *dY = newDevCharVec(Ysize);

	__img_affine(NULL, dX, IH, IW, dY, OH, OW, r00, r01, r02, r10, r11, r12, N, C);

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

void affine_testSpeed(int N, int IH, int IW, int C)
{
	float m00 = 1.1f, m01 = 0.3f, m02 = 10;
	float m10 = 0.5f, m11 = 0.9f, m12 = 10;

	int OH, OW;
	affine_output_size(IH, IW, OH, OW,
		m00, m01, m02,
		m10, m11, m12);

	float r00, r01, r02, r10, r11, r12;
	revert(//find the reverse of M
		m00, m01, m02, m10, m11, m12,
		r00, r01, r02, r10, r11, r12);

	int Xsize = N * IH*IW*C;
	int Ysize = N * OH*OW*C;

	printf("test correct affine:\n");
	printf("(N, IH, IW, C) = (%d, %d, %d, %d)\n", N, IH, IW, C);
	printf("(N, OH, OW, C) = (%d, %d, %d, %d)\n", N, OH, OW, C);
	printf("(Xsize, Ysize) = (%d, %d)\n", Xsize, Ysize);
	printf("(%f, %f, %f)\n", r00, r01, r02);
	printf("(%f, %f, %f)\n", r10, r11, r12);

	char *X = newRandomCharVec(Xsize);
	char *dX = newDevCharVec(X, Xsize);

	//GPU--------------------------------------------------
	cudaError_t error;

	char *dY = newDevCharVec(Ysize);

	int nIter = 1000;
	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		__img_affine(NULL, dX, IH, IW, dY, OH, OW, r00, r01, r02, r10, r11, r12, N, C);
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
