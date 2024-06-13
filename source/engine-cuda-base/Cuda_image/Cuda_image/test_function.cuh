#ifndef FUNCTION_CPU
#define FUNCTION_CPU

//=======================================================
void img_linear(
	float alpha, const char *X, float beta, 
	char *Y,
	int height, int width, int stride)
{
	unsigned char* uX = (unsigned char*)X;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
		{
			float y = alpha * uX[j] + beta;
			Y[j] = PIXEL_CLIP(y);
		}
		uX += stride; Y += stride;
	}
}

//=======================================================
void img_linear_dual_field(
	const char* X1, const float* X2, 
	float alpha, float beta, float gamma,
	char *Y,
	int height, int width, int stride)
{
	for (int i = 0; i < height; i++) {
		float x2 = X2[i];//x2[height]
		for (int j = 0; j < width; j++) {
			unsigned char x1 = X1[j];
			float y = alpha * x1 + beta * x2 + gamma;
			Y[j] = PIXEL_CLIP(y);
		}
		X1 += stride; Y += stride;
	}
}

void img_linear_dual_row(
	const char* X1, const float* X2,
	float alpha, float beta, float gamma,
	char *Y,
	int height, int width, int stride)
{
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			unsigned char x1 = X1[j];
			float x2 = X2[j];
			float y = alpha * x1 + beta * x2 + gamma;
			Y[j] = PIXEL_CLIP(y);
		}
		X1 += stride; Y += stride;
	}
}

//=======================================================
void img_linear2_div_field(
	const char* X,
	const float* X1,
	const float* X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float C,
	float *Y,
	int height, int width, int stride)
{
	for (int i = 0; i < height; i++) 
	{
		float x1 = X1[i];
		float x2 = X2[i];

		for (int j = 0; j < width; j++) 
		{
			unsigned char x = X[j];
			float y1 = (alpha1*x + beta1 * x1 + gamma1);
			float y2 = (alpha2*x2 + beta2);
			float y = y1 / y2 + C;
			Y[j] = y;
		}
		X += stride; Y += stride;
	}
}

void img_linear2_div_row(
	const char* X,
	const float* X1,
	const float* X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float C,
	float *Y,
	int height, int width, int stride)
{
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float x1 = X1[j];
			float x2 = X2[j];
			unsigned char x = X[j];
			float y1 = (alpha1*x + beta1 * x1 + gamma1);
			float y2 = (alpha2*x2 + beta2);
			float y = y1 / y2 + C;
			Y[j] = y;
		}
		X += stride; Y += stride;
	}
}


//=======================================================
void img_log(
	const char *X, 
	      char *Y,
	float C, float alpha, float beta,
	int height, int width, int stride)
{
	unsigned char* uX = (unsigned char*)X;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) 
		{
			float y = C * log(alpha * uX[j] + beta);
			Y[j] = PIXEL_CLIP(y);
		}
		uX += stride; Y += stride;
	}
}

void img_exp(
	const char *X,
	char *Y,
	float alpha, float beta, float C,
	int height, int width, int stride)
{
	unsigned char* uX = (unsigned char*)X;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
		{
			float y = exp(alpha * uX[j] + beta) + C;
			Y[j] = PIXEL_CLIP(y);
		}
		uX += stride; Y += stride;
	}
}

//=======================================================
void img_dualLinear2_divide(
	const char *X,
	const char *X1,
	const char *X2,
	float alpha1, float beta1, float gamma1,
	float alpha2, float beta2, float gamma2,
	float C,
	float *Y,
	int height, int width, int stride)
{
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
		{
			unsigned char x  =  X[j];
			unsigned char x1 = X1[j];
			unsigned char x2 = X2[j];

			float fy1 = alpha1 * x + beta1 * x1 + gamma1;
			float fy2 = alpha2 * x + beta2 * x2 + gamma2;
			Y[j] = (fy1 / fy2) + C;
		}
		X += stride; X1 += stride; X2 += stride;
		Y += stride;
	}
}


#endif

void testCorrect(int height, int width)
{
	int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
	int length = height * width;

	printf("test correct:\n");
	printf("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
	printf("(lengthv, length) = (%d, %d)\n", lengthv, length);

	char *X  = newRandomCharVec(lengthv, width, stride);
	char *X1 = newRandomCharVec(lengthv, width, stride);
	char *X2 = newRandomCharVec(lengthv, width, stride);

	char *dX  = newDevCharVec(X,  lengthv);
	char *dX1 = newDevCharVec(X1, lengthv);
	char *dX2 = newDevCharVec(X2, lengthv);

	float *X1_field = newRandomFloatVec(height), *dX1_field = newDevFloatVec(X1_field, height);
	float *X2_field = newRandomFloatVec(height), *dX2_field = newDevFloatVec(X2_field, height);

	float *X1_row = newRandomFloatVec(stride), *dX1_row = newDevFloatVec(X1_row, stride);
	float *X2_row = newRandomFloatVec(stride), *dX2_row = newDevFloatVec(X2_row, stride);

	float alpha  = next_float(0, 1), beta  = next_float(0, 1), gamma  = next_float(0, 1);
	float alpha1 = next_float(0, 1), beta1 = next_float(0, 1), gamma1 = next_float(0, 1);
	float C = next_float(0, 1);

	//char *Y1 = newCharVec(lengthv), *dY = newDevCharVec(lengthv);
	float *Y1 = newFloatVec(lengthv), *dY = newDevFloatVec(lengthv);
	cudaError_t error;

	//CPU--------------------------------------------------
	//img_linear(alpha, X, beta, Y1, height, width, stride);
	//img_log(X, Y1, C, alpha, beta, height, width, stride);
	//img_exp(X, Y1, alpha, beta, C, height, width, stride);
	//img_dualLinear2_divide(X, X1, X2, alpha, beta, gamma, alpha1, beta1, gamma1, C, Y1, height, width, stride);

	//img_linear_dual_field(X, X2_field, alpha, beta, C, Y1, height, width, stride);
	//img_linear_dual_row(X, X2_row, alpha, beta, C, Y1, height, width, stride);

	img_linear2_div_field(X, X1_field, X2_field, alpha, beta, gamma, alpha1, beta1, C, Y1, height, width, stride);
	//img_linear2_div_row(X, X1_row, X2_row, alpha, beta, gamma, alpha1, beta1, C, Y1, height, width, stride);

	cout << "CPU: "; println(Y1, 10);

	//GPU--------------------------------------------------
	//__img_linear2D(NULL, alpha, dX, beta, dY, lengthv, width, stride);
	//__img_log2D(NULL, dX, dY, C, alpha, beta, lengthv, width, stride);
	//__img_exp2D(NULL, dX, dY, alpha, beta, C, lengthv, width, stride);
	//__img_dualLinear2_divide(NULL, dX, dX1, dX2, alpha, beta, gamma, alpha1, beta1, gamma1, C, dY, lengthv, width, stride);

	//__img_linear_dual2D_field(NULL, dX, dX2_field, stride, alpha, beta, C, dY, lengthv, width, stride);
	//__img_linear_dual2D_row(NULL, dX, dX2_row, stride, alpha, beta, C, dY, lengthv, width, stride);

	__img_linear2_div2D_field(NULL, dX, dX1_field, dX2_field, stride, alpha, beta, gamma, alpha1, beta1, C, dY, lengthv, width, stride);
	//__img_linear2_div2D_row(NULL, dX, dX1_row, dX2_row, stride, alpha, beta, gamma, alpha1, beta1, C, dY, lengthv, width, stride);

	error = cudaDeviceSynchronize(); printError(error);

	//char *Y2 = new char[lengthv];
	//error = cudaMemcpy(Y2, dY, sizeof(char)*lengthv, cudaMemcpyDeviceToHost); printError(error);

	float *Y2 = new float[lengthv];
	error = cudaMemcpy(Y2, dY, sizeof(float)*lengthv, cudaMemcpyDeviceToHost); printError(error);
	cout << "GPU: "; println(Y2, 10);

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

	if (sp < 0.99f) exit(2);
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

	char *X = newRandomCharVec(lengthv, width, stride);
	char *X1 = newRandomCharVec(lengthv, width, stride);
	char *X2 = newRandomCharVec(lengthv, width, stride);

	char *dX = newDevCharVec(X, lengthv);
	char *dX1 = newDevCharVec(X1, lengthv);
	char *dX2 = newDevCharVec(X2, lengthv);

	float *X1_field = newRandomFloatVec(height), *dX1_field = newDevFloatVec(X1_field, height);
	float *X2_field = newRandomFloatVec(height), *dX2_field = newDevFloatVec(X2_field, height);
	
	float *X1_row = newRandomFloatVec(stride), *dX1_row = newDevFloatVec(X1_row, stride);
	float *X2_row = newRandomFloatVec(stride), *dX2_row = newDevFloatVec(X2_row, stride);

	//char *dY = newDevCharVec(lengthv);
	float *dY = newDevFloatVec(lengthv);

	int nIter = 1000;
	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		//__img_linear2D(NULL, 1.0f, dX, 0.0f, dY, lengthv, width, stride);
		//__img_log2D(NULL, dX, dY, 1.0f, 1.0f, 1.0f, lengthv, width, stride);
		//__img_exp2D(NULL, dX, dY, 1.0f, 1.0f, 1.0f, lengthv, width, stride);
		//__img_dualLinear2_divide(NULL, dX, dX1, dX2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, dY, lengthv, width, stride);

		//__img_linear_dual2D_field(NULL, dX, dX2_field, stride, 1.0f, 1.0f, 1.0f, dY, lengthv, width, stride);
		//__img_linear_dual2D_row(NULL, dX, dX2_row, stride, 1.0f, 1.0f, 1.0f, dY, lengthv, width, stride);

		//__img_linear2_div2D_field(NULL, dX, dX1_field, dX2_field, stride, 1, 1, 1, 1, 1, 1, dY, lengthv, width, stride);
		__img_linear2_div2D_row(NULL, dX, dX1_row, dX2_row, stride, 1, 1, 1, 1, 1, 1, dY, lengthv, width, stride);
	}
	error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = end - start;
	float time = 1.0f * div / nIter;
	
	//int data_size = (lengthv) * sizeof(char) * 2;
	//int data_size = (lengthv) * sizeof(char) * 4;
	//int data_size = (lengthv * sizeof(char)) * 3 + (lengthv * sizeof(float));
	//int data_size = (lengthv * sizeof(char) * 2) + (stride * sizeof(float));
	//int data_size = (lengthv * sizeof(char) * 2) + (stride * sizeof(float) * 2);
	int data_size = (lengthv * sizeof(char) * 1) + ((stride*2 + lengthv) * sizeof(float));

	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Size = " << 1.0f * data_size / 1024 / 1024 << ", "
		 << "Time = " << time << " mesc, "
		 << "Speed = " << speed << " GB/s" << endl;

	delete X;
}
