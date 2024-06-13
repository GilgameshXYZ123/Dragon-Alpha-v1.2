#pragma once

#ifndef CUDA_CHAR_H
#define CUDA_CHAR_H

#include "Cuda_char_set1D.cuh"
#include "Cuda_char_set2D.cuh"

#ifndef CUDA_CHAR_SET_CONSTANT
#define CUDA_CHAR_SET_CONSTANT

//Method:    set1D_char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set1D_1char__JJII(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jint value, jint length)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *dX = (char*)(intptr_t)address;
	__char_set1D(stream, dX, value, length);//return syncer
}

//Method:    set2D_char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D_1char__JJIIII(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jint value,//2D: stride > width
	jint height, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *dX = (char*)(intptr_t)address;
	__char_set2D(stream, dX, value, height, width, stride);//return syncer
}

#endif


//=======[transfer data: GPU -> CPU(1D)]==============================
#ifndef CUDA_CHAR_GET_GPU2CPU_1D
#define CUDA_CHAR_GET_GPU2CPU_1D

//Method:    get1D_char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get1D_1char(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jint length)
{
	//CUDA to C
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char* dX = (char*)(intptr_t)address;
	char *X = new char[length];
	cudaError_t error = cudaMemcpyAsync(X, dX, length,
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	env->SetByteArrayRegion(value, 0, length, (jbyte*)X);
	delete[] X;
}

//Method:    get1D_v2_char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get1D_1v2_1char(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jlong buf_address,
	jint length)
{
	//CUDA to C
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *dX = (char*)(intptr_t)address;
	char *X = (char*)(intptr_t)buf_address;
	cudaError_t error = cudaMemcpyAsync(X, dX, length,
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	env->SetByteArrayRegion(value, 0, length, (jbyte*)X);
}

#endif


//=======[transfer data: CPU -> GPU(2D) with zero padding]============
#ifndef CUDA_CHAR_GET_GPU2CPU_2D
#define CUDA_CHAR_GET_GPU2CPU_2D

//Method:    get2D_char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get2D_1char(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jint height, jint width, jint stride)
{
	//Java to C: stride > width
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int lengthv = height * stride;
	char *dX = (char*)(intptr_t)address;
	char *X = new char[lengthv];

	//CUDA to C
	cudaError_t error = cudaMemcpyAsync(X, dX, lengthv,
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	if (width > 63)
	{
		char *tX = X;
		for (int i = 0, start = 0; i < height; i++) {
			env->SetByteArrayRegion(value, start, width, (jbyte*)tX);
			start += width; tX += stride;
		}
	}
	else
	{
		//compact: lengthv(height * stride) -> length(height * width)
		char *X1 = X + width, *X2 = X + stride;
		for (int i = 1; i < height; i++) {
			for (int j = 0; j < width; j++) X1[j] = X2[j];
			X1 += width, X2 += stride;
		}
		env->SetByteArrayRegion(value, 0, height*width, (jbyte*)X);
	}
	delete[] X;
}

//Method:    get2D_v2_char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get2D_1v2_1char(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jlong buf_address,
	jint height, jint width, jint stride)
{
	//Java to C: stride = (width + 3) >> 2 << 2
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int lengthv = height * stride;
	char *dX = (char*)(intptr_t)address;
	char *X = (char*)(intptr_t)buf_address;

	//CUDA to C
	cudaError_t error = cudaMemcpyAsync(X, dX, lengthv,
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	if (width > 63)
	{
		char *tX = X;
		for (int i = 0, start = 0; i < height; i++) {
			env->SetByteArrayRegion(value, start, width, (jbyte*)tX);
			start += width; tX += stride;
		}
	}
	else
	{
		//compact: lengthv(height * stride) -> length(height * width)
		char *X1 = X + width, *X2 = X + stride;
		for (int i = 1; i < height; i++) {
			for (int j = 0; j < width; j++) X1[j] = X2[j];
			X1 += width; X2 += stride;
		}

		env->SetByteArrayRegion(value, 0, height * width, (jbyte*)X);
	}
}

//Method:    get2D_v2_char_W3S4
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get2D_1v2_1char_1W3S4(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jlong buf_address,
	jint height)
{
	int length = height * 3;//width  = 3
	int lengthv = height * 4;//height = 4

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *X = (char*)(intptr_t)buf_address;
	char *dX = (char*)(intptr_t)address;

	cudaError_t error = cudaMemcpyAsync(X, dX, lengthv,
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//Java to C: compact: lengthv(height * stride) -> length(height * width)
	char *X1 = X + 3, *X2 = X + 4;
	for (int i = 1; i < height; i++) {
		*(char_2*)X1 = *(char_2*)X2;
		X1[2] = X2[2];
		X1 += 3; X2 += 4;
	}

	env->SetByteArrayRegion(value, 0, length, (jbyte*)X);
}

#endif


//======[transfer data: CPU -> GPU(1D)]===============================
#ifndef CUDA_CHAR_SET_CPU2GPU_1D
#define CUDA_CHAR_SET_CPU2GPU_1D

//Method:    set1D_char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set1D_1char(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jint length)
{
	//Java to C
	char *X = (char*)env->GetByteArrayElements(value, NULL);

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char* dX = (char*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, length,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	env->ReleaseByteArrayElements(value, (jbyte*)X, JNI_ABORT);
}

//Method:    set1D_v2_char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set1D_1v2_1char(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jlong buf_address,
	jint length)
{
	//Java to C
	char *X = (char*)(intptr_t)buf_address;
	env->GetByteArrayRegion(value, 0, length, (jbyte*)X);

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *dX = (char*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, length,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);
}

#endif


//======[transfer data: CPU -> GPU(2D) with zero padding]=============
#ifndef CUDA_CHAR_SET_CPU2GPU_2D
#define CUDA_CHAR_SET_CPU2GPU_2D

//Method:    set2D_char(normal version)
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D_1char__JJ_3BIII(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jint height, jint width, jint stride)
{
	//Java to C
	int length = height * width, lengthv = height * stride;
	char *X = new char[lengthv];
	env->GetByteArrayRegion(value, 0, length, (jbyte*)X);

	char *X1 = X + lengthv, *X2 = X + length;//go to the end of Tensor

	for (int i = 0; i < height; i++) {
		for (int j = width; j < stride; j++) *(--X1) = 0;
		for (int j = 0; j < width; j++) *(--X1) = *(--X2);
	}

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *dX = (char*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, lengthv,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	delete[] X;
}

//Method:    set2D_v2_char(fast version: with pinned-mem buf)
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D_1v2_1char(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jlong buf_address,
	jint height, jint width, jint stride)
{
	//Java to C
	int length = height * width, lengthv = height * stride;
	char* X = (char*)(intptr_t)buf_address;

	env->GetByteArrayRegion(value, 0, length, (jbyte*)X);

	char *X1 = X + lengthv, *X2 = X + length;//go to the end of Tensor
	for (int i = 0; i < height; i++) {
		for (int j = width; j < stride; j++) *(--X1) = 0;
		for (int j = 0; j < width; j++) *(--X1) = *(--X2);
	}

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *dX = (char*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, lengthv,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);
}

//Method:    set2D_v2_char_W3S4: optimized channel for JPEG pictures with 3 channels
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D_1v2_1char_1W3S4(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jlong buf1_address, jlong buf2_address,
	jint height)//width = 3, height = 4
{
	int length = height * 3;//width  = 3
	int lengthv = height * 4;//height = 4

	//Java to C
	char *X1 = (char*)(intptr_t)buf1_address;//[height,  width:3]
	char *X2 = (char*)(intptr_t)buf2_address;//[height, stride:4]

	env->GetByteArrayRegion(value, 0, length, (jbyte*)X1);
	__zeroPad_W3S4(X1, X2, height);//X1 -> X2

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *dX = (char*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X2, lengthv,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);
}

#endif

#endif