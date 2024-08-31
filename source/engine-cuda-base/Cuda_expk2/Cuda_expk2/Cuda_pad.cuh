#pragma once

#ifndef CUDA_PAD_H
#define CUDA_PAD_H

#include "Cuda_pad2D.cuh"
#include "Cuda_pad3D.cuh"
#include "Cuda_pad4D.cuh"

//Method:    pad2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_pad2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jint IN, jint IC, 
	jlong dY_address, jint ON, jint OC,
	jint pn0, jint pc0)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dY = (float*)(intptr_t)dY_address;
	__pad2d(stream, 
		dX, IN, IC, 
		dY, ON, OC,
		pn0, pc0);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    pad3D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_pad3D(JNIEnv *env, jclass cls, 
	jlong stream_address,
	jlong dX_address, jint IN, jint IW, jint IC,
	jlong dY_address, jint ON, jint OW, jint OC,
	jint pn0, jint pw0, jint pc0)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dY = (float*)(intptr_t)dY_address;
	__pad3d(stream,
		dX, IN, IW, IC,
		dY, ON, OW, OC,
		pn0, pw0, pc0);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    pad3D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_pad4D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jint IN, jint IH, jint IW, jint IC, 
	jlong dY_address, jint ON, jint OH, jint OW, jint OC, 
	jint pn0, jint ph0, jint pw0, jint pc0)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dY = (float*)(intptr_t)dY_address;
	__pad4d(stream,
		dX, IN, IH, IW, IC,
		dY, ON, OH, OW, OC, 
		pn0, ph0, pw0, pc0);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

#endif
