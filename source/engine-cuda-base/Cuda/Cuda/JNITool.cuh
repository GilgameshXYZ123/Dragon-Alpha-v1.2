#pragma once

#ifndef JNI_TOOL_H
#define JNI_TOOL_H

#define printError(error) \
	{if(error!=cudaSuccess) printf("error: %d  -  %s\n", error, cudaGetErrorName(error));}

#define handleError(error) \
	{if(error!=cudaSuccess) throwCudaException(env, error);}

#define handleError(error) \
	{if(error!=cudaSuccess) throwCudaException(env, error);}

const char* CudaException_class = "z/dragon/engine/cuda/impl/CudaException";

//passed
JNIEXPORT void JNICALL throwCudaException(JNIEnv* env, 
	cudaError_t stat)
{
	jclass cls = env->FindClass(CudaException_class);
	jmethodID constructor = env->GetMethodID(cls, "<init>", "(I)V");
	jobject e = env->NewObject(cls, constructor, stat);
	env->Throw((jthrowable)e);
}

//passed
JNIEXPORT void JNICALL throwCudaException(JNIEnv* env, 
	cudaError_t stat, const char *msg)
{
	jclass cls = env->FindClass(CudaException_class);
	jmethodID constructor = env->GetMethodID(cls, "<init>", "(ILjava/lang/String;)V");
	jobject e = env->NewObject(cls, constructor, stat, env->NewStringUTF(msg));
	env->Throw((jthrowable)e);
}

JNIEXPORT void JNICALL throwException(JNIEnv* env,
	const char *msg)
{
	jclass cls = env->FindClass("java/lang/Exception");
	jmethodID constructor = env->GetMethodID(cls, "<int>", "(Ljava/lang/String;)V");
	jobject e = env->NewObject(cls, constructor, env->NewStringUTF(msg));
	env->Throw((jthrowable)e);
}

#endif
