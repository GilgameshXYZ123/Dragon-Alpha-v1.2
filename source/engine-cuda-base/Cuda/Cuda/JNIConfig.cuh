#pragma once

#ifndef JNI_CONFIG_H
#define JNI_CONFIG_H

static int JNI_VERSION = JNI_VERSION_1_8;

static JavaVM* JVM = NULL;
static JavaVMAttachArgs JVM_cudaStreamCallback_args;
static JNIEnv* JVM_cudaStreamCallback_env = NULL;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
	JVM = vm;

	JVM_cudaStreamCallback_args.version = JNI_VERSION;
	JVM_cudaStreamCallback_args.name = "CUDA_streamCallback";
	JVM_cudaStreamCallback_args.group = NULL;

	return JNI_VERSION;
}

#endif
