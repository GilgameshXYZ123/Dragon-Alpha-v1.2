/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class z_dragon_engine_cuda_impl_math_Cuda_random */

#ifndef _Included_z_dragon_engine_cuda_impl_math_Cuda_random
#define _Included_z_dragon_engine_cuda_impl_math_Cuda_random
#ifdef __cplusplus
extern "C" {
#endif
	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_random
	 * Method:    bernouli2D
	 * Signature: (JJIFFFIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_bernouli2D
	(JNIEnv *, jclass, jlong, jlong, jint, jfloat, jfloat, jfloat, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_random
	 * Method:    leakyRelu_bernouli_mul2D
	 * Signature: (JJJJFIFFFIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_leakyRelu_1bernouli_1mul2D
	(JNIEnv *, jclass, jlong, jlong, jlong, jlong, jfloat, jint, jfloat, jfloat, jfloat, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_random
	 * Method:    bernouli_mul2D
	 * Signature: (JJJJIFFFIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_bernouli_1mul2D
	(JNIEnv *, jclass, jlong, jlong, jlong, jlong, jint, jfloat, jfloat, jfloat, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_random
	 * Method:    uniform2D
	 * Signature: (JJIFFIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_uniform2D
	(JNIEnv *, jclass, jlong, jlong, jint, jfloat, jfloat, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_random
	 * Method:    sparse_uniform2D
	 * Signature: (JJIIFFFIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_sparse_1uniform2D
	(JNIEnv *, jclass, jlong, jlong, jint, jint, jfloat, jfloat, jfloat, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_random
	 * Method:    gaussian2D
	 * Signature: (JJIIFFIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_gaussian2D
	(JNIEnv *, jclass, jlong, jlong, jint, jint, jfloat, jfloat, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_random
	 * Method:    sparse_gaussian2D
	 * Signature: (JJIIIFFFIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_sparse_1gaussian2D
	(JNIEnv *, jclass, jlong, jlong, jint, jint, jint, jfloat, jfloat, jfloat, jint, jint, jint);

#ifdef __cplusplus
}
#endif
#endif
