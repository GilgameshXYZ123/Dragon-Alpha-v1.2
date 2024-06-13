/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class z_dragon_engine_cuda_impl_math_Cuda_image */

#ifndef _Included_z_dragon_engine_cuda_impl_math_Cuda_image
#define _Included_z_dragon_engine_cuda_impl_math_Cuda_image
#ifdef __cplusplus
extern "C" {
#endif
	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    linear2D_pixel2float
	 * Signature: (JFJFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_linear2D_1pixel2float
	(JNIEnv *, jclass, jlong, jfloat, jlong, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    linear2D_float2pixel
	 * Signature: (JFJFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_linear2D_1float2pixel
	(JNIEnv *, jclass, jlong, jfloat, jlong, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_dualLinear2_div2D
	 * Signature: (JJJJFFFFFFFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1dualLinear2_1div2D
	(JNIEnv *, jclass, jlong, jlong, jlong, jlong, jfloat, jfloat, jfloat, jfloat, jfloat, jfloat, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_linear2_div2D_field
	 * Signature: (JJJJIFFFFFFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1linear2_1div2D_1field
	(JNIEnv *, jclass, jlong, jlong, jlong, jlong, jint, jfloat, jfloat, jfloat, jfloat, jfloat, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_linear2_div2D_row
	 * Signature: (JJJJIFFFFFFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1linear2_1div2D_1row
	(JNIEnv *, jclass, jlong, jlong, jlong, jlong, jint, jfloat, jfloat, jfloat, jfloat, jfloat, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_linear2D
	 * Signature: (JFJFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1linear2D
	(JNIEnv *, jclass, jlong, jfloat, jlong, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_linear_dual2D_row
	 * Signature: (JJJIFFFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1linear_1dual2D_1row
	(JNIEnv *, jclass, jlong, jlong, jlong, jint, jfloat, jfloat, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_linear_dual2D_field
	 * Signature: (JJJIFFFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1linear_1dual2D_1field
	(JNIEnv *, jclass, jlong, jlong, jlong, jint, jfloat, jfloat, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_quadratic2D
	 * Signature: (JJFFFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1quadratic2D
	(JNIEnv *, jclass, jlong, jlong, jfloat, jfloat, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_log2D
	 * Signature: (JFFJFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1log2D
	(JNIEnv *, jclass, jlong, jfloat, jfloat, jlong, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_exp2D
	 * Signature: (JFJFFJIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1exp2D
	(JNIEnv *, jclass, jlong, jfloat, jlong, jfloat, jfloat, jlong, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_pad
	 * Signature: (JJIIIJIIIIIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1pad
	(JNIEnv *, jclass, jlong, jlong, jint, jint, jint, jlong, jint, jint, jint, jint, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_trim
	 * Signature: (JJIIIJIIIIIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1trim
	(JNIEnv *, jclass, jlong, jlong, jint, jint, jint, jlong, jint, jint, jint, jint, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_transpose2D
	 * Signature: (JJJIIIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1transpose2D
	(JNIEnv *, jclass, jlong, jlong, jlong, jint, jint, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_transpose3D
	 * Signature: (JJJIIIIIIIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1transpose3D
	(JNIEnv *, jclass, jlong, jlong, jlong, jint, jint, jint, jint, jint, jint, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_transpose4D
	 * Signature: (JJJIIIIIIIIIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1transpose4D
	(JNIEnv *, jclass, jlong, jlong, jlong, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_resize
	 * Signature: (JJIIJIIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1resize
	(JNIEnv *, jclass, jlong, jlong, jint, jint, jlong, jint, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_affine
	 * Signature: (JJIIJIIFFFFFFII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1affine
	(JNIEnv *, jclass, jlong, jlong, jint, jint, jlong, jint, jint, jfloat, jfloat, jfloat, jfloat, jfloat, jfloat, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_gappedMemcpy2D
	 * Signature: (JJIIJIIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1gappedMemcpy2D
	(JNIEnv *, jclass, jlong, jlong, jint, jint, jlong, jint, jint, jint, jint);

	/*
	 * Class:     z_dragon_engine_cuda_impl_math_Cuda_image
	 * Method:    img_extract_3channels
	 * Signature: (JJIJIIII)V
	 */
	JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1extract_13channels
	(JNIEnv *, jclass, jlong, jlong, jint, jlong, jint, jint, jint, jint);

#ifdef __cplusplus
}
#endif
#endif
