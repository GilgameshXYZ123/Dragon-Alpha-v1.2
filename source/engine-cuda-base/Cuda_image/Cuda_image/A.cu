#include "frame.cuh"
#include "Cuda_image.cuh"
#include "JNITool.cuh"
#include "micro.cuh"
#include "func_elementwise.cuh"
#include "func_reduce.cuh"
#include "func_tensor_trick.cuh"
using namespace std;
#include "test.cuh"


#ifdef COMPILE//<<<<complie-area--------------------------------------------------

#ifndef LINEAR2D_PIXEL_2_FLOAT
#define LINEAR2D_PIXEL_2_FLOAT

//Method:    linear2D_pixel2float
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_linear2D_1pixel2float(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	float* dY = (float*)(intptr_t)dY_address;
	__linear2D_pixel2float(stream, alpha, dX, beta, dY, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    linear2D_float2pixel
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_linear2D_1float2pixel(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	float *dX = (float*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__linear2D_float2pixel(stream, alpha, dX, beta, dY, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_dualLinear2_div2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1dualLinear2_1div2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dX1_address,
	jlong dX2_address,
	jfloat alpha1, jfloat beta1, jfloat gamma1,
	jfloat alpha2, jfloat beta2, jfloat gamma2,
	jfloat C,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char *dX1 = (char*)(intptr_t)dX1_address;
	char *dX2 = (char*)(intptr_t)dX2_address;
	float* dY = (float*)(intptr_t)dY_address;
	__img_dualLinear2_div2D(stream, 
		dX, dX1, dX2,
		alpha1, beta1, gamma1,
		alpha2, beta2, gamma2, C,
		dY, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//img_dualLinear2_noramlize2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1dualLinear2_1noramlize2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dX1_address,
	jlong dX2_address, jint row_lengthv,
	jfloat alpha1, jfloat beta1, jfloat gamma1,
	jfloat alpha2, jfloat beta2, jfloat gamma2,
	jfloat C,
	jlong dY_address,
	jint lengthv, jint width, jint stride) 
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float* dY  = (float*)(intptr_t)dY_address;
	__img_dualLinear2_normalize2D_row(stream, 
		dX, dX1, dX2, row_lengthv,
		alpha1, beta1, gamma1,
		alpha2, beta2, gamma2, C,
		dY, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_dualLinear2_noramlize2D_center
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1dualLinear2_1noramlize2D_1center(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, 
	jlong dX1_address, jlong dX2_address,
	jfloat alpha1, jfloat beta1, jfloat gamma1, 
	jfloat alpha2, jfloat beta2, jfloat gamma2, 
	jfloat C, 
	jlong dY_address,
	jint dim0, jint dim1, jint dim2,
	jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float* dY = (float*)(intptr_t)dY_address;
	__img_dualLinear2_normalize2D_center(stream,
		dX, dX1, dX2,
		alpha1, beta1, gamma1,
		alpha2, beta2, gamma2, C,
		dY, dim0, dim1, dim2, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}


//Method:    img_linear_dual2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1linear2_1div2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dX1_address,
	jlong dX2_address, jint row_lengthv,
	jfloat alpha1, jfloat beta1, jfloat gamma1,
	jfloat alpha2, jfloat beta2, jfloat C,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char* dX = (char*)(intptr_t)dX_address;
	float* dX1 = (float*)(intptr_t)dX1_address;
	float* dX2 = (float*)(intptr_t)dX2_address;
	float* dY = (float*)(intptr_t)dY_address;
	__img_linear2_div2D_row(stream, dX, dX1, dX2, row_lengthv,
		alpha1, beta1, gamma1, alpha2, beta2, C,
		dY, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_linear2_div_field
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1linear2_1div2D_1field(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, 
	jlong dX1_address,
	jlong dX2_address, jint row_lengthv, 
	jfloat alpha1, jfloat beta1, jfloat gamma1,
	jfloat alpha2, jfloat beta2, jfloat C, 
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char* dX = (char*)(intptr_t)dX_address;
	float* dX1 = (float*)(intptr_t)dX1_address;
	float* dX2 = (float*)(intptr_t)dX2_address;
	float* dY = (float*)(intptr_t)dY_address;
	__img_linear2_div2D_field(stream, dX, dX1, dX2, row_lengthv,
		alpha1, beta1, gamma1, alpha2, beta2, C,
		dY, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

#endif



#ifndef ELEMENT_WISE
#define ELEMENT_WISE

//Method:    img_linear2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1linear2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_linear2D(stream, alpha, dX, beta, dY, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_linear_dual2D_field
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1linear_1dual2D_1field(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address,
	jlong dX2_address, jint row_lengthv,
	jfloat alpha, jfloat beta, jfloat gamma,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX1 = (char*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_linear_dual2D_field(stream, 
		dX1, dX2, row_lengthv, 
		alpha, beta, gamma, dY, 
		lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_threshold2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1threshold2D(JNIEnv *env, jclass cls,
	jlong stream_address, jlong dX_address,
	jfloat alpha, jfloat v, jbyte v1, jbyte v2,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_threshold2D(stream, dX, alpha, v, v1, v2, dY, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_quadratic2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1quadratic2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jfloat alpha, jfloat beta, jfloat gamma,
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_quadratic2D(stream, dX, alpha, beta, gamma, dY, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    linear_dual2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_linear_1dual2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address,
	jlong dX2_address, jint row_lengthv, 
	jfloat alpha, jfloat beta, jfloat gamma,
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX1 = (char*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_linear_dual2D_row(stream, 
		dX1, dX2, row_lengthv,
		alpha, beta, gamma, dY,
		lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_log2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1log2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat C, jfloat alpha, jlong dX_address, jfloat beta, 
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_log2D(stream, dX, dY, C, alpha, beta, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_exp2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1exp2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta, jfloat C,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_exp2D(stream, dX, dY, alpha, beta, C, lengthv, width, stride);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

#endif



#ifndef TENSOR_TRICK
#define TENSOR_TRICK

//Method:    img_pad
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1pad(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jint OH, jint OW, jint OC,
	jlong dX_address, jint IH, jint IW, jint IC,
	jint N, jint ph0, jint pw0, jint pc0)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_pad(stream, dX, IH, IW, IC, dY, OH, OW, OC, N, ph0, pw0, pc0);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_trim
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1trim(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jint OH, jint OW, jint OC,
	jlong dX_address, jint IH, jint IW, jint IC,
	jint N, jint ph0, jint pw0, jint pc0)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	return __img_trim(stream, dX, IH, IW, IC, dY, OH, OW, OC, N, ph0, pw0, pc0);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_transpose2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1transpose2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jlong dY_address,
	jint Xdim1, jint Ydim1,
	jint strideX, jint strideY, 
	jint length)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_transpose2d(stream, dX, dY, Xdim1, Ydim1, strideX, strideY, length);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_transpose3D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1transpose3D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address, jlong dY_address,
	jint Xdim1, jint Xdim2,
	jint Ydim1, jint Ydim2,
	jint dimIndex1, jint dimIndex2,
	jint strideX, jint strideY, 
	jint length)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_transpose3d(stream, 
		dX, dY, 
		Xdim1, Xdim2, 
		Ydim1, Ydim2,
		dimIndex1, dimIndex2,
		strideX, strideY, length);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_transpose4D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1transpose4D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jlong dY_address,
	jint Xdim1, jint Xdim2, jint Xdim3,
	jint Ydim1, jint Ydim2, jint Ydim3,
	jint dimIndex1, jint dimIndex2,
	jint strideX, jint strideY,
	jint length)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_transpose4d(stream, 
		dX, dY,
		Xdim1, Xdim2, Xdim3,
		Ydim1, Ydim2, Ydim3,
		dimIndex1, dimIndex2,
		strideX, strideY, length);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_resize
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1resize(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address, jint IH, jint IW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint C)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_resize(stream, dX, IH, IW, dY, OH, OW, N, C);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_affine
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1affine(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address, jint IH, jint IW,
	jlong dY_address, jint OH, jint OW, 
	jfloat r00, jfloat r01, jfloat r02, 
	jfloat r10, jfloat r11, jfloat r12,
	jint N, jint C)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_affine(stream, 
		dX, IH, IW, dY, OH, OW,
		r00, r01, r02, r10, r11, r12, N, C);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_gappedMemcpy2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1gappedMemcpy2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address, jint Xstart, jint strideX,
	jlong dY_address, jint Ystart, jint strideY,
	jint width, jint length)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_gappedMemcpy2D(stream, 
		dX, Xstart, strideX, 
		dY, Ystart, strideY, 
		width, length);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

//Method:    img_extract_3channels
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1image_img_1extract_13channels(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jint IC,
	jlong dY_address, jint c0, jint c1, jint c2,
	jint lengthv)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(stream_address);
	char *dX = (char*)(intptr_t)dX_address;
	char* dY = (char*)(intptr_t)dY_address;
	__img_extract_3channels(stream, dX, IC, dY, c0, c1, c2, lengthv);
	cudaError_t error = cudaGetLastError(); handleError(error);
}

#endif

#endif//complie-area>>>>------------------------------------------------------------