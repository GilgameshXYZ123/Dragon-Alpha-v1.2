#include "frame.cuh"
#include "Cuda_conv3D.cuh"
#include "JNITool.cuh"
#include "micro.cuh"
#include "texture.cuh"
#include "conv3D.cuh"
using namespace std;
#include "test.cuh"


#ifdef COMPLIE//<<<<complie-area--------------------------------------------------

//Kernel Remode Functions
#ifndef JNI_CONV_3D_KERNEL_REMODE
#define JNI_CONV_3D_KERNEL_REMODE

//Method:    kernel_remode
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_kernel_1remode(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address, jlong dCW_address,
	jint FH, jint FW, jint OC, jint IC)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dW = (float*)(intptr_t)dW_address;
	float* dCW = (float*)(intptr_t)dCW_address;
	__kernel_remode(stream, dW, dCW, FH, FW, OC, IC);
}

#endif



//Implicit-GEMM: Common
#ifndef JNI_CONV_3D_GEMM_AREA
#define JNI_CONV_3D_GEMM_AREA

//Method:    conv3D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	__conv3D_Gemm(streams, index, length, 
		dX, IH, IW, 
		dW, FH, FW, 
		dY, OH, OW, 
		N, IC, OC, 
		sh, sw, ph, pw);
}

//Method:    conv3D_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	cudaTextureObject_t texX = floatTexture(dX, (N*IH*IW*IC), env);
	__conv3D_Gemm_tex(streams, index, length,
		texX, dX, IH, IW,
		dW, FH, FW,
		dY, OH, OW,
		N, IC, OC,
		sh, sw, ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
}

//Method:    conv3DV2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3DV2(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	if (!useTexture) env = NULL;
	__conv3D_GemmV2(env, streams, index, length,
		dX, IH, IW, (N*IH*IW*IC),//sizeX = N * IH * IW * IC
		dW, FH, FW,
		dY, OH, OW,
		IC, OC,
		sh, sw, ph, pw);
}

// Method:    conv3D_np
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1np(JNIEnv *env, jclass cls, 
	jlongArray streamArray, jint length, 
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jint FH, jint FW, 
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	__conv3D_Gemm_np(streams, index, length,
		dX, IH, IW,
		dW, FH, FW,
		dY, OH, OW,
		IC, OC,
		sh, sw);
}

//Method:    conv3D_W1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1W1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, 
	jlong dY_address, 
	jint N, jint IC, jint OC)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	__conv3D_W1(streams, index, length,
		dX, IH, IW, dW, dY, 
		N, IC, OC);
}

#endif



//Implicit-GEMM: Remode the kernel: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef JNI_CONV_3D_GEMMR_AREA
#define JNI_CONV_3D_GEMMR_AREA

#ifndef JNI_CONV_3D_GEMMR
#define JNI_CONV_3D_GEMMR

//Method:    conv3D_GemmR
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmR(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	__conv3D_GemmR(streams, index, length,
		dX, IH, IW,
		dW, dCW, FH, FW,
		dY, OH, OW,
		N, IC, OC,
		sh, sw, ph, pw);
}

//Method:    conv3D_GemmR_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmR_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int GN = GET_GN(OC), GM = GET_GM(N, OH, OW), GK = GET_GK(FH, FW, IC);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	for (int i = 0; i < n; i++) {
		conv3dGemmR(streams, index, length,
			dX, IH, IW,
			dW, dCW, FH, FW,
			dY, OH, OW,
			N, IC, OC,
			sh, sw, ph, pw,
			GN, GM_slice, GK,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		conv3dGemmR(streams, index, length,
			dX, IH, IW,
			dW, dCW, FH, FW,
			dY, OH, OW,
			N, IC, OC,
			sh, sw, ph, pw,
			GN, GMr, GK,
			0, j_index);
	}
}

#endif


#ifndef JNI_CONV_3D_GEMMR_TEXTURE
#define JNI_CONV_3D_GEMMR_TEXTURE

//Method:    conv3D_GemmR_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmR_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	cudaTextureObject_t texX = floatTexture(dX, (N*IH*IW*IC), env);
	__conv3D_GemmR_tex(streams, index, length,
		texX, dX, IH, IW,
		dW, dCW, FH, FW,
		dY, OH, OW,
		N, IC, OC,
		sh, sw, ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
}

//Method:    conv3D_GemmR_texture_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmR_1texture_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int GN = GET_GN(OC), GM = GET_GM(N, OH, OW), GK = GET_GK(FH, FW, IC);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;
	cudaTextureObject_t texX = floatTexture(dX, (N*IH*IW*IC), env);

	for (int i = 0; i < n; i++) {
		conv3dGemmR_texture(streams, index, length,
			texX, dX, IH, IW,
			dW, dCW, FH, FW,
			dY, OH, OW,
			N, IC, OC,
			sh, sw, ph, pw,
			GN, GM_slice, GK,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {
		conv3dGemmR_texture(streams, index, length,
			texX, dX, IH, IW,
			dW, dCW, FH, FW,
			dY, OH, OW,
			N, IC, OC,
			sh, sw, ph, pw,
			GN, GMr, GK,
			0, j_index);
	}
	cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
}

#endif


#ifndef JNI_CONV_3D_GEMMR_W1
#define JNI_CONV_3D_GEMMR_W1

//Method:    conv3D_GemmR_W1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmR_1W1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address,
	jlong dY_address,
	jint N, jint IC, jint OC)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float* dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	__conv3D_GemmR_W1(streams, index, length,
		dX, IH, IW, dW, dCW, dY,
		N, IC, OC);
}

//Method:    conv3D_GemmR_W1_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmR_1W1_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address,
	jlong dY_address,
	jint N, jint IC, jint OC,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float* dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int GN = GET_GN(OC), GM = GET_GM(N, IH, IW);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	for (int i = 0; i < n; i++) {
		conv3d_GemmR_W1(streams, index, length,
			dX, IH, IW,
			dW, dCW,
			dY, IC, OC,
			GN, GM_slice,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		conv3d_GemmR_W1(streams, index, length,
			dX, IH, IW,
			dW, dCW,
			dY, IC, OC,
			GN, GMr,
			0, j_index);
	}
}

#endif


#ifndef JNI_CONV_3D_GEMMV2R
#define JNI_CONV_3D_GEMMV2R

//Method:    conv3D_GemmV2R
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmV2R(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	if (!useTexture) env = NULL;
	__conv3D_GemmV2R(env, streams, index, length,
		dX, IH, IW, (N*IH*IW*IC),//sizeX = N * IH * IW * IC
		dW, dCW, FH, FW,
		dY, OH, OW,
		IC, OC,
		sh, sw, ph, pw);
}

#endif

#endif



//Implicit-GEMM: Remode the kernel: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef JNI_CONV_3D_IM2COL_WINOGRAD_AREA
#define JNI_CONV_3D_IM2COL_WINOGRAD_AREA

#ifndef JNI_CONV_3D_IM2COL_WINOGRAD_S8
#define JNI_CONV_3D_IM2COL_WINOGRAD_S8

//Method:    conv3D_Im2col_Winograd_s8_R_texture
JNIEXPORT jboolean JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1Im2col_1Winograd_1s8_1R_1texture(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX  = (float*)(intptr_t)dX_address;
	float *dW  = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY  = (float*)(intptr_t)dY_address;

	int index = 0;
	JNIEnv *tenv = (useTexture ? env : NULL);
	cudaTextureObject_t texX = floatTexture(dX, (N*IH*IW*IC), env);//[N, IH, IW, IC]
	bool flag = __conv3D_Im2col_Winograd_s8_tex(tenv, streams, index, length,
		dX, texX, IH, IW,
		dW, dCW, FH, FW,
		dY, OH, OW,
		N, IC, OC, 
		ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
	return flag;
}

//Method:    conv3D_Im2col_WinogradR_s8_texture_SGM
JNIEXPORT jboolean JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1Im2col_1WinogradR_1s8_1texture_1SGM(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX  = (float*)(intptr_t)dX_address;
	float *dW  = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY  = (float*)(intptr_t)dY_address;
	
	//split GM along the N axis will not affect the calculation of Im2col-Winograd and GemmR
	int N_slice = ((GM_slice / OW) >> 5 << 5) / OH; if (N_slice == 0) return false;//N * OH: 32 group
	int n = N / N_slice, Nr = N % N_slice;

	int index = 0; 
	JNIEnv *tenv = (useTexture ? env : NULL);
	int Xstride = N_slice * (IH * IW * IC);
	int Ystride = N_slice * (OH * OW * OC);

	for (int i = 0; i < n; i++) {
		cudaTextureObject_t texX = floatTexture(dX, Xstride, env);//[N_slice, IH, IW, IC]
		bool flag = __conv3D_Im2col_Winograd_s8_tex(tenv, streams, index, length,
			dX, texX, IH, IW,
			dW, dCW, FH, FW,
			dY, OH, OW,
			N_slice, IC, OC,
			ph, pw);
		cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
		if (!flag) return false;
		dX += Xstride; dY += Ystride;//N += N_slice
	}

	if (Nr) {//N % N_slice != 0
		if ((Nr * OH) > 31) {//(Nr * OH) >= 32
			cudaTextureObject_t texX = floatTexture(dX, (Nr*IH*IW*IC), env);//[Nr, IH, IW, IC]
			__conv3D_Im2col_Winograd_s8_tex(tenv, streams, index, length,
				dX, texX, IH, IW,
				dW, dCW, FH, FW,
				dY, OH, OW,
				Nr, IC, OC,
				ph, pw);
			cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
		}
		else if (useTexture) {
			index = 0;//save L2 cache: texX != X
			cudaTextureObject_t texX = floatTexture(dX, (Nr*IH*IW*IC), env);//[Nr, IH, IW, IC]
			__conv3D_GemmR_tex(streams, index, length,
				texX, dX, IH, IW,
				dW, dCW, FH, FW,
				dY, OH, OW,
				Nr, IC, OC,
				1, 1, ph, pw);
			cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
		}
		else {
			index = 0;//save L2 cache: texX != X
			__conv3D_GemmR(streams, index, length,
				dX, IH, IW,
				dW, dCW, FH, FW,
				dY, OH, OW,
				Nr, IC, OC,
				1, 1, ph, pw);
		}
	}

	return true;
}


#endif


#ifndef JNI_CONV_3D_IM2COL_WINOGRAD_S16
#define JNI_CONV_3D_IM2COL_WINOGRAD_S16

//Method:    conv3D_Im2col_Winograd_s16_R_texture
JNIEXPORT jboolean JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1Im2col_1Winograd_1s16_1R_1texture(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW, 
	jint N, jint IC, jint OC, 
	jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX  = (float*)(intptr_t)dX_address;
	float *dW  = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY  = (float*)(intptr_t)dY_address;

	int index = 0;
	JNIEnv *tenv = (useTexture ? env : NULL);
	cudaTextureObject_t texX = floatTexture(dX, (N*IH*IW*IC), env);
	bool flag = __conv3D_Im2col_Winograd_s16_tex(tenv, streams, index, length,
		dX, texX, IH, IW,
		dW, dCW, FH, FW,
		dY, OH, OW, 
		N, IC, OC,
		ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
	return flag;
}

//Method:    conv3D_Im2col_WinogradR_s16_texture_SGM
JNIEXPORT jboolean JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1Im2col_1WinogradR_1s16_1texture_1SGM(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW, 
	jint N, jint IC, jint OC,
	jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;

	//split GM along the N axis will not affect the calculation of Im2col-Winograd and GemmR
	int N_slice = ((GM_slice / OW) >> 5 << 5) / OH; if (N_slice == 0) return false;//N * OH: 32 group
	int n = N / N_slice, Nr = N % N_slice;
	
	int index = 0;
	JNIEnv *tenv = (useTexture ? env : NULL);
	int Xstride = N_slice * (IH * IW * IC);
	int Ystride = N_slice * (OH * OW * OC);

	for (int i = 0; i < n; i++) {
		cudaTextureObject_t texX = floatTexture(dX, Xstride, env);//[N_slice, IH, IW, IC]
		bool flag = __conv3D_Im2col_Winograd_s16_tex(tenv, streams, index, length,
			dX, texX, IH, IW,
			dW, dCW, FH, FW,
			dY, OH, OW,
			N_slice, IC, OC,
			ph, pw);
		cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
		if (!flag) return false;
		dX += Xstride; dY += Ystride;//N += N_slice
	}

	if (Nr) {//N % N_slice != 0
		if ((Nr * OH) > 31) {//(Nr * OH) >= 32
			cudaTextureObject_t texX = floatTexture(dX, (Nr*IH*IW*IC), env);//[Nr, IH, IW, IC]
			__conv3D_Im2col_Winograd_s16_tex(tenv, streams, index, length,
				dX, texX, IH, IW,
				dW, dCW, FH, FW,
				dY, OH, OW,
				Nr, IC, OC,
				ph, pw);
			cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
		}
		else if (useTexture) {
			cudaTextureObject_t texX = floatTexture(dX, (Nr*IH*IW*IC), env);//[Nr, IH, IW, IC]
			__conv3D_GemmR_tex(streams, index, length,
				texX, dX, IH, IW,
				dW, dCW, FH, FW,
				dY, OH, OW,
				Nr, IC, OC,
				1, 1, ph, pw);
			cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
		}
		else {
			__conv3D_GemmR(streams, index, length,
				dX, IH, IW,
				dW, dCW, FH, FW,
				dY, OH, OW,
				Nr, IC, OC,
				1, 1, ph, pw);
		}
	}

	return true;
}

#endif

#endif

#endif//complie-area>>>>------------------------------------------------------------
