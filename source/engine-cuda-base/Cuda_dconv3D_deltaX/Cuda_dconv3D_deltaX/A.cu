#include "frame.cuh"
#include "JNITool.cuh"
#include "Cuda_dconv3D_deltaX.cuh"
#include "micro.cuh"
#include "texture.cuh"
#include "dconv3D_dX.cuh"
using namespace std;
#include "test.cuh"


#ifdef COMPLIE//<<<<complie-area--------------------------------------------------

//zero_padding: Y[N, OH, OW, OC] -> Y[N, OHp, OWp, OC]
#ifndef JNI_DECONV_3D_DELTAX_ZERO_PADDING
#define JNI_DECONV_3D_DELTAX_ZERO_PADDING

#ifndef JNI_DECONV_3D_DELTAX_S1
#define JNI_DECONV_3D_DELTAX_S1

//Method:    dconv3D_deltaX_s1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1s1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	
	int index = 0;
	__dconv3D_deltaX_ZeroPadding_s1(streams, index, length,
		d_deltaY, OH, OW,
		dW, FH, FW,
		d_deltaX, IH, IW,
		N, IC, OC,
		ph, pw);
}

//Method:    dconv3D_deltaX_s1_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1s1_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	int GN = GET_GN_ZeroPadding(IC);
	int GM = GET_GM_ZeroPadding(N, IH, IW);
	int GK = GET_GK_ZeroPadding(OC, FH, FW);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	for (int i = 0; i < n; i++) {
		dconv3d_deltaX_ZeroPadding_s1(streams, index, length,
			d_deltaY, OH, OW,
			dW, FH, FW,
			d_deltaX, IH, IW,
			N, IC, OC,
			ph, pw,
			GN, GM_slice, GK,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		dconv3d_deltaX_ZeroPadding_s1(streams, index, length,
			d_deltaY, OH, OW,
			dW, FH, FW,
			d_deltaX, IH, IW,
			N, IC, OC,
			ph, pw,
			GN, GMr, GK,
			0, j_index);
	}
}

#endif


#ifndef JNI_DECONV_3D_DELTAX_S1_TEXTURE
#define JNI_DECONV_3D_DELTAX_S1_TEXTURE

//Method:    dconv3D_deltaX_s1_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1s1_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	
	int index = 0;
	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OC*OH*OW), env);
	__dconv3D_deltaX_ZeroPadding_s1_tex(streams, index, length,
		texDy, d_deltaY, OH, OW,
		dW, FH, FW,
		d_deltaX, IH, IW,
		N, IC, OC,
		ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
}

//Method:    dconv3D_deltaX_s1_texture_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1s1_1texture_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	int GN = GET_GN_ZeroPadding(IC);
	int GM = GET_GM_ZeroPadding(N, IH, IW);
	int GK = GET_GK_ZeroPadding(OC, FH, FW);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OC*OH*OW), env);
	for (int i = 0; i < n; i++) {
		dconv3d_deltaX_ZeroPadding_s1_texture(streams, index, length,
			texDy, d_deltaY, OH, OW,
			dW, FH, FW,
			d_deltaX, IH, IW,
			N, IC, OC,
			ph, pw,
			GN, GM_slice, GK,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		dconv3d_deltaX_ZeroPadding_s1_texture(streams, index, length,
			texDy, d_deltaY, OH, OW,
			dW, FH, FW,
			d_deltaX, IH, IW,
			N, IC, OC,
			ph, pw,
			GN, GMr, GK,
			0, j_index);
	}
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
}

#endif


#ifndef JNI_DECONV_3D_DELTAX_W1
#define JNI_DECONV_3D_DELTAX_W1

//Method:    dconv3D_deltaX_W1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1W1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address,
	jlong dW_address,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	int index = 0;
	__dconv3D_deltaX_W1(streams, index, length,
		d_deltaY,
		dW, 
		d_deltaX, IH, IW, 
		N, IC, OC);
}

//Method:    dconv3D_deltaX_W1_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1W1_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address,
	jlong dW_address,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	int GN = GET_GN_ZeroPadding(IC);
	int GM = GET_GM_ZeroPadding(N, IH, IW);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	for (int i = 0; i < n; i++) {
		dconv3d_deltaX_W1(streams, index, length,
			d_deltaY, dW, d_deltaX,
			IC, OC,
			GN, GM_slice,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		dconv3d_deltaX_W1(streams, index, length,
			d_deltaY, dW, d_deltaX,
			IC, OC,
			GN, GMr,
			0, j_index);
	}
}

#endif


#ifndef JNI_DECONV_3D_DELTAX_V2_S1
#define JNI_DECONV_3D_DELTAX_V2_S1

//Method:    dconv3D_deltaX_V2_s1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1V2_1s1(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	int index = 0;
	if (!useTexture) env = NULL;//useTexture = false, env = null
	__dconv3D_deltaX_ZeroPaddingV2_s1(env, streams, index, length,
		d_deltaY, OH, OW, (N*OH*OW*OC),//sizeY = N*OH*OW*OC
		dW, FH, FW,
		d_deltaX, IH, IW,
		N, IC, OC,
		ph, pw);
}

#endif

#endif



//kernel split: W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
#ifndef JNI_DECONV_3D_DELTAX_KERNEL_SPLIT
#define JNI_DECONV_3D_DELTAX_KERNEL_SPLIT

#ifndef JNI_DECONV_3D_DELTAX_KS_REMODE
#define JNI_DECONV_3D_DELTAX_KS_REMODE

//Method:    ks_remode
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_ks_1remode(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address, jint FH, jint FW,
	jlong dCW_address, jint CFH, jint CFW,
	jint OC, jint IC, jint sh, jint sw)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dW = (float*)(intptr_t)dW_address;
	float* dCW = (float*)(intptr_t)dCW_address;
	__ks_remode(stream, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);
}

//Method:    ks_remodev2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_ks_1remodev2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address, jint FH, jint FW,
	jlong dCW_address, jint CFH, jint CFW, 
	jint OC, jint IC, jint sh, jint sw)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dW = (float*)(intptr_t)dW_address;
	float* dCW = (float*)(intptr_t)dCW_address;
	__ks_remodev2(stream, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);
}

#endif


#ifndef JNI_DECONV_3D_DELTAX_KS
#define JNI_DECONV_3D_DELTAX_KS

//Method:    dconv3D_deltaX_kernelSplit
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1kernelSplit(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	
	KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
	int index = 0;
	__dconv3D_deltaX_ksR(streams, index, length,
		d_deltaY, OH, OW,
		dCW, FH, FW, CWstride,
		d_deltaX, IH, IW, 
		N, IC, OC,
		sh, sw, ph, pw);
}

//Method:    dconv3D_deltaX_kernelSplit_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1kernelSplit_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	for (int i = 0; i < n; i++) {
		dconv3D_deltaX_ksR(streams, index, length,
			d_deltaY, OH, OW,
			dCW, FH, FW, CWstride,
			d_deltaX, IH, IW,
			N, IC, OC,
			sh, sw, ph, pw,
			GN, GM_slice,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		dconv3D_deltaX_ksR(streams, index, length,
			d_deltaY, OH, OW,
			dCW, FH, FW, CWstride,
			d_deltaX, IH, IW,
			N, IC, OC,
			sh, sw, ph, pw,
			GN, GMr,
			0, j_index);
	}
}

#endif


#ifndef JNI_DECONV_3D_DELTAX_KS_IMSR
#define JNI_DECONV_3D_DELTAX_KS_IMSR

//Method:    dconv3D_deltaX_ksImsR
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksImsR(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
	int index = 0;
	__dconv3D_deltaX_ksImsR(streams, index, length,
		d_deltaY, OH, OW,
		dCW, FH, FW, CWstride,
		d_deltaX, IH_slice, IW_slice,
		IC, OC,
		sh, sw, ph, pw);
}

//Method:    dconv3D_deltaX_ksImsR_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksImsR_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	for (int i = 0; i < n; i++) {
		dconv3D_deltaX_ksImsR(streams, index, length,
			d_deltaY, OH, OW,
			dCW, FH, FW, CWstride,
			d_deltaX, IH_slice, IW_slice,
			IC, OC,
			sh, sw, ph, pw,
			GN, GM_slice,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		dconv3D_deltaX_ksImsR(streams, index, length,
			d_deltaY, OH, OW,
			dCW, FH, FW, CWstride,
			d_deltaX, IH_slice, IW_slice,
			IC, OC,
			sh, sw, ph, pw,
			GN, GMr,
			0, j_index);
	}
}

#endif


#ifndef JNI_DECONV_3D_DELTAX_KS_IMSR_TEXTURE
#define JNI_DECONV_3D_DELTAX_KS_IMSR_TEXTURE

//Method:    dconv3D_deltaX_ksImsR_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksImsR_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
	int index = 0;
	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OH*OW*OC), env);
	__dconv3D_deltaX_ksImsR_tex(streams, index, length,
		texDy, d_deltaY, OH, OW,
		dCW, FH, FW, CWstride,
		d_deltaX, IH_slice, IW_slice,
		IC, OC, 
		sh, sw, ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
}

//Method:    dconv3D_deltaX_ksImsR_texture_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksImsR_1texture_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OH*OW*OC), env);
	for (int i = 0; i < n; i++) {
		dconv3D_deltaX_ksImsR_texture(streams, index, length,
			texDy, d_deltaY, OH, OW,
			dCW, FH, FW, CWstride,
			d_deltaX, IH_slice, IW_slice,
			IC, OC,
			sh, sw, ph, pw,
			GN, GM_slice,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		dconv3D_deltaX_ksImsR_texture(streams, index, length,
			texDy, d_deltaY, OH, OW,
			dCW, FH, FW, CWstride,
			d_deltaX, IH_slice, IW_slice,
			IC, OC,
			sh, sw, ph, pw,
			GN, GMr,
			0, j_index);
	}
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
}

#endif


#ifndef JNI_DECONV_3D_DELTAX_KS_IMS2R
#define JNI_DECONV_3D_DELTAX_KS_IMS2R

//Method:    dconv3D_deltaX_ksIms2R
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksIms2R(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims2_init(N, IH, IW, FH, FW, OC, IC);
	int index = 0;
	__dconv3D_deltaX_ksIms2R(streams, index, length,
		d_deltaY, OH, OW,
		dCW, FH, FW, CWstride,
		d_deltaX, IH_slice, IW_slice,
		N, IC, OC,
		ph, pw);
}

//Method:    dconv3D_deltaX_ksIms2R_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksIms2R_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims2_init(N, IH, IW, FH, FW, OC, IC);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	for (int i = 0; i < n; i++) {
		dconv3D_deltaX_ksIms2R(streams, index, length,
			d_deltaY, OH, OW,
			dCW, FH, FW, CWstride,
			d_deltaX, IH_slice, IW_slice,
			N, IC, OC,
			ph, pw,
			GN, GM_slice,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		dconv3D_deltaX_ksIms2R(streams, index, length,
			d_deltaY, OH, OW,
			dCW, FH, FW, CWstride,
			d_deltaX, IH_slice, IW_slice,
			N, IC, OC,
			ph, pw,
			GN, GMr,
			0, j_index);
	}
}

#endif


#ifndef JNI_DECONV_3D_DELTAX_KS_IMS2R_TEXTURE
#define JNI_DECONV_3D_DELTAX_KS_IMS2R_TEXTURE

//Method:    dconv3D_deltaX_ksIms2R_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksIms2R_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims2_init(N, IH, IW, FH, FW, OC, IC);
	int index = 0;
	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OH*OW*OC), env);
	__dconv3D_deltaX_ksIms2R_tex(streams, index, length,
		texDy, d_deltaY, OH, OW,
		dCW, FH, FW, CWstride,
		d_deltaX, IH_slice, IW_slice,
		IC, OC,
		ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
}

//Method:    dconv3D_deltaX_ksIms2R_texture_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksIms2R_1texture_1SGM(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims2_init(N, IH, IW, FH, FW, OC, IC);
	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OH*OW*OC), env);
	for (int i = 0; i < n; i++) {
		dconv3D_deltaX_ksIms2R_texture(streams, index, length,
			texDy, d_deltaY, OH, OW,
			dCW, FH, FW, CWstride,
			d_deltaX, IH_slice, IW_slice,
			N, IC, OC,
			ph, pw,
			GN, GM_slice,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		dconv3D_deltaX_ksIms2R_texture(streams, index, length,
			texDy, d_deltaY, OH, OW,
			dCW, FH, FW, CWstride,
			d_deltaX, IH_slice, IW_slice,
			N, IC, OC,
			ph, pw,
			GN, GMr,
			0, j_index);
	}
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
}

#endif


#ifndef JNI_DECONV_3D_DELTAX_KSV2_IMS2R
#define JNI_DECONV_3D_DELTAX_KSV2_IMS2R

//Method:    dconv3D_deltaX_ksV2_Ims2R
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksV2_1Ims2R(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC, 
	jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_KernelSplit]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	V2_Ims2_init(N, IH, IW, FH, FW, OC, IC);
	int index = 0;
	__dconv3D_deltaX_ksV2_Ims2R(env, streams, index, length,
		d_deltaY, OH, OW, (N*OH*OW*OC),//sizeY = N*OH*OW*OC
		dCW, FH, FW, CWstride,
		d_deltaX, IH_slice, IW_slice,
		IC, OC, ph, pw);
}

#endif

#endif



//cross add: reversed forward porpagation
#ifndef JNI_DECONV_3D_DELTAX_CROSS_ADD
#define JNI_DECONV_3D_DELTAX_CROSS_ADD

//Method:    dconv3D_deltaX_crossAdd
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1crossAdd(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_CrossAdd]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	//stage1: set deltaX = 0
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)(streams[0]);//2L = sizeof(float)
	cudaError_t error = cudaMemsetAsync(d_deltaX, 0, (N*IH*IW*IC) << 2L, stream1); handleError(error);

	//stage2: reversed convolution:
	cudaEvent_t event; error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	for (int i = 0; i < length; i++) {//wait the end of stage1
		cudaStream_t stream = (cudaStream_t)(intptr_t)(streams[i]);
		error = cudaStreamWaitEvent(stream, event, cudaEventWaitDefault); handleError(error);
	}
	
	int index = 0;
	__dconv3D_deltaX_CrossAdd(streams, index, length,
		d_deltaY, OH, OW, 
		dW, FH, FW, 
		d_deltaX, IH, IW, 
		N, IC, OC, 
		sh, sw, ph, pw);
}


//Method:    dconv3D_deltaX_crossAdd_SGM
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1crossAdd_1SGM(JNIEnv *env, jclass cls, 
	jlongArray streamArray, jint length, 
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw,
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_CrossAdd]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	//stage1: set deltaX = 0
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)(streams[0]);//2L = sizeof(float)
	cudaError_t error = cudaMemsetAsync(d_deltaX, 0, (N*IH*IW*IC) << 2L, stream1); handleError(error);
	
	//stage2: reversed convolution:
	cudaEvent_t event; error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	for (int i = 0; i < length; i++) {//wait the end of stage1
		cudaStream_t stream = (cudaStream_t)(intptr_t)(streams[i]);
		error = cudaStreamWaitEvent(stream, event, cudaEventWaitDefault); handleError(error);
	}

	const int GN = GET_GN_CrossAdd(OC);
	const int GM = GET_GM_CrossAdd(N, OH, OW);
	const int GK = GET_GK_CrossAdd(FH, FW, IC);

	int n = GM / GM_slice, GMr = GM % GM_slice;
	int index = 0, j_index = 0;

	for (int i = 0; i < n; i++) {
		dconv3d_deltaX_CrossAdd(streams, index, length,
			d_deltaY, OH, OW,
			dW, FH, FW,
			d_deltaX, IH, IW,
			IC, OC, 
			sh, sw, ph, pw, 
			GN, GM_slice, GK,
			0, j_index);
		j_index += GM_slice;
	}

	if (GMr) {//GM % GM_slice != 0
		dconv3d_deltaX_CrossAdd(streams, index, length,
			d_deltaY, OH, OW,
			dW, FH, FW,
			d_deltaX, IH, IW,
			IC, OC, 
			sh, sw, ph, pw,
			GN, GMr, GK,
			0, j_index);
	}
}

#endif



//kernel split: W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
#ifndef JNI_DECONV_3D_DELTAX_IM2COL_WINOGRAD_AREA
#define JNI_DECONV_3D_DELTAX_IM2COL_WINOGRAD_AREA

#ifndef JNI_DECONV_3D_DELTAX_IM2COL_WINOGRAD_S8
#define JNI_DECONV_3D_DELTAX_IM2COL_WINOGRAD_S8

//Method:    Im2col_Winograd_s8_texture
JNIEXPORT jboolean JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_Im2col_1Winograd_1s8_1texture(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length, 
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW, 
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	int index = 0;
	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OH*OW*OC), env);//[N, OH, OW, OC]
	JNIEnv *tenv = (useTexture ? env : NULL);
	bool flag = __dconv3D_deltaX_Im2col_Winograd_s8_tex(tenv, streams, index, length,
		d_deltaY, texDy, OH, OW,
		dW, FH, FW,
		d_deltaX, IH, IW, 
		N, IC, OC,
		ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
	return flag;
}

//Method:    Im2col_Winograd_s8_texture_SGM
JNIEXPORT jboolean JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_Im2col_1Winograd_1s8_1texture_1SGM(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length, 
	jlong d_deltaY_address, jint OH, jint OW, 
	jlong dW_address, jint FH, jint FW, 
	jlong d_deltaX_address, jint IH, jint IW, 
	jint N, jint IC, jint OC, 
	jint ph, jint pw, 
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	
	//split GM along the N axis will not affect the calculation of Im2col-Winograd and GemmR
	int N_slice = ((GM_slice / IW) >> 5 << 5) / IH; if (N_slice == 0) return false;//N * IH: 32 group
	int n = N / N_slice, Nr = N % N_slice;

	int index = 0;
	JNIEnv *tenv = (useTexture ? env : NULL);
	int Ystride = N_slice * (OH * OW * OC);
	int Xstride = N_slice * (IH * IW * IC);

	for (int i = 0; i < n; i++) {
		cudaTextureObject_t texDy = floatTexture(d_deltaY, Ystride, env);//[N_slice, OH, OW, OC]
		bool flag = __dconv3D_deltaX_Im2col_Winograd_s8_tex(tenv, streams, index, length,
			d_deltaY, texDy, OH, OW,
			dW, FH, FW,
			d_deltaX, IH, IW,
			N_slice, IC, OC,
			ph, pw);
		cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
		if (!flag) return false;
		d_deltaX += Xstride; d_deltaY += Ystride;//N += N_slice
	}

	if (Nr) {//N % N_slice != 0
		if ((Nr * IH) > 31) {//(Nr * IH) >= 32
			cudaTextureObject_t texDy = floatTexture(d_deltaY, (Nr*OH*OW*OC), env);//[Nr, OH, OW, OC]
			__dconv3D_deltaX_Im2col_Winograd_s8_tex(tenv, streams, index, length,
				d_deltaY, texDy, OH, OW,
				dW, FH, FW,
				d_deltaX, IH, IW,
				Nr, IC, OC,
				ph, pw);
			cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
		}
		else if (useTexture) {
			index = 0;//save L2 cache: texDy != deltaY
			cudaTextureObject_t texDy = floatTexture(d_deltaY, (Nr*OH*OW*OC), env);//[Nr, OH, OW, OC]
			__dconv3D_deltaX_ZeroPadding_s1_tex(streams, index, length,
				texDy, d_deltaY, OH, OW,
				dW, FH, FW,
				d_deltaX, IH, IW,
				Nr, IC, OC,
				ph, pw);
			cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
		}
		else {
			index = 0;//save L2 cache: texDy != deltaY
			__dconv3D_deltaX_ZeroPadding_s1(streams, index, length,
				d_deltaY, OH, OW,
				dW, FH, FW,
				d_deltaX, IH, IW,
				Nr, IC, OC,
				ph, pw);
		}
	}

	return true;
}

#endif


#ifndef JNI_DECONV_3D_DELTAX_IM2COL_WINOGRAD_S16
#define JNI_DECONV_3D_DELTAX_IM2COL_WINOGRAD_S16


//Method:    Im2col_Winograd_s16_texture
JNIEXPORT jboolean JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_Im2col_1Winograd_1s16_1texture(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	int index = 0;
	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OH*OW*OC), env);//[N, OH, OW, OC]
	JNIEnv *tenv = (useTexture ? env : NULL);
	bool flag = __dconv3D_deltaX_Im2col_Winograd_s16_tex(tenv, streams, index, length,
		d_deltaY, texDy, OH, OW,
		dW, FH, FW,
		d_deltaX, IH, IW,
		N, IC, OC,
		ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
	return flag;
}

// Method:    Im2col_Winograd_s16_texture_SGM
JNIEXPORT jboolean JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_Im2col_1Winograd_1s16_1texture_1SGM(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw, 
	jint GM_slice)
{
	jlong streams[MAX_STREAM_SIZE_ZeroPadding]; env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	//split GM along the N axis will not affect the calculation of Im2col-Winograd and GemmR
	int N_slice = ((GM_slice / IW) >> 5 << 5) / IH; if (N_slice == 0) return false;//N * IH: 32 group
	int n = N / N_slice, Nr = N % N_slice;

	int index = 0;
	JNIEnv *tenv = (useTexture ? env : NULL);
	int Ystride = N_slice * (OH * OW * OC);
	int Xstride = N_slice * (IH * IW * IC);

	for (int i = 0; i < n; i++) {
		cudaTextureObject_t texDy = floatTexture(d_deltaY, Ystride, env);//[N_slice, OH, OW, OC]
		bool flag = __dconv3D_deltaX_Im2col_Winograd_s8_tex(tenv, streams, index, length,
			d_deltaY, texDy, OH, OW,
			dW, FH, FW,
			d_deltaX, IH, IW,
			N_slice, IC, OC,
			ph, pw);
		cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
		if (!flag) return false;
		d_deltaX += Xstride; d_deltaY += Ystride;//N += N_slice
	}

	if (Nr) {//N % N_slice != 0
		if ((Nr * IH) > 31) {//(Nr * IH) >= 32
			cudaTextureObject_t texDy = floatTexture(d_deltaY, (Nr*OH*OW*OC), env);//[Nr, OH, OW, OC]
			__dconv3D_deltaX_Im2col_Winograd_s8_tex(tenv, streams, index, length,
				d_deltaY, texDy, OH, OW,
				dW, FH, FW,
				d_deltaX, IH, IW,
				Nr, IC, OC,
				ph, pw);
			cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
		}
		else if (useTexture) {
			index = 0;//save L2 cache: texDy != deltaY
			cudaTextureObject_t texDy = floatTexture(d_deltaY, (Nr*OH*OW*OC), env);//[Nr, OH, OW, OC]
			__dconv3D_deltaX_ZeroPadding_s1_tex(streams, index, length,
				texDy, d_deltaY, OH, OW,
				dW, FH, FW,
				d_deltaX, IH, IW,
				Nr, IC, OC,
				ph, pw);
			cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
		}
		else {
			index = 0;//save L2 cache: texDy != deltaY
			__dconv3D_deltaX_ZeroPadding_s1(streams, index, length,
				d_deltaY, OH, OW,
				dW, FH, FW,
				d_deltaX, IH, IW,
				Nr, IC, OC,
				ph, pw);
		}
	}

	return true;
}


#endif

#endif

#endif//complie-area>>>>------------------------------------------------------------