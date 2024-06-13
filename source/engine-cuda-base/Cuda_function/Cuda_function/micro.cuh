#pragma once

#ifndef MICRO_H
#define MICRO_H

#ifndef TEXTURE_FUNCTION
#define TEXTURE_FUNCTION

cudaTextureObject_t createFloat4Texture(float *X, long sizeX)
{

	cudaResourceDesc rdesc;
	memset(&rdesc, 0, sizeof(rdesc));
	rdesc.resType = cudaResourceTypeLinear;
	rdesc.res.linear.devPtr = X;
	rdesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	rdesc.res.linear.desc.x = 32;
	rdesc.res.linear.desc.y = 32;
	rdesc.res.linear.desc.z = 32;
	rdesc.res.linear.desc.w = 32;
	rdesc.res.linear.sizeInBytes = sizeX * sizeof(float);

	cudaTextureDesc tdesc;
	memset(&tdesc, 0, sizeof(tdesc));
	tdesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t texW = NULL;
	cudaCreateTextureObject(&texW, &rdesc, &tdesc, NULL);
	return texW;
}

#endif 


#ifndef COMMON_MICRO_H
#define COMMON_MICRO_H

#define GRID_MAX 8192
#define LENGTHV_MAX 4194303 //8192 * 512 - 1 > 8192 * 128

#define COPY4(a, b) {(a).x = (b).x; (a).y = (b).y; (a).z = (b).z; (a).w = (b).w;}

#define MAX_INT(x, y) ((x)^(((x)^(y))& -((x)<(y))))
#define MIN_INT(x, y) ((y)^(((x)^(y))& -((x)<(y))))

#define PI  (3.141592f)
#define RPI (0.3183099f)

#define FLOAT_MAX (3.4028235E38)
#define FLOAT_MIN (-3.4028234E38)

#define F32_4_0 float4{0, 0, 0, 0}

#define MOVE_E(v) {(v) = (v) + 1e-9f - 2.0f*((v)<0.0f)*1e-9f;}

#define simdLinear4(b, alpha, a, beta) {\
	(b).x = (a).x *(alpha) + (beta);\
	(b).y = (a).y *(alpha) + (beta);\
	(b).z = (a).z *(alpha) + (beta);\
	(b).w = (a).w *(alpha) + (beta);}

//pay attention to nan caused by 0
#define within_width(v, index4, stride, width) {\
	v.x *= ((index4    ) % stride) < width;\
	v.y *= ((index4 + 1) % stride) < width;\
	v.z *= ((index4 + 2) % stride) < width;\
	v.w *= ((index4 + 3) % stride) < width;}

//use more resource, but can zero nan caused by zero: 1, if within with
#define within_width_zero_nan(v, index4, table, stride, width) {\
	table[1] = v;\
	v.x = table[((index4    ) % stride) < width].x;\
	v.y = table[((index4 + 1) % stride) < width].y;\
	v.z = table[((index4 + 2) % stride) < width].z;\
	v.w = table[((index4 + 3) % stride) < width].w;}

#define simdMul4(c, a, b) {\
	(c).x = (a).x * (b).x;\
	(c).y = (a).y * (b).y;\
	(c).z = (a).z * (b).z;\
	(c).w = (a).w * (b).w; }

#endif


#ifndef FUNCTION_MICRO_H
#define FUNCTION_MICRO_H

#define PIXEL_CLIP(x) ((x<255.0f && x>0.0f)*x + (x>=255.0f)*255.0f)
#define CLIP(x, vmin, vmax) ((x<=vmin)*vmin + (x<vmax && x>vmin)*x + (x>=vmax)*vmax)

#define SIGN(x) (((x)>0.0f) - ((x)<0.0f))

#define ELU(x, k) ((x>0.0f)*x + (x<0.0f)*k*(expf(x) - 1.0f))
#define ELU_DERI(y, alpha, beta) (alpha + (y<=0.0f)*(y + beta))

//#define LEAKY_RELU(x, k) (x * ((x>0)*(1.0f-k) + (k))) // this expression may reduce accuarcy
#define LEAKY_RELU(x, k) ((x) * (((x) > 0.0f) + (k)*((x) < 0.0f)))
#define LEAKY_RELU2(x, k) ((x) > 0.0f ? (x) : (k)*(x))
#define LEAKY_RELU_DERI(y, k) (1.0f + (y<=0.0f)*k) //k = k-1.0f

#define RELU(x) fmaxf(x, 0.0f)
#define RELU_DERI(y) (y > 0.0f)

#define SOFTPLUS_DERI(y) (1.0f - expf(-y))

#define SIGMOID_DERI(y) (y * (1.0f - y))

#define TANH_DERI(y) (1.0f - y*y)

#define float_to_pixel(x) ( (x>0.0f && x<1.0f)*255.0f*x + (x>=1.0f)*255.0f )

#endif

#endif