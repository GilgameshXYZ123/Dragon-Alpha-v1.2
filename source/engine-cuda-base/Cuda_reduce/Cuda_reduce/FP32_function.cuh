#pragma once

#ifndef FP32_FUNCTION_H
#define FP32_FUNCTION_H

class fp32_func {
public:
	__device__ __host__ fp32_func() {}
	__device__ __host__ virtual float function(float x) = 0;//y = f(x)
	__device__ __host__ virtual float inverse(float y) = 0;//x = f^(-1)(x)
	__device__ __host__ virtual float derivative_v1(float y) = 0;//f'(y): V1, holdY()
	__device__ __host__ virtual float derivative_v2(float x) = 0;//f'(x): V2, holdX()
};


#ifndef FP32_FUNCTION_RELU_GROUPS
#define FP32_FUNCTION_RELU_GROUPS

class fp32_func_Relu : public fp32_func {
public:
	__device__ __host__ __forceinline__ fp32_func_Relu() {}
	__device__ __host__ __forceinline__ virtual float function(float x) { return (x >= 0.0f) * x; }
	__device__ __host__ __forceinline__ virtual float inverse(float y) { return NAN; }//this is not allowed
	__device__ __host__ __forceinline__ virtual float derivative_v1(float y) { return (y >= 0.0f); }
	__device__ __host__ __forceinline__ virtual float derivative_v2(float x) { return (x >= 0.0f); }
};


class fp32_func_LeakyRelu : public fp32_func {
	float k, rk;//k > 0
public:
	__device__ __host__ __forceinline__ fp32_func_LeakyRelu(float k) { this->k = k; this->rk = 1.0f / k; }
	__device__ __host__ __forceinline__ virtual float function(float x) { bool flag = (x >= 0.0f); return flag * x + !flag * k * x; }
	__device__ __host__ __forceinline__ virtual float inverse(float y) { bool flag = (y >= 0.0f); return flag * y + !flag * rk * y; }
	__device__ __host__ __forceinline__ virtual float derivative_v1(float y) { bool flag = (y >= 0.0f); return flag + !flag * k; }
	__device__ __host__ __forceinline__ virtual float derivative_v2(float x) { bool flag = (x >= 0.0f); return flag + !flag * k; }
};


class fp32_func_Elu : public fp32_func {
	float alpha, ralpha, k, rk;//alpha > 0, k > 0
public:
	__device__ __host__ __forceinline__ fp32_func_Elu(float alpha, float k) {
		this->alpha = alpha;
		this->ralpha = 1.0f / alpha;
		this->k = k;
		this->rk = 1.0f / k;
	}

	__device__ __host__ __forceinline__ virtual float function(float x) {//y = f(x)
		bool flag = (x >= 0.0f);
		float y = flag * x + !flag * k * (expf(x) - 1.0f);
		return alpha * y;
	}

	__device__ __host__ __forceinline__ virtual float inverse(float y) {//x = f^(-1)(x)
		bool flag = (y >= 0.0f);
		y = y * ralpha;
		return flag * y + !flag * log1pf(y * rk);
	}

	__device__ __host__ __forceinline__ virtual float derivative_v1(float y) {//f'(y): V1, holdY()
		bool flag = (y >= 0.0f);
		return flag * alpha + !flag * (y + alpha * k);
	}

	__device__ __host__ __forceinline__ virtual float derivative_v2(float x) {//f'(x): V2, holdX() 
		bool flag = (x >= 0.0f);
		float dy = flag + !flag * k * (expf(x));
		return alpha * dy;
	}
};


class fp32_func_Softplus : public fp32_func {
public:
	__device__ __host__ __forceinline__ fp32_func_Softplus() {}
	__device__ __host__ __forceinline__ virtual float function(float x) { return log1pf(expf(x)); }
	__device__ __host__ __forceinline__ virtual float inverse(float y) { return logf(expf(y) - 1.0f); }
	__device__ __host__ __forceinline__ virtual float derivative_v1(float y) { return 1.0f - expf(-y); }
	__device__ __host__ __forceinline__ virtual float derivative_v2(float x) { return 1.0f / (1.0f + expf(-x)); }
};


class fp32_func_Gelu : public fp32_func {
public:
	__device__ __host__ __forceinline__ fp32_func_Gelu() {}
	__device__ __host__ __forceinline__ virtual float function(float x) {//y = f(x)
		float u = -1.5957692f * x * (1.0f + 0.044715f * x * x);
		return x / (1.0f + expf(u));
	}

	__device__  __host__ __forceinline__ virtual float inverse(float y) { return NAN; }//this is not allowed
	__device__  __host__ __forceinline__ virtual float derivative_v1(float y) { return NAN; }//this is not allowed
	__device__  __host__ __forceinline__ virtual float derivative_v2(float x) {//f'(x): V2, holdX() 
		float u = -1.5957692f * x * (1.0f + 0.044715f * x * x);
		float expu = expf(u), A1 = 1.0f / (expu + 1.0f), B1 = expu * A1;
		float expm = expf(-u), B2 = 1.0f / (expm + 1.0f), A2 = expm * B2;

		float As[2] = { A2, A1 }, Bs[2] = { B2, B1 };
		char flag = (x > 0.0f); float A = As[flag], B = Bs[flag];
		return A * (1.0f - B * (u - 0.14270963f * x * x * x));
	}
};

#endif


#ifndef FP32_FUNCTION_EXP_GROUPS
#define FP32_FUNCTION_EXP_GROUPS

class fp32_func_Sigmoid : public fp32_func {
public:
	__device__ __host__ fp32_func_Sigmoid() {}
	__device__ __host__ __forceinline__ virtual float function(float x) { return 1.0f / (1.0f + expf(-x)); }
	__device__ __host__ __forceinline__ virtual float inverse(float y) { return logf(y / (1.0f - y)); }
	__device__ __host__ __forceinline__ virtual float derivative_v1(float y) { return y * (1.0f - y); }
	__device__ __host__ __forceinline__ virtual float derivative_v2(float x) {
		float y = 1.0f / (1.0f + expf(-x));
		return y * (1.0f - y);
	}
};


class fp32_func_Tanh : public fp32_func {
public:
	__device__ __host__ __forceinline__ fp32_func_Tanh() {}
	__device__ __host__ __forceinline__ virtual float function(float x) { return 2.0f / (1.0f + expf(-2.0f * x)) - 1.0f; }
	__device__ __host__ __forceinline__ virtual float inverse(float y) { return 0.5f * logf((1.0f + y) / (1.0f - y)); }
	__device__ __host__ __forceinline__ virtual float derivative_v1(float y) { return 1.0f - (y * y); }
	__device__ __host__ __forceinline__ virtual float derivative_v2(float x) {//f'(x): V2, holdX()
		float y = 2.0f / (1.0f + expf(-2.0f * x)) - 1.0f;
		return 1.0f - (y * y);
	}
};

#endif


const int MAX_FP32_Func_Param_length = 3;
const int FP32_Func_MIN = 0;
const int FP32_Func_MAX = 6;


const int FP32_Func_Relu = 0;
const int FP32_Func_LeakyRelu = 1;
const int FP32_Func_Elu = 2;
const int FP32_Func_Softplus = 3;
const int FP32_Func_Gelu = 4;
const int FP32_Func_Sigmoid = 5;
const int FP32_Func_Tanh = 6;


#define new_fp32_func(fp32_func_type, fp32_func_param0, fp32_func_param1, fp32_func_param2)\
	create_fp32_func<fp32_func_type>(fp32_func_param0, fp32_func_param1, fp32_func_param2)

template<int fp32_func_type>
__host__ __device__ __forceinline__ fp32_func* create_fp32_func(
	float fp32_func_param0,
	float fp32_func_param1,
	float fp32_func_param2)
{
	if (fp32_func_type == FP32_Func_Relu)      return new fp32_func_Relu();
	if (fp32_func_type == FP32_Func_LeakyRelu) return new fp32_func_LeakyRelu(fp32_func_param0);
	if (fp32_func_type == FP32_Func_Elu)       return new fp32_func_Elu(fp32_func_param0, fp32_func_param1);
	if (fp32_func_type == FP32_Func_Softplus)  return new fp32_func_Softplus();
	if (fp32_func_type == FP32_Func_Gelu)      return new fp32_func_Gelu();
	if (fp32_func_type == FP32_Func_Sigmoid)   return new fp32_func_Sigmoid();
	if (fp32_func_type == FP32_Func_Tanh)      return new fp32_func_Tanh();
	return NULL;
}


//======[compile optimization]=======================
#ifndef FP32_FUNCTION_FORWARD
#define FP32_FUNCTION_FORWARD

template<int fp32_func_type>
__host__ __device__ __forceinline__ float fp32_func_forward(
	float x, float param0, float param1, float param2)
{
	if (fp32_func_type == FP32_Func_Relu) return (x >= 0.0f) * x;
	if (fp32_func_type == FP32_Func_LeakyRelu) {
		float k = param0;
		bool flag = (x >= 0.0f);
		return flag * x + !flag * k * x;
	}
	if (fp32_func_type == FP32_Func_Elu) {
		float alpha = param0, k = param1;
		bool flag = (x >= 0.0f);
		float y = flag * x + !flag * k * (expf(x) - 1.0f);
		return alpha * y;
	}
	if (fp32_func_type == FP32_Func_Softplus) return log1pf(expf(x));
	if (fp32_func_type == FP32_Func_Gelu) {
		float u = -1.5957692f * x * (1.0f + 0.044715f * x * x);
		return x / (1.0f + expf(u));
	}
	if (fp32_func_type == FP32_Func_Sigmoid) return 1.0f / (1.0f + expf(-x));
	if (fp32_func_type == FP32_Func_Tanh) return 2.0f / (1.0f + expf(-2.0f * x)) - 1.0f;
	return NAN;
}

#endif


#ifndef FP32_FUNCTION_IEVERSE
#define FP32_FUNCTION_IEVERSE

template<int fp32_func_type>
__host__ __device__ __forceinline__ float fp32_func_inverse(
	float x, float param0, float param1, float param2)
{
	//if (fp32_func_type == FP32_Func_Relu) return NAN;
	if (fp32_func_type == FP32_Func_LeakyRelu) {
		float k = param0;
		bool flag = (x >= 0.0f);
		return flag * x + !flag * k * x;
	}
	if (fp32_func_type == FP32_Func_Elu) {
		float alpha = param0, k = param1;
		float ralpha = 1.0f / alpha, rk = 1.0f / k;
		bool flag = (x >= 0.0f);
		x = x * ralpha;
		return flag * x + !flag * log1pf(x * rk);
	}
	if (fp32_func_type == FP32_Func_Softplus) return logf(expf(x) - 1.0f);
	//if (fp32_func_type == FP32_Func_Gelu) return NAN;
	if (fp32_func_type == FP32_Func_Sigmoid) return logf(x / (1.0f - x));
	if (fp32_func_type == FP32_Func_Tanh) return 0.5f * logf((1.0f + x) / (1.0f - x));
	return NAN;
}

#endif


#ifndef FP32_FUNCTION_DEVIVATIVE_V1
#define FP32_FUNCTION_DEVIVATIVE_V1

template<int fp32_func_type>
__host__ __device__ __forceinline__ float fp32_func_derivative_v1(
	float x, float param0, float param1, float param2)
{
	if (fp32_func_type == FP32_Func_Relu) return (x >= 0.0f);
	if (fp32_func_type == FP32_Func_LeakyRelu) {
		float k = param0;
		bool flag = (x >= 0.0f);
		return flag + !flag * k;
	}
	if (fp32_func_type == FP32_Func_Elu) {
		float alpha = param0, k = param1;
		bool flag = (x >= 0.0f);
		return flag * alpha + !flag * (x + alpha * k);
	}
	if (fp32_func_type == FP32_Func_Softplus) return 1.0f - expf(-x);
	//if (fp32_func_type == FP32_Func_Gelu) return NAN;
	if (fp32_func_type == FP32_Func_Sigmoid) return x * (1.0f - x);
	if (fp32_func_type == FP32_Func_Tanh) return 1.0f - (x * x);
	return NAN;
}

#endif


#ifndef FP32_FUNCTION_DEVIVATIVE_V2
#define FP32_FUNCTION_DEVIVATIVE_V2

template<int fp32_func_type>
__host__ __device__ __forceinline__ float fp32_func_derivative_v2(
	float x, float param0, float param1, float param2)
{
	if (fp32_func_type == FP32_Func_Relu) return (x >= 0.0f);
	if (fp32_func_type == FP32_Func_LeakyRelu) {
		float k = param0;
		bool flag = (x >= 0.0f);
		return flag + !flag * k;
	}
	if (fp32_func_type == FP32_Func_Elu) {
		float alpha = param0, k = param1;
		bool flag = (x >= 0.0f);
		float dy = flag + !flag * k * (expf(x));
		return alpha * dy;
	}
	if (fp32_func_type == FP32_Func_Softplus) return 1.0f / (1.0f + expf(-x));
	if (fp32_func_type == FP32_Func_Gelu) {
		float u = -1.5957692f * x * (1.0f + 0.044715f * x * x);
		float expu = expf(u), A1 = 1.0f / (expu + 1.0f), B1 = expu * A1;
		float expm = expf(-u), B2 = 1.0f / (expm + 1.0f), A2 = expm * B2;

		float As[2] = { A2, A1 }, Bs[2] = { B2, B1 };
		char flag = (x > 0.0f); float A = As[flag], B = Bs[flag];
		return A * (1.0f - B * (u - 0.14270963f * x * x * x));
	}
	if (fp32_func_type == FP32_Func_Sigmoid) {
		float y = 1.0f / (1.0f + expf(-x));
		return y * (1.0f - y);
	}
	if (fp32_func_type == FP32_Func_Tanh) {
		float y = 2.0f / (1.0f + expf(-2.0f * x)) - 1.0f;
		return 1.0f - (y * y);
	}
	return NAN;
}

#endif

#endif