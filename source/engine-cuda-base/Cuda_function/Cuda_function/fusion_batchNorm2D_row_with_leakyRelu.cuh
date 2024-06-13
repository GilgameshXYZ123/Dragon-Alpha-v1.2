#pragma once

#ifndef BATCH_NORM_2D_ROW_H_WITH_LEAKY_RELU
#define BATCH_NORM_2D_ROW_H_WITH_LEAKY_RELU

//(1) lengthv = height * stride
//(2) stride = (width + 3)/4*4
//(3) [lenghtv, stride] % 4 == 0
//(4) affined = false
#ifndef BATCH_NORM_2D_ROW_WITH_LEAKY_RELU_CALL
#define BATCH_NORM_2D_ROW_WITH_LEAKY_RELU_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

//lengthv < 256
#define batchNorm2d_row_with_leakyRelu_k4_small(stream, X, X_mean, X_var, eps, row_lengthv, Y, k, lengthv) \
	batchNorm2D_row_with_leakyRelu_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, X_mean, X_var, eps, row_lengthv, Y, k, lengthv, width, stride)

//common
#define batchNorm2d_row_with_leakyRelu_k4(stream, LB, LT, X, X_mean, X_var, eps, row_lengthv, Y, k, lengthv) \
	batchNorm2D_row_with_leakyRelu_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, X_mean, X_var, eps, row_lengthv, Y, k, lengthv, width, stride)

//lengthv > lengthv_max
#define batchNorm2d_row_with_leakyRelu_k4_max(stream, X, X_mean, X_var, eps, row_lengthv, Y, k, lengthv) \
	batchNorm2D_row_with_leakyRelu_kernel_4\
		<<< GRID_MAX, 32, 0, stream >>>\
			(X, X_mean, X_var, eps, row_lengthv, Y, k, lengthv, width, stride)

#endif


#ifndef BATCH_NORM_2D_ROW_WITH_LEAKY_RELU_KERNEL
#define BATCH_NORM_2D_ROW_WITH_LEAKY_RELU_KERNEL

//=======[Document]==================================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_var[M], A[M], B[M]
//<4> k: negative_slope
//
//[Forward Propagation]:
//(1) X_mean = mean_each_field(X)
//(2) X_var = variance_each_field(X^2)
//(3) X_std = sqrt(X_var + eps)
//(4) Y1 = X_norm = (X - X_mean) / X_std
//(6) Y2 = leakyRelu(Y1, k)
//=======[Document]==================================================

__global__ void batchNorm2D_row_with_leakyRelu_kernel_4(
	const float* __restrict__ X,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_var, float eps, int row_lengthv,
	      float* __restrict__ Y, float k,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = F32_4_0;//(x_var == 0) will cause NaN
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data------------------------------------------------------
		const int field_index4 = index4 % row_lengthv;
		const float4 x = *(float4*)(X + index4);
		const float4 x_mean = *(float4*)(X_mean + field_index4);
		const float4 x_var = *(float4*)(X_var + field_index4);

		//compute result-------------------------------------------------
		float4 x_norm;//X_norm = (X - X_mean) / sqrt(X_var + eps)
		x_norm.x = (x.x - x_mean.x) * rsqrtf(x_var.x + eps);
		x_norm.y = (x.y - x_mean.y) * rsqrtf(x_var.y + eps);
		x_norm.z = (x.z - x_mean.z) * rsqrtf(x_var.z + eps);
		x_norm.w = (x.w - x_mean.w) * rsqrtf(x_var.w + eps);

		float4 y;//Y = leakyRelu(X_norm, k)
		y.x = LEAKY_RELU(x_norm.x, k);
		y.y = LEAKY_RELU(x_norm.y, k);
		y.z = LEAKY_RELU(x_norm.z, k);
		y.w = LEAKY_RELU(x_norm.w, k);

		//write data-----------------------------------------------------
		within_width_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __batchNorm2D_row_with_leakyRelu(cudaStream_t stream,
	const float* X,
	const float* X_mean,
	const float* X_var, float eps, int row_lengthv,
	float* __restrict__ Y, float k,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { batchNorm2d_row_with_leakyRelu_k4_small(stream, X, X_mean, X_var, eps, row_lengthv, Y, k, lengthv); return; }
	if (lengthv > LENGTHV_MAX) { batchNorm2d_row_with_leakyRelu_k4_max(stream, X, X_mean, X_var, eps, row_lengthv, Y, k, lengthv); return; }
	batchNorm2d_row_with_leakyRelu_k4(stream, 5, 2, X, X_mean, X_var, eps, row_lengthv, Y, k, lengthv);
}

#endif