//(32 * 32) feature-map================================================
			//<1> IC = 64, OC = 64
			//winograd: Size = 9, Time = 1.81338 msec, Performace = 10658.2 GFlop/s
			//GemmR:     Size = 9, Time = 1.95829 msec, Performace = 9869.53 GFlop/s
			//<2> IC =  128, OC = 128:
			//winograd: Size = 18, Time = 3.55475 msec, Performace = 10874.1 GFlop/s
			//GemmR:    Size = 18, Time = 3.47753 msec, Performace = 11115.6 GFlop/s
			//<3> IC = 256, OC = 256
			//winograd: Size = 72, Time = 13.7044 msec, Performace = 11282.4 GFlop/s
			//GemmR:    Size = 72, Time = 13.535 msec, Performace = 11423.6 GFlop/s

			//32 <= IC <= 64: winograd
			//(56 * 56) feature - map================================================
			//<1> IC = 64, OC = 64
			//winograd:
			//GemmR: 
			//<2> IC =  128, OC = 128:
			//winograd: Size = 13.7812, Time = 2.7444 msec, Performace = 10783.8 GFlop/s
			//GemmR:    Size = 13.7812, Time = 2.70029 msec, Performace = 10959.9 GFlop/s

			//<1> Size = 55.125, Time = 10.5597 msec, Performace = 11210.5 GFlop/s
			//<2> GemmR: Size = 55.125, Time = 10.749 msec, Performace = 11013.2 GFlop/s

			

#ifndef XENO1_WINOGRAD_F2x3_KERNEL_C1
#define XENO1_WINOGRAD_F2x3_KERNEL_C1

#define xeno_winograd_2x3_c1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_C1<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.84899 msec, Performace = 10452.9 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel_C1(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		float4 d0 = winograd_f2x3_d(x0, x1, x2, x3);
		float4 d1 = winograd_f2x3_d(x2, x3, x4, x5);
		Ds[buf][tx][ty] = d0;
		Ds[buf][tx][ty + STEP2] = d1;

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh*IC3 + wic)*OC;
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		WinoGrad_produce_G(Gs[buf][ty][tx], w0.x, w1.x, w2.x);
		WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], w0.y, w1.y, w2.y);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];

				//d0 * {g0, g1, g2, g3}
				v0.x += g0.x*d0.x; v0.y += g1.x*d0.x; v0.z += g2.x*d0.x; v0.w += g3.x*d0.x;
				v1.x -= g0.w*d0.w; v1.y -= g1.w*d0.w; v1.z -= g2.w*d0.w; v1.w -= g3.w*d0.w;
				t0.x += g0.y*d0.y; t0.y += g1.y*d0.y; t0.z += g2.y*d0.y; t0.w += g3.y*d0.y;
				t1.x += g0.z*d0.z; t1.y += g1.z*d0.z; t1.z += g2.z*d0.z; t1.w += g3.z*d0.z;

				//d1 * {g0, g1, g2, g3}
				v2.x += g0.x*d1.x; v2.y += g1.x*d1.x; v2.z += g2.x*d1.x; v2.w += g3.x*d1.x;
				v3.x -= g0.w*d1.w; v3.y -= g1.w*d1.w; v3.z -= g2.w*d1.w; v3.w -= g3.w*d1.w;
				t2.x += g0.y*d1.y; t2.y += g1.y*d1.y; t2.z += g2.y*d1.y; t2.w += g3.y*d1.y;
				t3.x += g0.z*d1.z; t3.y += g1.z*d1.z; t3.z += g2.z*d1.z; t3.w += g3.z*d1.z;

				//d2 * {g0, g1, g2, g3}
				v4.x += g0.x*d2.x; v4.y += g1.x*d2.x; v4.z += g2.x*d2.x; v4.w += g3.x*d2.x;
				v5.x -= g0.w*d2.w; v5.y -= g1.w*d2.w; v5.z -= g2.w*d2.w; v5.w -= g3.w*d2.w;
				t4.x += g0.y*d2.y; t4.y += g1.y*d2.y; t4.z += g2.y*d2.y; t4.w += g3.y*d2.y;
				t5.x += g0.z*d2.z; t5.y += g1.z*d2.z; t5.z += g2.z*d2.z; t5.w += g3.z*d2.z;

				//d3 * {g0, g1, g2, g3}
				v6.x += g0.x*d3.x; v6.y += g1.x*d3.x; v6.z += g2.x*d3.x; v6.w += g3.x*d3.x;
				v7.x -= g0.w*d3.w; v7.y -= g1.w*d3.w; v7.z -= g2.w*d3.w; v7.w -= g3.w*d3.w;
				t6.x += g0.y*d3.y; t6.y += g1.y*d3.y; t6.z += g2.y*d3.y; t6.w += g3.y*d3.y;
				t7.x += g0.z*d3.z; t7.y += g1.z*d3.z; t7.z += g2.z*d3.z; t7.w += g3.z*d3.z;
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			float4 d0 = winograd_f2x3_d(x0, x1, x2, x3);
			float4 d1 = winograd_f2x3_d(x2, x3, x4, x5);
			Ds[buf][tx][ty] = d0;
			Ds[buf][tx][ty + STEP2] = d1;

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = (fh*IC3 + wic)*OC;
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			WinoGrad_produce_G(Gs[buf][ty][tx], w0.x, w1.x, w2.x);
			WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], w0.y, w1.y, w2.y);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];

			//d0 * {g0, g1, g2, g3}
			v0.x += g0.x*d0.x; v0.y += g1.x*d0.x; v0.z += g2.x*d0.x; v0.w += g3.x*d0.x;
			v1.x -= g0.w*d0.w; v1.y -= g1.w*d0.w; v1.z -= g2.w*d0.w; v1.w -= g3.w*d0.w;
			t0.x += g0.y*d0.y; t0.y += g1.y*d0.y; t0.z += g2.y*d0.y; t0.w += g3.y*d0.y;
			t1.x += g0.z*d0.z; t1.y += g1.z*d0.z; t1.z += g2.z*d0.z; t1.w += g3.z*d0.z;

			//d1 * {g0, g1, g2, g3}
			v2.x += g0.x*d1.x; v2.y += g1.x*d1.x; v2.z += g2.x*d1.x; v2.w += g3.x*d1.x;
			v3.x -= g0.w*d1.w; v3.y -= g1.w*d1.w; v3.z -= g2.w*d1.w; v3.w -= g3.w*d1.w;
			t2.x += g0.y*d1.y; t2.y += g1.y*d1.y; t2.z += g2.y*d1.y; t2.w += g3.y*d1.y;
			t3.x += g0.z*d1.z; t3.y += g1.z*d1.z; t3.z += g2.z*d1.z; t3.w += g3.z*d1.z;

			//d2 * {g0, g1, g2, g3}
			v4.x += g0.x*d2.x; v4.y += g1.x*d2.x; v4.z += g2.x*d2.x; v4.w += g3.x*d2.x;
			v5.x -= g0.w*d2.w; v5.y -= g1.w*d2.w; v5.z -= g2.w*d2.w; v5.w -= g3.w*d2.w;
			t4.x += g0.y*d2.y; t4.y += g1.y*d2.y; t4.z += g2.y*d2.y; t4.w += g3.y*d2.y;
			t5.x += g0.z*d2.z; t5.y += g1.z*d2.z; t5.z += g2.z*d2.z; t5.w += g3.z*d2.z;

			//d3 * {g0, g1, g2, g3}
			v6.x += g0.x*d3.x; v6.y += g1.x*d3.x; v6.z += g2.x*d3.x; v6.w += g3.x*d3.x;
			v7.x -= g0.w*d3.w; v7.y -= g1.w*d3.w; v7.z -= g2.w*d3.w; v7.w -= g3.w*d3.w;
			t6.x += g0.y*d3.y; t6.y += g1.y*d3.y; t6.z += g2.y*d3.y; t6.w += g3.y*d3.y;
			t7.x += g0.z*d3.z; t7.y += g1.z*d3.z; t7.z += g2.z*d3.z; t7.w += g3.z*d3.z;
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


#ifndef XENO1_WINOGRAD_F2x3_KERNEL_C2
#define XENO1_WINOGRAD_F2x3_KERNEL_C2

#define xeno_winograd_2x3_c2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_C2<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.84899 msec, Performace = 10452.9 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel_C2(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		float4 d0 = winograd_f2x3_d(x0, x1, x2, x3);
		float4 d1 = winograd_f2x3_d(x2, x3, x4, x5);
		Ds[buf][tx][ty] = d0;
		Ds[buf][tx][ty + STEP2] = d1;

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh*IC3 + wic)*OC;
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		WinoGrad_produce_G(Gs[buf][ty][tx], w0.x, w1.x, w2.x);
		WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], w0.y, w1.y, w2.y);
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];

				//{d0, d1} * {g0, g1, g2, g3}
				v0.x += g0.x*d0.x; v0.y += g1.x*d0.x; v0.z += g2.x*d0.x; v0.w += g3.x*d0.x;
				v1.x -= g0.w*d0.w; v1.y -= g1.w*d0.w; v1.z -= g2.w*d0.w; v1.w -= g3.w*d0.w;
				v2.x += g0.x*d1.x; v2.y += g1.x*d1.x; v2.z += g2.x*d1.x; v2.w += g3.x*d1.x;
				v3.x -= g0.w*d1.w; v3.y -= g1.w*d1.w; v3.z -= g2.w*d1.w; v3.w -= g3.w*d1.w;

				//{d2, d3} * {g0, g1, g2, g3}
				v4.x += g0.x*d2.x; v4.y += g1.x*d2.x; v4.z += g2.x*d2.x; v4.w += g3.x*d2.x;
				v5.x -= g0.w*d2.w; v5.y -= g1.w*d2.w; v5.z -= g2.w*d2.w; v5.w -= g3.w*d2.w;
				v6.x += g0.x*d3.x; v6.y += g1.x*d3.x; v6.z += g2.x*d3.x; v6.w += g3.x*d3.x;
				v7.x -= g0.w*d3.w; v7.y -= g1.w*d3.w; v7.z -= g2.w*d3.w; v7.w -= g3.w*d3.w;

				//{d0, d1} * {g0, g1, g2, g3}
				t0.x += g0.y*d0.y; t0.y += g1.y*d0.y; t0.z += g2.y*d0.y; t0.w += g3.y*d0.y;
				t1.x += g0.z*d0.z; t1.y += g1.z*d0.z; t1.z += g2.z*d0.z; t1.w += g3.z*d0.z;
				t2.x += g0.y*d1.y; t2.y += g1.y*d1.y; t2.z += g2.y*d1.y; t2.w += g3.y*d1.y;
				t3.x += g0.z*d1.z; t3.y += g1.z*d1.z; t3.z += g2.z*d1.z; t3.w += g3.z*d1.z;

				//{d2, d3} * {g0, g1, g2, g3}
				t4.x += g0.y*d2.y; t4.y += g1.y*d2.y; t4.z += g2.y*d2.y; t4.w += g3.y*d2.y;
				t5.x += g0.z*d2.z; t5.y += g1.z*d2.z; t5.z += g2.z*d2.z; t5.w += g3.z*d2.z;
				t6.x += g0.y*d3.y; t6.y += g1.y*d3.y; t6.z += g2.y*d3.y; t6.w += g3.y*d3.y;
				t7.x += g0.z*d3.z; t7.y += g1.z*d3.z; t7.z += g2.z*d3.z; t7.w += g3.z*d3.z;
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			float4 d0 = winograd_f2x3_d(x0, x1, x2, x3);
			float4 d1 = winograd_f2x3_d(x2, x3, x4, x5);
			Ds[buf][tx][ty] = d0;
			Ds[buf][tx][ty + STEP2] = d1;

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = (fh*IC3 + wic)*OC;
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			WinoGrad_produce_G(Gs[buf][ty][tx], w0.x, w1.x, w2.x);
			WinoGrad_produce_G(Gs[buf][ty][tx + STEP2], w0.y, w1.y, w2.y);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];

			//{d0, d1} * {g0, g1, g2, g3}
			v0.x += g0.x*d0.x; v0.y += g1.x*d0.x; v0.z += g2.x*d0.x; v0.w += g3.x*d0.x;
			v1.x -= g0.w*d0.w; v1.y -= g1.w*d0.w; v1.z -= g2.w*d0.w; v1.w -= g3.w*d0.w;
			v2.x += g0.x*d1.x; v2.y += g1.x*d1.x; v2.z += g2.x*d1.x; v2.w += g3.x*d1.x;
			v3.x -= g0.w*d1.w; v3.y -= g1.w*d1.w; v3.z -= g2.w*d1.w; v3.w -= g3.w*d1.w;

			//{d2, d3} * {g0, g1, g2, g3}
			v4.x += g0.x*d2.x; v4.y += g1.x*d2.x; v4.z += g2.x*d2.x; v4.w += g3.x*d2.x;
			v5.x -= g0.w*d2.w; v5.y -= g1.w*d2.w; v5.z -= g2.w*d2.w; v5.w -= g3.w*d2.w;
			v6.x += g0.x*d3.x; v6.y += g1.x*d3.x; v6.z += g2.x*d3.x; v6.w += g3.x*d3.x;
			v7.x -= g0.w*d3.w; v7.y -= g1.w*d3.w; v7.z -= g2.w*d3.w; v7.w -= g3.w*d3.w;

			//{d0, d1} * {g0, g1, g2, g3}
			t0.x += g0.y*d0.y; t0.y += g1.y*d0.y; t0.z += g2.y*d0.y; t0.w += g3.y*d0.y;
			t1.x += g0.z*d0.z; t1.y += g1.z*d0.z; t1.z += g2.z*d0.z; t1.w += g3.z*d0.z;
			t2.x += g0.y*d1.y; t2.y += g1.y*d1.y; t2.z += g2.y*d1.y; t2.w += g3.y*d1.y;
			t3.x += g0.z*d1.z; t3.y += g1.z*d1.z; t3.z += g2.z*d1.z; t3.w += g3.z*d1.z;

			//{d2, d3} * {g0, g1, g2, g3}
			t4.x += g0.y*d2.y; t4.y += g1.y*d2.y; t4.z += g2.y*d2.y; t4.w += g3.y*d2.y;
			t5.x += g0.z*d2.z; t5.y += g1.z*d2.z; t5.z += g2.z*d2.z; t5.w += g3.z*d2.z;
			t6.x += g0.y*d3.y; t6.y += g1.y*d3.y; t6.z += g2.y*d3.y; t6.w += g3.y*d3.y;
			t7.x += g0.z*d3.z; t7.y += g1.z*d3.z; t7.z += g2.z*d3.z; t7.w += g3.z*d3.z;
		}
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif



//STAGE1: for decompse  v, t
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_C3
#define XENO1_WINOGRAD_F2x3_KERNEL_C3

#define xeno_winograd_2x3_c3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_C3<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.87129 msec, Performace = 10328.3 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel_C3(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		float4 d0 = winograd_f2x3_d(x0, x1, x2, x3);
		float4 d1 = winograd_f2x3_d(x2, x3, x4, x5);
		Ds[buf][tx][ty        ] = float4{ d0.x, d0.w, d1.x, d1.w };//d0, d2: {x, w} for v
		Ds[buf][tx][ty + STEP2] = float4{ d0.y, d0.z, d1.y, d1.z };//d1, d3: {y, z} for t

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh*IC3 + wic)*OC;
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
		float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
		Gs[buf][ty][tx] = g0;
		Gs[buf][ty][tx + STEP2] = g1;
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx        ], g2 = Gs[buf][ik + STEP][tx        ];
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];

				float4 d0 = Ds[buf][ik][ty        ], d2 = Ds[buf][ik + STEP][ty        ];//{x, w} for v;
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];//{y, z} for t

				//{d0, d1} * {g0, g1, g2, g3}
				v0.x += g0.x*d0.x; v0.y += g1.x*d0.x; v0.z += g2.x*d0.x; v0.w += g3.x*d0.x;
				v1.x -= g0.w*d0.y; v1.y -= g1.w*d0.y; v1.z -= g2.w*d0.y; v1.w -= g3.w*d0.y;
				v2.x += g0.x*d0.z; v2.y += g1.x*d0.z; v2.z += g2.x*d0.z; v2.w += g3.x*d0.z;
				v3.x -= g0.w*d0.w; v3.y -= g1.w*d0.w; v3.z -= g2.w*d0.w; v3.w -= g3.w*d0.w;

				//{d2, d3} * {g0, g1, g2, g3}
				v4.x += g0.x*d2.x; v4.y += g1.x*d2.x; v4.z += g2.x*d2.x; v4.w += g3.x*d2.x;
				v5.x -= g0.w*d2.y; v5.y -= g1.w*d2.y; v5.z -= g2.w*d2.y; v5.w -= g3.w*d2.y;
				v6.x += g0.x*d2.z; v6.y += g1.x*d2.z; v6.z += g2.x*d2.z; v6.w += g3.x*d2.z;
				v7.x -= g0.w*d2.w; v7.y -= g1.w*d2.w; v7.z -= g2.w*d2.w; v7.w -= g3.w*d2.w;

				//==========================================================================

				//{d0, d1} * {g0, g1, g2, g3}
				t0.x += g0.y*d1.x; t0.y += g1.y*d1.x; t0.z += g2.y*d1.x; t0.w += g3.y*d1.x;
				t1.x += g0.z*d1.y; t1.y += g1.z*d1.y; t1.z += g2.z*d1.y; t1.w += g3.z*d1.y;
				t2.x += g0.y*d1.z; t2.y += g1.y*d1.z; t2.z += g2.y*d1.z; t2.w += g3.y*d1.z;
				t3.x += g0.z*d1.w; t3.y += g1.z*d1.w; t3.z += g2.z*d1.w; t3.w += g3.z*d1.w;

				//{d2, d3} * {g0, g1, g2, g3}
				t4.x += g0.y*d3.x; t4.y += g1.y*d3.x; t4.z += g2.y*d3.x; t4.w += g3.y*d3.x;
				t5.x += g0.z*d3.y; t5.y += g1.z*d3.y; t5.z += g2.z*d3.y; t5.w += g3.z*d3.y;
				t6.x += g0.y*d3.z; t6.y += g1.y*d3.z; t6.z += g2.y*d3.z; t6.w += g3.y*d3.z;
				t7.x += g0.z*d3.w; t7.y += g1.z*d3.w; t7.z += g2.z*d3.w; t7.w += g3.z*d3.w;
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			float4 d0 = winograd_f2x3_d(x0, x1, x2, x3);
			float4 d1 = winograd_f2x3_d(x2, x3, x4, x5);
			Ds[buf][tx][ty        ] = float4{ d0.x, d0.w, d1.x, d1.w };//{x, w} for v;
			Ds[buf][tx][ty + STEP2] = float4{ d0.y, d0.z, d1.y, d1.z };//{y, z} for t

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = (fh*IC3 + wic)*OC;
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
			float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
			Gs[buf][ty][tx] = g0;
			Gs[buf][ty][tx + STEP2] = g1;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];

			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];//{x, w} for v;
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];//{y, z} for t

			//{d0, d1} * {g0, g1, g2, g3}
			v0.x += g0.x*d0.x; v0.y += g1.x*d0.x; v0.z += g2.x*d0.x; v0.w += g3.x*d0.x;
			v1.x -= g0.w*d0.y; v1.y -= g1.w*d0.y; v1.z -= g2.w*d0.y; v1.w -= g3.w*d0.y;
			v2.x += g0.x*d0.z; v2.y += g1.x*d0.z; v2.z += g2.x*d0.z; v2.w += g3.x*d0.z;
			v3.x -= g0.w*d0.w; v3.y -= g1.w*d0.w; v3.z -= g2.w*d0.w; v3.w -= g3.w*d0.w;

			//{d2, d3} * {g0, g1, g2, g3}
			v4.x += g0.x*d2.x; v4.y += g1.x*d2.x; v4.z += g2.x*d2.x; v4.w += g3.x*d2.x;
			v5.x -= g0.w*d2.y; v5.y -= g1.w*d2.y; v5.z -= g2.w*d2.y; v5.w -= g3.w*d2.y;
			v6.x += g0.x*d2.z; v6.y += g1.x*d2.z; v6.z += g2.x*d2.z; v6.w += g3.x*d2.z;
			v7.x -= g0.w*d2.w; v7.y -= g1.w*d2.w; v7.z -= g2.w*d2.w; v7.w -= g3.w*d2.w;

			//==========================================================================

			//{d0, d1} * {g0, g1, g2, g3}
			t0.x += g0.y*d1.x; t0.y += g1.y*d1.x; t0.z += g2.y*d1.x; t0.w += g3.y*d1.x;
			t1.x += g0.z*d1.y; t1.y += g1.z*d1.y; t1.z += g2.z*d1.y; t1.w += g3.z*d1.y;
			t2.x += g0.y*d1.z; t2.y += g1.y*d1.z; t2.z += g2.y*d1.z; t2.w += g3.y*d1.z;
			t3.x += g0.z*d1.w; t3.y += g1.z*d1.w; t3.z += g2.z*d1.w; t3.w += g3.z*d1.w;

			//{d2, d3} * {g0, g1, g2, g3}
			t4.x += g0.y*d3.x; t4.y += g1.y*d3.x; t4.z += g2.y*d3.x; t4.w += g3.y*d3.x;
			t5.x += g0.z*d3.y; t5.y += g1.z*d3.y; t5.z += g2.z*d3.y; t5.w += g3.z*d3.y;
			t6.x += g0.y*d3.z; t6.y += g1.y*d3.z; t6.z += g2.y*d3.z; t6.w += g3.y*d3.z;
			t7.x += g0.z*d3.w; t7.y += g1.z*d3.w; t7.z += g2.z*d3.w; t7.w += g3.z*d3.w;
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//STAGE2: for decompse  v, t(BEST)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_C4
#define XENO1_WINOGRAD_F2x3_KERNEL_C4

#define xeno_winograd_2x3_c4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_C4<LB, (1<<LB)-1, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.73562 msec, Performace = 11135.7 GFlop/s
template<int LB, int BLOCK_m1, int STEP, int STEP2, int STEP_m1, 
		int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel_C4(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		float4 d0 = winograd_f2x3_d(x0, x1, x2, x3);
		float4 d1 = winograd_f2x3_d(x2, x3, x4, x5);
		Ds[buf][tx][ty        ] = float4{ d0.x, d0.w, d1.x, d1.w };//d0, d2: {x, w} for v
		Ds[buf][tx][ty + STEP2] = float4{ d0.y, d0.z, d1.y, d1.z };//d1, d3: {y, z} for t

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh*IC3 + wic)*OC;
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
		float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
		Gs[buf][ty][tx        ] = float4{ g0.x, g0.w, g1.x, g1.w };//g0, d2: {x, w} for v
		Gs[buf][ty][tx + STEP2] = float4{ g0.y, g0.z, g1.y, g1.z };//d1, d3: {y, z} for t
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				//g0 -> {g0, g1}, g2 -> {g1, g2}
				float4 g0 = Gs[buf][ik][tx        ], g2 = Gs[buf][ik + STEP][tx        ];//{x, w} for v;
				float4 d0 = Ds[buf][ik][ty        ], d2 = Ds[buf][ik + STEP][ty        ];//{x, w} for v;
				
				//{d0, d1} * {g0, g1, g2, g3}
				v0.x += g0.x*d0.x; v0.y += g0.z*d0.x; v0.z += g2.x*d0.x; v0.w += g2.z*d0.x;
				v1.x -= g0.y*d0.y; v1.y -= g0.w*d0.y; v1.z -= g2.y*d0.y; v1.w -= g2.w*d0.y;
				v2.x += g0.x*d0.z; v2.y += g0.z*d0.z; v2.z += g2.x*d0.z; v2.w += g2.z*d0.z;
				v3.x -= g0.y*d0.w; v3.y -= g0.w*d0.w; v3.z -= g2.y*d0.w; v3.w -= g2.w*d0.w;

				//{d2, d3} * {g0, g1, g2, g3}
				v4.x += g0.x*d2.x; v4.y += g0.z*d2.x; v4.z += g2.x*d2.x; v4.w += g2.z*d2.x;
				v5.x -= g0.y*d2.y; v5.y -= g0.w*d2.y; v5.z -= g2.y*d2.y; v5.w -= g2.w*d2.y;
				v6.x += g0.x*d2.z; v6.y += g0.z*d2.z; v6.z += g2.x*d2.z; v6.w += g2.z*d2.z;
				v7.x -= g0.y*d2.w; v7.y -= g0.w*d2.w; v7.z -= g2.y*d2.w; v7.w -= g2.w*d2.w;

				//==========================================================================
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];//{y, z} for t
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];//{y, z} for t

				//{d0, d1} * {g0, g1, g2, g3}
				t0.x += g1.x*d1.x; t0.y += g1.z*d1.x; t0.z += g3.x*d1.x; t0.w += g3.z*d1.x;
				t1.x += g1.y*d1.y; t1.y += g1.w*d1.y; t1.z += g3.y*d1.y; t1.w += g3.w*d1.y;
				t2.x += g1.x*d1.z; t2.y += g1.z*d1.z; t2.z += g3.x*d1.z; t2.w += g3.z*d1.z;
				t3.x += g1.y*d1.w; t3.y += g1.w*d1.w; t3.z += g3.y*d1.w; t3.w += g3.w*d1.w;

				//{d2, d3} * {g0, g1, g2, g3}
				t4.x += g1.x*d3.x; t4.y += g1.z*d3.x; t4.z += g3.x*d3.x; t4.w += g3.z*d3.x;
				t5.x += g1.y*d3.y; t5.y += g1.w*d3.y; t5.z += g3.y*d3.y; t5.w += g3.w*d3.y;
				t6.x += g1.x*d3.z; t6.y += g1.z*d3.z; t6.z += g3.x*d3.z; t6.w += g3.z*d3.z;
				t7.x += g1.y*d3.w; t7.y += g1.w*d3.w; t7.z += g3.y*d3.w; t7.w += g3.w*d3.w;
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			float4 d0 = winograd_f2x3_d(x0, x1, x2, x3);
			float4 d1 = winograd_f2x3_d(x2, x3, x4, x5);
			Ds[buf][tx][ty] = float4{ d0.x, d0.w, d1.x, d1.w };//{x, w} for v;
			Ds[buf][tx][ty + STEP2] = float4{ d0.y, d0.z, d1.y, d1.z };//{y, z} for t

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = (fh*IC3 + wic)*OC;
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
			float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
			Gs[buf][ty][tx        ] = float4{ g0.x, g0.w, g1.x, g1.w };
			Gs[buf][ty][tx + STEP2] = float4{ g0.y, g0.z, g1.y, g1.z };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			//g0 -> {g0, g1}, g2 -> {g1, g2}
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];//{x, w} for v;
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];//{x, w} for v;

			//{d0, d1} * {g0, g1, g2, g3}
			v0.x += g0.x*d0.x; v0.y += g0.z*d0.x; v0.z += g2.x*d0.x; v0.w += g2.z*d0.x;
			v1.x -= g0.y*d0.y; v1.y -= g0.w*d0.y; v1.z -= g2.y*d0.y; v1.w -= g2.w*d0.y;
			v2.x += g0.x*d0.z; v2.y += g0.z*d0.z; v2.z += g2.x*d0.z; v2.w += g2.z*d0.z;
			v3.x -= g0.y*d0.w; v3.y -= g0.w*d0.w; v3.z -= g2.y*d0.w; v3.w -= g2.w*d0.w;

			//{d2, d3} * {g0, g1, g2, g3}
			v4.x += g0.x*d2.x; v4.y += g0.z*d2.x; v4.z += g2.x*d2.x; v4.w += g2.z*d2.x;
			v5.x -= g0.y*d2.y; v5.y -= g0.w*d2.y; v5.z -= g2.y*d2.y; v5.w -= g2.w*d2.y;
			v6.x += g0.x*d2.z; v6.y += g0.z*d2.z; v6.z += g2.x*d2.z; v6.w += g2.z*d2.z;
			v7.x -= g0.y*d2.w; v7.y -= g0.w*d2.w; v7.z -= g2.y*d2.w; v7.w -= g2.w*d2.w;

			//==========================================================================
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];//{y, z} for t
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];//{y, z} for t

			//{d0, d1} * {g0, g1, g2, g3}
			t0.x += g1.x*d1.x; t0.y += g1.z*d1.x; t0.z += g3.x*d1.x; t0.w += g3.z*d1.x;
			t1.x += g1.y*d1.y; t1.y += g1.w*d1.y; t1.z += g3.y*d1.y; t1.w += g3.w*d1.y;
			t2.x += g1.x*d1.z; t2.y += g1.z*d1.z; t2.z += g3.x*d1.z; t2.w += g3.z*d1.z;
			t3.x += g1.y*d1.w; t3.y += g1.w*d1.w; t3.z += g3.y*d1.w; t3.w += g3.w*d1.w;

			//{d2, d3} * {g0, g1, g2, g3}
			t4.x += g1.x*d3.x; t4.y += g1.z*d3.x; t4.z += g3.x*d3.x; t4.w += g3.z*d3.x;
			t5.x += g1.y*d3.y; t5.y += g1.w*d3.y; t5.z += g3.y*d3.y; t5.w += g3.w*d3.y;
			t6.x += g1.x*d3.z; t6.y += g1.z*d3.z; t6.z += g3.x*d3.z; t6.w += g3.z*d3.z;
			t7.x += g1.y*d3.w; t7.y += g1.w*d3.w; t7.z += g3.y*d3.w; t7.w += g3.w*d3.w;
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//NOT_GOOD
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_C5
#define XENO1_WINOGRAD_F2x3_KERNEL_C5

#define xeno_winograd_2x3_c5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_C5<LB, (1<<LB)-1, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.73562 msec, Performace = 11135.7 GFlop/s
template<int LB, int BLOCK_m1, int STEP, int STEP2, int STEP_m1,
	int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel_C5(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0, tgroup1}

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		float4 d0 = winograd_f2x3_d(x0, x1, x2, x3);
		float4 d1 = winograd_f2x3_d(x2, x3, x4, x5);
		Ds[buf][tx][ty] = float4{ d0.x, d0.w, d1.x, d1.w };//d0, d2: {x, w} for v
		Ds[buf][tx][ty + STEP2] = float4{ d0.y, d0.z, d1.y, d1.z };//d1, d3: {y, z} for t

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh*IC3 + wic)*OC;
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
		float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
		Gs[buf][ty][tx] = float4{ g0.x, g0.w, g1.x, g1.w };//g0, d2: {x, w} for v
		Gs[buf][ty][tx + STEP2] = float4{ g0.y, g0.z, g1.y, g1.z };//d1, d3: {y, z} for t
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				//g0 -> {g0, g1}, g2 -> {g1, g2}
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];//{x, w} for v;
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];//{x, w} for v;

				//{d0, d1} * {g0, g1, g2, g3}
				v0.x += g0.x*d0.x; v0.y += g0.z*d0.x; v0.z += g2.x*d0.x; v0.w += g2.z*d0.x;
				v1.x -= g0.y*d0.y; v1.y -= g0.w*d0.y; v1.z -= g2.y*d0.y; v1.w -= g2.w*d0.y;
				v2.x += g0.x*d0.z; v2.y += g0.z*d0.z; v2.z += g2.x*d0.z; v2.w += g2.z*d0.z;
				v3.x -= g0.y*d0.w; v3.y -= g0.w*d0.w; v3.z -= g2.y*d0.w; v3.w -= g2.w*d0.w;

				//{d2, d3} * {g0, g1, g2, g3}
				v4.x += g0.x*d2.x; v4.y += g0.z*d2.x; v4.z += g2.x*d2.x; v4.w += g2.z*d2.x;
				v5.x -= g0.y*d2.y; v5.y -= g0.w*d2.y; v5.z -= g2.y*d2.y; v5.w -= g2.w*d2.y;
				v6.x += g0.x*d2.z; v6.y += g0.z*d2.z; v6.z += g2.x*d2.z; v6.w += g2.z*d2.z;
				v7.x -= g0.y*d2.w; v7.y -= g0.w*d2.w; v7.z -= g2.y*d2.w; v7.w -= g2.w*d2.w;
			}
			float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
			float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];//{y, z} for t
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];//{y, z} for t

				//{d0, d1} * {g0, g1, g2, g3}
				t0.x += g1.x*d1.x; t0.y += g1.z*d1.x; t0.z += g3.x*d1.x; t0.w += g3.z*d1.x;
				t1.x += g1.y*d1.y; t1.y += g1.w*d1.y; t1.z += g3.y*d1.y; t1.w += g3.w*d1.y;
				t2.x += g1.x*d1.z; t2.y += g1.z*d1.z; t2.z += g3.x*d1.z; t2.w += g3.z*d1.z;
				t3.x += g1.y*d1.w; t3.y += g1.w*d1.w; t3.z += g3.y*d1.w; t3.w += g3.w*d1.w;

				//{d2, d3} * {g0, g1, g2, g3}
				t4.x += g1.x*d3.x; t4.y += g1.z*d3.x; t4.z += g3.x*d3.x; t4.w += g3.z*d3.x;
				t5.x += g1.y*d3.y; t5.y += g1.w*d3.y; t5.z += g3.y*d3.y; t5.w += g3.w*d3.y;
				t6.x += g1.x*d3.z; t6.y += g1.z*d3.z; t6.z += g3.x*d3.z; t6.w += g3.z*d3.z;
				t7.x += g1.y*d3.w; t7.y += g1.w*d3.w; t7.z += g3.y*d3.w; t7.w += g3.w*d3.w;
			}
			winograd_f2x3_VT4(v0, v1, t0, t1);
			winograd_f2x3_VT4(v2, v3, t2, t3);
			winograd_f2x3_VT4(v4, v5, t4, t5);
			winograd_f2x3_VT4(v6, v7, t6, t7);
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			float4 d0 = winograd_f2x3_d(x0, x1, x2, x3);
			float4 d1 = winograd_f2x3_d(x2, x3, x4, x5);
			Ds[buf][tx][ty] = float4{ d0.x, d0.w, d1.x, d1.w };//{x, w} for v;
			Ds[buf][tx][ty + STEP2] = float4{ d0.y, d0.z, d1.y, d1.z };//{y, z} for t

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = (fh*IC3 + wic)*OC;
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
			float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
			Gs[buf][ty][tx] = float4{ g0.x, g0.w, g1.x, g1.w };
			Gs[buf][ty][tx + STEP2] = float4{ g0.y, g0.z, g1.y, g1.z };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			//g0 -> {g0, g1}, g2 -> {g1, g2}
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];//{x, w} for v;
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];//{x, w} for v;

			//{d0, d1} * {g0, g1, g2, g3}
			v0.x += g0.x*d0.x; v0.y += g0.z*d0.x; v0.z += g2.x*d0.x; v0.w += g2.z*d0.x;
			v1.x -= g0.y*d0.y; v1.y -= g0.w*d0.y; v1.z -= g2.y*d0.y; v1.w -= g2.w*d0.y;
			v2.x += g0.x*d0.z; v2.y += g0.z*d0.z; v2.z += g2.x*d0.z; v2.w += g2.z*d0.z;
			v3.x -= g0.y*d0.w; v3.y -= g0.w*d0.w; v3.z -= g2.y*d0.w; v3.w -= g2.w*d0.w;

			//{d2, d3} * {g0, g1, g2, g3}
			v4.x += g0.x*d2.x; v4.y += g0.z*d2.x; v4.z += g2.x*d2.x; v4.w += g2.z*d2.x;
			v5.x -= g0.y*d2.y; v5.y -= g0.w*d2.y; v5.z -= g2.y*d2.y; v5.w -= g2.w*d2.y;
			v6.x += g0.x*d2.z; v6.y += g0.z*d2.z; v6.z += g2.x*d2.z; v6.w += g2.z*d2.z;
			v7.x -= g0.y*d2.w; v7.y -= g0.w*d2.w; v7.z -= g2.y*d2.w; v7.w -= g2.w*d2.w;
		}
		float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
		float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];//{y, z} for t
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];//{y, z} for t

			//{d0, d1} * {g0, g1, g2, g3}
			t0.x += g1.x*d1.x; t0.y += g1.z*d1.x; t0.z += g3.x*d1.x; t0.w += g3.z*d1.x;
			t1.x += g1.y*d1.y; t1.y += g1.w*d1.y; t1.z += g3.y*d1.y; t1.w += g3.w*d1.y;
			t2.x += g1.x*d1.z; t2.y += g1.z*d1.z; t2.z += g3.x*d1.z; t2.w += g3.z*d1.z;
			t3.x += g1.y*d1.w; t3.y += g1.w*d1.w; t3.z += g3.y*d1.w; t3.w += g3.w*d1.w;

			//{d2, d3} * {g0, g1, g2, g3}
			t4.x += g1.x*d3.x; t4.y += g1.z*d3.x; t4.z += g3.x*d3.x; t4.w += g3.z*d3.x;
			t5.x += g1.y*d3.y; t5.y += g1.w*d3.y; t5.z += g3.y*d3.y; t5.w += g3.w*d3.y;
			t6.x += g1.x*d3.z; t6.y += g1.z*d3.z; t6.z += g3.x*d3.z; t6.w += g3.z*d3.z;
			t7.x += g1.y*d3.w; t7.y += g1.w*d3.w; t7.z += g3.y*d3.w; t7.w += g3.w*d3.w;
		}
		winograd_f2x3_VT4(v0, v1, t0, t1);
		winograd_f2x3_VT4(v2, v3, t2, t3);
		winograd_f2x3_VT4(v4, v5, t4, t5);
		winograd_f2x3_VT4(v6, v7, t6, t7);
	}

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//STAGE2: for decompse  v, t(BEST)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_C6
#define XENO1_WINOGRAD_F2x3_KERNEL_C6

#define xeno_winograd_2x3_c6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_C6<LB, (1<<LB)-1, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.73562 msec, Performace = 11135.7 GFlop/s
template<int LB, int BLOCK_m1, int STEP, int STEP2, int STEP_m1, 
	int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_f2x3_kernel_C6(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	      float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(2 << LB) + 1];//{tgroup0}

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		Ds[buf][tx][ty        ] = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };//{d0.x, -d0.w, d1.x, -d1.w} => d0, d2
		Ds[buf][tx][ty + STEP2] = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };//{d0.y,  d0.z, d1.y,  d1.z} => d1, d3

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh*IC3 + wic)*OC;
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
		float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
		Gs[buf][ty][tx        ] = float4{ g0.x, g0.w, g1.x, g1.w };//g0, d2: {x, w} for v
		Gs[buf][ty][tx + STEP2] = float4{ g0.y, g0.z, g1.y, g1.z };//d1, d3: {y, z} for t
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				//g0 -> {g0, g1}, g2 -> {g1, g2}
				float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];//{x, w} for v;
				float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];//{x, w} for v;

				//{d0, d1} * {g0, g1, g2, g3}
				v0.x += g0.x*d0.x; v0.y += g0.z*d0.x; v0.z += g2.x*d0.x; v0.w += g2.z*d0.x;//d0.x = +
				v1.x += g0.y*d0.y; v1.y += g0.w*d0.y; v1.z += g2.y*d0.y; v1.w += g2.w*d0.y;//d0.y = -
				v2.x += g0.x*d0.z; v2.y += g0.z*d0.z; v2.z += g2.x*d0.z; v2.w += g2.z*d0.z;//d0.z = +
				v3.x += g0.y*d0.w; v3.y += g0.w*d0.w; v3.z += g2.y*d0.w; v3.w += g2.w*d0.w;//d0.w = -

				//{d2, d3} * {g0, g1, g2, g3}
				v4.x += g0.x*d2.x; v4.y += g0.z*d2.x; v4.z += g2.x*d2.x; v4.w += g2.z*d2.x;//d2.x = +
				v5.x += g0.y*d2.y; v5.y += g0.w*d2.y; v5.z += g2.y*d2.y; v5.w += g2.w*d2.y;//d2.y = -
				v6.x += g0.x*d2.z; v6.y += g0.z*d2.z; v6.z += g2.x*d2.z; v6.w += g2.z*d2.z;//d2.z = +
				v7.x += g0.y*d2.w; v7.y += g0.w*d2.w; v7.z += g2.y*d2.w; v7.w += g2.w*d2.w;//d2.w = -

				//==========================================================================
				float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];//{y, z} for t
				float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];//{y, z} for t

				//{d0, d1} * {g0, g1, g2, g3}
				t0.x += g1.x*d1.x; t0.y += g1.z*d1.x; t0.z += g3.x*d1.x; t0.w += g3.z*d1.x;
				t1.x += g1.y*d1.y; t1.y += g1.w*d1.y; t1.z += g3.y*d1.y; t1.w += g3.w*d1.y;
				t2.x += g1.x*d1.z; t2.y += g1.z*d1.z; t2.z += g3.x*d1.z; t2.w += g3.z*d1.z;
				t3.x += g1.y*d1.w; t3.y += g1.w*d1.w; t3.z += g3.y*d1.w; t3.w += g3.w*d1.w;

				//{d2, d3} * {g0, g1, g2, g3}
				t4.x += g1.x*d3.x; t4.y += g1.z*d3.x; t4.z += g3.x*d3.x; t4.w += g3.z*d3.x;
				t5.x += g1.y*d3.y; t5.y += g1.w*d3.y; t5.z += g3.y*d3.y; t5.w += g3.w*d3.y;
				t6.x += g1.x*d3.z; t6.y += g1.z*d3.z; t6.z += g3.x*d3.z; t6.w += g3.z*d3.z;
				t7.x += g1.y*d3.w; t7.y += g1.w*d3.w; t7.z += g3.y*d3.w; t7.w += g3.w*d3.w;
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			Ds[buf][tx][ty        ] = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };//{d0.x, -d0.w, d1.x, -d1.w} => d0, d2
			Ds[buf][tx][ty + STEP2] = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };//{d0.y,  d0.z, d1.y,  d1.z} => d1, d3

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = (fh*IC3 + wic)*OC;
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
			float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
			Gs[buf][ty][tx] = float4{ g0.x, g0.w, g1.x, g1.w };
			Gs[buf][ty][tx + STEP2] = float4{ g0.y, g0.z, g1.y, g1.z };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			//g0 -> {g0, g1}, g2 -> {g1, g2}
			float4 g0 = Gs[buf][ik][tx], g2 = Gs[buf][ik + STEP][tx];//{x, w} for v;
			float4 d0 = Ds[buf][ik][ty], d2 = Ds[buf][ik + STEP][ty];//{x, w} for v;

			//{d0, d1} * {g0, g1, g2, g3}
			v0.x += g0.x*d0.x; v0.y += g0.z*d0.x; v0.z += g2.x*d0.x; v0.w += g2.z*d0.x;//d0.x = +
			v1.x += g0.y*d0.y; v1.y += g0.w*d0.y; v1.z += g2.y*d0.y; v1.w += g2.w*d0.y;//d0.y = -
			v2.x += g0.x*d0.z; v2.y += g0.z*d0.z; v2.z += g2.x*d0.z; v2.w += g2.z*d0.z;//d0.z = +
			v3.x += g0.y*d0.w; v3.y += g0.w*d0.w; v3.z += g2.y*d0.w; v3.w += g2.w*d0.w;//d0.w = -

			//{d2, d3} * {g0, g1, g2, g3}
			v4.x += g0.x*d2.x; v4.y += g0.z*d2.x; v4.z += g2.x*d2.x; v4.w += g2.z*d2.x;//d2.x = +
			v5.x += g0.y*d2.y; v5.y += g0.w*d2.y; v5.z += g2.y*d2.y; v5.w += g2.w*d2.y;//d2.y = -
			v6.x += g0.x*d2.z; v6.y += g0.z*d2.z; v6.z += g2.x*d2.z; v6.w += g2.z*d2.z;//d2.z = +
			v7.x += g0.y*d2.w; v7.y += g0.w*d2.w; v7.z += g2.y*d2.w; v7.w += g2.w*d2.w;//d2.w = -

			//==========================================================================
			float4 g1 = Gs[buf][ik][tx + STEP2], g3 = Gs[buf][ik + STEP][tx + STEP2];//{y, z} for t
			float4 d1 = Ds[buf][ik][ty + STEP2], d3 = Ds[buf][ik + STEP][ty + STEP2];//{y, z} for t

			//{d0, d1} * {g0, g1, g2, g3}
			t0.x += g1.x*d1.x; t0.y += g1.z*d1.x; t0.z += g3.x*d1.x; t0.w += g3.z*d1.x;
			t1.x += g1.y*d1.y; t1.y += g1.w*d1.y; t1.z += g3.y*d1.y; t1.w += g3.w*d1.y;
			t2.x += g1.x*d1.z; t2.y += g1.z*d1.z; t2.z += g3.x*d1.z; t2.w += g3.z*d1.z;
			t3.x += g1.y*d1.w; t3.y += g1.w*d1.w; t3.z += g3.y*d1.w; t3.w += g3.w*d1.w;

			//{d2, d3} * {g0, g1, g2, g3}
			t4.x += g1.x*d3.x; t4.y += g1.z*d3.x; t4.z += g3.x*d3.x; t4.w += g3.z*d3.x;
			t5.x += g1.y*d3.y; t5.y += g1.w*d3.y; t5.z += g3.y*d3.y; t5.w += g3.w*d3.y;
			t6.x += g1.x*d3.z; t6.y += g1.z*d3.z; t6.z += g3.x*d3.z; t6.w += g3.z*d3.z;
			t7.x += g1.y*d3.w; t7.y += g1.w*d3.w; t7.z += g3.y*d3.w; t7.w += g3.w*d3.w;
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//STAGE2: for decompse  v, t(BEST)
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_C7
#define XENO1_WINOGRAD_F2x3_KERNEL_C7

#define xeno_winograd_2x3_c7(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_C7<LB, (1<<LB)-1, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.72823 msec, Performace = 11183.3 GFlop/s
template<int LB, int BLOCK_m1, int STEP, int STEP2, int STEP_m1,
	int IC, int IC2, int IC3, int IC4, int IC5>
	__global__ void xeno_winograd_f2x3_kernel_C7(
		cudaTextureObject_t X, int IH, int IW,
		const float* __restrict__ CW,//[FH, FW, IC, OC]
		float* __restrict__ Y, int OH_OW, int OW, int OC,
		int ph, int pw,//sh = sw = 1
		int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Gst[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dst[2][1 << LB][(1 << LB) + 1];

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		Dsv[buf][tx][ty] = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };//{d0.x, -d0.w, d1.x, -d1.w} => d0, d2
		Dst[buf][tx][ty] = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };//{d0.y,  d0.z, d1.y,  d1.z} => d1, d3

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh*IC3 + wic)*OC;
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
		float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
		Gsv[buf][ty][tx] = float4{ g0.x, g0.w, g1.x, g1.w };//g0, d2: {x, w} for v
		Gst[buf][ty][tx] = float4{ g0.y, g0.z, g1.y, g1.z };//d1, d3: {y, z} for t
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				//g0 -> {g0, g1}, g2 -> {g1, g2}
				float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
				float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

				//{d0, d1} * {g0, g1, g2, g3}
				v0.x += g0.x*d0.x; v0.y += g0.z*d0.x; v0.z += g2.x*d0.x; v0.w += g2.z*d0.x;//d0.x = +
				v1.x += g0.y*d0.y; v1.y += g0.w*d0.y; v1.z += g2.y*d0.y; v1.w += g2.w*d0.y;//d0.y = -
				v2.x += g0.x*d0.z; v2.y += g0.z*d0.z; v2.z += g2.x*d0.z; v2.w += g2.z*d0.z;//d0.z = +
				v3.x += g0.y*d0.w; v3.y += g0.w*d0.w; v3.z += g2.y*d0.w; v3.w += g2.w*d0.w;//d0.w = -

				//{d2, d3} * {g0, g1, g2, g3}
				v4.x += g0.x*d2.x; v4.y += g0.z*d2.x; v4.z += g2.x*d2.x; v4.w += g2.z*d2.x;//d2.x = +
				v5.x += g0.y*d2.y; v5.y += g0.w*d2.y; v5.z += g2.y*d2.y; v5.w += g2.w*d2.y;//d2.y = -
				v6.x += g0.x*d2.z; v6.y += g0.z*d2.z; v6.z += g2.x*d2.z; v6.w += g2.z*d2.z;//d2.z = +
				v7.x += g0.y*d2.w; v7.y += g0.w*d2.w; v7.z += g2.y*d2.w; v7.w += g2.w*d2.w;//d2.w = -

				//==========================================================================
				float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
				float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

				//{d0, d1} * {g0, g1, g2, g3}
				t0.x += g1.x*d1.x; t0.y += g1.z*d1.x; t0.z += g3.x*d1.x; t0.w += g3.z*d1.x;
				t1.x += g1.y*d1.y; t1.y += g1.w*d1.y; t1.z += g3.y*d1.y; t1.w += g3.w*d1.y;
				t2.x += g1.x*d1.z; t2.y += g1.z*d1.z; t2.z += g3.x*d1.z; t2.w += g3.z*d1.z;
				t3.x += g1.y*d1.w; t3.y += g1.w*d1.w; t3.z += g3.y*d1.w; t3.w += g3.w*d1.w;

				//{d2, d3} * {g0, g1, g2, g3}
				t4.x += g1.x*d3.x; t4.y += g1.z*d3.x; t4.z += g3.x*d3.x; t4.w += g3.z*d3.x;
				t5.x += g1.y*d3.y; t5.y += g1.w*d3.y; t5.z += g3.y*d3.y; t5.w += g3.w*d3.y;
				t6.x += g1.x*d3.z; t6.y += g1.z*d3.z; t6.z += g3.x*d3.z; t6.w += g3.z*d3.z;
				t7.x += g1.y*d3.w; t7.y += g1.w*d3.w; t7.z += g3.y*d3.w; t7.w += g3.w*d3.w;
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			Dsv[buf][tx][ty] = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };//{d0.x, -d0.w, d1.x, -d1.w} => d0, d2
			Dst[buf][tx][ty] = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };//{d0.y,  d0.z, d1.y,  d1.z} => d1, d3

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = (fh*IC3 + wic)*OC;
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float4 g0; WinoGrad_produce_G(g0, w0.x, w1.x, w2.x);
			float4 g1; WinoGrad_produce_G(g1, w0.y, w1.y, w2.y);
			Gsv[buf][ty][tx] = float4{ g0.x, g0.w, g1.x, g1.w };
			Gst[buf][ty][tx] = float4{ g0.y, g0.z, g1.y, g1.z };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
			float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

			//{d0, d1} * {g0, g1, g2, g3}
			v0.x += g0.x*d0.x; v0.y += g0.z*d0.x; v0.z += g2.x*d0.x; v0.w += g2.z*d0.x;//d0.x = +
			v1.x += g0.y*d0.y; v1.y += g0.w*d0.y; v1.z += g2.y*d0.y; v1.w += g2.w*d0.y;//d0.y = -
			v2.x += g0.x*d0.z; v2.y += g0.z*d0.z; v2.z += g2.x*d0.z; v2.w += g2.z*d0.z;//d0.z = +
			v3.x += g0.y*d0.w; v3.y += g0.w*d0.w; v3.z += g2.y*d0.w; v3.w += g2.w*d0.w;//d0.w = -

			//{d2, d3} * {g0, g1, g2, g3}
			v4.x += g0.x*d2.x; v4.y += g0.z*d2.x; v4.z += g2.x*d2.x; v4.w += g2.z*d2.x;//d2.x = +
			v5.x += g0.y*d2.y; v5.y += g0.w*d2.y; v5.z += g2.y*d2.y; v5.w += g2.w*d2.y;//d2.y = -
			v6.x += g0.x*d2.z; v6.y += g0.z*d2.z; v6.z += g2.x*d2.z; v6.w += g2.z*d2.z;//d2.z = +
			v7.x += g0.y*d2.w; v7.y += g0.w*d2.w; v7.z += g2.y*d2.w; v7.w += g2.w*d2.w;//d2.w = -

			//==========================================================================
			float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
			float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

			//{d0, d1} * {g0, g1, g2, g3}
			t0.x += g1.x*d1.x; t0.y += g1.z*d1.x; t0.z += g3.x*d1.x; t0.w += g3.z*d1.x;
			t1.x += g1.y*d1.y; t1.y += g1.w*d1.y; t1.z += g3.y*d1.y; t1.w += g3.w*d1.y;
			t2.x += g1.x*d1.z; t2.y += g1.z*d1.z; t2.z += g3.x*d1.z; t2.w += g3.z*d1.z;
			t3.x += g1.y*d1.w; t3.y += g1.w*d1.w; t3.z += g3.y*d1.w; t3.w += g3.w*d1.w;

			//{d2, d3} * {g0, g1, g2, g3}
			t4.x += g1.x*d3.x; t4.y += g1.z*d3.x; t4.z += g3.x*d3.x; t4.w += g3.z*d3.x;
			t5.x += g1.y*d3.y; t5.y += g1.w*d3.y; t5.z += g3.y*d3.y; t5.w += g3.w*d3.y;
			t6.x += g1.x*d3.z; t6.y += g1.z*d3.z; t6.z += g3.x*d3.z; t6.w += g3.z*d3.z;
			t7.x += g1.y*d3.w; t7.y += g1.w*d3.w; t7.z += g3.y*d3.w; t7.w += g3.w*d3.w;
		}
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//Decompose shared memory for t, v
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_C8
#define XENO1_WINOGRAD_F2x3_KERNEL_C8

#define xeno_winograd_2x3_c8(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_C8<LB, (1<<LB)-1, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 9, Time = 1.73298 msec, Performace = 11152.7 GFlop/s
template<int LB, int BLOCK_m1, int STEP, int STEP2, int STEP_m1,
	int IC, int IC2, int IC3, int IC4, int IC5>
	__global__ void xeno_winograd_f2x3_kernel_C8(
		cudaTextureObject_t X, int IH, int IW,
		const float* __restrict__ CW,//[FH, FW, IC, OC]
		float* __restrict__ Y, int OH_OW, int OW, int OC,
		int ph, int pw,//sh = sw = 1
		int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Gst[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dst[2][1 << LB][(1 << LB) + 1];

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	j0 = j0 * OC + oc0;

#pragma once //compute area---------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 1 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		const int xic = (tx & STEP_m1);//with the same ty
		const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		Dsv[buf][tx][ty] = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
		Dst[buf][tx][ty] = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

		//load 2 group from CW
		const int wic = (ty & STEP_m1);//with the same tx
		const int woffset = (fh*IC3 + wic)*OC;
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
		Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
				float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

				winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2); 
				winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

				float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
				float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

				winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
				winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
			}
			buf ^= 1;

			//load 1 group from X
			const int xic = oic + (tx & STEP_m1);
			const int xoffset = ((tn0*IH + tih0 + fh)*IW + tiw0)*IC + xic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			Dsv[buf][tx][ty] = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
			Dst[buf][tx][ty] = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

			//load 2 group from CW
			const int wic = oic + (ty & STEP_m1);
			const int woffset = (fh*IC3 + wic)*OC;
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
			Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
			float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

			winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
			winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

			float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
			float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

			winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
			winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
		}
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//[STANDARD]
//Decompose shared memory for t, v
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_C9
#define XENO1_WINOGRAD_F2x3_KERNEL_C9

#define xeno_winograd_2x3_c9(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_f2x3_kernel_C9<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size =  9, Time = 1.73173 msec, Performace = 11160.7 GFlop/s
//LB = 4: Size = 18, Time = 3.45507 msec, Performace = 11187.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5> 
__global__ void xeno_winograd_f2x3_kernel_C9(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Gst[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dst[2][1 << LB][(1 << LB) + 1];

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += (ty & STEP_m1)*OC + oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);
	j0 = j0 * OC + oc0;

#pragma once //compute area------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from X
		bool lh0 = (tih0 >= -fh) && (tih0 + fh < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		Dsv[buf][tx][ty] = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
		Dst[buf][tx][ty] = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

		//load 2 group from CW
		const int woffset = fh * IC3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
		Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
				float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

				winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
				winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

				float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
				float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

				winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
				winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
			}
			buf ^= 1;

			//load 1 group from X
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			Dsv[buf][tx][ty] = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
			Dst[buf][tx][ty] = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };

			//load 2 group from CW
			const int woffset = (fh*IC3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
			Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
			float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

			winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
			winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

			float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
			float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

			winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
			winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
		}
	}
	
	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), IC % (BLOCK_SIZE / 2) == 0
//LB = 4, IC % 8 == 0
#ifndef XENO1_WINOGRAD_F2x3_KERNEL_C10
#define XENO1_WINOGRAD_F2x3_KERNEL_C10

#define xeno_winograd_2x3_c10(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	xeno_winograd_2x3_kernel_c10<LB, (1<<LB>>1), (1<<LB), (1<<LB>>1)-1,\
		IC, (IC << 1), (IC * 3), (IC << 2), (IC * 5)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, OC, ph, pw, oc_index, j_index)

//LB = 4: Size =  9, Time = 1.73173 msec, Performace = 11160.7 GFlop/s
//LB = 4: Size = 18, Time = 3.45507 msec, Performace = 11187.8 GFlop/s
template<int LB, int STEP, int STEP2, int STEP_m1, int IC, int IC2, int IC3, int IC4, int IC5>
__global__ void xeno_winograd_2x3_kernel_c10(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW, int OC,
	int ph, int pw,//sh = sw = 1
	int oc_index, int j_index)
{
	const int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dsv[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Gst[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Dst[2][1 << LB][(1 << LB) + 1];

	//compute 4*8 results
	float4 v0 = F32_4_0, v1 = F32_4_0, v2 = F32_4_0, v3 = F32_4_0;
	float4 v4 = F32_4_0, v5 = F32_4_0, v6 = F32_4_0, v7 = F32_4_0;
	float4 t0 = F32_4_0, t1 = F32_4_0, t2 = F32_4_0, t3 = F32_4_0;
	float4 t4 = F32_4_0, t5 = F32_4_0, t6 = F32_4_0, t7 = F32_4_0;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += (ty & STEP_m1)*OC + oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tih0 = toh0 - ph, tiw0 = tow0 - pw;
	const int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + (tx & STEP_m1);
	j0 = j0 * OC + oc0;

#pragma once //compute area------------------------------------------
	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from CW
		const int woffset = fh * IC3 * OC;//with the same tx
		float2 w0 = *(float2*)(CW + woffset);
		float2 w1 = *(float2*)(CW + woffset + Wstride);
		float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
		float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
		float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
		Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
		Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };

		//load 2 group from X
		bool ly0 = (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = (tiw0 >= -5) && (tiw0 + 5 < IW);
		const int xoffset = X0 + fh * IW * IC;//with the same ty
		float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
		float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
		float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
		float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
		float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
		float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
		Dsv[buf][tx][ty] = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
		Dst[buf][tx][ty] = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };
		__syncthreads();

		for (int oic = STEP; oic < IC; oic += STEP)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
				float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

				winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
				winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

				float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
				float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

				winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
				winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
				winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
			}
			buf ^= 1;

			//load 2 group from CW
			const int woffset = (fh*IC3 + oic) * OC;//fh, ic, fw, oc
			float2 w0 = *(float2*)(CW + woffset);
			float2 w1 = *(float2*)(CW + woffset + Wstride);
			float2 w2 = *(float2*)(CW + woffset + (Wstride << 1));
			float gst0 = 0.5f*(w0.x + w1.x + w2.x), gst1 = gst0 - w1.x;
			float gst2 = 0.5f*(w0.y + w1.y + w2.y), gst3 = gst2 - w1.y;
			Gsv[buf][ty][tx] = float4{ w0.x, w2.x, w0.y, w2.y };
			Gst[buf][ty][tx] = float4{ gst0, gst1, gst2, gst3 };

			//load 1 group from X
			const int xoffset = X0 + fh * IW * IC + oic;
			float x0 = tex1Dfetch<float>(X, IF_int(ly0, xoffset, -1));
			float x1 = tex1Dfetch<float>(X, IF_int(ly1, xoffset + IC, -1));
			float x2 = tex1Dfetch<float>(X, IF_int(ly2, xoffset + IC2, -1));
			float x3 = tex1Dfetch<float>(X, IF_int(ly3, xoffset + IC3, -1));
			float x4 = tex1Dfetch<float>(X, IF_int(ly4, xoffset + IC4, -1));
			float x5 = tex1Dfetch<float>(X, IF_int(ly5, xoffset + IC5, -1));
			Dsv[buf][tx][ty] = float4{ x0 - x2, x3 - x1, x2 - x4, x5 - x3 };
			Dst[buf][tx][ty] = float4{ x1 + x2, x2 - x1, x3 + x4, x4 - x3 };
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gsv[buf][ik][tx], g2 = Gsv[buf][ik + STEP][tx];//{x, w} for v;
			float4 d0 = Dsv[buf][ik][ty], d2 = Dsv[buf][ik + STEP][ty];//{x, w} for v;

			winograd_f2x3_simdMM4(v0, v1, d0.x, d0.y, g0, g2);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v2, v3, d0.z, d0.w, g0, g2);
			winograd_f2x3_simdMM4(v4, v5, d2.x, d2.y, g0, g2);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(v6, v7, d2.z, d2.w, g0, g2);

			float4 g1 = Gst[buf][ik][tx], g3 = Gst[buf][ik + STEP][tx];//{y, z} for t
			float4 d1 = Dst[buf][ik][ty], d3 = Dst[buf][ik + STEP][ty];//{y, z} for t

			winograd_f2x3_simdMM4(t0, t1, d1.x, d1.y, g1, g3);//{d0, d1} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t2, t3, d1.z, d1.w, g1, g3);
			winograd_f2x3_simdMM4(t4, t5, d3.x, d3.y, g1, g3);//{d2, d3} * {g0, g1, g2, g3}
			winograd_f2x3_simdMM4(t6, t7, d3.z, d3.w, g1, g3);
		}
		buf ^= 1;
	}

	winograd_f2x3_VT4(v0, v1, t0, t1);
	winograd_f2x3_VT4(v2, v3, t2, t3);
	winograd_f2x3_VT4(v4, v5, t4, t5);
	winograd_f2x3_VT4(v6, v7, t6, t7);

	const int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	const int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0; *(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2; *(float4*)(Y + j3) = v3;
	*(float4*)(Y + j4) = v4; *(float4*)(Y + j5) = v5;
	*(float4*)(Y + j6) = v6; *(float4*)(Y + j7) = v7;
}


#endif