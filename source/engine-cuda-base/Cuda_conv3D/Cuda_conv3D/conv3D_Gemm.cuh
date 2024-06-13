#pragma once

#ifndef CONV_3D_GEMM_H
#define CONV_3D_GEMM_H

//======[GEMM]=================================================
#include "conv3D_Gemm_kernel.cuh"
#include "conv3D_Gemm_kernel_EX.cuh"
#include "conv3D_Gemm_kernel_EX2.cuh"
#include "conv3D_Gemm_kernel_texture.cuh"
#include "conv3D_Gemm_kernel_texture2.cuh"
#include "conv3D_Gemm_kernel_no_padding.cuh"
#include "conv3D_Gemm_kernel_no_padding_EX.cuh"
#include "conv3D_Gemm_sernel.cuh"
#include "conv3D_kernel_W1.cuh"
//#include "conv3D_sernel_W1.cuh"

//======[GEMM V2]==============================================
#include "conv3D_GemmV2_kernel.cuh"

#endif