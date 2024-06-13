#pragma once

#ifndef FRAME_H
#define FRAME_H

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
#include "texture_fetch_functions.h"

#include <iostream>
#include <jni.h>
#include <math.h>
#include <time.h>

//#define COMPLIE 1

//#define ENABLE_CONV3D_GEMM_UERNEL_32X32_RC 1

#define ENBALE_CONV3D_WINOGRAD_F6X3R_CHANNEL_TEMPLATE 1//F(6, 3)
#define ENBALE_CONV3D_WINOGRAD_F5X4R_CHANNEL_TEMPLATE 1//F(5, 4)
#define ENBALE_CONV3D_WINOGRAD_F3X6R_CHANNEL_TEMPLATE 1//F(3, 6)
#define ENBALE_CONV3D_WINOGRAD_F2X7R_CHANNEL_TEMPLATE 1//F(2, 7)

#define ENBALE_CONV3D_WINOGRAD_FAX7R_CHANNEL_TEMPLATE 1//F(10, 7)
#define ENBALE_CONV3D_WINOGRAD_F9X8R_CHANNEL_TEMPLATE 1//F( 9, 8)
#define ENBALE_CONV3D_WINOGRAD_F8X9R_CHANNEL_TEMPLATE 1//F( 8, 9)

#endif