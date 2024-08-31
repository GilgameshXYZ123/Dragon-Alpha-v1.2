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

//#include "cooperative_groups.h"
//#include "cooperative_groups/memcpy_async.h"

#define COMPILE 1

#define ENABLE_DECONV3D_ZERO_PADDING_S1_32X32C 1

#define ENABLE_DECONV3D_WINOGRAD_F6X3_CHANNEL_TEMPLATE 1
#define ENABLE_DECONV3D_WINOGRAD_F3X6_CHANNEL_TEMPLATE 1
#define ENABLE_DECONV3D_WINOGRAD_F2X7_CHANNEL_TEMPLATE 1

#define ENABLE_DECONV3D_WINOGRAD_FAX7_CHANNEL_TEMPLATE 1
#define ENABLE_DECONV3D_WINOGRAD_F9X8_CHANNEL_TEMPLATE 1
#define ENABLE_DECONV3D_WINOGRAD_F8X9_CHANNEL_TEMPLATE 1

#endif