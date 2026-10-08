// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/common/rtcore-common.h"
#include "rtcore/cudaCommon/cuda-common.h"
#include "rtcore/cuda/AppInterface.h"
#include "rtcore/cudaCommon/ComputeInterface.h"
// only pipeline programs should ever see trace functions
#if BARNEY_DEVICE_PROGRAM
// #if RTC_DEVICE_CODE 
# include "rtcore/cuda/TraceInterface.h"
#endif

/*! tells the barney/native/ and barney/anari/ components that use
    this backend which kind of backend it is. this allows downstream
    layers to enable/disable stuff that should only be applicable to
    this backend (such as anari nv-framebuffer extensions */
#if BARNEY_HAVE_CUDA
# define BARNEY_RTC_CUDA 1
#endif


