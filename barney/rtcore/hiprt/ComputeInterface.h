// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#pragma once

#include "rtcore/cudaCommon/ComputeInterface.h"


# define __rtc_global __global__
# define __rtc_launch(myRTC,kernel,nb,bs,...)                           \
  { rtc::SetActiveGPU forDuration(myRTC); if (nb) kernel<<<nb,bs,0,myRTC->stream>>>(rtc::ComputeInterface(), __VA_ARGS__); }
