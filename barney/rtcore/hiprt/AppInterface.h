// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#pragma once

#include "rtcore/hiprt/Device.h"
#include "rtcore/hiprt/Buffer.h"
#include "rtcore/cudaCommon/ComputeKernel.h"
#include "rtcore/hiprt/TraceKernel.h"
#include "rtcore/hiprt/Geom.h"
#include "rtcore/hiprt/Group.h"
#include "rtcore/cudaCommon/TextureData.h"
#include "rtcore/cudaCommon/Texture.h"


#define RTC_IMPORT_USER_GEOM(moduleName,typeName,DD,has_ah,has_ch)      \
  extern ::BARNEY_NS::rtc::GeomType *createGeomType_##typeName(::BARNEY_NS::rtc::Device *);

#define RTC_IMPORT_TRIANGLES_GEOM(moduleName,typeName,DD,has_ah,has_ch) \
  extern rtc::GeomType *createGeomType_##typeName(rtc::Device *);
