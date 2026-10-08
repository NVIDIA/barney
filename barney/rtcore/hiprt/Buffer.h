// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#pragma once

#include "rtcore/cudaCommon/Device.h"

namespace BARNEY_NS {
  namespace rtc {

    // a plain device buffer; identical to the cuda backend's Buffer but bound to
    // CudaDeviceBase so the hiprt and cuda backends do not share a Device
    // type (each backend owns its rtc::Device).
    struct Buffer {
      Buffer(CudaDeviceBase *device,
             size_t numBytes,
             const void *initValues);
      virtual ~Buffer();

      void *getDD() const { return d_data; }
      void upload(const void *data, size_t numBytes, size_t offset=0);
      void resize(size_t newNumBytes);
      void *d_data = 0;
      CudaDeviceBase *const device;
    };

  }
}
