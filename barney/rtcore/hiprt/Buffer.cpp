// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#include "rtcore/hiprt/Buffer.h"

namespace BARNEY_NS {
  namespace rtc {


    Buffer::Buffer(CudaDeviceBase *device,
                   size_t numBytes,
                   const void *initValues)
      : device(device)
    {
      if (numBytes == 0) return;
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL(Malloc((void**)&d_data,numBytes));
      if (initValues)
        BARNEY_CUDA_CALL(Memcpy(d_data,initValues,numBytes,cudaMemcpyDefault));
    }

    Buffer::~Buffer()
    {
      if (!d_data) return;
      BARNEY_CUDA_CALL_NOTHROW(Free(d_data));
    }

    void Buffer::upload(const void *data, size_t numBytes, size_t offset)
    {
      if (!d_data) return;
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL(Memcpy(((char *)d_data)+offset,data,numBytes,cudaMemcpyDefault));
    }

    void Buffer::resize(size_t numBytes)
    {
      SetActiveGPU forDuration(device);
      if (d_data)
        BARNEY_CUDA_CALL(Free(d_data));
      BARNEY_CUDA_CALL(Malloc((void**)&d_data,numBytes));
    }

  }
}
