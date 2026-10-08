// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#include "rtcore/hiprt/Device.h"
#include "rtcore/hiprt/Group.h"
#include "rtcore/hiprt/TraceKernel.h"
#include "rtcore/hiprt/Geom.h"
#include "rtcore/hiprt/GeomType.h"
#include "rtcore/hiprt/Buffer.h"

#include <hiprt/hiprt.h>
#include <stdexcept>

namespace BARNEY_NS {
  namespace rtc {

    int physicalDeviceCount()
    {
      int count = 0;
      BARNEY_CUDA_CALL(GetDeviceCount(&count));
      return count;
    }

    static void hiprtCheck(hiprtError e, const char *where)
    {
      if (e != hiprtSuccess)
        throw std::runtime_error(std::string("HIPRT error in ")+where
                                 +" code "+std::to_string((int)e));
    }
#define HIPRT_CALL(call,where) rtc::hiprtCheck(call,where)

    Device::Device(int physicalGPU)
      : CudaDeviceBase(physicalGPU)
    {
      SetActiveGPU forDuration(this);

      // Bind HIPRT to THIS device's HIP context/device so every BVH build and
      // the trace kernels see the same device pointers barney allocates (this
      // avoids a "fresh context => degenerate BVH, zero hits" failure mode). These
      // are HIP driver-API calls used only on the HIP build, so they are spelled
      // directly rather than through the cuda_to_hip compat aliases.
      hipDevice_t hdev = 0;
      (void)hipDeviceGet(&hdev, physicalID);
      hipCtx_t hctx = nullptr;
      if (hipCtxGetCurrent(&hctx) != hipSuccess || hctx == nullptr) {
        (void)hipDevicePrimaryCtxRetain(&hctx, hdev);
        (void)hipCtxSetCurrent(hctx);
      }

      hiprtContextCreationInput ci{};
      ci.deviceType = hiprtDeviceAMD;
      ci.ctxt   = hctx;
      ci.device = hdev;
      HIPRT_CALL(hiprtCreateContext(HIPRT_API_VERSION, ci, hiprtCtx),
                 "hiprtCreateContext");
    }

    Device::~Device()
    {
      if (hiprtCtx)
        hiprtDestroyContext(hiprtCtx);
    }

    rtc::AccelHandle getAccelHandle(Group *ig)
    { return ig->getDD(); }

    void Device::freeGroup(Group *g)
    { delete g; }

    Denoiser *Device::createDenoiser()
    { return nullptr; }

    Buffer *Device::createBuffer(size_t numBytes, const void *initValues)
    { return new Buffer(this,numBytes,initValues); }

    void Device::freeBuffer(Buffer *b)
    { delete b; }

    void Device::freeGeomType(GeomType *gt)
    { delete gt; }

    void Device::freeGeom(Geom *g)
    { delete g; }

    Group *Device::createTrianglesGroup(const std::vector<Geom *> &geoms)
    { return new TrianglesGeomGroup(this,geoms); }

    Group *Device::createUserGeomsGroup(const std::vector<Geom *> &geoms)
    { return new UserGeomGroup(this,geoms); }

    Group *Device::createInstanceGroup(const std::vector<Group *> &groups,
                                       const std::vector<int>      &instIDs,
                                       const std::vector<affine3f> &xfms)
    { return new InstanceGroup(this,groups,instIDs,xfms); }

    void Device::buildPipeline() {}
    void Device::buildSBT() {}

  }
}
