// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#pragma once

#include "rtcore/cudaCommon/Device.h"

// host-side HIPRT API; the device-side traversal lives in TraceInterface.h
#include <hiprt/hiprt.h>

namespace BARNEY_NS {
  namespace rtc {

    int physicalDeviceCount();


    // forward decls; the geometry/SBT/program data model is backend-generic
    // compute (no OptiX, no HIPRT) and lives in this backend's Geom/GeomType/
    // Buffer (CudaDeviceBase-based). Only BVH build + ray traversal differ
    // (HIPRT here, the cuBQL software walk in the cuda backend), so only Group
    // and the device-side TraceInterface are HIPRT-specific.
    struct Buffer;
    struct Geom;
    struct GeomType;
    struct TrianglesGeom;
    struct UserGeom;
    struct UserGeomType;
    struct TrianglesGeomType;
    struct Group;
    struct Denoiser;
    struct TraceKernel2D;



    struct Device : public CudaDeviceBase {
      Device(int physicalGPU);
      virtual ~Device();

      std::string toString() const
      { return "rtc::Device(physical="+std::to_string(physicalID)+")"; }

      /*! the HIPRT context, created over this Device's HIP device + stream;
          shared by every Group's BVH build and the trace kernels so they see
          the same device pointers (binding to this Device's HIP context avoids
          a "fresh context => degenerate BVH" failure mode). */
      hiprtContext hiprtCtx = nullptr;

      void freeGeomType(GeomType *);
      void freeGeom(Geom *);

      Group *createTrianglesGroup(const std::vector<Geom *> &geoms);
      Group *createUserGeomsGroup(const std::vector<Geom *> &geoms);
      Group *createInstanceGroup(const std::vector<Group *> &groups,
                                 const std::vector<int>      &instIDs,
                                 const std::vector<affine3f> &xfms);
      void freeGroup(Group *);
      void buildPipeline();
      void buildSBT();
      Buffer *createBuffer(size_t numBytes, const void *initValues = 0);
      void freeBuffer(Buffer *);
      Denoiser *createDenoiser();
    };

    /*! HIPRT has no denoiser; mirror the cuda backend's no-op passthrough (the
        renderer runs un-denoised, exactly as on the software backend). OIDN-GPU
        on ROCm is a deferred enhancement. */
    struct Denoiser
    {
      Denoiser(Device* device) : device(device) {}
      ~Denoiser() = default;
      void resize(vec2i dims) { outputDims = dims; }
      void run(float blendFactor) {}
      Device* const device;

      vec4f *in_rgba = 0;
      vec4f *out_rgba = 0;
      vec3f *in_normal = 0;

      bool upscaleMode = false;
      vec2i outputDims = {0,0};
    };

    rtc::AccelHandle getAccelHandle(Group *ig);
  }
}
