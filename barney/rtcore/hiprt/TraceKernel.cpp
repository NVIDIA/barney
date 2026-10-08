// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>
//
// HIPRT func-table callbacks and the host-side trace-kernel launcher.
//
// HIPRT's inline device traversal (pulled in via <hiprt/impl/hiprt_device_impl.h>)
// references two user-provided device functions, intersectFunc() and filterFunc().
// They are the hook by which barney's per-geometry intersect/anyHit programs run
// during traversal: HIPRT walks the BVH and, for each custom-geom candidate, calls
// intersectFunc -> the intersect thunk (which runs barney's intersect program once
// and folds the hit), and for each triangle candidate calls filterFunc -> the
// filter thunk (triangle test + anyHit + fold). We route both into barney's
// existing function-pointer SBT dispatch via the TraceInterface payload, so HIPRT
// supplies the BVH/traversal while barney supplies the shading -- the "keep the
// megakernel" first cut.
//
// This TU is compiled -fgpu-rdc and device-linked into barney_hiprt_programs, so
// these device symbols resolve for the trace kernel in the same module.

#include "rtcore/hiprt/Device.h"
#include "rtcore/hiprt/TraceKernel.h"

#define BARNEY_DEVICE_PROGRAM 1
#include "rtcore/hiprt/TraceInterface.h"

#include <hiprt/impl/hiprt_device_impl.h>

// ------------------------------------------------------------------
// device-side func-table callbacks (required by hiprt_device_impl.h)
// ------------------------------------------------------------------
HIPRT_DEVICE bool intersectFunc(uint32_t          /*geomType*/,
                                uint32_t          /*rayType*/,
                                const hiprtFuncTableHeader& /*tableHeader*/,
                                const hiprtRay&   /*ray*/,
                                void*             payload,
                                hiprtHit&         hit)
{
  // custom (user) geometry: run barney's intersect program for this candidate
  // prim exactly once and fold the hit. Returns false unconditionally so HIPRT
  // discards this leaf and keeps enumerating (the filter is never called for
  // custom geoms, so the hit-writing intersect runs only once per prim).
  auto *ti = (::BARNEY_NS::rtc::TraceInterface *)payload;
  return ti->hiprtIntersectThunk(hit);
}

HIPRT_DEVICE bool filterFunc(uint32_t          /*geomType*/,
                             uint32_t          /*rayType*/,
                             const hiprtFuncTableHeader& /*tableHeader*/,
                             const hiprtRay&   /*ray*/,
                             void*             payload,
                             const hiprtHit&   hit)
{
  // any-hit (triangle geometry only): recompute barycentrics, run the geom's
  // anyHit program, and fold this hit into barney's accepted state, then ALWAYS
  // reject (return true) so HIPRT enumerates every hit along the ray and we keep
  // the closest non-rejected one ourselves. This is the record-and-ignore
  // filter pattern. (Custom geoms fold in the intersect thunk and never reach
  // here, so their hit-writing intersect runs only once per prim.)
  auto *ti = (::BARNEY_NS::rtc::TraceInterface *)payload;
  ti->hiprtFilterThunk(hit);
  return true;
}

namespace BARNEY_NS {
  namespace rtc {

    // ------------------------------------------------------------------
    // a process-wide 1x1 HIPRT func table; barney carries all per-geom state in
    // the SBT, so HIPRT needs no per-(geomType,rayType) data -- the callbacks
    // above reach everything through the TraceInterface payload. Created lazily
    // per device and cached.
    void *getFuncTable(Device *device)
    {
      static thread_local hiprtFuncTable table = nullptr;
      static thread_local hiprtContext   forCtx = nullptr;
      if (table && forCtx == device->hiprtCtx) return table;
      hiprtCreateFuncTable(device->hiprtCtx, 1, 1, table);
      hiprtFuncDataSet fds{};
      hiprtSetFuncTable(device->hiprtCtx, table, 0, 0, fds);
      forCtx = device->hiprtCtx;
      return table;
    }

    // ------------------------------------------------------------------
    TraceKernel2D::TraceKernel2D(Device *device,
                                 size_t sizeOfLP,
                                 TraceLaunchFct traceLaunchFct)
      : device(device), traceLaunchFct(traceLaunchFct), sizeOfLP(sizeOfLP)
    {
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL(Malloc((void **)&d_lpData,sizeOfLP));
    }

    TraceKernel2D::~TraceKernel2D()
    {
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL_NOTHROW(Free(d_lpData));
    }

    void TraceKernel2D::launch(vec2i launchDims, const void *kernelData)
    {
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL(MemcpyAsync(d_lpData,kernelData,sizeOfLP,
                                   cudaMemcpyDefault,device->stream));
      traceLaunchFct(device,launchDims,d_lpData);
    }

  }
}
