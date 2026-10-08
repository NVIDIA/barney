// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>
//
// Device-side ray-tracing interface for the HIPRT (hardware-RT) backend. The
// shading surface (getPrimitiveIndex/getInstanceID/getWorldRayOrigin/the
// transform helpers/...) is identical to the software (cuda) backend; the only
// difference is traceRay(), which walks HIPRT's BVH (a hiprtScene built by
// rtcore/hiprt/Group) instead of the cuBQL software two-level walk, then runs
// barney's existing per-geometry function-pointer dispatch (intersect/anyHit/
// closestHit programs) on the returned hit. This keeps barney's megakernel +
// function-pointer SBT model; HIPRT supplies BVH build + traversal + hit
// enumeration only.
//
// HIPRT device traversal is provided by including <hiprt/impl/hiprt_device_impl.h>
// (the full inline traversal implementation), so the trace kernel links against
// libhiprt without going through HIPRT's runtime JIT. The kernel TU must also
// define the func-table callbacks intersectFunc()/filterFunc() (see the
// RTC_HIPRT_TRACEKERNEL macro) that route HIPRT's custom-primitive intersection
// and any-hit filtering back into barney's intersect/anyHit programs.

#pragma once

#include "rtcore/cudaCommon/ComputeInterface.h"
#include "rtcore/hiprt/TraceKernel.h"
#include "rtcore/hiprt/Group.h"
#include "rtcore/hiprt/ProgramInterface.h"
#include "rtcore/cudaCommon/Device.h"

#if RTC_DEVICE_CODE
# include <hiprt/hiprt_device.h>
#endif

namespace BARNEY_NS {
  namespace rtc {

    inline __device__
    bool TraceInterface::intersectTriangle(const vec3f v0,
                                           const vec3f v1,
                                           const vec3f v2,
                                           bool dbg)
    {
      // barney's triangle test, reused verbatim from the software backend so
      // the barycentrics/tMax convention matches the rest of barney exactly.
      vec3f e1 = v1-v0;
      vec3f e2 = v2-v0;
      vec3f N = cross(e1,e2);
      if (fabsf(dot(object.dir,N)) < 1e-12f) return false;
      float t = -dot(object.org-v0,N)/dot(object.dir,N);
      if (t <= 0.f || t >= current.tMax) return false;
      vec3f P = object.org - v0 + t*object.dir;
      float e1u,e2u,Pu, e1v,e2v,Pv;
      if (fabsf(N.x) >= max(fabsf(N.y),fabsf(N.z))) {
        e1u = e1.y; e2u = e2.y; Pu = P.y;
        e1v = e1.z; e2v = e2.z; Pv = P.z;
      } else if (fabsf(N.y) > fabsf(N.z)) {
        e1u = e1.x; e2u = e2.x; Pu = P.x;
        e1v = e1.z; e2v = e2.z; Pv = P.z;
      } else {
        e1u = e1.x; e2u = e2.x; Pu = P.x;
        e1v = e1.y; e2v = e2.y; Pv = P.y;
      }
      auto det = [](float a, float b, float c, float d) -> float
      { return a*d - c*b; };
      if (det(e1u,e1v,e2u,e2v) == 0.f) return false;
      float u = det(Pu,e2u,Pv,e2v)/det(e1u,e2u,e1v,e2v);
      float v = det(e1u,Pu,e1v,Pv)/det(e1u,e2u,e1v,e2v);
      if ((u < 0.f) || (v < 0.f) || ((u+v) >= 1.f)) return false;
      current.triangleBarycentrics = vec2f{ u,v };
      current.tMax = t;
      return true;
    }

#if RTC_DEVICE_CODE
    /*! set up the per-hit instance/object-space + SBT state for the candidate
        (instanceID, geometry-local primID) HIPRT just produced. Mirrors the
        cuda backend's intersectPrim preamble. Returns the SBT header. */
    inline __device__
    Geom::SBTHeader *TraceInterface::setupHit(uint32_t instID, uint32_t localPrimID)
    {
      InstanceGroup::DeviceRecord *model
        = (InstanceGroup::DeviceRecord *)hiprtWorld;
      currentInstance = model->instanceRecords + instID;
      current.instID = instID;
      object.org = xfmPoint (currentInstance->worldToObjectXfm, world.org);
      object.dir = xfmVector(currentInstance->worldToObjectXfm, world.dir);

      const GeomGroup::DeviceRecord *group = &currentInstance->group;
      GeomGroup::Prim prim = group->prims[localPrimID];
      current.primID = prim.primID;
      current.geomID = prim.geomID;
      current.tMax   = accepted.tMax;

      uint8_t *geomSBT = group->sbt + prim.geomID * group->sbtEntrySize;
      Geom::SBTHeader *header = (Geom::SBTHeader *)geomSBT;
      this->geomData = (header+1);
      return header;
    }

    /*! fold the hit currently described by `current`/`header` into `accepted`,
        running the geom's anyHit program first (which may reject via
        ignoreIntersection). Shared by the custom-geom intersect thunk and the
        triangle filter thunk so a hit is committed exactly once, with the same
        anyHit/reject/closest-wins logic as the software backend's intersectPrim. */
    inline __device__
    void TraceInterface::foldAcceptedHit(Geom::SBTHeader *header)
    {
      rejectThisHit = false;
      if (header->ah) header->ah(*this);
      if (!rejectThisHit) {
        accepted.tMax = current.tMax;
        accepted.primID = current.primID;
        accepted.geomID = current.geomID;
        accepted.instID = current.instID;
        accepted.triangleBarycentrics = current.triangleBarycentrics;
        acceptedSBT = header;
      }
    }

    /*! HIPRT intersectFunc hook for CUSTOM geometry. Runs barney's intersect
        program for this candidate prim EXACTLY ONCE and folds the hit here
        (matching the software backend's single intersectPrim per prim). For
        has_ch=false geoms (cylinders/cones/capsules) the intersect program
        itself writes the hit (material BSDF, hitIDs); running it once per
        candidate -- not again in the filter -- is what keeps that hit-write from
        corrupting memory as the custom-prim count grows. The closest hit wins
        because the intersect tests against current.tMax (= accepted.tMax), so a
        farther prim early-returns before writing.

        Returns FALSE unconditionally so HIPRT discards this leaf and keeps
        walking the BVH (the record-and-ignore enumerate-all pattern); the filter
        is therefore never invoked for custom geoms, so the intersect runs only
        once. Kept __noinline__ to bound the trace kernel's register footprint. */
    inline __device__ __attribute__((noinline))
    bool TraceInterface::hiprtIntersectThunk(::hiprtHit &hit)
    {
      Geom::SBTHeader *header = setupHit(hit.instanceID, hit.primID);
      header->user.intersect(*this);
      if (current.tMax < accepted.tMax)
        foldAcceptedHit(header);
      return false;
    }

    /*! HIPRT filterFunc hook (any-hit) -- reached only for TRIANGLE geometry
        (the custom intersect thunk returns false, so HIPRT never calls the
        filter for custom geoms). Recompute the barycentrics with barney's own
        triangle test so they match the rest of barney, run anyHit, and fold the
        hit. Always returns true to the caller so HIPRT keeps enumerating. */
    inline __device__ __attribute__((noinline))
    void TraceInterface::hiprtFilterThunk(const ::hiprtHit &hit)
    {
      Geom::SBTHeader *header = setupHit(hit.instanceID, hit.primID);
      vec3i indices = header->triangles.indices[current.primID];
      vec3f v0 = header->triangles.vertices[indices.x];
      vec3f v1 = header->triangles.vertices[indices.y];
      vec3f v2 = header->triangles.vertices[indices.z];
      if (!intersectTriangle(v0,v1,v2)) return;
      foldAcceptedHit(header);
    }
#endif

    inline __device__
    void TraceInterface::traceRay(rtc::AccelHandle _world,
                                  vec3f org,
                                  vec3f dir,
                                  float t0,
                                  float t1,
                                  void *prdPtr)
    {
#if RTC_DEVICE_CODE
      if (fabsf(dir.x) < 1e-6f) dir.x = 1e-6f;
      if (fabsf(dir.y) < 1e-6f) dir.y = 1e-6f;
      if (fabsf(dir.z) < 1e-6f) dir.z = 1e-6f;

      prd = prdPtr;
      world.org = org;
      world.dir = dir;
      tMin = t0;
      accepted.tMax = t1;
      accepted.primID = -1;
      accepted.instID = -1;
      acceptedSBT = 0;
      hiprtWorld = _world;

      InstanceGroup::DeviceRecord *model
        = (InstanceGroup::DeviceRecord *)_world;
      if (!model || !model->scene) return;

      ::hiprtRay ray;
      ray.origin    = { org.x, org.y, org.z };
      ray.direction = { dir.x, dir.y, dir.z };
      ray.minT      = t0;
      ray.maxT      = t1;

      // The any-hit traversal enumerates hits; for custom geoms HIPRT calls
      // intersectFunc -> hiprtIntersectThunk (runs barney's intersect once and
      // folds the hit), for triangles filterFunc -> hiprtFilterThunk (runs the
      // triangle test + anyHit and folds). Both fold the closest non-rejected
      // hit into this TraceInterface (the payload) and tell HIPRT to discard the
      // leaf so it keeps walking. This is the record-and-ignore pattern;
      // barney's own anyHit programs (transparent/clip geometry) run during
      // traversal. The intersect runs exactly once per candidate prim (as in the
      // software backend), so has_ch=false geoms write their hit only once.
      ::hiprtSceneTraversalAnyHit tr(model->scene, ray, hiprtFullRayMask,
                                     hiprtTraversalHintDefault,
                                     /*payload*/ this,
                                     (hiprtFuncTable)hiprtFuncTableHandle,
                                     /*rayType*/ 0);
      hiprtHit hit = tr.getNextHit();
      // The filter has already folded every hit into `accepted`; the returned
      // hit is the final one (or invalid). All shading state lives in this
      // TraceInterface (the payload), not in a kernel-stack array.
      (void)hit;

      if (acceptedSBT && acceptedSBT->ch) {
        current = accepted;
        this->geomData = (acceptedSBT+1);
        currentInstance = model->instanceRecords + accepted.instID;
        object.org = xfmPoint (currentInstance->worldToObjectXfm, world.org);
        object.dir = xfmVector(currentInstance->worldToObjectXfm, world.dir);
        acceptedSBT->ch(*this);
      }
#endif
    }

    inline __device__
    vec3f TraceInterface::transformNormalFromObjectToWorldSpace(vec3f v) const
    { return xfmVector(currentInstance->objectToWorldXfm.l,(const owl::common::vec3f &)v); }

    inline __device__
    vec3f TraceInterface::transformPointFromObjectToWorldSpace(vec3f v) const
    { return xfmPoint(currentInstance->objectToWorldXfm,(const owl::common::vec3f &)v); }

    inline __device__
    vec3f TraceInterface::transformVectorFromObjectToWorldSpace(vec3f v) const
    { return xfmVector(currentInstance->objectToWorldXfm.l,(const owl::common::vec3f &)v); }

    inline __device__
    vec3f TraceInterface::transformNormalFromWorldToObjectSpace(vec3f v) const
    { return xfmVector(currentInstance->worldToObjectXfm.l,(const owl::common::vec3f &)v); }

    inline __device__
    vec3f TraceInterface::transformPointFromWorldToObjectSpace(vec3f v) const
    { return xfmPoint(currentInstance->worldToObjectXfm,(const owl::common::vec3f &)v); }

    inline __device__
    vec3f TraceInterface::transformVectorFromWorldToObjectSpace(vec3f v) const
    { return xfmVector(currentInstance->worldToObjectXfm.l,(const owl::common::vec3f &)v); }

  }
}

// The trace kernel: one launched HIP kernel doing raygen + traversal + shading
// (HIPRT's single-kernel model). __launch_bounds__(256) keeps a small declared
// block size to bound per-thread VGPRs through the HIPRT traversal call
// (barney launches 16x16=256).
#if RTC_DEVICE_CODE
# define RTC_HIPRT_TRACEKERNEL(name,Class)                              \
  __global__ void __launch_bounds__(256)                               \
  rtc_hiprt_run_##name(::BARNEY_NS::rtc::TraceInterface ti)                \
  {                                                                     \
    Class::run(ti);                                                     \
  }
#else
# define RTC_HIPRT_TRACEKERNEL(name,Class)                              \
  __global__ void rtc_hiprt_run_##name(::BARNEY_NS::rtc::TraceInterface ti);
#endif

#define RTC_EXPORT_TRACE2D(name,Class)                                  \
  RTC_HIPRT_TRACEKERNEL(name,Class)                                     \
  void rtc_hiprt_launch_##name(rtc::Device *device,                    \
                               vec2i dims,                              \
                               const void *lpData)                      \
  {                                                                     \
    vec2i bs(16,16);                                                    \
    vec2i nb = divRoundUp(dims,bs);                                     \
    ::BARNEY_NS::rtc::TraceInterface ti;                                    \
    ti.lpData = lpData;                                                 \
    ti.hiprtFuncTableHandle = rtc::getFuncTable(device);         \
    rtc_hiprt_run_##name                                               \
      <<<dim3{(unsigned)nb.x,(unsigned)nb.y,1u},                       \
         dim3{(unsigned)bs.x,(unsigned)bs.y,1u},                       \
         0,device->stream>>>(ti);                                       \
  }
