// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include <owl/owl.h>
#include "rtcore/cudaCommon/ComputeInterface.h"
#include "rtcore/cuda/TraceKernel.h"
#include "rtcore/cuda/Group.h"

#include "rtcore/cuda/ProgramInterface.h"
#include "rtcore/cudaCommon/Device.h"

#include <cuBQL/bvh.h>
#include <cuBQL/math/Ray.h>
#include <cuBQL/queries/triangleData/Triangle.h>
#include <cuBQL/queries/triangleData/math/rayTriangleIntersections.h>
#include <cuBQL/traversal/rayQueries.h>

namespace BARNEY_NS {
  namespace rtc {

    inline __device__
    void TraceInterface::traceRay(rtc::AccelHandle _world,
                                  vec3f org,
                                  vec3f dir,
                                  float t0,
                                  float t1,
                                  void *prdPtr)
    {
      bool dbg = false;
      
      using Triangle3f = cuBQL::triangle_t<float>;
      using RayTriangleIntersection = cuBQL::RayTriangleIntersection_t<float>;
      
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

      InstanceGroup::DeviceRecord *model
        = (InstanceGroup::DeviceRecord *)_world;
      // Rejecting invalid instance transforms can leave an empty world.
      if (!model || !model->instanceRecords)
        return;
      
      ::cuBQL::ray3f ray((const cuBQL::vec3f&)world.org,
                         (const cuBQL::vec3f&)world.dir,
                         t0,t1);

      auto intersectPrim = [&,this](uint32_t primIdx) -> float
      {
        GeomGroup::DeviceRecord *group 
          = (GeomGroup::DeviceRecord *)&this->currentInstance->group;
        GeomGroup::Prim *prims = group->prims;
        GeomGroup::Prim prim = prims[primIdx];
        current.primID = prim.primID;
        current.geomID = prim.geomID;
        current.tMax = accepted.tMax;
        uint8_t *geomSBT = group->sbt + prim.geomID  * group->sbtEntrySize;
        Geom::SBTHeader *header
          = (Geom::SBTHeader *)geomSBT;
        this->geomData = (header+1);
        if (group->isTrianglesGroup) {
          vec3i indices = header->triangles.indices[prim.primID];
          vec3f v0 = header->triangles.vertices[indices.x];
          vec3f v1 = header->triangles.vertices[indices.y];
          vec3f v2 = header->triangles.vertices[indices.z];

          Triangle3f tri;
          tri.a = (const cuBQL::vec3f&)v0;
          tri.b = (const cuBQL::vec3f&)v1;
          tri.c = (const cuBQL::vec3f&)v2;

          RayTriangleIntersection isec;
          bool hadHit = isec.compute(ray,tri,dbg);
          
          if (!hadHit)
            return accepted.tMax;
          this->current.triangleBarycentrics.x = isec.u;
          this->current.triangleBarycentrics.y = isec.v;
          this->current.tMax = isec.t;
        } else {
          if (header->user.intersect)
            header->user.intersect(*this);
          if (current.tMax >= accepted.tMax)
            return accepted.tMax;
        }
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
        return accepted.tMax;
      };
      auto enterBlas = [&,this,model]
        (cuBQL::ray3f &out_ray,
         cuBQL::bvh3f &out_bvh,
         int instID) 
      {
        this->current.instID  = instID;
        this->currentInstance = model->instanceRecords+current.instID;
        this->object.org
          = xfmPoint(currentInstance->worldToObjectXfm,world.org);
        this->object.dir
          = xfmVector(currentInstance->worldToObjectXfm,world.dir);
        (vec3f&)out_ray.origin    = object.org;
        (vec3f&)out_ray.direction = object.dir;
        out_bvh = {0,0,0,0};
        out_bvh.nodes = currentInstance->group.bvhNodes;
      };
      auto leaveBlas = [this]() -> void {
        currentInstance = 0;
      };

      ::cuBQL::shrinkingRayQuery::twoLevel::forEachPrim
          (enterBlas,leaveBlas,intersectPrim,model->bvh,ray);

      if (acceptedSBT && acceptedSBT->ch) {
        current = accepted;
        this->geomData = (acceptedSBT+1);
        // Restore currentInstance for closestHit transform calls:
        // leaveBlas() zeroed it, but closestHit needs it for
        // object-to-world space transforms.
        this->currentInstance = model->instanceRecords + accepted.instID;
        acceptedSBT->ch(*this);
      }
    }
    
    inline __device__
    vec3f TraceInterface::transformNormalFromObjectToWorldSpace(vec3f v) const
    {
      return xfmVector(currentInstance->worldToObjectXfm.l.transposed(),
                       (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformPointFromObjectToWorldSpace(vec3f v) const
    { 
      return xfmPoint(currentInstance->objectToWorldXfm,
                      (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformVectorFromObjectToWorldSpace(vec3f v) const
    { 
      return xfmVector(currentInstance->objectToWorldXfm.l,
                       (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformNormalFromWorldToObjectSpace(vec3f v) const
    {
      return xfmVector(currentInstance->objectToWorldXfm.l.transposed(),
                       (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformPointFromWorldToObjectSpace(vec3f v) const
    { 
      return xfmPoint(currentInstance->worldToObjectXfm,
                      (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformVectorFromWorldToObjectSpace(vec3f v) const
    { 
      return xfmVector(currentInstance->worldToObjectXfm.l,
                       (const owl::common::vec3f &)v);
    }
    
  }
}

#if RTC_DEVICE_CODE
# define RTC_CUDA_TRACEKERNEL(name,Class)               \
  __global__                                            \
  void rtc_cuda_run_##name(rtc::TraceInterface ti)      \
  {                                                     \
    Class::run(ti);                                     \
  }                                                             
#else
# define RTC_CUDA_TRACEKERNEL(name,Class)               \
  __global__                                            \
  void rtc_cuda_run_##name(rtc::TraceInterface ti);
#endif

#define RTC_EXPORT_TRACE2D(name,Class)                          \
  RTC_CUDA_TRACEKERNEL(name,Class)                              \
  void rtc_cuda_launch_##name(rtc::Device *device,              \
                              ::owl::common::vec2i dims,        \
                              const void *lpData)               \
  {                                                             \
    ::owl::common::vec2i bs(16,16);                             \
    ::owl::common::vec2i nb                                     \
        = ::owl::common::divRoundUp(dims,bs);                   \
    rtc::TraceInterface ti;                                     \
    ti.lpData = lpData;                                         \
    rtc_cuda_run_##name                                         \
      <<<dim3{(unsigned)nb.x,(unsigned)nb.y,(unsigned)1},       \
      dim3{(unsigned)bs.x,(unsigned)bs.y,(unsigned)1},          \
      0,device->stream>>>(ti);                                  \
  }                                                             \
  


