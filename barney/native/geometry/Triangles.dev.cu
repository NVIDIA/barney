// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <bit>

#include "native/geometry/Triangles.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::native::OptixGlobals);

namespace BARNEY_NS {
  namespace native {
    
    struct TrianglesPrograms {
#if RTC_DEVICE_CODE
      static inline __rtc_device
      void closestHit(rtc::TraceInterface &rt)
      {}

      static inline __rtc_device
      void anyHit(rtc::TraceInterface &ti)
      {
        auto &ray = *(Ray *)ti.getPRD();

        const OptixGlobals &globals = OptixGlobals::get(ti);
        const World::DD &world = globals.world;
#ifdef NDEBUG
        bool dbg = false;
#else
        bool dbg = ray.dbg();
#endif
        // if (dbg) printf("=======================================================\n");
        auto &self = *(Triangles::DD*)ti.getProgramData();
        const float u = ti.getTriangleBarycentrics().x;
        const float v = ti.getTriangleBarycentrics().y;
        int primID    = ti.getPrimitiveIndex();
        int instID    = ti.getInstanceID();
        float depth   = ti.getRayTmax();

        // Cut-plane: reject hits on the invisible side
        if (OptixGlobals::hitOnInvisibleSide(globals, depth, ti)) {
          ti.ignoreIntersection();
          return;
        }

        // ------------------------------------------------------------------
        if (globals.hitIDs) {
          /* ID buffer rendering writes IDs no matter what transparency */
          const int rayID
            = ti.getLaunchIndex().x
            + ti.getLaunchDims().x
            * ti.getLaunchIndex().y;
          if (depth < globals.hitIDs[rayID].depth) {
            globals.hitIDs[rayID].primID = primID;
            globals.hitIDs[rayID].instID
              = world.instIDToUserInstID
              ? world.instIDToUserInstID[instID]
              : instID;
            globals.hitIDs[rayID].objID  = self.userID;
            globals.hitIDs[rayID].depth  = depth;
          }
        }
        // ------------------------------------------------------------------

      
        vec3i triangle = self.indices[primID];
        vec3f v0 = self.vertices[triangle.x];
        vec3f v1 = self.vertices[triangle.y];
        vec3f v2 = self.vertices[triangle.z];
        // Geometric face normal is the seed/fallback; a per-vertex
        // vertex.normal (if any) overrides it via normalAttribute in
        // setHitAttributes below, exactly like tangents.
        const vec3f n = normalize(cross(v1 - v0, v2 - v0));
        const vec3f osN = normalize(n);
        // n = ti.transformNormalFromObjectToWorldSpace(n);
        // n = normalize(n);

        // ------------------------------------------------------------------
        // get texture coordinates
        // ------------------------------------------------------------------
        const vec3f osP  = (1.f-u-v)*v0 + u*v1 + v*v2;
        vec3f P  = ti.transformPointFromObjectToWorldSpace(osP);

        HitAttributes hitData;
        hitData.worldPosition   = P;
        hitData.objectPosition  = osP;
        hitData.objectNormal    = make_vec4f(osN);
        hitData.primID          = primID;
        hitData.instID          = instID;
        hitData.t               = depth;
        hitData.isShadowRay     = ray.isShadowRay;

        auto interpolator
          = [triangle,u,v,primID,dbg](const GeometryAttribute::DD &attrib,
                                      bool faceVarying) -> vec4f
          {
            vec3i indices
              = faceVarying
              ? (vec3i(3*primID)+vec3i(0,1,2))
              : triangle;
            const vec4f value_a = attrib.fromArray.valueAt(indices.x,dbg);
            const vec4f value_b = attrib.fromArray.valueAt(indices.y,dbg);
            const vec4f value_c = attrib.fromArray.valueAt(indices.z,dbg);
            const vec4f ret = (1.f-u-v)*value_a + u*value_b + v*value_c;
            return ret;
          };
        self.setHitAttributes(hitData,interpolator,world,dbg);
        hitData.worldNormal
          = ti.transformNormalFromObjectToWorldSpace
          ((const vec3f&)hitData.objectNormal);
        // Tangents transform as vectors (model matrix), unlike normals. w
        // carries glTF handedness; it is interpolated per-vertex so collapse
        // it back to a sign. w==0 means no tangent array was set.
        if (hitData.objectTangent.w != 0.f) {
          const vec3f wt = ti.transformVectorFromObjectToWorldSpace(
            (const vec3f &)hitData.objectTangent);
          const float handedness = (hitData.objectTangent.w < 0.f) ? -1.f : 1.f;
          hitData.worldTangent = make_vec4f(wt);
          hitData.worldTangent.w = handedness;
        }

        const DeviceMaterial &material
          = world.materials[self.materialID];
      
        const float coverage
          = material.coverage(hitData,world.samplers,dbg);
        if (coverage < 1.f) {
          ray.rngSeed.next(std::bit_cast<uint32_t>(osP.x));
          ray.rngSeed.next(std::bit_cast<uint32_t>(osP.y));
          ray.rngSeed.next(std::bit_cast<uint32_t>(osP.z));
          Random rng(ray.rngSeed,290374u);
          if (rng() > coverage) {
            ti.ignoreIntersection();
            return;
          }
        }
        material.setHit(ray,hitData,world.samplers,dbg);
      }
#endif
    };
  
    RTC_EXPORT_TRIANGLES_GEOM(Triangles,Triangles::DD,TrianglesPrograms,true,false);
  }
}

