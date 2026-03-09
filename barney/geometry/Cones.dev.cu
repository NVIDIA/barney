// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/geometry/Cones.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  using namespace BARNEY_NS::render;

  inline __rtc_device float sqrt(float f) { return sqrtf(f); }
  inline __rtc_device float inversesqrt(float f) { return 1./sqrtf(f); }
  inline __rtc_device float length2(vec3f v) { return dot(v,v); }
  
  struct ConesPrograms {
#if RTC_DEVICE_CODE
    /*! bounding box program */
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
      auto &self = *(Cones::DD*)geomData;
      const vec2i pidx
        = self.indices
        ? self.indices[primID]
        : (2 * primID + vec2i(0, 1));
    
      const vec3f pa = (const vec3f&)self.vertices[pidx.x];
      const vec3f pb = (const vec3f&)self.vertices[pidx.y];
    
      const float ra = self.radii[pidx.x];
      const float rb = self.radii[pidx.y];
      box3f aBox(pa-ra,pa+ra);
      box3f bBox(pb-ra,pb+rb);
    
      bounds.lower = min(aBox.lower,bBox.lower);
      bounds.upper = max(aBox.upper,bBox.upper);
    }

    /*! closest-hit program - doesn't do anything because we do all the
      work in IS prog, but needs to exist to make optix happy */
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &ti)
    {
      /* nothing - already set in isec */
    }
    
    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    {
      /* nothing - already set in isec */
    }
  
    /*! largely stolen from VisRTX */
    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
      Ray &ray    = *(Ray*)ti.getPRD();
      const auto &self
        = *(Cones::DD*)ti.getProgramData();
      const int primID = ti.getPrimitiveIndex();
      const int instID = ti.getInstanceID();
      const OptixGlobals &globals = OptixGlobals::get(ti);
      const World::DD &world = globals.world;
        
      render::HitAttributes hitData;
      hitData.primID          = primID;
      hitData.instID          = instID;
      const DeviceMaterial &material
        = world.materials[self.materialID];
      hitData.t = ray.tMax;
      float ray_tmin = ti.getRayTmin();
      
      vec3f ro  = ti.getObjectRayOrigin();
      vec3f rd  = ti.getObjectRayDirection();
    
      const vec2i idx
        = self.indices
        ? self.indices[primID]
        : (2 * primID + vec2i(0, 1));
    
      const auto p0 = (const vec3f &)self.vertices[idx.x];
      const auto p1 = (const vec3f &)self.vertices[idx.y];

      const float ra = self.radii[idx.x];
      const float rb = self.radii[idx.y];

      const vec3f ba = p1 - p0;
      const vec3f oa = ro - p0;
      const vec3f ob = ro - p1;

      const float m0 = dot(ba, ba);
      const float m1 = dot(oa, ba);
      const float m2 = dot(ob, ba);
      const float m3 = dot(rd, ba);

      float lerp_t = 0.f;

      // interpolator for anari-style color/attribute interpolation
      auto interpolator = [&](const GeometryAttribute::DD &attrib,
                              bool faceVarying) -> vec4f
      {
        int idx_x = faceVarying ? 2*primID+0 : idx.x;
        int idx_y = faceVarying ? 2*primID+1 : idx.y;
        
        const vec4f value_a = attrib.fromArray.valueAt(idx_x);
        const vec4f value_b = attrib.fromArray.valueAt(idx_y);
        
        const vec4f ret = (1.f-lerp_t)*value_a + lerp_t*value_b;
        return ret;
      };

      if (m1 < 0.f) {
        if (length2(oa * m3 - rd * m1) < (ra * ra * m3 * m3)) {
          float t = -m1 / m3;
          if (t > ray_tmin && t < hitData.t) {
            lerp_t = 0.f;
            vec3f N = normalize(-ba * inversesqrt(m0));
            vec3f P = (vec3f)ro+t*rd;

            hitData.t               = t;
            hitData.objectPosition  = P;
            hitData.objectNormal    = make_vec4f(N);
            hitData.worldPosition
              = ti.transformPointFromObjectToWorldSpace(P);
            hitData.worldNormal
              = normalize(ti.transformNormalFromObjectToWorldSpace(N));
          
            // trigger the anari attribute evaluation
            self.setHitAttributes(hitData,interpolator,world,ray.dbg());
          
            // ... store the hit in the ray, rqs-style ...
            material.setHit(ray,hitData,world.samplers,ray.dbg());

            // write hit IDs for AOV channels
            if (globals.hitIDs) {
              const int rayID
                = ti.getLaunchIndex().x
                + ti.getLaunchDims().x
                * ti.getLaunchIndex().y;
              globals.hitIDs[rayID].primID = primID;
              globals.hitIDs[rayID].instID
                = world.instIDToUserInstID
                ? world.instIDToUserInstID[instID]
                : instID;
              globals.hitIDs[rayID].objID  = self.userID;
            }
          
            // Cut-plane: reject hits on the invisible side
            if (OptixGlobals::hitOnInvisibleSide(globals, hitData.t, ti))
              return;

            // .... and let optix know we did have a hit.
            ti.reportIntersection(hitData.t, 0);
            return;
          }
        }
      } else if (m2 > 0.0f) {
        if (length2(ob * m3 - rd * m2) < (rb * rb * m3 * m3)) {
          lerp_t = 1.f;
          float t = -m2 / m3;
          if (t > ray_tmin && t < hitData.t) {
            vec3f N = normalize(ba * inversesqrt(m0));
            vec3f P = (vec3f)ro+t*rd;

            hitData.t               = t;
            hitData.objectPosition  = P;
            hitData.objectNormal    = make_vec4f(N);
            hitData.worldPosition
              = ti.transformPointFromObjectToWorldSpace(P);
            hitData.worldNormal
              = normalize(ti.transformNormalFromObjectToWorldSpace((const vec3f&)N));
                    
            // trigger the anari attribute evaluation
            self.setHitAttributes(hitData,interpolator,world,ray.dbg());
          
            // ... store the hit in the ray, rqs-style ...
            material.setHit(ray,hitData,world.samplers,ray.dbg());

            // write hit IDs for AOV channels
            if (globals.hitIDs) {
              const int rayID
                = ti.getLaunchIndex().x
                + ti.getLaunchDims().x
                * ti.getLaunchIndex().y;
              globals.hitIDs[rayID].primID = primID;
              globals.hitIDs[rayID].instID
                = world.instIDToUserInstID
                ? world.instIDToUserInstID[instID]
                : instID;
              globals.hitIDs[rayID].objID  = self.userID;
            }
          
            // Cut-plane: reject hits on the invisible side
            if (OptixGlobals::hitOnInvisibleSide(globals, hitData.t, ti))
              return;

            // .... and let optix know we did have a hit.
            ti.reportIntersection(hitData.t, 0);
            return;
          }
        }
      }

      const float m4 = dot(rd, oa);
      const float m5 = dot(oa, oa);
      const float rr = ra - rb;
      const float hy = m0 + rr * rr;

      float k2 = m0 * m0 - m3 * m3 * hy;
      float k1 = m0 * m0 * m4 - m1 * m3 * hy + m0 * ra * (rr * m3 * 1.0f);
      float k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * m1 * 2.0f - m0 * ra);

      const float h = k1 * k1 - k2 * k0;
      if (h < 0.0f)
        return;

      const float t = (-k1 - sqrtf(h)) / k2;
      const float y = m1 + t * m3;
      if (y > 0.0f && y < m0 && t > ray_tmin && t < hitData.t) {
        vec3f N = normalize(m0 * (m0 * (oa + t * rd) + rr * ba * ra) - ba * hy * y);
        vec3f P = (vec3f)ro+t*rd;
        lerp_t = y / m0;
        //lerp_t = dot(P-p1,p0-p1)/dot(p0-p1,p0-p1);
        lerp_t = clamp(lerp_t,0.f,1.f);
        hitData.primID          = primID;
        hitData.t               = t;
        hitData.objectPosition  = P;
        hitData.objectNormal    = make_vec4f(N);
        hitData.worldPosition
          = ti.transformPointFromObjectToWorldSpace(P);
        hitData.worldNormal
          = normalize(ti.transformNormalFromObjectToWorldSpace((const vec3f&)N));
        
        // trigger the anari attribute evaluation
        self.setHitAttributes(hitData,interpolator,world,ray.dbg());
        
        // ... store the hit in the ray, rqs-style ...
        material.setHit(ray,hitData,world.samplers,ray.dbg());

        // write hit IDs for AOV channels
        if (globals.hitIDs) {
          const int rayID
            = ti.getLaunchIndex().x
            + ti.getLaunchDims().x
            * ti.getLaunchIndex().y;
          globals.hitIDs[rayID].primID = primID;
          globals.hitIDs[rayID].instID
            = world.instIDToUserInstID
            ? world.instIDToUserInstID[instID]
            : instID;
          globals.hitIDs[rayID].objID  = self.userID;
        }
        
        // Cut-plane: reject hits on the invisible side
        if (OptixGlobals::hitOnInvisibleSide(globals, hitData.t, ti))
          return;

        // .... and let optix know we did have a hit.
        ti.reportIntersection(hitData.t, 0);
      }
    }
#endif
  };
  
  RTC_EXPORT_USER_GEOM(Cones,Cones::DD,ConesPrograms,false,false);
} // ::BARNEY_NS

