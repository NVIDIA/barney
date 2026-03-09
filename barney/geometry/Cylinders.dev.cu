// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/geometry/Cylinders.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  using namespace BARNEY_NS::render;

  /* ray - rounded cone intersection. */
  inline __rtc_device
  bool intersectRoundedCone(const vec3f  pa, const vec3f  pb,
                            const float  ra, const float  rb,
                            const vec3f ray_org,
                            const vec3f ray_dir,
                            float& hit_t,
                            vec3f& isec_normal)
  {
    const vec3f& ro = ray_org;//ray.origin;
    const vec3f& rd = ray_dir;//ray.direction;

    vec3f  ba = pb - pa;
    vec3f  oa = ro - pa;
    vec3f  ob = ro - pb;
    float  rr = ra - rb;
    float  m0 = dot(ba, ba);
    float  m1 = dot(ba, oa);
    float  m2 = dot(ba, rd);
    float  m3 = dot(rd, oa);
    float  m5 = dot(oa, oa);
    float  m6 = dot(ob, rd);
    float  m7 = dot(ob, ob);

    float d2 = m0 - rr * rr;

    float k2 = d2 - m2 * m2;
    if (k2 == 0.f) return false;
    
    float k1 = d2 * m3 - m1 * m2 + m2 * rr * ra;
    float k0 = d2 * m5 - m1 * m1 + m1 * rr * ra * 2.0 - m0 * ra * ra;

    float h = k1 * k1 - k0 * k2;
    if (h < 0.f) return false;
    float t = (-sqrtf(h) - k1) / k2;

    float y = m1 - ra * rr + t * m2;
    if (y > 0.f && y < d2)
      {
        hit_t = t;
        isec_normal = normalize(d2 * (oa + t * rd) - ba * y);
        return true;
      }

    // Caps. 
    float h1 = m3 * m3 - m5 + ra * ra;
    if (h1 > 0.f)
      {
        t = -m3 - sqrtf(h1);
        hit_t = t;
        isec_normal = normalize((oa + t * rd) / ra);
        return true;
      }
    return false;
  }
  
  struct CylindersPrograms {
#if RTC_DEVICE_CODE
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &ti,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
      const Cylinders::DD &geom = *(const Cylinders::DD *)geomData;
      vec2i idx = geom.indices[primID];
      vec3f a = geom.vertices[idx.x];
      vec3f b = geom.vertices[idx.y];
      float ra, rb;
      ra = rb = geom.radii[primID];
      box3f box_a = {a-ra,a+ra};
      box3f box_b = {b-rb,b+rb};
      bounds.lower = min(box_a.lower,box_b.lower);
      bounds.upper = max(box_a.upper,box_b.upper);
    }
  
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
  
    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
      // capped
      const int primID = ti.getPrimitiveIndex();
      const int instID = ti.getInstanceID();
      const auto &self
        = *(Cylinders::DD*)ti.getProgramData();
      Ray &ray    = *(Ray*)ti.getPRD();

      const OptixGlobals &globals = OptixGlobals::get(ti);
      const World::DD &world = globals.world;
#ifdef NDEBUG
      bool dbg = 0;
#else
      bool dbg = ray.dbg();
#endif      
      const vec2i idx = self.indices[primID];
      const vec3f v0  = self.vertices[idx.x];
      const vec3f v1  = self.vertices[idx.y];
      
      const float radius
        = self.radii[primID];
      
      const vec3f ray_org  = ti.getObjectRayOrigin();
      const vec3f ray_dir  = ti.getObjectRayDirection();
      float hit_t      = ti.getRayTmax();
      const float ray_tmin = ti.getRayTmin();
      const float ray_tmax = ti.getRayTmax();
    
      const vec3f d = ray_dir;
      const vec3f s = v1 - v0; // axis
      const vec3f sxd = cross(s, d);
      const float a = dot(sxd, sxd); // (s x d)^2
      if (a == 0.f)
        return;
    
      const vec3f f = v0 - ray_org;
      const vec3f sxf = cross(s, f);
      const float ra = 1.f/a;
      const float ts = dot(sxd, sxf) * ra; // (sd)(s x f) / (s x d)^2, in ray-space
      const vec3f fp = f - ts * d; // f' = v0 - closest point to axis
    
      const float s2 = dot(s, s); // s^2
      const vec3f perp = cross(s, fp); // s x f'
      const float c = radius*radius * s2 - dot(perp, perp); //  r^2 s^2 - (s x f')^2
      if (c < 0.f)
        return;

      float td = sqrtf(c * ra);
      const float tube_t0 = ts - td;
      const float tube_t1 = ts + td;
      
      // clip to cylinder caps
      const float sf = dot(s, f);
      const float sd = dot(s, d);

      float cap_t0 = -1e20f;
      float cap_t1 = +1e20f;
      float cap_t_v0 = cap_t0;
      float cap_t_v1 = cap_t1;
      if (sd == 0.f) {
        if (dot(ray_org-v0,v1-v0) < 0.f) return;
        if (dot(ray_org-v1,v0-v1) < 0.f) return;
      } else {
        const float rsd = 1.f/(sd);
        cap_t_v0 = sf * rsd;
        cap_t_v1 = cap_t_v0 + s2 * rsd;
        cap_t0 = min(cap_t_v0,cap_t_v1);
        cap_t1 = max(cap_t_v0,cap_t_v1);
      }
      
      const float t0 = max(cap_t0,tube_t0);
      const float t1 = min(cap_t1,tube_t1);
      if (t0 > t1) return;

      vec3f objectN = 0.f;
      if (ray_tmin <= t0 && t0 <= ray_tmax) {
        // front side hit:
        ray.tMax = t0;
        td *= -1.f;
        float hit_surf_u = (ray.tMax * sd - sf) * 1.f/(s2);
        if (t0 == cap_t0) {
          objectN
            = (cap_t0 == cap_t_v0)
            ? -s
            : s;
        } else {
          objectN = (td * d - fp - hit_surf_u * s);          
        }
      } else if (ray_tmin <= t1 && t1 <= ray_tmax) {
        ray.tMax = t1;
        float hit_surf_u = (ray.tMax * sd - sf) * 1.f/(s2);
        if (t0 == cap_t1) {
          objectN
            = (cap_t0 == cap_t_v0)
            ? -s
            : s;
        } else {
          objectN = (td * d - fp - hit_surf_u * s);          
        }
      } else
        return;

      vec3f objectP = ray_org + ray.tMax * ray_dir;
      float t_hit = ray.tMax;
      objectP = objectP + 1e-4f * normalize(objectN);

      float lerp_t
        = dot(objectP-v0,v1-v0)
        / (dot(v1-v0,v1-v0));
      lerp_t = max(0.f,min(1.f,lerp_t));

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

      if (dbg)
        printf("hit normal %f %f %f\n",
               objectN.x,
               objectN.y,
               objectN.z);
      render::HitAttributes hitData;
      hitData.objectPosition  = objectP;
      hitData.worldPosition   = ti.transformPointFromObjectToWorldSpace(objectP);
      hitData.objectNormal    = make_vec4f(normalize(objectN));
      hitData.worldNormal     = normalize(ti.transformNormalFromObjectToWorldSpace((const vec3f&)hitData.objectNormal));
      hitData.primID          = primID;
      hitData.instID          = instID;
      hitData.t               = t_hit;

      self.setHitAttributes(hitData,interpolator,world,ray.dbg());

      const DeviceMaterial &material
        = world.materials[self.materialID];
      material.setHit(ray,hitData,world.samplers,ray.dbg());
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
      if (OptixGlobals::hitOnInvisibleSide(globals, ray.tMax, ti))
        return;

      ti.reportIntersection(ray.tMax, 0);
    }
#endif
  };
  
  RTC_EXPORT_USER_GEOM(Cylinders,Cylinders::DD,CylindersPrograms,false,false);  
}


