// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/geometry/Spheres.h"
#include "barney/render/HitIDs.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  using namespace BARNEY_NS::render;
  
  inline __rtc_device
  float safe_eps(float f, vec3f v)
  {
    return max(f,1e-6f*reduce_max(abs(v)));
  }

  inline __rtc_device
  float safe_eps(float f, float v)
  {
    return max(f,1e-6f*fabsf(v));
  }
  
  
  struct SpheresPrograms {
#if RTC_DEVICE_CODE 
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &rt,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
      const Spheres::DD &geom = *(const Spheres::DD *)geomData;
      vec3f origin = geom.origins[primID];
      float radius = geom.radii?geom.radii[primID]:geom.defaultRadius;
      bounds.lower = origin - radius;
      bounds.upper = origin + radius;
    }
    
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &ti)
    {
      auto &ray = *(Ray*)ti.getPRD();
      auto &self = *(Spheres::DD*)ti.getProgramData();
      const OptixGlobals &globals = OptixGlobals::get(ti);
      const World::DD &world = globals.world;
      int primID = ti.getPrimitiveIndex();
      int instID = ti.getInstanceID();

#ifdef NDEBUG
      bool dbg = false;
#else
      bool dbg = ray.dbg();
#endif
      
      float t_hit = ti.getRayTmax(); 

      vec3f org = ti.getWorldRayOrigin(); 
      vec3f dir = ti.getWorldRayDirection();
      /*! isec code has temporarily stored object-space hit position
        in ray.P, see below! */
      vec3f objectP = ray.P;
      vec3f worldP = ti.transformPointFromObjectToWorldSpace(objectP);
      
      vec3f objectCenter
        = self.origins[primID];
      vec3f objectN
        = (objectP == objectCenter)
        ? vec3f(1.f,0.f,0.f)
        : (objectP - objectCenter);
      float objectRadius = self.radii?self.radii[primID]:self.defaultRadius;

      /* shift object-space hit a bit away from the sphere */
      float eps = 1e-6f;
      eps = safe_eps(eps,objectRadius);
      eps = safe_eps(eps,objectP);
      objectP = objectCenter + (objectRadius+eps)*objectN;
    
      vec3f worldN
        = ti.transformVectorFromObjectToWorldSpace(objectN);

      render::HitAttributes hitData;
      hitData.worldPosition   = worldP;

      hitData.objectPosition  = objectP;
      hitData.worldNormal     = normalize(worldN);
      hitData.objectNormal    = make_vec4f(normalize(objectN));
      hitData.primID          = primID;
      hitData.instID          = instID;
      hitData.t               = t_hit;
      if (self.colors)
        (vec3f&)hitData.color = self.colors[primID];
    
      auto interpolator = [&](const GeometryAttribute::DD &attrib,
                              bool faceVarying) -> vec4f
      { // doesn't make sense, but anari sdk assumes for spheres
        // per-vtx is same as per-prim
        vec4f v = attrib.fromArray.valueAt(hitData.primID,dbg);
        return v;
      };
      self.setHitAttributes(hitData,interpolator,world,dbg);

      const DeviceMaterial &material
        = world.materials[self.materialID];
      material.setHit(ray,hitData,world.samplers,dbg);
    }
  
    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    {
      /* nothing - already set in isec */
    }
  
    static inline __rtc_device
    void intersect(rtc::TraceInterface &ti)
    {
      const int primID = ti.getPrimitiveIndex();//optixGetPrimitiveIndex();
      const auto &self = *(const Spheres::DD*)ti.getProgramData();
      // = owl::getProgramData<Spheres::DD>();
      auto &ray = *(Ray*)ti.getPRD();//owl::getPRD<Ray>();
      
      vec3f center = self.origins[primID];
      float radius = self.radii?self.radii[primID]:self.defaultRadius;
      
      // with "move the origin" trick; see Ray Tracing Gems 2
      // const vec3f old_org  = optixGetObjectRayOrigin();
      const vec3f old_org  = ti.getObjectRayOrigin();
      // const vec3f dir  = optixGetObjectRayDirection();
      const vec3f dir  = ti.getObjectRayDirection();
      vec3f org = old_org;
      float t_move = max(0.f,length(center - old_org)-3.f*radius);
      org = org + t_move * dir;
      float t_max = ti.getRayTmax() - t_move;
      if (t_max < 0.f) return;
    
      float hit_t = t_max;

      float tmin = max(0.f,ti.getRayTmin()-t_move);
      const vec3f oc = org - center;
      const float a = dot(dir,dir);
      const float b = dot(oc, dir);
      const float c = dot(oc, oc) - radius * radius;
      const float discriminant = b * b - a * c;
    
      if (discriminant < 0.f) return;
    
      {
        float temp = (-b - sqrtf(discriminant)) / a;
        if (temp < hit_t && temp > tmin) 
          hit_t = temp;
      }
      
      {
        float temp = (-b + sqrtf(discriminant)) / a;
        if (temp < hit_t && temp > tmin) 
          hit_t = temp;
      }
      

      
      if (hit_t < t_max) {
        // "abuse" ray.P to store local sphere coordinate
        vec3f osPositionOfHit = /*shifted!*/org + /*shifted!*/hit_t*dir;
        ray.P = osPositionOfHit;
      
        hit_t += t_move;

        // Cut-plane: reject hits on the invisible side
        if (OptixGlobals::hitOnInvisibleSide(
              OptixGlobals::get(ti), hit_t, ti))
          return;

        // ------------------------------------------------------------------
        ti.reportIntersection(hit_t, 0);
        const OptixGlobals &globals = OptixGlobals::get(ti);
        if (globals.hitIDs) {
          /* ID buffer rendering writes IDs no matter what transparency */
          const World::DD &world = globals.world;
          float depth = hit_t;
          int instID    = ti.getInstanceID();
        
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
            globals.hitIDs[rayID].instID
              = 13+world.rank;
            globals.hitIDs[rayID].objID  = self.userID;
            globals.hitIDs[rayID].depth  = depth;
          }
        }
        // ------------------------------------------------------------------

      }
    }
#endif
  };
  RTC_EXPORT_USER_GEOM(Spheres,Spheres::DD,SpheresPrograms,false,true);
}


