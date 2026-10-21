// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/half.h"
#include "native/packedBSDF/PackedBSDF.h"
#include "native/common/random.h"

namespace BARNEY_NS {
  namespace native {
    /* path state/shade info that does _not_ go over the network */

#define NEW_RNG 1

    struct PathState {
      vec3h    throughput;
      int32_t  pixelID;
      float    misWeight;
      int      numDiffuseBounces;
      int      numVolumeBounces;
    };

    struct RayOnly {
      vec3f    org;
      vec3f    dir;
      float    tMax;
      struct {
        uint32_t isInMedium : 1;
        uint32_t isSpecular : 1;
        uint32_t isShadowRay: 1;
        uint32_t dbg        : 1;
        uint32_t crosshair  : 1;
      };
    };

    struct HitOnly {
      float tHit;
      vec3f       P;
      vec3h       shadingNormal;
      vec3h       geometricNormal;
      uint16_t bsdfType   : 4;
      PackedBSDF::Data hitBSDF;
    };
    
#if RTC_DEVICE_CODE
    inline __rtc_device vec3f normalizeSurfaceNormal(vec3f n)
    {
      // Scaling first avoids float overflow/underflow for valid inverse
      // transforms. Preserve the zero normal that identifies volume hits.
      const float scale = max(fabsf(n.x),max(fabsf(n.y),fabsf(n.z)));
      if (scale == 0.f) return n;
      n = n / scale;
      return normalize(n);
    }
#endif

    struct Ray {
#if RTC_DEVICE_CODE
      inline __rtc_device void setVolumeHit(vec3f P, float t, vec3f albedo,
                                            float anisotropy,
                                            float scatteringAlbedo);
      inline __rtc_device void setVolumeHitWithEmission(vec3f P, float t,
                                                        vec3f albedo,
                                                        float anisotropy,
                                                        float scatteringAlbedo,
                                                        vec3f emission);
      inline __rtc_device void setVolumeHit(vec3f P, float t, vec3f albedo);
      inline __rtc_device PackedBSDF getBSDF() const;
      inline __rtc_device void setHit(vec3f P, vec3f N, float t,
                                    const PackedBSDF &packedBSDF,
                                    vec3f geometricNormal = vec3f(0.f));
      
      inline __rtc_device bool hadHit() const { return bsdfType != PackedBSDF::NONE; }
      inline __rtc_device void clearHit(float newTMax = BARNEY_INF)
      { bsdfType = PackedBSDF::NONE; tMax = newTMax; }
      
      inline __rtc_device void packNormal(vec3f N);
      inline __rtc_device vec3f unpackNormal() const;
      inline __rtc_device vec3f getShadingNormal() const  { return unpackNormal(); }
      inline __rtc_device vec3f getGeometricNormal() const
      {
        const vec3f n = (vec3f)geometricNormal;
        return dot(n,n) > 0.f ? normalize(n) : n;
      }
#endif

      vec3f   org;
      vec3f   dir;
      float   tMax;
      RNGSeed rngSeed;

      vec3f       P;
      vec3h       shadingNormal;
      vec3h       geometricNormal;
      struct {
        uint16_t bsdfType   : 4;
        uint16_t isInMedium : 1;
        uint16_t isSpecular : 1;
        uint16_t isShadowRay: 1;
        uint16_t crosshair  : 1;
        uint16_t _dbg       : 1;
      };
      inline __rtc_device bool dbg() const {
        return _dbg;
      }
      
      union {
        PackedBSDF::Data hitBSDF;
        bn_float4 missColor;
      };
    };
  
#if RTC_DEVICE_CODE
    inline __rtc_device PackedBSDF Ray::getBSDF() const
    {
      return PackedBSDF((PackedBSDF::Type)bsdfType,hitBSDF);
    }
    
    inline __rtc_device void Ray::setHit(vec3f P, vec3f N, float t,
                                       const PackedBSDF &packedBSDF,
                                       vec3f geometricNormal)
    {
      this->packNormal(N);
      const vec3f ng = normalizeSurfaceNormal(geometricNormal);
      this->geometricNormal = dot(ng,ng) > 0.f
        ? ng : normalizeSurfaceNormal(N);
      this->hitBSDF   = packedBSDF.data;
      this->bsdfType  = packedBSDF.type;
      this->tMax      = t;
      this->P         = P;
    }
    
    inline __rtc_device
    void Ray::setVolumeHit(vec3f P,
                           float t,
                           vec3f albedo,
                           float anisotropy,
                           float scatteringAlbedo)
    {
      if (this->dbg()) printf("setting volume hit %f %f %f g=%f\n",
                            albedo.x, albedo.y, albedo.z, anisotropy);
      setHit(P, vec3f(0.f), t,
             packedBSDF::HGPhase(albedo, scatteringAlbedo, anisotropy));
    }

    inline __rtc_device
    void Ray::setVolumeHitWithEmission(vec3f P,
                                       float t,
                                       vec3f albedo,
                                       float anisotropy,
                                       float scatteringAlbedo,
                                       vec3f emission)
    {
      packedBSDF::HGPhase phase(albedo, scatteringAlbedo, anisotropy);
      (vec3f&)phase.emission = emission;
      setHit(P, vec3f(0.f), t, phase);
    }

    inline __rtc_device void Ray::setVolumeHit(vec3f P,
                                             float t,
                                             vec3f albedo)
    {
      if (this->dbg()) printf("setting volume hit %f %f %f\n",
                            albedo.x, albedo.y, albedo.z);
      setHit(P, vec3f(0.f), t, packedBSDF::SimplePhase(albedo));
    }

    inline __rtc_device
    void makeShadowRay(Ray &ray,
                       PathState &state,
                       vec3f _tp, vec3f _org, vec3f _dir, float len)
    {
      ray.bsdfType = PackedBSDF::NONE;
      ray.isShadowRay = 1;
      // iw: issue #66: initialize the dbg and crosshair flags; they
      // can still be overwritten but shouldn't be left uninitialized.
      ray._dbg        = 0;
      ray.crosshair   = 0;

      ray.dir = _dir;
      ray.org = _org;
      ray.tMax = len;
      state.throughput = _tp;
    }

    inline __rtc_device void Ray::packNormal(vec3f N)
    {
      this->shadingNormal = normalizeSurfaceNormal(N);
    }
  
    inline __rtc_device vec3f Ray::unpackNormal() const
    {
      const vec3f n = (vec3f)shadingNormal;
      return dot(n,n) > 0.f ? normalize(n) : n;
    }
    
    inline __rtc_device
    bool boxTest(vec3f org, vec3f dir,
                 float &t0, float &t1,
                 const box3f &box)
    {
      vec3f t_lo = (box.lower - org) * rcp(dir);
      vec3f t_hi = (box.upper - org) * rcp(dir);
      vec3f t_nr = min(t_lo,t_hi);
      vec3f t_fr = max(t_lo,t_hi);
      t0 = max(t0,reduce_max(t_nr));
      t1 = min(t1,reduce_min(t_fr));
      return t0 <= t1;
    }

    inline __rtc_device
    bool boxTest(const Ray &ray,
                 range1f &tRange,
                 const box3f &box)
    {
      vec3f org = ray.org;
      vec3f dir = ray.dir;
      return boxTest(org,dir,tRange.lower,tRange.upper,box);
    }

    inline __rtc_device
    bool boxTest(float &t0, float &t1,
                 box3f box,
                 const vec3f org,
                 const vec3f dir)
    {
      vec3f t_lo = (box.lower - org) * rcp(dir);
      vec3f t_hi = (box.upper - org) * rcp(dir);
      vec3f t_nr = min(t_lo,t_hi);
      vec3f t_fr = max(t_lo,t_hi);
      t0 = max(t0,reduce_max(t_nr));
      t1 = min(t1,reduce_min(t_fr));
      return t0 <= t1;
    }

    inline __rtc_device
    bool boxTest(float &t0, float &t1,
                 box4f box,
                 const vec3f org,
                 const vec3f dir)
    {
      return boxTest(t0,t1,box3f({box.lower.x,box.lower.y,box.lower.z},
                                 {box.upper.x,box.upper.y,box.upper.z}),
                     org,dir);
    }
#endif
    
  }  
}
