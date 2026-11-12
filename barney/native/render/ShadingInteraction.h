// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// some functions taken from OSPRay, under this lincense:
// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "native/common/barney-common.h"
#include "native/common/half.h"
#include "native/common/Texture.h"
#include "native/common/Data.h"
#include "native/common/math.h"

namespace BARNEY_NS {
  namespace native {
    using Random = BARNEY_NS::native::Random2;
    
    // World-space hit data; wo and sampled wi point away from the hit.
    // geometricNormal controls sidedness and offsets, not the shading-lobe projection.
    // See doc/Scattering.md for the evaluation and sampling contract.
    struct ShadingInteraction {
      vec3f geometricNormal; // Geometric normal oriented toward wo.
      vec3f shadingNormal; // Unoriented interpolated normal, before material normal maps.
      vec3f wo;
      vec3f P;
      bool  insideMedium;

      // Flip the entire original shading frame, not each mapped lobe toward wo.
      inline __rtc_device vec3f orientShadingNormal(vec3f normal) const
      { return dot(geometricNormal,shadingNormal) < 0.f ? -normal : normal; }
    };

    struct EvalRes {
      inline __both__ EvalRes() : value(0.f), pdf(0.f) {}
      inline __rtc_device EvalRes(vec3f v, float p) : value(v),pdf(p) {}
      static inline __rtc_device EvalRes zero() { return { vec3f(0.f),0.f }; }
      inline __rtc_device bool valid() const { return pdf > 0.f && isfinite(pdf); }
      // Raw BSDF/phase value from eval(); per-lobe cosine included by
      // evalProjected() for surfaces only. Neither includes division by pdf.
      vec3f value;
      float pdf; // Marginal continuous density in solid angle; excludes deltas.
    };

    /*! result of scattering a ray on a differential surface,
        according to a BSDF */
    struct ScatterResult {
      typedef enum {
        INVALID  = 0,
        NONE,
        DIFFUSE,
        SPECULAR,
        GLOSSY,
        VOLUME,
      } Type;
      enum Event { NO_EVENT, REFLECTION, TRANSMISSION, VOLUME_EVENT };
      inline __rtc_device bool valid() const
      {
        return event != NO_EVENT && isfinite(dir.x) && isfinite(dir.y)
          && isfinite(dir.z) && dot(dir,dir) > 0.f
          && isfinite(weight.x) && isfinite(weight.y) && isfinite(weight.z)
          && (delta ? pdf == 0.f : (pdf > 0.f && isfinite(pdf)));
      }

      // Complete radiance multiplier, including sampling compensation and
      // surface projection. Apply directly; do not multiply by cosine/pdf.
      vec3f weight = vec3f(0.f);
      vec3f dir = vec3f(0.f);
      float pdf = 0.f; // Solid-angle density; zero for delta events.
      float eta = 1.f; // Incident / transmitted refractive index.
      bool delta = false;
      Event event = NO_EVENT;
      ScatterResult::Type type = ScatterResult::INVALID;
    };


    inline __rtc_device
    vec3f sampleCosineWeightedHemisphere(vec3f shadingNormal, Random &random)
    {
      while (1) {
        vec3f p = 2.f*vec3f(random(),random(),random()) - vec3f(1.f);
        if (dot(p,p) > 1.f) continue;
        return normalize(normalize(p)+shadingNormal);
      }
    }

    /*! Spherical coordinates for unit direction v (theta = polar angle from +Z). */
    inline __rtc_device float sphericalTheta(const vec3f &v)
    {
      return acosf(clamp(v.z, -1.f, 1.f));
    }

    inline __rtc_device float sphericalPhi(const vec3f &v)
    {
      float p = atan2f(v.y, v.x);
      return (p < 0.f) ? (p + float(2.f * M_PI)) : p;
    }

    inline __rtc_device
    float luminance(vec3f c)
    {
      return 0.212671f*c.x + 0.715160f*c.y + 0.072169f*c.z;
    }
    
    inline __rtc_device
    vec3f reflect(vec3f v, vec3f n)
    {
      return v - (2.f*dot(v,n))*n;
    }
    
    inline __rtc_device
    vec3f refract(vec3f v, vec3f n, float eta)
    {
      float dotValue = dot(n,v);
      float k = 1.f-eta*eta*(1.f-dotValue*dotValue);
      return (k >= 0.f)
        ? (eta*v - (eta*dotValue + sqrtf(k))*n)
        : vec3f(0.f);
    }

    inline __rtc_device
    bool all_zero(vec3f v) { return v.x==0 && v.y==0 && v.z == 0; }
    


    inline __rtc_device
    vec3f cartesian(float phi, float sinTheta, float cosTheta)
    {
      float sinPhi, cosPhi;
#ifdef _WIN32
      sinPhi = sinf(phi);
      cosPhi = cosf(phi);
#elif __APPLE__
      __sincosf(phi, &sinPhi, &cosPhi);
#else
      sincosf(phi, &sinPhi, &cosPhi);
#endif
      return vec3f(cosPhi * sinTheta,
                   sinPhi * sinTheta,
                   cosTheta);
    }

    inline __rtc_device
    vec3f cartesian(const float phi, const float cosTheta)
    {
      return cartesian(phi, cos2sin(cosTheta), cosTheta);
    }




    inline __rtc_device
    vec3f cosineSampleHemisphere(const vec2f s)
    {
      const float phi = TWO_PI * s.x;
      const float cosTheta = sqrtf(s.y);
      const float sinTheta = sqrtf(1.0f - s.y);
      return cartesian(phi, sinTheta, cosTheta);
    }

    inline __rtc_device
    float cosineSampleHemispherePDF(const vec3f &dir)
    {
      return dir.z * ONE_OVER_PI;
    }

    inline __rtc_device
    float cosineSampleHemispherePDF(float cosTheta)
    {
      return cosTheta * ONE_OVER_PI;
    }

    inline __rtc_device
    vec3f randomDirection(Random &rng)
    {
      vec3f v;
      while (true) {
        v.x = 1.f-2.f*rng();
        v.y = 1.f-2.f*rng();
        v.z = 1.f-2.f*rng();
        if (dot(v,v) <= 1.f)
          return normalize(v);
      }
    }

    
  }
}
