// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/render/DG.h"

namespace BARNEY_NS {
  namespace native {
    namespace packedBSDF {

      /*! homogeneous phase function with Henyey-Greenstein anisotropy. */
      struct HGPhase {
        inline HGPhase() = default;
        inline __rtc_device HGPhase(vec3f color, float scatteringAlbedo=.9f,
                                  float g=0.f);

        inline __rtc_device
        float pdf(DG dg, vec3f wi, bool dbg) const;
        
        inline __rtc_device
        EvalRes eval(DG dg, vec3f wi, bool dbg) const;
        
        inline __rtc_device
        void scatter(ScatterResult &scatter,
                     const DG &dg,
                     Random &random,
                     bool dbg) const;

        inline __rtc_device
        float hgPhase(float cosTheta) const;

        inline __rtc_device EvalRes evalProjected(DG dg, vec3f wi, bool dbg) const
        { return eval(dg,wi,dbg); }

        static inline __rtc_device
        vec3f sampleDirection(vec3f wi_in, float g, Random &random);
        
        rtc::float3 albedo;
        float g;
        rtc::float3 emission;
      };

      inline __rtc_device
      HGPhase::HGPhase(vec3f color, float scatteringAlbedo, float _g)
      {
        (vec3f&)this->albedo = scatteringAlbedo * color;
        this->g = _g;
        (vec3f&)this->emission = vec3f(0.f);
      }

      inline __rtc_device
      float HGPhase::hgPhase(float cosTheta) const
      {
        if (fabsf(g) < 1e-3f)
          return ONE_OVER_FOUR_PI;
        float g2 = g*g;
        float denom = 1.f + g2 - 2.f*g*cosTheta;
        return ONE_OVER_FOUR_PI * (1.f - g2) / (denom * sqrtf(denom));
      }
      
      inline __rtc_device
      float HGPhase::pdf(DG dg, vec3f wi, bool dbg) const
      {
        vec3f wi_in = normalize(-dg.wo);
        float cosTheta = dot(wi_in, normalize(wi));
        return hgPhase(cosTheta);
      }
        
      inline __rtc_device
      EvalRes HGPhase::eval(DG dg, vec3f wi, bool dbg) const
      {
        float density = pdf(dg, wi, dbg);
        return EvalRes(density * (const vec3f&)albedo, density);
      }

      inline __rtc_device
      vec3f HGPhase::sampleDirection(vec3f wi_in, float g, Random &random)
      {
        float u0 = random();
        float u1 = random();
        float cosTheta;
        if (fabsf(g) < 1e-3f)
          cosTheta = 1.f - 2.f*u0;
        else {
          float sqrTerm = (1.f - g*g) / (1.f + g - 2.f*g*u0);
          cosTheta = (1.f + g*g - sqrTerm*sqrTerm) / (2.f * g);
        }
        float sinTheta = sqrtf(max(0.f, 1.f - cosTheta*cosTheta));
        float phi = TWO_PI * u1;
        vec3f local = vec3f(cosf(phi)*sinTheta,
                            sinf(phi)*sinTheta,
                            cosTheta);
        vec3f w = normalize(wi_in);
        vec3f u, v;
        if (fabsf(w.x) > 0.1f)
          u = normalize(cross(vec3f(0.f,1.f,0.f), w));
        else
          u = normalize(cross(vec3f(1.f,0.f,0.f), w));
        v = cross(w, u);
        return normalize(local.x*u + local.y*v + local.z*w);
      }

      inline __rtc_device
      void HGPhase::scatter(ScatterResult &scatter,
                          const DG &dg,
                          Random &random,
                          bool dbg) const
      {
        scatter = {};
        vec3f wi_in = normalize(-dg.wo);
        scatter.dir = HGPhase::sampleDirection(wi_in, g, random);
        float density = hgPhase(dot(wi_in, scatter.dir));
        scatter.pdf = density;
        scatter.weight = (const vec3f&)albedo;
        scatter.event = ScatterResult::VOLUME_EVENT;
        scatter.type = ScatterResult::VOLUME;
      }

      /*! implements a homogenous phase function that scatters equally
          in all directions, with given average reflectance and
          color */
      struct SimplePhase {
        inline SimplePhase() = default;
        inline __rtc_device SimplePhase(vec3f color, float avg_reflectance=.7f);
        inline __rtc_device EvalRes evalProjected(DG dg, vec3f wi, bool dbg) const
        { return eval(dg,wi,dbg); }

        inline __rtc_device
        float pdf(DG dg, vec3f wi, bool dbg) const;
        
        inline __rtc_device
        EvalRes eval(DG dg, vec3f wi, bool dbg) const;
        
        inline __rtc_device
        void scatter(ScatterResult &scatter,
                     const DG &dg,
                     Random &random,
                     bool dbg) const;
        
        rtc::float3 albedo;
      };

      inline __rtc_device
      SimplePhase::SimplePhase(vec3f color, float avg_reflectance)
      {
        (vec3f&)this->albedo = avg_reflectance * color;
      }
      
      inline __rtc_device
      float SimplePhase::pdf(DG dg, vec3f wi, bool dbg) const
      { return ONE_OVER_FOUR_PI; }
        
      inline __rtc_device
      EvalRes SimplePhase::eval(DG dg, vec3f wi, bool dbg) const
      {
        float density = ONE_OVER_FOUR_PI;
        return EvalRes(density*
                       (const vec3f&)albedo,density);
      }

      inline __rtc_device
      void SimplePhase::scatter(ScatterResult &scatter,
                          const DG &dg,
                          Random &random,
                          bool dbg) const
      {
        scatter = {};
        float r1 = random();
        float r2 = random(); 
        float x = 2.f*cosf(TWO_PI*r1)*sqrtf(r2*(1.f-r2));
        float y = 2.f*sinf(TWO_PI*r1)*sqrtf(r2*(1.f-r2));
        float z = (1.f-2.f*r2);
        float density = ONE_OVER_FOUR_PI;
        scatter.pdf = density;
        scatter.weight = (const vec3f&)albedo;
        scatter.event = ScatterResult::VOLUME_EVENT;
        scatter.dir = vec3f(x,y,z);
        scatter.type = ScatterResult::VOLUME;
      }
      
    }
  }
}
