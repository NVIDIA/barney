// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/light/Light.h"
#include "native/DeviceGroup.h"
#include "native/common/math.h"
#include "native/render/DG.h"

/*! for debugging only - if enabled the envmap light will return a
  constant radiance no matter what map was specified */
// #define FORCE_CONSTANT_ENV 1

namespace BARNEY_NS {
  namespace native {
    
    using Random = BARNEY_NS::native::Random2;

    struct EnvMapLight : public Light {
      typedef std::shared_ptr<EnvMapLight> SP;
      EnvMapLight(Context *context,
                  const DevGroup::SP &devices);
      virtual ~EnvMapLight();
    
      struct DD {
#if RTC_DEVICE_CODE
        inline __rtc_device float pdf(vec3f dir, bool dbg=false) const;
        inline __rtc_device Light::Sample sample(Random &r, bool dbg=false) const;
        // inline __rtc_device vec3f  eval(vec3f dir, bool dbg=false) const;
        /*! converts from a given pixel's coordinates into the
          world-space vector that poitns to the center of that
          pixel */
        inline __rtc_device vec3f  uvToWorld(float sx, float sy) const;
        inline __rtc_device vec3f  pixelToWorld(vec2i pixelID) const;
        inline __rtc_device vec2i  worldToPixel(vec3f worldDir) const;
#endif
        linear3f            toWorld;
        linear3f            toLocal;
        rtc::TextureObject  texture = 0;
        vec2i               dims;
        float               scale;
        const float        *cdf_y = 0;
        const float        *allCDFs_x = 0;
      };

      DD getDD(Device *device, const affine3f &xfm);

      // ==================================================================
      std::string toString() const override { return "EnvMapLight"; }
    
      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      bool set1f(const std::string &member, const float &value) override;
      bool set2i(const std::string &member, const vec2i &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      bool setObject(const std::string &member, const Object::SP &value) override;
      /*! @} */
      // ------------------------------------------------------------------

    private:
      void computeCDFs();
    public: // =========== PLD STUFF ===========
      struct PLD {
        rtc::Buffer  *cdf_y = 0;
        rtc::Buffer  *allCDFs_x = 0;
      
        rtc::ComputeKernel2D *computeWeights_xy;
        rtc::ComputeKernel1D *computeCDFs_doLine;
        rtc::ComputeKernel1D *normalize_cdf_y;
      };

      PLD *getPLD(Device *);
      std::vector<PLD> perLogical;
    
    public: // =========== parameters ===========
      struct {
        Texture::SP texture;
        vec3f       direction { 1.f, 0.f, 0.f };
        vec3f       up        { 0.f, 0.f, 1.f };
        float       scale = 1.f;
      } params;
      Texture::SP texture;

      linear3f   toWorld;
      linear3f   toLocal;
    
      vec2i      dims{-1,-1};
    };



#if RTC_DEVICE_CODE
    // #if BARNEY_DEVICE_PROGRAM
    inline __rtc_device
    float cdfGetPDF(int position, const float *cdf, int N)
    {
      float f_at_position = cdf[position];
      float f_before_position
        = position
        ? cdf[position-1]
        : 0.f;
    
      return N*(f_at_position-f_before_position);
    }

    inline __rtc_device
    int sampleCDF(const float *cdf, int N, float v,
                  float &pdf, bool dbg = false)
    {
      int begin = 0;
      int end   = N;
      while ((end - begin) > 1) {
        int mid = (begin+end)/2;
        float f_mid = cdf[mid-1];

        if (v < f_mid)
          end = mid;
        else
          begin = mid;
      }
      int position = begin;
      float f_at_position = cdf[begin];
      float f_before_position
        = begin
        ? cdf[begin-1]
        : 0.f;
    
      pdf = N*(f_at_position-f_before_position);
      return position;
    }
  
    inline __rtc_device float
    EnvMapLight::DD::pdf(vec3f dir, bool dbg) const
    {
#if FORCE_CONSTANT_ENV
      return ONE_OVER_FOUR_PI;
#endif
  
      if (!texture) return 0.f;
    
      vec2i pixel = worldToPixel(dir);
      float pdf_y = cdfGetPDF(pixel.y,cdf_y,dims.y);

      float pdf_x = cdfGetPDF(pixel.x,allCDFs_x+pixel.y*dims.x,dims.x);

      const float theta = sphericalTheta(normalize(xfmVector(toLocal,dir)));

      return pdf_x * pdf_y * 1.f/(TWO_PI*ONE_PI*sinf(theta));
    }
  
    inline __rtc_device vec2i
    EnvMapLight::DD::worldToPixel(vec3f worldDir) const
    {
      vec3f localDir = xfmVector(toLocal,worldDir);
    
      float theta = sphericalTheta(localDir);
      float phi   = sphericalPhi(localDir);
      const float invPi  = ONE_OVER_PI;
      const float inv2Pi = ONE_OVER_TWO_PI;
      vec2f uv(phi * inv2Pi, theta * invPi);
    
      int ix = (int)clamp(uv.x*(float)dims.x,0.f,dims.x-1.f);
      int iy = (int)clamp(uv.y*(float)dims.y,0.f,dims.y-1.f);

      return {ix,iy};
    }
  
    inline __rtc_device vec3f
    EnvMapLight::DD::pixelToWorld(vec2i pixelID) const
    {
      const float f_x   = (pixelID.x+.5f)/dims.x;
      const float phi   = TWO_PI * f_x;
      const float f_y   = (pixelID.y+.5f)/dims.y;
      const float theta = ONE_PI * f_y;
    
      vec3f dir;
      dir.z = cosf(theta);
      dir.x = cosf(phi)*sinf(theta);
      dir.y = sinf(phi)*sinf(theta);

      return xfmVector(toWorld,dir);
    }

    inline __rtc_device vec3f
    EnvMapLight::DD::uvToWorld(float f_x, float f_y) const
    {
      const float phi   = TWO_PI * f_x;
      const float theta = ONE_PI * f_y;
    
      vec3f dir;
      dir.z = cosf(theta);
      dir.x = cosf(phi)*sinf(theta);
      dir.y = sinf(phi)*sinf(theta);

      return xfmVector(toWorld,dir);
    }

    inline __rtc_device Light::Sample
    EnvMapLight::DD::sample(Random &r, bool dbg) const
    {
      Light::Sample sample;
#if FORCE_CONSTANT_ENV
      sample.direction = randomDirection(r);
      sample.radiance = vec3f(1.f);
      sample.pdf = ONE_OVER_FOUR_PI;
      return sample;
#endif
      if (!texture) return {};

      float r_y = r();
      float r_x = r();
      float pdf_y;
      int iy = sampleCDF(cdf_y,dims.y,r_y,pdf_y,dbg);
      float pdf_x;
      int ix = sampleCDF(allCDFs_x+dims.x*iy,dims.x,r_x,pdf_x,dbg);

#if 1
      float sx = (ix+r())/dims.x;
      float sy = (iy+r())/dims.y;
#else
      float sx = (ix+.5f)/dims.x;
      float sy = (iy+.5f)/dims.y;
#endif
      vec4f fromTex = rtc::tex2D<vec4f>(texture,sx,sy);
      sample.radiance = scale * (vec3f&)fromTex;
      sample.direction = uvToWorld(sx,sy);
    
      float rel_y = sy;
      const float theta = ONE_PI * rel_y;
      sample.pdf
        = pdf_x*pdf_y
        * 1.f/(TWO_PI*ONE_PI*sinf(theta));
      sample.environment = true;
    
      sample.distance = BARNEY_INF;
      return sample;
    }
#endif
  }
}
