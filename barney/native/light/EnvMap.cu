// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/light/EnvMap.h"
#include "native/ModelSlot.h"
#include "native/Context.h"
#include "barney_rtc.h"

namespace BARNEY_NS {
  namespace native {

    RTC_IMPORT_COMPUTE2D(computeWeights_xy);
    RTC_IMPORT_COMPUTE1D(computeCDFs_doLine);
    RTC_IMPORT_COMPUTE1D(normalize_cdf_y);
  
    EnvMapLight::PLD *EnvMapLight::getPLD(Device *device)
    { return &perLogical[device->contextRank()]; }

    /*! computes an importance sampling weight for each pixel; gets
      called with one thread per pixel, in a 2d launch */
    struct ComputeWeights_xy {
      float *allLines_cdf_x;
      rtc::TextureObject texture;
      vec2i textureDims;

#if RTC_DEVICE_CODE
      inline __rtc_device
      void run(const rtc::ComputeInterface &ci);
#endif
    };
  
    /*! this kernel does one thread per line, then this one thread does
      entire line. not great, but we're not doing this per frame,
      anyway */
    struct ComputeCDFs_doLine {
      float *cdf_y;
      float *allLines_cdf_x;
      vec2i textureDims;

#if RTC_DEVICE_CODE
      inline __rtc_device
      void run(const rtc::ComputeInterface &ci);
#endif
    };

  
    /*! run by a single thread, to normalize the cdf_y */
    struct Normalize_cdf_y {
      float       *cdf_y;
      const float *allLines_cdf_x;
      vec2i        textureDims;
    
#if RTC_DEVICE_CODE
      inline __rtc_device
      void run(const rtc::ComputeInterface &ci);
#endif
    };


#if RTC_DEVICE_CODE
    inline __rtc_device
    void Normalize_cdf_y::run(const rtc::ComputeInterface &ci)
    {
      if (ci.getThreadIdx().x != 0) return;
        
      float sum = 0.f;
      for (int i=0;i<textureDims.y;i++)
        sum += cdf_y[i];
      if (!(sum > 0.f)) {
        for (int i=0;i<textureDims.y;i++)
          cdf_y[i] = float(i+1)/float(textureDims.y);
        return;
      }
      float rcp_sum = 1.f/sum;
        
      sum = 0.f;
      for (int i=0;i<textureDims.y;i++) {
        sum += cdf_y[i];
        cdf_y[i] = sum * rcp_sum;
      }
      cdf_y[textureDims.y-1] = 1.f;
    }
  
    inline __rtc_device
    void ComputeWeights_xy::run(const rtc::ComputeInterface &ci)
    {
      int ix = ci.getThreadIdx().x
        + ci.getBlockIdx().x
        * ci.getBlockDim().x;
      int iy = ci.getThreadIdx().y
        + ci.getBlockIdx().y
        * ci.getBlockDim().y;
        
      if (ix >= textureDims.x) return;
      if (iy >= textureDims.y) return;
        
      auto importance = [&](vec4f v)->float
      { return max(max(v.x,v.y),v.z); };
        
      float weight = 0.f;
      for (int iiy=0;iiy<=2;iiy++)
        for (int iix=0;iix<=2;iix++) {
          vec4f fromTex
            = rtc::tex2D<vec4f>(texture,
                                (ix+iix*.5f)/(textureDims.x),
                                (iy+iiy*.5f)/(textureDims.y));
          weight = max(weight,importance(fromTex));
        }
        
      allLines_cdf_x[ix+textureDims.x*iy] = weight;
    }
  
    inline __rtc_device
    void ComputeCDFs_doLine::run(const rtc::ComputeInterface &ci)
    {
      int tid
        = ci.getThreadIdx().x
        + ci.getBlockIdx().x * ci.getBlockDim().x;
        
      int y = tid;
      if (y >= textureDims.y) return;
      float *thisLine_pdf = allLines_cdf_x + y * textureDims.x;
        
      float sum = 0.f;
      for (int ix=0;ix<textureDims.x;ix++) 
        sum += thisLine_pdf[ix];
        
      if (!(sum > 0.f)) {
        for (int ix=0;ix<textureDims.x;ix++)
          thisLine_pdf[ix] = float(ix+1)/float(textureDims.x);
        cdf_y[y] = 0.f;
        return;
      }
      float rcp_sum = 1.f/sum;
      sum = 0.f;
      for (int ix=0;ix<textureDims.x;ix++) {
        sum += thisLine_pdf[ix];
        thisLine_pdf[ix] = sum * rcp_sum;
      }
      thisLine_pdf[textureDims.x-1] = 1.f;
        
      float rel_y = (y+.5f) / textureDims.y;
        
      const float theta = ONE_PI * rel_y;
        
      float relativeWeightOfLine = sum * sinf(theta);
      cdf_y[y] = relativeWeightOfLine;
    }
#endif
  
    EnvMapLight::DD EnvMapLight::getDD(Device *device,
                                       const affine3f &xfm) 
    {
      DD dd;
      dd.dims = dims;
      if (texture) {
        PLD *pld = getPLD(device);
        dd.texture
          = texture->getDD(device);
        dd.cdf_y
          = (const float *)pld->cdf_y->getDD();
        dd.allCDFs_x
          = (const float *)pld->allCDFs_x->getDD();
      } else {
        dd.texture = 0;
        dd.cdf_y = 0;
        dd.allCDFs_x = 0;
      }
      dd.scale   = params.scale;
      dd.toWorld = toWorld;
      dd.toLocal = toLocal;
      return dd;
    }

    void EnvMapLight::commit()
    {
#if 1
      toWorld.vz = -normalize(params.up);
      toWorld.vy = -normalize(cross(toWorld.vz,params.direction));
      toWorld.vx = normalize(cross(toWorld.vy,toWorld.vz));
#else
      toWorld.vz = normalize(params.up);
      toWorld.vy = normalize(cross(toWorld.vz,params.direction));
      toWorld.vx = normalize(cross(toWorld.vy,toWorld.vz));
#endif
      toLocal    = rcp(toWorld);
      assert(params.texture);
      texture    = params.texture;
      computeCDFs();
    }

    void EnvMapLight::computeCDFs()
    {
      assert(texture);
      dims = texture->getDims();
    
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;

        if (pld->cdf_y)
          rtc->freeBuffer(pld->cdf_y);
        pld->cdf_y
          = rtc->createBuffer(dims.y*sizeof(float));
      
        if (pld->allCDFs_x)
          rtc->freeBuffer(pld->allCDFs_x);
        pld->allCDFs_x
          = rtc->createBuffer(dims.x*dims.y*sizeof(float));

        /* computes an importance sampling weight for each pixel; gets
           called with one thread per pixel, in a 2d launch */
        {
          ComputeWeights_xy kernelData = {
            (float*)pld->allCDFs_x->getDD(),
            texture->getDD(device),
            dims
          };
          vec2ui bs = 16;
          vec2ui nb = divRoundUp(vec2ui(dims),bs);
          pld->computeWeights_xy
            ->launch(nb,bs,&kernelData);
        }
        /* this kernel will do one thread per line, then this one thread does
           entire line. not great, but we're not doing this per frame,
           anyway */
        {
          ComputeCDFs_doLine kernelData = {
            (float*)pld->cdf_y->getDD(),
            (float*)pld->allCDFs_x->getDD(),
            dims
          };
          pld->computeCDFs_doLine
            ->launch(dims.y,1024,
                     &kernelData);
        }
        /* run by a single thread, to normalize the cdf_y */
        {
          Normalize_cdf_y kernelArgs = {
            (float*)pld->cdf_y->getDD(),
            (float*)pld->allCDFs_x->getDD(),
            dims
          };
          pld->normalize_cdf_y
            ->launch(1,1,&kernelArgs);
        }
        device->rtc->sync();
      }
    }
  
  
    EnvMapLight::EnvMapLight(Context *context,
                             const DevGroup::SP &devices)
      : Light(context,devices)
    {
      perLogical.resize(devices->numLogical);
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;
        pld->cdf_y
          = rtc->createBuffer(sizeof(float));
        pld->allCDFs_x
          = rtc->createBuffer(sizeof(float));
      
        pld->computeWeights_xy
          = createCompute_computeWeights_xy(rtc);
        pld->computeCDFs_doLine
          = createCompute_computeCDFs_doLine(rtc);
        pld->normalize_cdf_y
          = createCompute_normalize_cdf_y(rtc);
      }
    }

    EnvMapLight::~EnvMapLight()
    {
      BN_TRACK_LEAKS(std::cout << "#barney: ~EnvMapLight deconstructing"
                     << std::endl);
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;
        if (pld->cdf_y)
          rtc->freeBuffer(pld->cdf_y);
      
        if (pld->allCDFs_x)
          rtc->freeBuffer(pld->allCDFs_x);
      }
    }

  
    bool EnvMapLight::set1f(const std::string &member,
                            const float &value) 
    {
      if (Light::set1f(member,value))
        return true;
    
      if (member == "scale") {
        params.scale = value;
        return true;
      }

      return false;
    }

    bool EnvMapLight::set2i(const std::string &member,
                            const vec2i &value) 
    {
      if (Light::set2i(member,value))
        return true;
    
      return false;
    }

    bool EnvMapLight::set3f(const std::string &member,
                            const vec3f &value) 
    {
      if (Light::set3f(member,value))
        return true;
    
      if (member == "direction") {
        params.direction = value;
        return true;
      }
      if (member == "up") {
        params.up = value;
        return true;
      }

      return false;
    }

    bool EnvMapLight::setObject(const std::string &member,
                                const Object::SP &value) 
    {
      if (Light::setObject(member,value))
        return true;
    
      if (member == "texture") {
        params.texture = value->as<Texture>();
        return true;
      }
      return false;
    }
  
    RTC_EXPORT_COMPUTE2D(computeWeights_xy,ComputeWeights_xy);
    RTC_EXPORT_COMPUTE1D(computeCDFs_doLine,ComputeCDFs_doLine);
    RTC_EXPORT_COMPUTE1D(normalize_cdf_y,Normalize_cdf_y);
  }
}

