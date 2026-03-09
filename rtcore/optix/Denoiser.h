// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/optix/Device.h"

namespace rtc {
  namespace optix {

    /*! abstract interface to a denoiser. implementation(s) depend of
        which optix version and/or oidn are available */
    struct Denoiser {
      Denoiser(Device* device) : device(device) {}
      virtual ~Denoiser() = default;
      virtual void resize(vec2i dims) = 0;
      virtual void run(float blendFactor) = 0;
      vec4f *out_rgba  = 0;
      vec4f *in_rgba   = 0;
      vec3f *in_normal = 0;
      Device* const device;

      /*! if true, the denoiser will use UPSCALE2X mode: input is at
          render resolution, output is at 2x render resolution. The
          caller must set this before calling resize(). */
      bool upscaleMode = false;

      /*! the actual output dimensions. When upscaleMode is false this
          equals the input dims; when true it is 2x the input dims. Set
          by resize(). */
      vec2i outputDims = {0,0};
    };

#if OPTIX_VERSION >= 80000
    /*! denoising using optix 8 built-in denoiser. only available for
        optix 8 or newer */
    struct Optix8Denoiser : public Denoiser {
      Optix8Denoiser(Device *device);
      virtual ~Optix8Denoiser();
      void resize(vec2i dims) override;
      void run(float blendFactor) override;
      
      vec2i                numPixels;
      OptixDenoiser        denoiser = {};
      OptixDenoiserOptions denoiserOptions;
      void                *denoiserScratch = 0;
      void                *denoiserState   = 0;
      OptixDenoiserSizes   denoiserSizes;

      /*! tracks the mode the OptixDenoiser was created with, so we
          know when to destroy + recreate */
      bool currentUpscaleMode = false;

      /*! set to false when denoiser creation or weight loading fails
          (e.g. missing/incompatible nvoptix.bin); resize() and run()
          become no-ops so the renderer can proceed without denoising */
      bool available = true;

      /*! destroy and recreate the OptixDenoiser if upscaleMode
          changed since the last call */
      void recreateIfNeeded();
    };
#endif
    
    template<typename T>
    void printDev(const char *text, const T *t)
    {
      T onHost;
      cudaMemcpy(&onHost,t,sizeof(T),cudaMemcpyDefault);
      std::cout << text << " : " << onHost << std::endl;
    }
  }
}

