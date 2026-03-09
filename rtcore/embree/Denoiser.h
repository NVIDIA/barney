// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/embree/Device.h"

#if BARNEY_OIDN_CPU
# include <OpenImageDenoise/oidn.h>
#endif

namespace rtc {
  namespace embree {

    struct Denoiser {
      Denoiser(Device *device) : rtc(device) {}
      virtual ~Denoiser() = default;
      virtual void resize(vec2i dims) = 0;
      virtual void run(float blendFactor) = 0;

      vec4f *out_rgba  = 0;
      vec4f *in_rgba   = 0;
      vec3f *in_normal = 0;
                             
      Device *const rtc;

      bool upscaleMode = false;
      vec2i outputDims = {0,0};
    };
    
#if BARNEY_OIDN_CPU
    /*! oidn-based CPU denoiser */
    struct DenoiserOIDN : public Denoiser
    {
      DenoiserOIDN(Device *device);
      virtual ~DenoiserOIDN();
      
      void resize(vec2i size) override;
      void run(float blendFactor) override;
      
    private:
      void freeMem();

      vec2i         numPixels { 0,0 };
      
      OIDNDevice oidnDevice = 0;
      OIDNFilter filter = 0;
    };
#endif
  }
}
