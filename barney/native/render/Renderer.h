// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Object.h"
#include "native/common/Texture.h"

namespace BARNEY_NS {
  namespace native {
    
    struct Device;
  
    /*! the base class for _any_ other type of object/actor in the
      barney class hierarchy */
    struct Renderer : public Object {
      typedef std::shared_ptr<Renderer> SP;

      struct DD {
        vec4f              bgColor;
        rtc::TextureObject bgTexture;
        float              ambientRadiance;
        int                pathsPerPixel;
        vec4f              cutPlane;
        int                maxVolumeBounces;
        int                volumeMultiScatter;
      };
    
      Renderer(Context *context);
      virtual ~Renderer() {}

      /*! pretty-printer for printf-debugging */
      std::string toString() const override;

      static SP create(Context *context);

      DD getDD(Device *device) const;
    
      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      bool setObject(const std::string &member,
                     const std::shared_ptr<Object> &value) override;
      bool set1i(const std::string &member, const int &value) override;
      bool set1f(const std::string &member, const float &value) override;
      bool set4f(const std::string &member, const vec4f &value) override;
      /*! @} */
      // ------------------------------------------------------------------

      struct { /* no default values here, they all get set on host anyway */
        vec4f       bgColor;
        vec4f       cutPlane;
        Texture::SP bgTexture;
        int         pathsPerPixel;
        float       ambientRadiance;
        int         crosshairs;
        int         maxVolumeBounces;
        int         volumeMultiScatter;
        int         sampleLimit;
      } staged;
      vec4f       bgColor         = vec4f(0,0,0,1.f);
      Texture::SP bgTexture       = 0;
      int         pathsPerPixel   = 1;
      float       ambientRadiance = 1.f;
      int         crosshairs      = 0;
      vec4f       cutPlane        = vec4f(0,0,0,-1e30f);
      int         maxVolumeBounces = 8;
      int         volumeMultiScatter = 0;
      // maximum number of samples accumulated:
      int         sampleLimit = 1024;
    };

  }
}
