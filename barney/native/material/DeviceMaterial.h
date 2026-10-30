// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/render/Ray.h"
#include "native/packedBSDF/PackedBSDF.h"
#include "native/render/HitAttributes.h"
#include "native/material/AnariMatte.h"
#include "native/material/AnariPBR.h"
#include "native/material/NVisii.h"
#include "native/material/Glass.h"

namespace BARNEY_NS {
  namespace native {
      
    struct DeviceMaterial {
      typedef enum {
        INVALID=0,
        TYPE_AnariMatte,
        TYPE_AnariPBR,
        TYPE_NVisii,
        TYPE_Glass
      } Type;

#if RTC_DEVICE_CODE
      inline __rtc_device
      PackedBSDF createBSDF(const HitAttributes &hitData,
                            const Sampler::DD *samplers,
                            bool dbg=false) const;
    private:
      inline __rtc_device
      float rawOpacity(const HitAttributes &hitData,
                       const Sampler::DD *samplers,
                       bool dbg=false) const;

    public:
      inline __rtc_device
      float coverage(const HitAttributes &hitData,
                     const Sampler::DD *samplers,
                     bool dbg=false) const;

      inline __rtc_device
      void setHit(Ray &ray,
                  const HitAttributes &hitData,
                  const Sampler::DD *samplers,
                  bool dbg=false) const;
#endif      
      Type type;
      AlphaMode alphaMode = AlphaMode::Opaque;
      float alphaCutoff = 0.5f;
      union {
        AnariPBR::DD   anariPBR;
        AnariMatte::DD anariMatte;
        NVisii::DD     nvisii;
        Glass::DD      glass;
      };
    };

#if RTC_DEVICE_CODE
    inline __rtc_device
    PackedBSDF DeviceMaterial::createBSDF(const HitAttributes &hitData,
                                          const Sampler::DD *samplers,
                                          bool dbg) const
    {
      if (type == TYPE_AnariMatte)
        return anariMatte.createBSDF(hitData,samplers,dbg);
      if (type == TYPE_AnariPBR)
        return anariPBR.createBSDF(hitData,samplers,dbg);
      if (type == TYPE_NVisii)
        return nvisii.createBSDF(hitData,samplers,dbg);
      if (type == TYPE_Glass)
        return glass.createBSDF(hitData,samplers,dbg);
#ifndef NDEBUG
      printf("#bn: DeviceMaterial::createBSDF encountered an invalid "
             "device material type (%i); most likely this is the app"
             " not having properly committed its material\n",(int)type);
#endif
      return packedBSDF::Invalid();
    }

    inline __rtc_device
    float DeviceMaterial::rawOpacity(const HitAttributes &hitData,
                                     const Sampler::DD *samplers,
                                     bool dbg) const
    {
      if (type == TYPE_AnariMatte)
        return anariMatte.getOpacity(hitData,samplers,dbg);
      if (type == TYPE_AnariPBR)
        return anariPBR.getOpacity(hitData,samplers,dbg);
      if (type == TYPE_NVisii)
        return nvisii.getOpacity(hitData,samplers,dbg);
      if (type == TYPE_Glass)
        return glass.getOpacity(hitData,samplers,dbg);
      return 1.f;
    }

    inline __rtc_device
    float DeviceMaterial::coverage(const HitAttributes &hitData,
                                   const Sampler::DD *samplers,
                                   bool dbg) const
    {
      if (alphaMode == AlphaMode::Opaque)
        return 1.f;
      const float opacity = rawOpacity(hitData,samplers,dbg);
      if (alphaMode == AlphaMode::Mask)
        return (opacity < alphaCutoff) ? 0.f : 1.f;
      // Blend: opacity doubles as the stochastic keep probability, so it
      // must be clamped into [0,1] even if a texture returns out-of-range
      // values.
      return clamp(opacity, 0.f, 1.f);
    }

    inline __rtc_device
    void DeviceMaterial::setHit(Ray &ray,
                                const HitAttributes &hitData,
                                const Sampler::DD *samplers,
                                bool dbg) const
    {
      ray.setHit(hitData.worldPosition,hitData.worldShadingNormal,
                 hitData.t,createBSDF(hitData,samplers,dbg),
                 hitData.worldGeometricNormal);
    }
#endif
  }
}
