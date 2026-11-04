// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/barney-common.h"
#include "native/packedBSDF/PackedBSDF.h"
#include "native/packedBSDF/NVisii.h"
#include "native/render/HitAttributes.h"
#include "native/material/HostMaterial.h"

namespace BARNEY_NS {
  namespace native {
      
    struct AnariMatte : public HostMaterial {      
      struct DD {
#if RTC_DEVICE_CODE
        inline __rtc_device
        PackedBSDF createBSDF(const HitAttributes &hitData,
                              const Sampler::DD *samplers,
                              bool dbg) const;
        inline __rtc_device
        float getOpacity(const HitAttributes &hitData,
                         const Sampler::DD *samplers,
                         bool dbg) const;
#endif
        PossiblyMappedParameter::DD color;
        PossiblyMappedParameter::DD opacity;
      };
      AnariMatte(LDGContext *context);
      virtual ~AnariMatte() = default;

      bool setString(const std::string &member,
                     const std::string &value) override;
      bool set1f(const std::string &member,
                 const float &value) override;
      bool set3f(const std::string &member,
                 const vec3f &value) override;
      bool set4f(const std::string &member,
                 const vec4f &value) override;
      bool setObject(const std::string &member,
                     const Object::SP &value) override;
      
      std::string toString() const override { return "AnariMatte"; }
      
      DeviceMaterial getDD(Device *device) override;
      
      PossiblyMappedParameter color   = vec3f(.8f);
      PossiblyMappedParameter opacity = 1.f;
    };
      
#if RTC_DEVICE_CODE
    inline __rtc_device
    PackedBSDF AnariMatte::DD::createBSDF(const HitAttributes &hitData,
                                          const Sampler::DD *samplers,
                                          bool dbg) const
    {
      vec4f baseColor = this->color.eval(hitData,samplers,dbg);
      packedBSDF::Lambertian bsdf;
      (vec3f&)bsdf.albedo = (const vec3f&)baseColor;
      return bsdf;
    }

    inline __rtc_device
    float AnariMatte::DD::getOpacity(const HitAttributes &hitData,
                                    const Sampler::DD *samplers,
                                    bool dbg) const
    {
      vec4f baseColor = this->color.eval(hitData,samplers,dbg);
      vec4f opacity = this->opacity.eval(hitData,samplers,dbg);
      return baseColor.w * opacity.x;
    }
#endif    
  }
}
