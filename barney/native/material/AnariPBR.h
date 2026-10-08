// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/packedBSDF/PackedBSDF.h"
#include "native/material/HostMaterial.h"
#include "native/packedBSDF/PhysicallyBased.h"
#include <limits>

namespace BARNEY_NS {
  namespace native {
    
    struct AnariPBR : public HostMaterial {
      struct DD {
#if RTC_DEVICE_CODE
#ifdef __HIPCC__
        // Work around AMDGPU compiler return-address bug by forcing inlining
        // Possibly related to https://github.com/llvm/llvm-project/issues/224205
        __forceinline__ __rtc_device
#else
       inline __rtc_device
#endif
        PackedBSDF createBSDF(const HitAttributes &hitData,
                              const Sampler::DD *samplers,
                              bool dbg) const;
        inline __rtc_device
        float getOpacity(const HitAttributes &hitData,
                         const Sampler::DD *samplers,
                         bool dbg) const;
#endif
        PossiblyMappedParameter::DD baseColor;
        PossiblyMappedParameter::DD metallic;
        PossiblyMappedParameter::DD opacity;
        PossiblyMappedParameter::DD roughness;
        PossiblyMappedParameter::DD transmission;
        PossiblyMappedParameter::DD ior;
        PossiblyMappedParameter::DD emission;
        PossiblyMappedParameter::DD normal;
        PossiblyMappedParameter::DD occlusion;
        PossiblyMappedParameter::DD specular;
        PossiblyMappedParameter::DD specularColor;
        PossiblyMappedParameter::DD clearcoat;
        PossiblyMappedParameter::DD clearcoatRoughness;
        PossiblyMappedParameter::DD clearcoatNormal;
        PossiblyMappedParameter::DD thickness;
        PossiblyMappedParameter::DD attenuationDistance;
        PossiblyMappedParameter::DD attenuationColor;
        PossiblyMappedParameter::DD sheenColor;
        PossiblyMappedParameter::DD sheenRoughness;
        PossiblyMappedParameter::DD iridescence;
        PossiblyMappedParameter::DD iridescenceIor;
        PossiblyMappedParameter::DD iridescenceThickness;
      };
      
      AnariPBR(LDGContext *context);
      virtual ~AnariPBR() = default;
      
      std::string toString() const override { return "AnariPBR"; }
      
      DeviceMaterial getDD(Device *device) override;

      bool setObject(const std::string &member,
                     const Object::SP &value) override;
      bool setString(const std::string &member,
                     const std::string &value) override;
      bool set1f(const std::string &member,
                 const float &value) override;
      bool set3f(const std::string &member,
                 const vec3f &value) override;
      bool set4f(const std::string &member,
                 const vec4f &value) override;
      
      PossiblyMappedParameter baseColor    = vec3f(1.f,1.f,1.f);
      PossiblyMappedParameter metallic     = 1.f;
      PossiblyMappedParameter opacity      = 1.f;
      PossiblyMappedParameter roughness    = 1.f;
      PossiblyMappedParameter transmission = 0.f;
      PossiblyMappedParameter ior          = 1.5f;
      PossiblyMappedParameter emission     = vec3f(0.f,0.f,0.f);
      // Default to the geometric world normal; resolved to a world-space
      // shading normal in createBSDF (so a normal-map sampler perturbs it
      // through a TBN frame instead of being read as world space).
      PossiblyMappedParameter normal;
      PossiblyMappedParameter occlusion           = 1.f;
      PossiblyMappedParameter specular            = 0.f;
      PossiblyMappedParameter specularColor       = vec3f(1.f,1.f,1.f);
      PossiblyMappedParameter clearcoat           = 0.f;
      PossiblyMappedParameter clearcoatRoughness  = 0.f;
      PossiblyMappedParameter clearcoatNormal;
      PossiblyMappedParameter thickness           = 0.f;
      PossiblyMappedParameter attenuationDistance
        = std::numeric_limits<float>::infinity();
      PossiblyMappedParameter attenuationColor    = vec3f(1.f,1.f,1.f);
      PossiblyMappedParameter sheenColor          = vec3f(0.f,0.f,0.f);
      PossiblyMappedParameter sheenRoughness      = 0.f;
      PossiblyMappedParameter iridescence         = 0.f;
      PossiblyMappedParameter iridescenceIor      = 1.3f;
      PossiblyMappedParameter iridescenceThickness = 0.f;
    };
      
#if RTC_DEVICE_CODE
#ifdef USE_HIP
    __forceinline__ __rtc_device
#else
    inline __rtc_device
#endif
    PackedBSDF AnariPBR::DD::createBSDF(const HitAttributes &hitData,
                                        const Sampler::DD *samplers,
                                        bool dbg) const
    {
      vec4f baseColor    = this->baseColor   .eval(hitData,samplers,dbg);
      vec4f metallic     = this->metallic    .eval(hitData,samplers,dbg);
      vec4f roughness    = this->roughness   .eval(hitData,samplers,dbg);
      vec4f transmission = this->transmission.eval(hitData,samplers,dbg);
      vec4f ior          = this->ior         .eval(hitData,samplers,dbg);
      vec4f normal       = this->normal      .eval(hitData,samplers,dbg);
      vec4f emission     = this->emission    .eval(hitData,samplers,dbg);
      vec4f occlusion    = this->occlusion   .eval(hitData,samplers,dbg);
      vec4f specular     = this->specular    .eval(hitData,samplers,dbg);
      vec4f specularColor = this->specularColor.eval(hitData,samplers,dbg);
      vec4f clearcoat    = this->clearcoat   .eval(hitData,samplers,dbg);
      vec4f clearcoatRoughness
        = this->clearcoatRoughness.eval(hitData,samplers,dbg);
      vec4f clearcoatNormal
        = this->clearcoatNormal.eval(hitData,samplers,dbg);
      vec4f thickness    = this->thickness   .eval(hitData,samplers,dbg);
      vec4f attenuationDistance
        = this->attenuationDistance.eval(hitData,samplers,dbg);
      vec4f attenuationColor
        = this->attenuationColor.eval(hitData,samplers,dbg);
      vec4f sheenColor   = this->sheenColor  .eval(hitData,samplers,dbg);
      vec4f sheenRoughness
        = this->sheenRoughness.eval(hitData,samplers,dbg);
      vec4f iridescence  = this->iridescence .eval(hitData,samplers,dbg);
      vec4f iridescenceIor
        = this->iridescenceIor.eval(hitData,samplers,dbg);
      vec4f iridescenceThickness
        = this->iridescenceThickness.eval(hitData,samplers,dbg);

      packedBSDF::PhysicallyBased bsdf;
      bsdf.setDefaults();
      {
        const vec3f bc = (const vec3f&)baseColor;
        bsdf.baseColor = bc;
      }
      // Resolve the shading normal to world space. The geometric world
      // normal is the fallback and the frame a normal map is perturbed
      // through.
      const vec3f Ng = normalize((const vec3f&)hitData.worldNormal);
      if (this->normal.type == PossiblyMappedParameter::SAMPLER) {
        const vec3f ts = normalize(((const vec3f)normal)*2.f-1.f);
        const vec3f N = packedBSDF::physicallybased::applyNormalMap
          (ts,Ng,hitData.worldTangent);
        bsdf.normal = N;
      } else {
        const vec3f N = normalize((const vec3f)normal);
        const vec3f Nw = (dot(N,N) > 1e-12f) ? N : Ng;
        bsdf.normal = Nw;
      }
      // Clearcoat normal: same TBN treatment as the base normal.
      if (this->clearcoatNormal.type == PossiblyMappedParameter::SAMPLER) {
        const vec3f ts = normalize(((const vec3f)clearcoatNormal)*2.f-1.f);
        const vec3f Nc = packedBSDF::physicallybased::applyNormalMap
          (ts,Ng,hitData.worldTangent);
        bsdf.clearcoatNormal = Nc;
      } else {
        const vec3f Nc = normalize((const vec3f)clearcoatNormal);
        const vec3f Nw = (dot(Nc,Nc) > 1e-12f) ? Nc : Ng;
        bsdf.clearcoatNormal = Nw;
      }
      bsdf.metallic = metallic.x;
      bsdf.roughness = roughness.x;
      bsdf.transmission = transmission.x;
      // Stored raw; the BSDF derives eta from dg.insideMedium at shading time
      // (eta = insideMedium ? ior : 1/ior).
      // Clamp away degenerate ior <= 0, which would make eta = 1/ior infinite
      // and poison computeDielectricF0 with NaNs.
      bsdf.ior = fmaxf(ior.x, 1e-3f);
      bsdf.emissive = rtc::float3(emission.x,emission.y,emission.z);
      bsdf.occlusion = occlusion.x;
      bsdf.specular = specular.x;
      bsdf.specularColor
        = vec3f(specularColor.x,specularColor.y,specularColor.z);
      bsdf.clearcoat = clearcoat.x;
      bsdf.clearcoatRoughness = clearcoatRoughness.x;
      bsdf.thickness = thickness.x;
      bsdf.attenuationDistance = attenuationDistance.x;
      bsdf.attenuationColor
        = vec3f(attenuationColor.x,attenuationColor.y,
                attenuationColor.z);
      bsdf.sheenColor = vec3f(sheenColor.x,sheenColor.y,sheenColor.z);
      bsdf.sheenRoughness = sheenRoughness.x;
      bsdf.iridescence = iridescence.x;
      bsdf.iridescenceIor = iridescenceIor.x;
      bsdf.iridescenceThickness = iridescenceThickness.x;
      
      return bsdf;
    }

    inline __rtc_device
    float AnariPBR::DD::getOpacity(const HitAttributes &hitData,
                                  const Sampler::DD *samplers,
                                  bool dbg) const
    {
      vec4f baseColor = this->baseColor.eval(hitData,samplers,dbg);
      vec4f opacity = this->opacity.eval(hitData,samplers,dbg);
      return baseColor.w * opacity.x;
    }
#endif
    
  }
}
