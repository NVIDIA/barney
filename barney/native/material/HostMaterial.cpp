// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/Context.h"
#include "native/material/DeviceMaterial.h"
#include "native/material/AnariPBR.h"
#include "native/material/AnariMatte.h"
#include "native/ModelSlot.h"

namespace BARNEY_NS {
  namespace native {
    
    PossiblyMappedParameter::DD
    PossiblyMappedParameter::getDD(Device *device) 
    {
      PossiblyMappedParameter::DD dd;
      dd.type = type;
      switch(type) {
      case SAMPLER:
        dd.samplerID = sampler ? sampler->samplerID : -1;
        break;
      case ATTRIBUTE:
        dd.attribute = attribute;
        break;
      case VALUE:
        (vec4f&)dd.value = value;
        break;
      case INVALID:
        (vec4f&)dd.value = vec4f(0.f,0.f,0.f,0.f);
        break;
      }
      return dd;
    }
    
    void PossiblyMappedParameter::set(const vec3f  &v)
    {
      set(vec4f(v.x,v.y,v.z,1.f));
    }

    void PossiblyMappedParameter::set(const float &v)
    {
      set(vec4f(v,0.f,0.f,1.f));
    }

    void PossiblyMappedParameter::set(const vec4f &v)
    {
      type    = VALUE;
      sampler = {};
      value   = v;
    }

    void PossiblyMappedParameter::set(Sampler::SP s)
    {
      // A null sampler (mistyped or since-removed object) must not be
      // recorded as SAMPLER: getDD() would store samplerID -1 and the device
      // side would index samplers[-1] on eval.
      if (!s) {
        type  = INVALID;
        value = vec4f(0.f,0.f,0.f,1.f);
        return;
      }
      type = SAMPLER;
      sampler   = s;
    }

    void PossiblyMappedParameter::set(const std::string &attributeName)
    {
      sampler = {};
      type    = ATTRIBUTE;
      attribute = parseAttribute(attributeName);
    }
    
    HostMaterial::HostMaterial(LDGContext *slotContext)
      : Object(slotContext->context),
        devices(slotContext->devices),
        materialRegistry(slotContext->materialRegistry),
        materialID(slotContext->materialRegistry->allocate())
    {
      assert(slotContext->context);
    }

    HostMaterial::~HostMaterial()
    {
      BN_TRACK_LEAKS(std::cout << "#barney: ~HostMaterial deconstructing"
                     << std::endl);
      materialRegistry->release(materialID);
    }
    
    HostMaterial::SP HostMaterial::create(LDGContext *slotContext,
                                          const std::string &type)
    {
      if (type == "AnariMatte" || type == "matte")
        return std::make_shared<AnariMatte>(slotContext); 
      if (type == "physicallyBased" || type == "AnariPBR")
        return std::make_shared<AnariPBR>(slotContext); 
      return std::make_shared<AnariPBR>(slotContext); 
    }

    bool HostMaterial::setString(const std::string &member,
                                 const std::string &value)
    {
      if (Object::setString(member,value)) return true;
      if (member == "alphaMode") {
        alphaModeSet = true;
        if (value == "opaque")
          alphaMode = AlphaMode::Opaque;
        else if (value == "mask")
          alphaMode = AlphaMode::Mask;
        else if (value == "blend")
          alphaMode = AlphaMode::Blend;
        else {
          alphaModeSet = false;
          warn_unsupported_member("string", "alphaMode");
        }
        return true;
      }
      return false;
    }

    bool HostMaterial::set1f(const std::string &member, const float &value)
    {
      if (Object::set1f(member,value)) return true;
      if (member == "alphaCutoff") {
        alphaCutoffSet = true;
        alphaCutoff = value;
        return true;
      }
      return false;
    }

    bool HostMaterial::set1i(const std::string &member, const int &value)
    {
      if (Object::set1i(member,value)) return true;
      // Internal wire names used by the anari wrapper (setBNCoverage) to
      // convey parameter-presence bookkeeping; the 'bn:' prefix keeps them
      // from colliding with application parameter names.
      if (member == "bn:alphaModeSet") {
        alphaModeSet = value != 0;
        return true;
      }
      if (member == "bn:alphaCutoffSet") {
        alphaCutoffSet = value != 0;
        return true;
      }
      return false;
    }

    void HostMaterial::packCoverage(DeviceMaterial &dd,
                                    bool opacityIdenticallyOne) const
    {
      // ANARI 1.2 has no alphaMode parameter; coverage is derived:
      //  - an explicitly set legacy alphaMode always wins (1.0/1.1
      //    compatibility),
      //  - alphaCutoff alone implies Mask,
      //  - anything else defaults to Opaque, the documented ANARI default;
      //    apps wanting stochastic blending set alphaMode = "blend".
      AlphaMode mode;
      if (alphaModeSet) {
        mode = alphaMode;
        if (mode == AlphaMode::Mask) {
          const float c = alphaCutoffSet ? alphaCutoff : 0.5f;
          if (c <= 0.f)
            mode = AlphaMode::Opaque;
        }
      } else if (alphaCutoffSet) {
        mode = (alphaCutoff <= 0.f) ? AlphaMode::Opaque : AlphaMode::Mask;
      } else {
        mode = AlphaMode::Opaque;
      }
      // A blend material whose opacity is provably constant 1 is
      // indistinguishable from Opaque (coverage() returns 1 either way) and
      // lets anyHit skip the opacity evaluation entirely.
      if (mode == AlphaMode::Blend && opacityIdenticallyOne)
        mode = AlphaMode::Opaque;

      dd.alphaMode = mode;
      dd.alphaCutoff = alphaCutoffSet ? alphaCutoff : 0.5f;
    }

    void HostMaterial::commit()
    {
      for (auto device : *devices) {
        DeviceMaterial dd = getDD(device);
        materialRegistry->setMaterial(materialID,dd,device);
      }
      hasBeenCommittedAtLeastOnce = true;      
    }
  
  }
}
