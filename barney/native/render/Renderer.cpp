// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/render/Renderer.h"
#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {
    
    /*! the base class for _any_ other type of object/actor in the
      barney class hierarchy */
    Renderer::Renderer(Context *context)
      : Object(context)
    {}

    /*! pretty-printer for printf-debugging */
    std::string Renderer::toString() const
    {
      return "barney::Renderer";
    }

    Renderer::SP Renderer::create(Context *context)
    { return std::make_shared<Renderer>(context); }

    void Renderer::commit()
    {
      bgColor            = staged.bgColor;
      ambientRadiance    = staged.ambientRadiance;
      pathsPerPixel      = staged.pathsPerPixel;
      bgTexture          = staged.bgTexture;
      cutPlane           = staged.cutPlane;
      maxVolumeBounces   = staged.maxVolumeBounces;
      volumeMultiScatter = staged.volumeMultiScatter;
      sampleLimit        = staged.sampleLimit;
    }
  
    bool Renderer::setObject(const std::string &member,
                             const std::shared_ptr<Object> &value) 
    {
      if (Object::setObject(member,value))
        return true;
      if (member == "bgTexture") {
        staged.bgTexture = value ? value->as<Texture>() : 0;
        return true;
      }
      return false;
    }
  
    bool Renderer::set1f(const std::string &member, const float &value)
    {
      if (Object::set1f(member,value))
        return true;
      if (member == "ambientRadiance") {
        staged.ambientRadiance = value;
        return true;
      }
      return false;
    }
  
    bool Renderer::set1i(const std::string &member, const int &value)
    {
      if (Object::set1i(member,value))
        return true;
      if (member == "pathsPerPixel") {
        staged.pathsPerPixel = value;
        return true;
      }
      if (member == "crosshairs") {
        staged.crosshairs = value;
        return true;
      }
      if (member == "sampleLimit") {
        staged.sampleLimit = value;
        return true;
      }
      if (member == "maxVolumeBounces") {
        staged.maxVolumeBounces = value;
        return true;
      }
      if (member == "volumeMultiScatter") {
        staged.volumeMultiScatter = value;
        return true;
      }
      return false;
    }
  
    bool Renderer::set4f(const std::string &member, const vec4f &value)
    {
      if (Object::set4f(member,value))
        return true;
      if (member == "bgColor") {
        staged.bgColor = value;
        return true;
      }
      if (member == "cutPlane") {
        staged.cutPlane = value;
        return true;
      }
      return false;
    }

    Renderer::DD Renderer::getDD(Device *device) const
    {
      Renderer::DD dd;
      dd.bgColor = bgColor;
      dd.bgTexture
        = bgTexture ? bgTexture->getTextureObject(device)
        : 0;
      dd.ambientRadiance = ambientRadiance;
      dd.pathsPerPixel = pathsPerPixel;
      dd.cutPlane = cutPlane;
      dd.maxVolumeBounces = maxVolumeBounces;
      dd.volumeMultiScatter = volumeMultiScatter;
      return dd;
    }

  }
}
