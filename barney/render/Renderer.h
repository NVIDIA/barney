// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/Object.h"
#include "barney/common/Texture.h"

namespace BARNEY_NS {

  struct Device;
  
  /*! the base class for _any_ other type of object/actor in the
      barney class hierarchy */
  struct Renderer : public barney_api::Renderer {//Object {
    typedef std::shared_ptr<Renderer> SP;

    struct DD {
      vec4f              bgColor;
      rtc::TextureObject bgTexture;
      float              ambientRadiance;
      int                pathsPerPixel;
      vec4f              cutPlane;
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

    struct {
      Texture::SP bgTexture       = 0;
      vec4f       bgColor         = vec4f(0,0,0,1.f);
      int         pathsPerPixel   = 1;
      float       ambientRadiance = 1.f;
      int         crosshairs      = 0;
      vec4f       cutPlane        = vec4f(0,0,0,-1e30f);
    } staged;
    vec4f       bgColor         = vec4f(0,0,0,1.f);
    Texture::SP bgTexture       = 0;
    int         pathsPerPixel   = 1;
    float       ambientRadiance = 1.f;
    int         crosshairs      = 0;
    vec4f       cutPlane        = vec4f(0,0,0,-1e30f);
  };

}
