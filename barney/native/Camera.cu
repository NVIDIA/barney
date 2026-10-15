// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/Camera.h"
#include "native/Context.h"
#include <cstring>

namespace BARNEY_NS {
  namespace native {

  Camera::Camera(Context *owner)
    : Object(owner)
  {}

  bool Camera::set1i(const std::string &member, const int &value)
  {
    if (member == "motion.enabled") {
      motionEnabled = (value != 0);
      return true;
    }
    return false;
  }

  bool Camera::set4f(const std::string &member, const vec4f &value)
  {
    if (member == "imageRegion") {
      imageRegion = value;
      return true;
    }
    return false;
  }

  bool Camera::set4x4f(const std::string &member, const vec4f *value)
  {
    if (member == "motion.viewProjection") {
      std::memcpy(motionCurrViewProj, value, 16 * sizeof(float));
      return true;
    }
    if (member == "motion.previousViewProjection") {
      std::memcpy(motionPrevViewProj, value, 16 * sizeof(float));
      return true;
    }
    return false;
  }

  void Camera::commitBaseDeviceData()
  {
    dd.imageRegion = imageRegion;
    dd.haveMotionMatrices = motionEnabled;
    std::memcpy(dd.currViewProj, motionCurrViewProj, 16 * sizeof(float));
    std::memcpy(dd.prevViewProj, motionPrevViewProj, 16 * sizeof(float));
  }

  // ##################################################################

  struct PerspectiveCamera : public Camera {
    PerspectiveCamera(Context *owner)
      : Camera(owner)
    {}
    virtual ~PerspectiveCamera() = default;

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    
    vec3f position  { 0, 0, 0 };
    vec3f direction { 0, 0, 1 };
    vec3f up        { 0, 1, 0 };
    float aspect    = 1.f;
    float fovy      = 60.f;
    float focusDistance  = 0.f;
    float apertureRadius = 0.f;
    /*! allows for specifying that ray should originate
      'minumdistance' units away from the actual lens center,
      default is 0 */
    float nearClip = 0.f;
  };
  
  bool PerspectiveCamera::set1f(const std::string &member, const float &value)
  {
    if (Camera::set1f(member,value))
      return true;
    if (member == "aspect") {
      aspect = value;
      return true;
    }
    if (member == "apertureRadius") {
      apertureRadius = value;
      return true;
    }
    if (member == "focusDistance") {
      focusDistance = value;
      return true;
    }
    if (member == "nearClip") {
      nearClip = value;
      return true;
    }
    if (member == "fovy") {
      fovy = value;
      return true;
    }
    return false;
  }
  
  bool PerspectiveCamera::set3f(const std::string &member, const vec3f &value)
  {
    if (Camera::set3f(member,value))
      return true;
    if (member == "position") {
      position = value;
      return true;
    }
    if (member == "direction") {
      direction = normalize(value);
      return true;
    }
    if (member == "up") {
      up = value;
      return true;
    }
    return false;
  }
    
  void PerspectiveCamera::commit()
  {
    vec3f from   = (const vec3f&)position;
    vec3f dir_00 = normalize(direction);
    
    vec3f dir_du = normalize(cross(dir_00, up));
    vec3f dir_dv = normalize(cross(dir_du, dir_00));

    dir_00 *= (float)(1.f / (2.0f * tanf((0.5f * fovy) * (float)M_PI / 180.0f)));
    // dir_00 -= 0.5f * dir_du;
    // dir_00 -= 0.5f * dir_dv;

    dd.type = Camera::PERSPECTIVE;
    dd.perspective.dir_00  = dir_00;
    dd.perspective.dir_du  = dir_du;
    dd.perspective.dir_dv  = dir_dv;
    dd.perspective.lens_00 = from;
    dd.perspective.focusDistance = focusDistance;
    dd.perspective.apertureRadius = apertureRadius;
    dd.perspective.nearClip = nearClip;
    commitBaseDeviceData();
  }
    

  // ##################################################################

  struct OrthographicCamera : public Camera {
    OrthographicCamera(Context *owner)
      : Camera(owner)
    {}
    virtual ~OrthographicCamera() = default;

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    
    vec3f position  { 0, 0, 0 };
    vec3f direction { 0, 0, 1 };
    vec3f up        { 0, 1, 0 };
    float aspect    = 1.f;
    float height    = 1.f;
    float ortho_near      = 0.f;
    float ortho_far       = BARNEY_INF;
  };
  
  bool OrthographicCamera::set1f(const std::string &member, const float &value)
  {
    if (Camera::set1f(member,value))
      return true;
    if (member == "aspect") {
      aspect = value;
      return true;
    }
    if (member == "height") {
      height = value;
      return true;
    }
    if (member == "near") {
      ortho_near = value;
      return true;
    }
    if (member == "far") {
      ortho_far = value;
      return true;
    }
    return false;
  }
  
  bool OrthographicCamera::set3f(const std::string &member, const vec3f &value)
  {
    if (Camera::set3f(member,value))
      return true;
    if (member == "position") {
      position = value;
      return true;
    }
    if (member == "direction") {
      direction = normalize(value);
      return true;
    }
    if (member == "up") {
      up = value;
      return true;
    }
    return false;
  }
    
  void OrthographicCamera::commit()
  {
    vec3f from   = (const vec3f&)position;
    vec3f dir_00 = normalize(direction);
    
    vec3f dir_du = normalize(cross(dir_00, up));
    vec3f dir_dv = normalize(cross(dir_du, dir_00));

    dd.type = Camera::ORTHOGRAPHIC;
    dd.orthographic.dir     = dir_00;
    dd.orthographic.org_00  = from;
    dd.orthographic.org_du  = dir_du;
    dd.orthographic.org_dv  = dir_dv;
    dd.orthographic.height  = height;
    dd.orthographic.aspect  = aspect;
    commitBaseDeviceData();
  }
    

  // ##################################################################

  struct OmniCamera : public Camera {
    OmniCamera(Context *owner)
      : Camera(owner)
    {}
    virtual ~OmniCamera() = default;

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    
    vec3f position  { 0, 0, 0 };
    vec3f direction { 0, 0, 1 };
    vec3f up        { 0, 1, 0 };
    float aspect    = 1.f;
    float height    = 1.f;
    float ortho_near      = 0.f;
    float ortho_far       = BARNEY_INF;
  };
  
  bool OmniCamera::set1f(const std::string &member,
                                    const float &value)
  {
    if (Camera::set1f(member,value))
      return true;
    return false;
  }
  
  bool OmniCamera::set3f(const std::string &member,
                                    const vec3f &value)
  {
    if (Camera::set3f(member,value))
      return true;
    if (member == "position") {
      position = value;
      return true;
    }
    if (member == "direction") {
      direction = normalize(value);
      return true;
    }
    if (member == "up") {
      up = value;
      return true;
    }
    return false;
  }
    
  void OmniCamera::commit()
  {
    vec3f dir_00 = normalize(direction);

    linear3f toWorld;
    toWorld.vz = -normalize(up);
    toWorld.vy = -normalize(cross(toWorld.vz,direction));
    toWorld.vx = normalize(cross(toWorld.vy,toWorld.vz));

    dd.type = Camera::OMNIDIRECTIONAL;
    dd.omni.toWorld.l = toWorld;
    dd.omni.toWorld.p = position;
    commitBaseDeviceData();
  }
    

  // ##################################################################
  Camera::SP Camera::create(Context *owner,
                            const std::string &type)
  {
    if (type == "perspective")
      return std::make_shared<PerspectiveCamera>(owner);
    if (type == "orthographic")
      return std::make_shared<OrthographicCamera>(owner);
    if (type == "omni")
      return std::make_shared<OmniCamera>(owner);
    
    owner->warn_unsupported_object("Camera",type);
    return {};
  }

  }
}
