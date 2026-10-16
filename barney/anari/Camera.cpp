// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "anari/Camera.h"

namespace BARNEY_NS {
  namespace anari {

    Camera::Camera(BarneyGlobalState *s)
      : Object(ANARI_CAMERA, s)
    {}

    Camera::~Camera()
    {
      if (m_barneyCamera)
        bnRelease(m_barneyCamera);
    }

    Camera *Camera::createInstance(std::string_view subtype, BarneyGlobalState *s)
    {
      if (subtype == "perspective")
        return new Perspective(s);
      if (subtype == "orthographic")
        return new Orthographic(s);
      if (subtype == "omnidirectional")
        return new Omni(s);
      return (Camera *)new UnknownObject(ANARI_CAMERA, subtype, s);
    }

    void Camera::commitParameters()
    {
      m_pos = getParam<math::float3>("position", math::float3(0.f, 0.f, 0.f));
      if (isnan(m_pos.x + m_pos.y + m_pos.z))
        reportMessage(ANARI_SEVERITY_ERROR,
                      "app set camera.position to NAN coordinates");

      m_dir = math::normalize
        (getParam<math::float3>("direction",math::float3(0.f, 0.f, 1.f)));
      if (isnan(m_dir.x + m_dir.y + m_dir.z))
        reportMessage(ANARI_SEVERITY_ERROR,
                      "app set camera.direction to NAN coordinates");

      m_up = math::normalize
        (getParam<math::float3>("up", math::float3(0.f, 1.f, 0.f)));
      if (isnan(m_up.x + m_up.y + m_up.z))
        reportMessage(ANARI_SEVERITY_ERROR,
                      "app set camera.up to NAN coordinates");
      m_imageRegion = math::float4(0.f, 0.f, 1.f, 1.f);
      getParam("imageRegion", ANARI_FLOAT32_BOX2, &m_imageRegion);

      const bool haveCurr = getParam("motion.viewProjection",
                                     ANARI_FLOAT32_MAT4, &m_currViewProj);
      const bool havePrev = getParam("motion.previousViewProjection",
                                     ANARI_FLOAT32_MAT4, &m_prevViewProj);
      m_haveMotionMatrices = haveCurr && havePrev;
    }

    void Camera::applyBaseCameraParameters()
    {
      bnSet4f(m_barneyCamera, "imageRegion",
              m_imageRegion.x, m_imageRegion.y,
              m_imageRegion.z, m_imageRegion.w);
      bnSet1i(m_barneyCamera, "motion.enabled", m_haveMotionMatrices ? 1 : 0);
      if (m_haveMotionMatrices) {
        bnSet4x4fv(m_barneyCamera, "motion.viewProjection",
                   (const bn_float4 *)&m_currViewProj);
        bnSet4x4fv(m_barneyCamera, "motion.previousViewProjection",
                   (const bn_float4 *)&m_prevViewProj);
      }
    }

    BNCamera Camera::barneyCamera() const
    {
      return m_barneyCamera;
    }

    // orthographic ///////////////////////////////////////////////////////////////////

    Orthographic::Orthographic(BarneyGlobalState *s) : Camera(s)
    {
      assert(deviceState());
      assert(deviceState()->tether);
      assert(deviceState()->tether->context);
      m_barneyCamera =
        bnCameraCreate(deviceState()->tether->context, "orthographic");
    }

    void Orthographic::commitParameters()
    {
      Camera::commitParameters();
      m_aspect = getParam<float>("aspect", 1.f);
      m_height = getParam<float>("height", 1.f);
      m_near   = getParam<float>("near", 1.f);
      m_far    = getParam<float>("far", 1.f);
    }

    void Orthographic::finalize()
    {
      bnSetVec(m_barneyCamera, "up", m_up);
      bnSetVec(m_barneyCamera, "position", m_pos);
      bnSetVec(m_barneyCamera, "direction", m_dir);
      bnSet1f(m_barneyCamera, "aspect", m_aspect);
      bnSet1f(m_barneyCamera, "height", m_height);
      bnSet1f(m_barneyCamera, "near", m_near);
      bnSet1f(m_barneyCamera, "far", m_far);
      applyBaseCameraParameters();
      bnCommit(m_barneyCamera);
    }

    // orthographic ///////////////////////////////////////////////////////////////////

    Omni::Omni(BarneyGlobalState *s) : Camera(s)
    {
      assert(deviceState());
      assert(deviceState()->tether);
      assert(deviceState()->tether->context);
      m_barneyCamera =
        bnCameraCreate(deviceState()->tether->context, "omni");
    }

    void Omni::commitParameters()
    {
      Camera::commitParameters();
      /*! nothing yet - in theory the omni camera has a 'layout'
        aprameter, but it can only take one value so we'll just ignore
        it for now */
    }

    void Omni::finalize()
    {
      bnSetVec(m_barneyCamera, "up", m_up);
      bnSetVec(m_barneyCamera, "position", m_pos);
      bnSetVec(m_barneyCamera, "direction", m_dir);
      applyBaseCameraParameters();
      bnCommit(m_barneyCamera);
    }

    // perspective ///////////////////////////////////////////////////////////////////

    Perspective::Perspective(BarneyGlobalState *s) : Camera(s)
    {
      assert(deviceState());
      assert(deviceState()->tether);
      assert(deviceState()->tether->context);
      m_barneyCamera =
        bnCameraCreate(deviceState()->tether->context, "perspective");
    }

    void Perspective::commitParameters()
    {
      Camera::commitParameters();
      m_aspect = getParam<float>("aspect", 1.f);
      m_fovy = getParam<float>("fovy", anari::radians(60.f));
      m_focusDistance = getParam<float>("focusDistance", 0.f);
      m_nearClip = getParam<float>("nearClip", 0.f);
      m_apertureRadius = getParam<float>("apertureRadius", 0.f);
    }

    void Perspective::finalize()
    {
      bnSetVec(m_barneyCamera, "up", m_up);
      bnSetVec(m_barneyCamera, "position", m_pos);
      bnSetVec(m_barneyCamera, "direction", m_dir);
      bnSet1f(m_barneyCamera, "aspect", m_aspect);
      bnSet1f(m_barneyCamera, "focusDistance", m_focusDistance);
      bnSet1f(m_barneyCamera, "nearClip", m_nearClip);
      bnSet1f(m_barneyCamera, "apertureRadius", m_apertureRadius);
      bnSet1f(m_barneyCamera, "fovy", anari::degrees(m_fovy));
      applyBaseCameraParameters();
      bnCommit(m_barneyCamera);
    }

  }
}

BARNEY_ANARI_TYPEFOR_DEFINITION(BARNEY_NS::anari::Camera *);
