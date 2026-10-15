// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/anari/Object.h"

namespace BARNEY_NS {
  namespace anari {

    /*! abstract base class for any anari camera type */
    struct Camera : public Object
    {
      Camera(BarneyGlobalState *s);
      ~Camera() override;

      virtual void commitParameters() override;

      static Camera *createInstance(std::string_view subtype,
                                    BarneyGlobalState *state);

      BNCamera barneyCamera() const;

    protected:
      math::float3 m_pos;
      math::float3 m_dir;
      math::float3 m_up;
      math::float4 m_imageRegion;

      bool         m_haveMotionMatrices = false;
      math::mat4   m_currViewProj;
      math::mat4   m_prevViewProj;

      /*! forwards base camera parameters to the underlying bnCamera. Subtypes
        call this from finalize() before bnCommit. */
      void applyBaseCameraParameters();

      BNCamera m_barneyCamera{nullptr};
    };

    // Subtypes ///////////////////////////////////////////////////////////////////

    /*! ANARI 'perspective' camera type */
    struct Perspective : public Camera
    {
      Perspective(BarneyGlobalState *s);

      void commitParameters() override;
      void finalize() override;

    private:
      float m_fovy{30.f};
      float m_aspect{1.f};
      float m_focusDistance = 0.f;
      float m_nearClip = 0.f;
      float m_apertureRadius = 0.f;
    };

    /*! ANARI 'orthographic' camera type - implements an
      orthographic/parallel projection camera. */
    struct Orthographic : public Camera
    {
      Orthographic(BarneyGlobalState *s);

      void commitParameters() override;
      void finalize() override;

    private:
      float m_aspect = 1.f;
      float m_height = 0.f;
      float m_near   = 0.f;
      float m_far    = std::numeric_limits<float>::infinity();
    };

    /*! ANARI 'omnidirectional' camera type - implements a camera type
      that is the direct opposite of a envmap light source */
    struct Omni : public Camera
    {
      Omni(BarneyGlobalState *s);

      void commitParameters() override;
      void finalize() override;

    private:
      /*! nothing yet - in theory the omni camera has a 'layout'
        aprameter, but it can only take one value so we'll just ignore
        it for now */
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Camera *, ANARI_CAMERA);
