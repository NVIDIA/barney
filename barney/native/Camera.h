// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/barney-common.h"
#include "native/common/math.h"
#include "native/Object.h"

namespace BARNEY_NS {
  namespace native {

    /*! the camera model we use in barney */
    struct Camera : public Object {
      typedef std::shared_ptr<Camera> SP;
      
      typedef enum {
        UNDEFINED=0,
        PERSPECTIVE,
        ORTHOGRAPHIC,
        OMNIDIRECTIONAL
      } Type;
      
      /*! device-data for the camera object; to avoid virtual functions
        this currently uses a 'type'-switch, so the camera code on the
        device will have to 'interpret' what the actual fields
        mean. on the host side, all derived cameras will have to fill
        in this one shared struct */
      struct DD {
        inline DD() : type(UNDEFINED) {};
        inline DD(const DD &) = default;
        inline ~DD() = default;
      
        Type  type = UNDEFINED;
        vec4f imageRegion {0.f, 0.f, 1.f, 1.f};
        
        /* Motion-vector reprojection matrices (world -> clip). When
           haveMotionMatrices is false, BN_FB_MOTION writes are skipped
           regardless of whether per-instance deltas were supplied. */
        bool     haveMotionMatrices = false;
        float    currViewProj[16]   = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        float    prevViewProj[16]   = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

        union {
          /* vector along u direction, for ONE pixel */
          struct {
            /*! vector from camera center to to lower-left pixel (i.e., pixel
              (0,0)) on the focal plane */
            vec3f dir_00;
            vec3f dir_du;
            /* vector along v direction, for ONE pixel */
            vec3f dir_dv;
            /*! lens center ... */
            vec3f lens_00;
            /* radius of lens, for DOF */
            float apertureRadius;
            /* distance to focal plane, for DOF */
            float focusDistance;
            /*! allows for specifying that ray should originate
              'minumdistance' units away from the actual lens center,
              default is 0 */
            float nearClip;
          } perspective;
          struct {
            vec3f org_00;
            vec3f org_du;
            vec3f org_dv;
            vec3f dir;
            float height;
            float aspect;
          } orthographic;
          struct {
            affine3f toWorld;
          } omni;
        };
      };
      DD dd;

      Camera(Context *owner);
      virtual ~Camera() = default;

      /*! subtypes call from their commit() so base camera state propagates
        into DD after subtype-specific fields are populated */
      void commitBaseDeviceData();
      
      bool set1i(const std::string &member, const int &value) override;
      bool set4f(const std::string &member, const vec4f &value) override;
      bool set4x4f(const std::string &member, const vec4f *value) override;
      
      static Camera::SP create(Context *owner, const std::string &type);

      DD getDD() { return dd; }
    protected:
      bool  motionEnabled = false;
      vec4f imageRegion {0.f, 0.f, 1.f, 1.f};
      float motionCurrViewProj[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
      float motionPrevViewProj[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
      
    };

  }
}
