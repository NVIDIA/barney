// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/material/DeviceMaterial.h"
#include "barney/render/Sampler.h"
#include "barney/render/HitAttributes.h"
// #if RTC_DEVICE_CODE
#if BARNEY_DEVICE_PROGRAM
# include "rtcore/TraceInterface.h"
#endif
#include "barney/render/World.h"
#include "barney/render/HitIDs.h"

namespace BARNEY_NS {
  namespace render {

    /*! defines all constant global launch parameter data. The struct
        type itself is also defined on the host (so its contents can
        be marshalled there, but the 'get()' method can only be
        available in device programs */
    struct OptixGlobals {
#if BARNEY_DEVICE_PROGRAM
// #if RTC_DEVICE_CODE
      static inline __rtc_device
      const OptixGlobals &get(const rtc::TraceInterface &dev);

      /*! Returns true if a hit at parameter hit_t is on the invisible
          side of the cutting plane and should be rejected. */
      static inline __rtc_device
      bool hitOnInvisibleSide(const OptixGlobals &globals,
                              float hit_t,
                              const rtc::TraceInterface &ti)
      {
        const vec4f &cp = globals.cutPlane;
        if (cp.w <= -1e28f) return false;
        vec3f hitPos = (vec3f)ti.getWorldRayOrigin()
                     + hit_t * (vec3f)ti.getWorldRayDirection();
        return dot(vec3f(cp.x, cp.y, cp.z), hitPos) + cp.w < 0.f;
      }
#endif

      /*! the current ray queue for the traceRays() kernel */
      Ray             *rays;
      /*! info for primid/geomid/instid info; may be null if not required */
      HitIDs          *hitIDs;

      /*! number of ryas in the queue */
      int              numRays;

      /*! this device's world to trace rays into */
      rtc::AccelHandle accel;
      World::DD        world;

      /*! cutting plane (nx,ny,nz,d); disabled when w <= -1e28 */
      vec4f            cutPlane{0.f, 0.f, 0.f, -1e30f};
    };
  }
}

namespace BARNEY_NS {
  namespace render {

// #if RTC_DEVICE_CODE
#if BARNEY_DEVICE_PROGRAM
    inline __rtc_device
    const OptixGlobals &OptixGlobals::get(const rtc::TraceInterface &ti)
    {
      return *(OptixGlobals*)ti.getLPData();
    }
#endif
  }
}
