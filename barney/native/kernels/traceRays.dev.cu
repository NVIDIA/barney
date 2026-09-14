// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/render/OptixGlobals.h"
#include "native/Context.h"
#include "native/GlobalModel.h"
#include "native/ModelSlot.h"
#include "native/render/SamplerRegistry.h"
#include "native/render/MaterialRegistry.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::native::OptixGlobals);

namespace BARNEY_NS {
  namespace native {

    struct TraceRays {
#if RTC_DEVICE_CODE
      inline __rtc_device static 
      void run(rtc::TraceInterface &ti);
#endif
    };

#if RTC_DEVICE_CODE
    inline __rtc_device  
    void TraceRays::run(rtc::TraceInterface &ti)
    {
      const int rayID
        = ti.getLaunchIndex().x
        + ti.getLaunchDims().x
        * ti.getLaunchIndex().y;

      auto &lp = OptixGlobals::get(ti);

      if (rayID >= lp.numRays)
        return;
      
      Ray &ray = lp.rays[rayID];
      
      vec3f dir = ray.dir;
      if (dir.x == 0.f) dir.x = 1e-6f;
      if (dir.y == 0.f) dir.y = 1e-6f;
      if (dir.z == 0.f) dir.z = 1e-6f;
      ti.traceRay(lp.accel,
                  ray.org,
                  dir,
                  0.f,
                  ray.tMax,
                  /* PRD */
                  (void *)&ray);
    }
#endif
  
    RTC_EXPORT_TRACE2D(traceRays,native::TraceRays);
  }
}

