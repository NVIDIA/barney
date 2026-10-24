// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/volume/NanoVDB.h" // Rebuild PTX when sampler/accelerator headers change.
#if BARNEY_HAVE_NANOVDB
#include "native/volume/MCAccelerator.h"
#include "barney_rtc.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::native::OptixGlobals);

namespace BARNEY_NS {
  namespace native {
    template<typename T>
    struct MCAccel_NanoVDB_Programs {
      static inline __rtc_device
      void bounds(const rtc::TraceInterface &ti,
                  const void *geomData,
                  owl::common::box3f &bounds,  
                  const int32_t primID)
      {
#if RTC_DEVICE_CODE
        MCVolumeAccel<NanoVDBDataSampler<T>>
          ::boundsProg(ti,geomData,bounds,primID);
#endif
      }
    
      static inline __rtc_device
      void intersect(rtc::TraceInterface &ti)
      {
#if RTC_DEVICE_CODE
        MCVolumeAccel<NanoVDBDataSampler<T>>
          ::isProg(ti);
#endif
      }
    
      static inline __rtc_device
      void closestHit(rtc::TraceInterface &ti)
      { /* nothing to do */ }
    
      static inline __rtc_device
      void anyHit(rtc::TraceInterface &ti)
      { /* nothing to do */ }
    };

    // OptiX creates a module per geometry type from this shared PTX. Keep
    // non-entry helpers local so those modules can coexist in one pipeline.
#if BARNEY_RTC_OPTIX
    namespace {
#endif
#define NANOVDB_EXPORT_GEOM(BuildType, Suffix, EnumName)                \
    RTC_EXPORT_USER_GEOM(                                               \
                          NanoVDBMC_##Suffix,                           \
                          typename MCVolumeAccel<NanoVDBDataSampler<BuildType>>::DD, \
                          MCAccel_NanoVDB_Programs<BuildType>,false,false);

    BARNEY_NANOVDB_FLOAT_TYPES(NANOVDB_EXPORT_GEOM)
#undef NANOVDB_EXPORT_GEOM
#if BARNEY_RTC_OPTIX
    }
#endif
  } // namespace native
} // namespace BARNEY_NS


#endif

