// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Object.h"
#include "native/common/Data.h"
#include "native/common/Texture.h"

namespace BARNEY_NS {
  namespace native {
    
    struct ModelSlot;
  
    struct Light : public Object {
      typedef std::shared_ptr<Light> SP;

      struct DD {
        vec3f color;
      };
    
      /*! what we return, during rendering, when we sample a light
        source */
      struct Sample {
        /* direction _to_ light */
        vec3f direction;
        /*! radiance coming _from_ dir */
        vec3f radiance;
        /*! distance to this light sample */
        float distance;
        // Conditional solid-angle density (zero for discrete directions).
        float pdf = 0.f;
        // Estimator denominator is selectionProbability * pdf for continuous
        // directions, or selectionProbability alone for delta directions.
        float selectionProbability = 0.f;
        bool delta = false;
        bool environment = false;
      };
  
    
      Light(Context *context, const DevGroup::SP &devices);

      std::string toString() const override { return "Light<>"; }

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      // bool set1f(const std::string &member, const float &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      /*! @} */
      // ------------------------------------------------------------------
    
      static Light::SP create(Context *context,
                              const DevGroup::SP &devices,
                              const std::string &name);

      vec3f color = vec3f(1.f);
      DevGroup::SP const devices;
    };

  }
}
