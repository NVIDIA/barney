// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "anari/light/Quad.h"

namespace BARNEY_NS {
  namespace anari {
  
    QuadLight::QuadLight(BarneyGlobalState *s) : Light(s) {}

    void QuadLight::commitParameters()
    {
      Light::commitParameters();
      m_position = getParam<math::float3>("position", math::float3(0.f, 0.f, 0.f));
      m_edge1 = getParam<math::float3>("edge1", math::float3(1.f, 0.f, 0.f));
      m_edge2 = getParam<math::float3>("edge2", math::float3(0.f, 1.f, 0.f));
      m_intensity = getParam<float>("intensity", 1.f);
    }

    const char *QuadLight::bnSubtype() const
    {
      return "quad";
    }

    void QuadLight::setBarneyParameters()
    {
      if (!m_bnLight)
        return;
      bnSetVec(m_bnLight, "corner", m_position);
      bnSetVec(m_bnLight, "edge0", m_edge1);
      bnSetVec(m_bnLight, "edge1", m_edge2);
      bnSetVec(m_bnLight, "emission", m_color*m_intensity);
      bnCommit(m_bnLight);
    }

  }
} // ::barney_device
