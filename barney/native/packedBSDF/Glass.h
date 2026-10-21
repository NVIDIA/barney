// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "native/render/DG.h"
#include "native/render/HitAttributes.h"

namespace BARNEY_NS {
namespace native {
namespace packedBSDF {
struct Glass
{
  float ior;
  rtc::float3 attenuation;

  inline __rtc_device float getOpacity(
      bool isShadowRay, bool, vec3f, vec3f, bool = false) const
  {
    return isShadowRay ? 0.f : 1.f;
  }
  inline __rtc_device float pdf(DG, vec3f, bool) const
  {
    return 0.f;
  }
  inline __rtc_device EvalRes eval(DG, vec3f, bool) const
  {
    return EvalRes::zero();
  }
  inline __rtc_device EvalRes evalProjected(DG dg, vec3f wi, bool dbg) const
  {
    return eval(dg, wi, dbg);
  }

  inline __rtc_device void scatter(
      ScatterResult &s, const DG &dg, Random &random, bool) const
  {
    s = {};
    const vec3f N = dg.orientShadingNormal(dg.Ns);
    const float c = dot(N, dg.wo);
    if (!(c > 0.f))
      return;
    const float eta = dg.insideMedium ? ior : 1.f / ior;
    const float sin2 = eta * eta * max(0.f, 1.f - c * c);
    const float ct = sqrtf(max(0.f, 1.f - sin2));
    const float rp = (eta * c - ct) / (eta * c + ct);
    const float rs = (c - eta * ct) / (c + eta * ct);
    const float F = sin2 >= 1.f ? 1.f : 0.5f * (rp * rp + rs * rs);
    const bool reflection = random() < F;
    s.dir = reflection ? reflect(-dg.wo, N) : refract(-dg.wo, N, eta);
    if ((dot(dg.Ng, s.dir) * dot(dg.Ng, dg.wo) > 0.f) != reflection)
      return;
    s.weight = vec3f(reflection ? 1.f : eta * eta);
    s.delta = true;
    s.eta = reflection ? 1.f : eta;
    s.event =
        reflection ? ScatterResult::REFLECTION : ScatterResult::TRANSMISSION;
    s.type = ScatterResult::SPECULAR;
  }
};
} // namespace packedBSDF
} // namespace native
} // namespace BARNEY_NS
