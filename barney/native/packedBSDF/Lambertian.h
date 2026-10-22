// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "native/render/ShadingInteraction.h"

namespace BARNEY_NS {
namespace native {
namespace packedBSDF {
struct Lambertian
{
  rtc::float3 albedo;

  inline __rtc_device vec3f normal(const ShadingInteraction &interaction) const
  {
    return interaction.orientShadingNormal(interaction.shadingNormal);
  }

  inline __rtc_device float pdf(ShadingInteraction interaction, vec3f wi, bool dbg) const
  {
    const vec3f N = normal(interaction);
    if (dot(N, interaction.wo) <= 0.f || dot(interaction.geometricNormal, wi) * dot(interaction.geometricNormal, interaction.wo) <= 0.f)
      return 0.f;
    return max(dot(N, wi), 0.f) * ONE_OVER_PI;
  }

  inline __rtc_device EvalRes eval(ShadingInteraction interaction, vec3f wi, bool dbg) const
  {
    const float p = pdf(interaction, wi, dbg);
    return p > 0.f ? EvalRes(rtc::load(albedo) * ONE_OVER_PI, p)
                   : EvalRes::zero();
  }

  inline __rtc_device EvalRes evalProjected(ShadingInteraction interaction, vec3f wi, bool dbg) const
  {
    EvalRes res = eval(interaction, wi, dbg);
    res.value *= max(dot(normal(interaction), wi), 0.f);
    return res;
  }

  inline __rtc_device void scatter(
      ScatterResult &s, const ShadingInteraction &interaction, Random &random, bool dbg) const
  {
    s = {};
    const vec3f N = normal(interaction);
    if (dot(N, interaction.wo) <= 0.f)
      return;
    s.dir = owl::common::xfmVector(owl::common::frame(N),
        cosineSampleHemisphere(vec2f(random(), random())));
    s.pdf = pdf(interaction, s.dir, dbg);
    if (!(s.pdf > 0.f))
      return;
    s.weight = rtc::load(albedo);
    s.type = ScatterResult::DIFFUSE;
    s.event = ScatterResult::REFLECTION;
  }
};
} // namespace packedBSDF
} // namespace native
} // namespace BARNEY_NS
