// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "native/render/ShadingInteraction.h"

namespace BARNEY_NS {
namespace native {
namespace packedBSDF {
namespace nvisii {
inline __rtc_device float pow5(float x)
{
  return x * x * x * x * x;
}
inline __rtc_device float fresnel(float c, float eta)
{
  const float sin2 = eta * eta * max(0.f, 1.f - c * c);
  if (sin2 >= 1.f)
    return 1.f;
  const float ct = sqrtf(1.f - sin2);
  const float a = (eta * c - ct) / (eta * c + ct);
  const float b = (c - eta * ct) / (c + eta * ct);
  return 0.5f * (a * a + b * b);
}
inline __rtc_device float D(vec3f h, vec2f a)
{
  const float q = h.x * h.x / (a.x * a.x) + h.y * h.y / (a.y * a.y) + h.z * h.z;
  return h.z > 0.f ? ONE_OVER_PI / (a.x * a.y * q * q) : 0.f;
}
inline __rtc_device float G1(vec3f v, vec2f a)
{
  if (v.z == 0.f)
    return 0.f;
  const float t = (a.x * a.x * v.x * v.x + a.y * a.y * v.y * v.y) / (v.z * v.z);
  return 2.f / (1.f + sqrtf(1.f + t));
}
inline __rtc_device vec3f sampleH(vec2f a, Random &r)
{
  const float phi = TWO_PI * r();
  const float u = min(r(), 1.f - 1e-7f);
  const float t = sqrtf(u / (1.f - u));
  return normalize(vec3f(a.x * t * cosf(phi), a.y * t * sinf(phi), 1.f));
}
inline __rtc_device float coatD(float c, float a)
{
  const float a2 = a * a;
  return (a2 - 1.f) / (ONE_PI * logf(a2) * (1.f + (a2 - 1.f) * c * c));
}
} // namespace nvisii

struct NVisii
{
  vec3h baseColor, subsurfaceColor;
  half metallic, specular, roughness, specularTint, anisotropy;
  half sheen, sheenTint, clearcoat, clearcoatGloss;
  half ior, specularTransmission, transmissionRoughness, flatness;

  inline __rtc_device void setDefaults()
  {
    baseColor = subsurfaceColor = vec3f(.8f);
    metallic = 0.f;
    specular = .5f;
    roughness = .5f;
    specularTint = anisotropy = sheen = clearcoat = flatness = 0.f;
    sheenTint = .5f;
    clearcoatGloss = 1.f - .03f * .03f;
    ior = 1.45f;
    specularTransmission = 0.f;
    transmissionRoughness = .04f;
  }
  inline __rtc_device vec3f getAlbedo(bool) const
  {
    return (vec3f)baseColor;
  }
  inline __rtc_device vec2f alpha() const
  {
    const float a = max(.002f, float(roughness) * float(roughness));
    const float aspect = sqrtf(1.f - .9f * float(anisotropy));
    return vec2f(max(.002f, a / aspect), max(.002f, a * aspect));
  }
  inline __rtc_device float transmissionAlpha() const
  {
    return max(
        .002f, float(transmissionRoughness) * float(transmissionRoughness));
  }
  inline __rtc_device float coatAlpha() const
  {
    return .1f * (1.f - float(clearcoatGloss)) + .002f * float(clearcoatGloss);
  }
  inline __rtc_device vec3f normal(const ShadingInteraction &interaction) const
  {
    return interaction.orientShadingNormal(interaction.shadingNormal);
  }
  inline __rtc_device vec4f probabilities() const
  {
    const float t = (1.f - float(metallic)) * float(specularTransmission);
    const float d =
        (1.f - float(metallic)) * (1.f - float(specularTransmission));
    // Sampling weights are a deterministic proposal, not lobe energies.
    const vec4f w(d + float(sheen), 1.f, float(clearcoat), t);
    return w / (w.x + w.y + w.z + w.w);
  }
  inline __rtc_device EvalRes evaluate(ShadingInteraction interaction, vec3f wi, bool projected) const
  {
    using namespace nvisii;
    const auto frame = owl::common::frame(normal(interaction));
    const vec3f v = owl::common::xfmVector(frame.transposed(), interaction.wo);
    const vec3f l = owl::common::xfmVector(frame.transposed(), wi);
    if (v.z <= 0.f || l.z == 0.f
        || dot(interaction.geometricNormal, wi) * dot(interaction.geometricNormal, interaction.wo) * l.z <= 0.f)
      return EvalRes::zero();
    const float eta = interaction.insideMedium ? float(ior) : 1.f / float(ior);
    const float t = (1.f - float(metallic)) * float(specularTransmission);
    const vec4f p = probabilities();
    const vec3f color = (vec3f)baseColor;
    const float lum = luminance(color);
    const vec3f tint = lum > 0.f ? color / lum : vec3f(1.f);
    const vec3f specTint =
        (1.f - float(specularTint)) * vec3f(1.f) + float(specularTint) * tint;
    const vec3f f0 = (1.f - float(metallic)) * .08f * float(specular) * specTint
        + float(metallic) * color;
    vec3f value(0.f);
    float density = 0.f;
    const vec2f a = alpha();
    const vec2f at(transmissionAlpha());
    if (l.z > 0.f) {
      const vec3f h = normalize(v + l);
      const float vh = dot(v, h);
      const float fi = pow5(1.f - l.z), fo = pow5(1.f - v.z);
      const float fd90 = .5f + 2.f * float(roughness) * vh * vh;
      const float fd =
          (1.f + (fd90 - 1.f) * fi) * (1.f + (fd90 - 1.f) * fo) * ONE_OVER_PI;
      const float fss90 = float(roughness) * vh * vh;
      const float fss = (1.f + (fss90 - 1.f) * fi) * (1.f + (fss90 - 1.f) * fo);
      const float ss =
          1.25f * (fss * (1.f / (l.z + v.z) - .5f) + .5f) * ONE_OVER_PI;
      value = ((1.f - float(flatness)) * color * fd
                  + float(flatness) * (vec3f)subsurfaceColor * ss)
          * (1.f - float(metallic)) * (1.f - float(specularTransmission));
      value += float(sheen)
          * ((1.f - float(sheenTint)) * vec3f(1.f) + float(sheenTint) * tint)
          * pow5(1.f - vh) * (1.f - float(metallic));
      const vec3f F = f0 + (vec3f(1.f) - f0) * pow5(1.f - vh);
      value +=
          (1.f - t) * F * (D(h, a) * G1(v, a) * G1(l, a) / (4.f * v.z * l.z));
      density = p.x * l.z * ONE_OVER_PI + p.y * D(h, a) * h.z / (4.f * vh);
      const float dc = coatD(h.z, coatAlpha());
      const float fc = .04f + .96f * pow5(1.f - vh);
      value += vec3f(.25f * float(clearcoat) * fc * dc * G1(v, vec2f(.25f))
          * G1(l, vec2f(.25f)) / (4.f * v.z * l.z));
      density += p.z * dc * h.z / (4.f * vh);
      if (t > 0.f) {
        const float ft = fresnel(vh, eta);
        value += vec3f(
            t * ft * D(h, at) * G1(v, at) * G1(l, at) / (4.f * v.z * l.z));
        density += p.w * ft * D(h, at) * h.z / (4.f * vh);
      }
    } else if (t > 0.f && eta != 1.f) {
      vec3f h = normalize(eta * v + l);
      if (h.z < 0.f)
        h = -h;
      const float vh = dot(v, h), lh = dot(l, h);
      if (vh <= 0.f || lh >= 0.f)
        return EvalRes::zero();
      const float f = fresnel(vh, eta);
      const float denom = eta * vh + lh;
      const float jacobian = fabsf(lh) / (denom * denom);
      value = color
          * (t * (1.f - f) * D(h, at) * G1(v, at) * G1(l, at) * fabsf(vh)
              * jacobian * eta * eta / (v.z * -l.z));
      density = p.w * (1.f - f) * D(h, at) * h.z * jacobian;
    }
    if (projected)
      value *= fabsf(l.z);
    return EvalRes(value, density);
  }
  inline __rtc_device EvalRes eval(ShadingInteraction interaction, vec3f wi, bool) const
  {
    return evaluate(interaction, wi, false);
  }
  inline __rtc_device EvalRes evalProjected(ShadingInteraction interaction, vec3f wi, bool) const
  {
    return evaluate(interaction, wi, true);
  }
  inline __rtc_device float pdf(ShadingInteraction interaction, vec3f wi, bool) const
  {
    return evaluate(interaction, wi, false).pdf;
  }

  inline __rtc_device void scatter(
      ScatterResult &s, const ShadingInteraction &interaction, Random &r, bool dbg) const
  {
    using namespace nvisii;
    s = {};
    const auto frame = owl::common::frame(normal(interaction));
    const vec3f v = owl::common::xfmVector(frame.transposed(), interaction.wo);
    if (v.z <= 0.f)
      return;
    const float eta = interaction.insideMedium ? float(ior) : 1.f / float(ior);
    const vec4f p = probabilities();
    const float u = r();
    vec3f l;
    s.type = ScatterResult::GLOSSY;
    bool transmission = false;
    if (u < p.x) {
      l = cosineSampleHemisphere(vec2f(r(), r()));
      s.type = ScatterResult::DIFFUSE;
    } else if (u < p.x + p.y) {
      const vec3f h = sampleH(alpha(), r);
      if (dot(v, h) <= 0.f)
        return;
      l = reflect(-v, h);
    } else if (u < p.x + p.y + p.z) {
      const float a = coatAlpha(), phi = TWO_PI * r();
      const float c2 = (1.f - powf(a * a, 1.f - r())) / (1.f - a * a);
      const float sn = sqrtf(max(0.f, 1.f - c2));
      const vec3f h(sn * cosf(phi), sn * sinf(phi), sqrtf(c2));
      if (dot(v, h) <= 0.f)
        return;
      l = reflect(-v, h);
    } else {
      if (eta == 1.f) {
        s.dir = -interaction.wo;
        s.weight = (vec3f)baseColor
            * ((1.f - float(metallic)) * float(specularTransmission) / p.w);
        s.delta = true;
        s.event = ScatterResult::TRANSMISSION;
        return;
      }
      const vec3f h = sampleH(vec2f(transmissionAlpha()), r);
      const float vh = dot(v, h);
      if (vh <= 0.f)
        return;
      if (r() < fresnel(vh, eta))
        l = reflect(-v, h);
      else {
        l = refract(-v, h, eta);
        transmission = true;
      }
    }
    if (transmission ? l.z >= 0.f : l.z <= 0.f)
      return;
    s.dir = normalize(owl::common::xfmVector(frame, l));
    const EvalRes e = evalProjected(interaction, s.dir, dbg);
    if (!(e.pdf > 0.f) || !isfinite(e.pdf))
      return;
    s.pdf = e.pdf;
    s.weight = e.value / e.pdf;
    s.event =
        transmission ? ScatterResult::TRANSMISSION : ScatterResult::REFLECTION;
    s.eta = transmission ? eta : 1.f;
  }
};
} // namespace packedBSDF
} // namespace native
} // namespace BARNEY_NS
