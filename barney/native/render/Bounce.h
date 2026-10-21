// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <bit>
#include "native/render/Ray.h"
#include "native/render/Renderer.h"
#include "native/render/World.h"

namespace BARNEY_NS {
namespace native {
#if RTC_DEVICE_CODE
inline __rtc_device float safe_eps(float f, vec3f v)
{
  return max(f, 1e-6f * reduce_max(abs(v)));
}

inline __rtc_device bool hasEnvironment(
    const World::DD &w, const Renderer::DD &r)
{
  return w.envMapLight.texture || r.ambientRadiance > 0.f;
}

inline __rtc_device int lightCount(const World::DD &w, const Renderer::DD &r)
{
  return w.numQuadLights + w.numDirLights + w.numPointLights
      + int(hasEnvironment(w, r));
}

inline __rtc_device float environmentProbability(
    const World::DD &w, const Renderer::DD &r, vec3f dir)
{
  if (!hasEnvironment(w, r))
    return 0.f;
  return (w.envMapLight.texture ? w.envMapLight.pdf(dir) : ONE_OVER_FOUR_PI)
      / float(lightCount(w, r));
}

inline __rtc_device bool sampleLights(Light::Sample &s,
    const World::DD &w,
    const Renderer::DD &r,
    vec3f P,
    Random &random)
{
  const int count = lightCount(w, r);
  if (!count)
    return false;
  int index = min(int(random() * count), count - 1);
  s = {};
  s.selectionProbability = 1.f / float(count);
  if (index < w.numQuadLights) {
    const auto &light = w.quadLights[index];
    const vec3f delta =
        light.corner + random() * light.edge0 + random() * light.edge1 - P;
    const float d2 = dot(delta, delta);
    if (!(d2 > 0.f) || !(light.area > 0.f))
      return false;
    s.distance = sqrtf(d2);
    s.direction = delta / s.distance;
    const float cosine = -dot(light.normal, s.direction);
    if (!(cosine > 0.f))
      return false;
    s.radiance = light.emission;
    s.pdf = d2 / (light.area * cosine);
  } else if ((index -= w.numQuadLights) < w.numDirLights) {
    const auto &light = w.dirLights[index];
    s.direction = -light.direction;
    s.distance = BARNEY_INF;
    s.radiance = light.color * light.radiance;
    s.delta = true;
  } else if ((index -= w.numDirLights) < w.numPointLights) {
    const auto &light = w.pointLights[index];
    const vec3f delta = light.position - P;
    const float d2 = dot(delta, delta);
    if (!(d2 > 0.f))
      return false;
    s.distance = sqrtf(d2);
    s.direction = delta / s.distance;
    s.radiance = light.radianceTowards(P) / d2;
    s.delta = true;
  } else {
    if (w.envMapLight.texture) {
      s = w.envMapLight.sample(random);
      s.selectionProbability = 1.f / float(count);
    } else {
      s.direction = randomDirection(random);
      s.distance = BARNEY_INF;
      s.radiance = vec3f(r.ambientRadiance);
      s.pdf = ONE_OVER_FOUR_PI;
    }
    s.environment = true;
  }
  return s.delta || (s.pdf > 0.f && isfinite(s.pdf));
}

inline __rtc_device vec3f radianceFromEnv(
    const World::DD &w, const Renderer::DD &r, const Ray &ray)
{
  const auto &env = w.envMapLight;
  if (!env.texture)
    return vec3f(r.ambientRadiance);
  const vec3f d = normalize(xfmVector(env.toLocal, normalize((vec3f)ray.dir)));
  const vec4f c = rtc::tex2D<vec4f>(env.texture,
      sphericalPhi(d) * ONE_OVER_TWO_PI,
      sphericalTheta(d) * ONE_OVER_PI);
  return env.scale * vec3f(c.x, c.y, c.z);
}

inline __rtc_device void bounce(const World::DD &world,
    const Renderer::DD &renderer,
    vec3f &fragment,
    Ray &ray,
    PathState &state,
    Ray &shadowRay,
    PathState &shadowState,
    int depth)
{
  const bool missed = !ray.hadHit();
  if (ray.isShadowRay) {
    if (missed)
      fragment = (vec3f)state.throughput * (float)state.misWeight;
    ray.tMax = -1.f;
    return;
  }
  if (missed) {
    fragment = depth == 0
        ? (world.envMapLight.texture
                  ? radianceFromEnv(world, renderer, ray)
                  : vec3f(ray.missColor.x, ray.missColor.y, ray.missColor.z))
        : (vec3f)state.throughput * radianceFromEnv(world, renderer, ray)
            * (float)state.misWeight;
    ray.tMax = -1.f;
    return;
  }
  const vec3f Ng = ray.getGeometricNormal();
  const bool volume = Ng == vec3f(0.f);
  const vec3f V = -normalize((vec3f)ray.dir);
  const bool front = dot(Ng, V) > 0.f;
  const vec3f Ngff = front ? Ng : -Ng;
  const PackedBSDF bsdf = ray.getBSDF();
  const vec3f incoming = state.throughput;
  fragment += incoming * bsdf.getEmission();
  DG dg;
  dg.P = ray.P;
  dg.Ng = Ngff;
  dg.Ns = ray.getShadingNormal();
  dg.wo = V;
  dg.insideMedium = ray.isInMedium;
  const float epsilon = safe_eps(1e-4f, dg.P);
  Random random(ray.rngSeed, std::bit_cast<uint32_t>(ray.tMax));
  Light::Sample light;
  if (sampleLights(light, world, renderer, dg.P, random)) {
    const EvalRes e = bsdf.evalProjected(dg, light.direction);
    const float probability =
        light.selectionProbability * (light.delta ? 1.f : light.pdf);
    if (probability > 0.f && reduce_max(e.value) > 0.f) {
      const vec3f offset =
          volume ? V : (dot(Ngff, light.direction) > 0.f ? Ngff : -Ngff);
      makeShadowRay(shadowRay,
          shadowState,
          incoming * e.value * light.radiance / probability,
          dg.P + epsilon * offset,
          light.direction,
          light.distance * (1.f - 2.f * epsilon));
      shadowRay.rngSeed = ray.rngSeed;
      ray.rngSeed.next(std::bit_cast<uint32_t>(ray.tMax));
      shadowRay._dbg = ray._dbg;
      shadowRay.crosshair = ray.crosshair;
      shadowState.pixelID = state.pixelID;
      // Quad lights have no intersectable geometry; there is no competing
      // BSDF estimator for them. Discrete directions never compete either.
      shadowState.misWeight = light.environment
          ? probability / (probability + e.pdf)
          : 1.f;
    }
  }
  ray.tMax = -1.f;
  if (depth >= 10)
    return;
  ScatterResult s;
  bsdf.scatter(s, dg, random);
  if (!s.valid())
    return;
  const bool transmission = s.event == ScatterResult::TRANSMISSION;
  if (!volume
      && (transmission ? dot(Ngff, s.dir) >= 0.f : dot(Ngff, s.dir) <= 0.f))
    return;
  if (s.event == ScatterResult::VOLUME_EVENT) {
#if BARNEY_USE_MULTI_SCATTERING
    if (!renderer.volumeMultiScatter
        || state.numVolumeBounces >= renderer.maxVolumeBounces)
      return;
    state.numVolumeBounces = state.numVolumeBounces + 1;
#else
    return;
#endif
  }
  if (s.type == ScatterResult::DIFFUSE) {
    if (state.numDiffuseBounces >= 1)
      return;
    state.numDiffuseBounces = state.numDiffuseBounces + 1;
  }
  ray.clearHit();
  ray.tMax = BARNEY_INF;
  ray.dir = normalize(s.dir);
  ray.org = dg.P
      + epsilon * (volume ? (vec3f)ray.dir : (transmission ? -Ngff : Ngff));
  if (transmission)
    ray.isInMedium = !ray.isInMedium;
  state.throughput = incoming * s.weight;
  const float lightPdf = environmentProbability(world, renderer, ray.dir);
  state.misWeight = s.delta ? 1.f : s.pdf / (s.pdf + lightPdf);
}
#endif

} // namespace native
} // namespace BARNEY_NS
