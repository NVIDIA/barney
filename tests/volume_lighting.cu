// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/render/Bounce.h"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace BARNEY_NS;
using namespace BARNEY_NS::native;

namespace volume_lighting_test {
constexpr double pi = 3.14159265358979323846;
constexpr int samples = 256;
enum Phase { HG, SIMPLE, SURFACE };
enum Lighting { AMBIENT, DIRECTIONAL, AREA, MIXED, NONE };

struct Parameters
{
  Phase phase = HG;
  Lighting lighting = AMBIENT;
  float g = 0.f;
  int multiScatter = 0;
  int maxBounces = 2;
  int volumeBounces = 0;
  int generation = 0;
  bool emissive = false;
  vec3f incoming = vec3f(8,4,2);
};

struct Result
{
  vec3f direct, continuation, emission, lightDirection, scatterDirection;
  int shadowValid, continuationValid, volumeBounces, diffuseBounces;
  float shadowMIS;
};

__rtc_global void evaluate(const rtc::ComputeInterface &ci,
                          Parameters p, Result *out)
#if !RTC_DEVICE_CODE
;
#else
{
  const int i = ci.getLaunchIndex().x;
  if (i >= samples) return;
  DirLight::DD directional{};
  directional.direction = vec3f(0,0,-1);
  directional.color = vec3f(1.f);
  directional.radiance = 2.f;
  QuadLight::DD area{};
  area.corner = vec3f(-.5f,-.5f,2.f);
  area.normal = vec3f(0,0,-1);
  area.area = 1.f;
  area.emission = vec3f(2.f);
  World::DD world{};
  Renderer::DD renderer{};
  renderer.ambientRadiance =
      p.lighting == AMBIENT || p.lighting == MIXED ? 2.f : 0.f;
  renderer.volumeMultiScatter = p.multiScatter;
  renderer.maxVolumeBounces = p.maxBounces;
  if (p.lighting == DIRECTIONAL || p.lighting == MIXED) {
    world.numDirLights = 1;
    world.dirLights = &directional;
  }
  if (p.lighting == AREA) {
    world.numQuadLights = 1;
    world.quadLights = &area;
  }
  Ray ray{}, shadow{};
  shadow.tMax = -1.f;
  ray.dir = vec3f(0,0,-1);
  ray.rngSeed.seed(i,0x197u);
  const vec3f tint(.125f,.5f,.875f);
  if (p.phase == SIMPLE) {
    ray.setVolumeHit(vec3f(0.f),1.f,tint);
  } else if (p.phase == SURFACE) {
    packedBSDF::Lambertian matte;
    matte.albedo = rtc::float3(tint.x,tint.y,tint.z);
    ray.setHit(vec3f(0.f),vec3f(0,0,1),1.f,matte,vec3f(0,0,1));
  } else {
    ray.setVolumeHitWithEmission(vec3f(0.f),1.f,tint,p.g,1.f,
        p.emissive ? vec3f(.25f,.5f,1.f) : vec3f(0.f));
  }
  PathState state{}, shadowState{};
  state.throughput = p.incoming;
  state.numVolumeBounces = p.volumeBounces;
  vec3f fragment(0.f);
  bounce(world,renderer,fragment,ray,state,shadow,shadowState,p.generation);
  Result r{};
  r.emission = fragment;
  r.shadowValid = shadow.valid();
  r.continuationValid = ray.valid();
  r.volumeBounces = state.numVolumeBounces;
  r.diffuseBounces = state.numDiffuseBounces;
  r.lightDirection = shadow.dir;
  r.scatterDirection = ray.dir;
  r.shadowMIS = shadowState.misWeight;
  Ray unused{};
  unused.tMax = -1.f;
  PathState unusedState{};
  // Synthesize unoccluded visibility/misses, but use the real integrator to
  // consume both estimators and apply their carried throughput/MIS weights.
  if (r.shadowValid) {
    shadow.clearHit();
    r.direct = vec3f(0.f);
    bounce(world,renderer,r.direct,shadow,shadowState,
           unused,unusedState,p.generation+1);
  }
  if (r.continuationValid) {
    ray.clearHit();
    r.continuation = vec3f(0.f);
    bounce(world,renderer,r.continuation,ray,state,
           unused,unusedState,p.generation+1);
  }
  out[i] = r;
}
#endif

struct Buffer
{
  rtc::Device &device;
  rtc::Buffer *buffer;
  explicit Buffer(rtc::Device &d)
      : device(d), buffer(d.createBuffer(samples*sizeof(Result),nullptr)) {}
  Buffer(const Buffer &) = delete;
  ~Buffer() { device.freeBuffer(buffer); }
  Result *data() { return static_cast<Result *>(buffer->getDD()); }
  std::vector<Result> read()
  {
    std::vector<Result> result(samples);
    device.copyAsync(result.data(),data(),result.size()*sizeof(Result));
    device.sync();
    return result;
  }
};

void require(bool ok, const std::string &message)
{
  if (!ok) throw std::runtime_error(message);
}

void near(vec3f actual, const std::array<double,3> &expected,
          const std::string &message)
{
  const float values[] = {actual.x,actual.y,actual.z};
  for (int c = 0; c < 3; ++c)
    require(std::isfinite(values[c]) && values[c] >= 0.f
        && std::abs(values[c]-expected[c]) <= 2e-6+2e-5*std::abs(expected[c]),
        message+" channel="+std::to_string(c)
        +" actual="+std::to_string(values[c])
        +" expected="+std::to_string(expected[c]));
}

double phaseDensity(const Parameters &p, vec3f direction)
{
  if (p.phase == SURFACE)
    return std::max(0.,double(direction.z))/pi;
  if (p.phase == SIMPLE || p.g == 0.f)
    return 1./(4.*pi);
  const double g = p.g;
  // Incident ray travels along -Z; wi points from the collision to the light.
  const double denominator = 1.+g*g+2.*g*double(direction.z);
  return (1.-g*g)/(4.*pi*std::pow(denominator,1.5));
}

void check(rtc::Device &device, Buffer &buffer,
           const std::string &name, Parameters p)
{
  __rtc_launch((&device),evaluate,2,128,p,buffer.data());
  const auto results = buffer.read();
  bool continues = p.generation < MAX_PATH_DEPTH;
  if (p.phase != SURFACE)
    continues = continues && p.multiScatter
        && p.volumeBounces < p.maxBounces;
#if !BARNEY_USE_MULTI_SCATTERING
  if (p.phase != SURFACE) continues = false;
#endif
  const bool ambient = p.lighting == AMBIENT || p.lighting == MIXED;
  const double selection = p.lighting == MIXED ? .5 : 1.;
  const double tint[] = {.125,.5,.875};
  const double incoming[] = {p.incoming.x,p.incoming.y,p.incoming.z};
  const double reflectance = p.phase == SIMPLE ? .7 : 1.;
  int environmentSamples = 0, directionalSamples = 0, shadowSamples = 0;
  for (const auto &r : results) {
    require(bool(r.continuationValid) == continues,name+" continuation policy");
    require(r.volumeBounces == p.volumeBounces
        + int(continues && p.phase != SURFACE),name+" volume count");
    require(r.diffuseBounces == int(continues && p.phase == SURFACE),
            name+" diffuse count");
    if (p.lighting == NONE)
      require(!r.shadowValid,name+" unexpected light");
    else if (p.phase != SURFACE)
      require(r.shadowValid,name+" missing shadow estimator");
    std::array<double,3> direct{}, continuation{}, emission{};
    if (r.shadowValid) {
      ++shadowSamples;
      require(r.shadowMIS == 1.f,name+" double shadow MIS");
      const bool delta = p.lighting == DIRECTIONAL
          || (p.lighting == MIXED && r.lightDirection == vec3f(0,0,1));
      const bool environment = !delta && p.lighting != AREA;
      environmentSamples += environment;
      directionalSamples += delta;
      double lightPDF = selection;
      if (environment)
        lightPDF *= 1./(4.*pi);
      else if (p.lighting == AREA) {
        // Unit square at z=2: p_omega = distance^2 / cos(theta).
        const double cosine = r.lightDirection.z;
        lightPDF *= 4./(cosine*cosine*cosine);
      }
      const double density = phaseDensity(p,r.lightDirection);
      const double denominator = lightPDF
          + (environment && continues ? density : 0.);
      for (int c = 0; c < 3; ++c)
        direct[c] = incoming[c]*tint[c]*reflectance*density*2./denominator;
    }
    if (continues && ambient) {
      const double density = phaseDensity(p,r.scatterDirection);
      const double weight = density/(density+selection/(4.*pi));
      for (int c = 0; c < 3; ++c)
        continuation[c] = incoming[c]*tint[c]*reflectance*2.*weight;
    }
    if (p.emissive) {
      const double emitted[] = {.25,.5,1.};
      for (int c = 0; c < 3; ++c) emission[c] = incoming[c]*emitted[c];
    }
    near(r.direct,direct,name+" direct");
    near(r.continuation,continuation,name+" continuation");
    near(r.emission,emission,name+" emission");
    if (p.phase == HG && p.g == 0.f && p.lighting == AMBIENT
        && p.incoming == vec3f(8,4,2))
      near(r.direct+r.continuation,{2.,4.,3.5},name+" furnace energy");
  }
  if (p.lighting != NONE)
    require(shadowSamples > 0,name+" no direct samples checked");
  if (p.lighting == MIXED)
    require(environmentSamples > 0 && directionalSamples > 0,
            name+" both light-selection branches must be exercised");
  std::cout << name << ": PASS (" << samples << " production bounces)\n";
}
} // namespace volume_lighting_test

int main()
{
  using namespace volume_lighting_test;
  try {
    rtc::Device device(0);
    std::cout << "Volume lighting device: " << device.toString() << '\n';
    Buffer buffer(device);
    Parameters p;
    check(device,buffer,"disabled isotropic",p);
    p.multiScatter = 1;
    p.maxBounces = 0;
    check(device,buffer,"zero volume budget",p);
    p.maxBounces = 2;
    p.volumeBounces = 2;
    check(device,buffer,"exhausted volume budget",p);
    p.volumeBounces = 1;
    check(device,buffer,"last volume bounce",p);
    p.volumeBounces = 0;
    p.generation = MAX_PATH_DEPTH - 1;
    check(device,buffer,"last global bounce",p);
    p.generation = MAX_PATH_DEPTH;
    check(device,buffer,"global depth limit",p);
    p.phase = SURFACE;
    check(device,buffer,"surface global depth limit",p);
    p = Parameters{};
    int cases = 7;
    for (Phase phase : {HG,SIMPLE})
      for (float g : {0.f,.6f,-.6f}) {
        if (phase == SIMPLE && g != 0.f) continue;
        for (int enabled : {0,1})
          for (Lighting lighting : {AMBIENT,DIRECTIONAL,AREA,MIXED,NONE}) {
            p.phase = phase;
            p.g = g;
            p.multiScatter = enabled;
            p.lighting = lighting;
            check(device,buffer,"phase="+std::to_string(phase)
                +" g="+std::to_string(g)+" enabled="+std::to_string(enabled)
                +" light="+std::to_string(lighting),p);
            ++cases;
          }
      }
    p = Parameters{};
    p.g = .6f;
    p.incoming = vec3f(1e6f,8.f,2.f);
    check(device,buffer,"HDR disabled anisotropic",p);
    p.emissive = true;
    p.lighting = NONE;
    check(device,buffer,"emission without lights",p);
    p.lighting = AMBIENT;
    check(device,buffer,"emission with direct lighting",p);
    p.multiScatter = 1;
    check(device,buffer,"emission with continuation",p);
    std::cout << "Volume lighting: PASS " << cases+4 << " cases\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Volume lighting: FAIL: " << e.what() << '\n';
    return 1;
  }
}
