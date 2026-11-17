// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/GlobalModel.h"
#include "native/geometry/Triangles.h"
#include "native/render/RayQueue.h"
#include <cmath>
#include <stdexcept>

namespace tangent_test {
using namespace BARNEY_NS;
using namespace BARNEY_NS::native;

inline void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

inline void expectNormal(vec3f actual, vec3f expected, const char *message)
{
  // Normals in the hit BSDF are stored as half precision.
  require(std::isfinite(actual.x) && std::isfinite(actual.y)
      && std::isfinite(actual.z) && length(actual-expected) < 1e-3f,message);
}

inline Sampler::SP constantNormal(Context &context, vec3f normal)
{
  auto sampler = context.createSampler(0,"transform");
  const vec4f zero[] = {vec4f(0.f),vec4f(0.f),vec4f(0.f),vec4f(0.f)};
  require(sampler->set4x4f("outTransform",zero),"normal transform rejected");
  require(sampler->set4f("outOffset",vec4f(normal.x,normal.y,normal.z,1.f)),
          "normal offset rejected");
  sampler->commit();
  return sampler;
}

inline std::vector<Ray> trace(Context &context, Geometry *geometry,
                              vec3f origin = vec3f(.2f,.3f,1.f),
                              affine3f transform = affine3f(owl::common::one))
{
  auto group = context.createGroup(0,&geometry,1,nullptr,0);
  group->build();
  auto model = context.createModel();
  Group *groupPtr = group.get();
  model->setInstances(0,&groupPtr,&transform,1);
  model->build(0);
  Ray input{};
  input.org = origin;
  input.dir = vec3f(0,0,-1);
  input.tMax = 100.f;
  input.bsdfType = PackedBSDF::NONE;
  for (auto *device : *geometry->devices) {
    device->syncPipelineAndSBT();
    device->rayQueue->resize(1);
    device->rayQueue->numActive = 1;
    device->rtc->copyAsync(device->rayQueue->traceAndShadeReadQueue.rays,
                          &input,sizeof(input));
  }
  context.traceRaysLocally(model.get(),0,false);
  std::vector<Ray> result;
  for (auto *device : *geometry->devices) {
    Ray hit{};
    device->rtc->copyAsync(&hit,device->rayQueue->traceAndShadeReadQueue.rays,
                          sizeof(hit));
    device->sync();
    require(hit.bsdfType != PackedBSDF::NONE,"test ray missed triangle");
    result.push_back(hit);
  }
  return result;
}

vec3f mappedNormal(const Ray &hit, bool clearcoat = false);
}
