// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/render/World.h"
#include "native/Context.h"
#include "native/material/DeviceMaterial.h"
#include "native/material/HostMaterial.h"
#include "native/render/MaterialRegistry.h"
#include "native/render/SamplerRegistry.h"

namespace BARNEY_NS {
  namespace native {

    World::World(LDGContext *ldgContext)
      : devices(ldgContext->devices),
        ldgContext(ldgContext)
    {
      perLogical.resize(devices->numLogical);
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;
        pld->quadLights = 0;
        pld->dirLights = 0;
        pld->pointLights = 0;
      }
    }
    
    World::~World()
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;
        if (pld->quadLights)   rtc->freeMem(pld->quadLights);
        if (pld->dirLights)    rtc->freeMem(pld->dirLights);
        if (pld->pointLights)  rtc->freeMem(pld->pointLights);
        if (pld->motionDeltas) rtc->freeMem(pld->motionDeltas);
        if (pld->instanceOrientations) rtc->freeMem(pld->instanceOrientations);
      }
    }

    World::PLD *World::getPLD(Device *device)
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }
    
    World::DD World::getDD(Device *device) 
    {
      PLD *pld = getPLD(device);
      DD dd;
      dd.quadLights
        = (QuadLight::DD *)pld->quadLights;
      dd.numQuadLights = pld->numQuadLights;
      dd.dirLights
        = (DirLight::DD *)pld->dirLights;
      dd.numDirLights = pld->numDirLights;
      dd.pointLights
        = (PointLight::DD *)pld->pointLights;
      dd.numPointLights = pld->numPointLights;
      dd.envMapLight
        = envMapLight.light
        ? envMapLight.light->getDD(device,envMapLight.xfm)
        : EnvMapLight::DD{};
      dd.rank            = ldgContext->context->myRank();
      dd.instanceOrientations = pld->instanceOrientations;
      dd.motionDeltas    = pld->motionDeltas;
      dd.numMotionDeltas = pld->numMotionDeltas;
      dd.samplers        = ldgContext->samplerRegistry->getDD(device);
      dd.materials       = ldgContext->materialRegistry->getDD(device);
      
      for (int i=0;i<5;i++)
        dd.instanceAttributes[i]
          = instanceAttributes[i]
          ? (const rtc::float4*)instanceAttributes[i]->getDD(device)
          : nullptr;
      return dd;
    }

    void World::set(const std::vector<QuadLight::DD> &quadLights)
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto pld = getPLD(device);
        auto rtc = device->rtc;
        rtc->freeMem(pld->quadLights);
        size_t numBytes = quadLights.size()*sizeof(quadLights[0]);
        pld->quadLights = (QuadLight::DD*)rtc->allocMem(numBytes);
        rtc->copy(pld->quadLights,quadLights.data(),numBytes);
        pld->numQuadLights = (int)quadLights.size();
      }
    }
    
    void World::set(const std::vector<DirLight::DD> &dirLights)
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto pld = getPLD(device);
        auto rtc = device->rtc;
        rtc->freeMem(pld->dirLights);
        size_t numBytes = dirLights.size()*sizeof(dirLights[0]);
        pld->dirLights = (DirLight::DD*)rtc->allocMem(numBytes);
        rtc->copy(pld->dirLights,dirLights.data(),numBytes);
        pld->numDirLights = (int)dirLights.size();
      }
    }

    void World::set(const std::vector<PointLight::DD> &pointLights)
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto pld = getPLD(device);
        auto rtc = device->rtc;
        rtc->freeMem(pld->pointLights);
        size_t numBytes = pointLights.size()*sizeof(pointLights[0]);
        pld->pointLights = (PointLight::DD*)rtc->allocMem(numBytes);
        rtc->copy(pld->pointLights,pointLights.data(),numBytes);
        pld->numPointLights = (int)pointLights.size();
      }
    }

    void World::set(EnvMapLight::SP envMapLight, const affine3f &xfm)
    {
      this->envMapLight.light = envMapLight;
      this->envMapLight.xfm = xfm;
    }

    void World::setInstanceOrientations(const std::vector<affine3f> &transforms)
    {
      // ModelSlot validates the effective transforms before publishing them.
      std::vector<float> orientations;
      orientations.reserve(transforms.size());
      for (const auto &xfm : transforms)
        orientations.push_back(xfm.l.det() < 0.f ? -1.f : 1.f);
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto pld = getPLD(device);
        auto rtc = device->rtc;
        if (pld->instanceOrientations) rtc->freeMem(pld->instanceOrientations);
        pld->instanceOrientations = nullptr;
        if (orientations.empty()) continue;
        const size_t numBytes = orientations.size() * sizeof(float);
        pld->instanceOrientations = (float *)rtc->allocMem(numBytes);
        rtc->copy(pld->instanceOrientations, orientations.data(), numBytes);
      }
    }

    void World::setMotionDeltas(const std::vector<affine3f> &deltas)
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto pld = getPLD(device);
        auto rtc = device->rtc;
        if (pld->motionDeltas) rtc->freeMem(pld->motionDeltas);
        pld->motionDeltas = nullptr;
        pld->numMotionDeltas = 0;
        if (deltas.empty()) continue;
        const size_t numBytes = deltas.size() * sizeof(affine3f);
        pld->motionDeltas = (affine3f *)rtc->allocMem(numBytes);
        rtc->copy(pld->motionDeltas, deltas.data(), numBytes);
        pld->numMotionDeltas = (int)deltas.size();
      }
    }

  } // ::BARNEY_NS::native
} // ::BARNEY_NS
