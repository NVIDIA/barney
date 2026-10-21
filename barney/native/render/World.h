// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/DeviceGroup.h"
#include "native/render/Sampler.h"
#include "native/light/EnvMap.h"
#include "native/light/DirLight.h"
#include "native/light/PointLight.h"
#include "native/light/QuadLight.h"

namespace BARNEY_NS {
  namespace native {

    struct LDGContext;
    struct DeviceMaterial;
    struct HostMaterial;

    /*! the rendering/path racing related part of a model that describes
      global render settings like light sources, background, envmap,
      etc

      iw - TODO probably want to mege this into LDGContext; this
      already has per-LDG sampler registry etc, so could/should also
      keep this data, too
    */
    struct World {
      typedef std::shared_ptr<World> SP;
      
      struct DD {
        int                  numQuadLights = 0;
        const QuadLight::DD *quadLights    = nullptr;
        int                  numDirLights  = 0;
        const DirLight::DD  *dirLights     = nullptr;
        int                  numPointLights  = 0;
        const PointLight::DD  *pointLights     = nullptr;
        int                 *instIDToUserInstID = 0;
        // Indexed by native input instance ID, like instance attributes.
        const float         *instanceOrientations = nullptr;
        const affine3f      *motionDeltas       = nullptr;
        int                  numMotionDeltas    = 0;

        const DeviceMaterial *materials;
        const Sampler::DD    *samplers;
        const rtc::float4    *instanceAttributes[5];
        EnvMapLight::DD       envMapLight;
        // uint32_t              rngSeed;
        uint32_t              rank;
      };
      struct {
        EnvMapLight::SP light;
        affine3f        xfm;
      } envMapLight;

      World(LDGContext *ldgContext);
      virtual ~World();

      void set(const std::vector<QuadLight::DD> &quadLights);
      void set(const std::vector<DirLight::DD> &dirLights);
      void set(const std::vector<PointLight::DD> &pointLights);
      void set(EnvMapLight::SP envMapLight, const affine3f &xfm);
      void setMotionDeltas(const std::vector<affine3f> &deltas);
      void setInstanceOrientations(const std::vector<affine3f> &transforms);

      PODData::SP instanceAttributes[5];
      PODData::SP instanceUserIDs;
      
      DD getDD(Device *device);

      struct PLD {
        QuadLight::DD *quadLights = 0;
        int numQuadLights = 0;
        DirLight::DD *dirLights = 0;
        int numDirLights = 0;
        PointLight::DD *pointLights = 0;
        int numPointLights = 0;
        float *instanceOrientations = nullptr;
        affine3f *motionDeltas = 0;
        int numMotionDeltas = 0;
      };
      PLD *getPLD(Device *device);
      
      std::vector<PLD> perLogical;
      DevGroup::SP const devices;
      LDGContext *const ldgContext;
    };

  }
}
