// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/ModelSlot.h"
#include "barney/GlobalModel.h"
#include "barney/common/Data.h"
#include "barney/common/Texture.h"
#include "barney/light/Light.h"
#include "barney/geometry/Geometry.h"

namespace BARNEY_NS {

  ModelSlot::PLD *ModelSlot::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }

  ModelSlot::ModelSlot(GlobalModel *_model,
                       const DevGroup::SP &devices,
                       int slotID)
    : SlottedObject((Context *)_model->context,
                    devices),
      model(_model),
      slotID(slotID),
      slotContext(((Context *)_model->context)->getSlot(slotID)),
      world(std::make_shared<render::World>(slotContext))
  {
    perLogical.resize(devices->numLogical);
  }

  ModelSlot::~ModelSlot()
  {
    for (auto device : *devices) {
      auto pld = getPLD(device);
      if (pld->instanceGroup) {
        device->rtc->freeGroup(pld->instanceGroup);
      }
    }
  }
  
  void ModelSlot::setInstances(barney_api::Group **groups,
                               const affine3f *xfms,
                               int numUserInstances)
  {
    instances.groups.resize(numUserInstances);
    instances.xfms.resize(numUserInstances);
    for (int i=0;i<numUserInstances;i++) {
      auto g = groups[i];
      instances.groups[i]
        = g
        ? g->shared_from_this()->as<Group>()
        : Group::SP{};
    }
    std::copy(xfms,xfms+numUserInstances,instances.xfms.data());
    for (auto device : *devices) {
      device->sbtDirty = true;
      auto pld = getPLD(device);
      if (pld->instanceGroup) {
        device->rtc->freeGroup(pld->instanceGroup);
        pld->instanceGroup = 0;
      }
    }
  }

  void ModelSlot::setInstanceAttributes(const std::string &which,
                                        const PODData::SP &data)
  {
    // assert(which >= 0 && which < 5);
    if (which == "instID")
      world->instanceUserIDs = data;
    else if (which == "attribute0")
      world->instanceAttributes[0] = data;
    else if (which == "attribute1")
      world->instanceAttributes[1] = data;
    else if (which == "attribute2")
      world->instanceAttributes[2] = data;
    else if (which == "attribute3")
      world->instanceAttributes[3] = data;
    else if (which == "attribute4" || which == "color")
      world->instanceAttributes[4] = data;
    else
      std::cout << "#barney: un-recognized instance attribute '" << which << "'" << std::endl;
      ;
  }

  void ModelSlot::updateWorldLightsFromInstances()
  {
    std::vector<QuadLight::DD> quadLights;
    std::vector<DirLight::DD>  dirLights;
    std::vector<PointLight::DD> pointLights;
    std::pair<EnvMapLight::SP,affine3f> envLight;

    for (int i = 0; i < (int)instances.groups.size(); i++) {
      Group *group = instances.groups[i].get();
      if (!group || !group->lights)
        continue;
      for (auto &light : group->lights->items) {
        if (!light)
          continue;
        if (QuadLight::SP quadLight = light->as<QuadLight>()) {
          quadLights.push_back(quadLight->getDD(instances.xfms[i]));
          continue;
        }
        if (DirLight::SP dirLight = light->as<DirLight>()) {
          dirLights.push_back(dirLight->getDD(instances.xfms[i]));
          continue;
        }
        if (PointLight::SP pointLight = light->as<PointLight>()) {
          pointLights.push_back(pointLight->getDD(instances.xfms[i]));
          continue;
        }
        if (EnvMapLight::SP el = light->as<EnvMapLight>()) {
          envLight = {el, instances.xfms[i]};
          continue;
        }
        throw std::runtime_error("un-handled type of light!?");
      }
    }
    world->set(envLight.first, envLight.second);
    world->set(quadLights);
    world->set(dirLights);
    world->set(pointLights);
  }

  void ModelSlot::flattenInstancesForDevice(Device *device,
                                            std::vector<rtc::Group *> *rtcGroups,
                                            std::vector<affine3f> &rtcTransforms,
                                            std::vector<int> *inputInstIDs)
  {
    for (int i = 0; i < (int)instances.groups.size(); i++) {
      Group *group = instances.groups[i].get();
      if (!group)
        continue;
      Group::PLD *groupPLD = group->getPLD(device);

      auto append = [&](rtc::Group *rtcGroup) {
        if (!rtcGroup)
          return;
        if (rtcGroups)
          rtcGroups->push_back(rtcGroup);
        rtcTransforms.push_back(instances.xfms[i]);
        if (inputInstIDs)
          inputInstIDs->push_back(i);
      };

      append(groupPLD->userGeomGroup);
      append(groupPLD->volumeGeomsGroup);
      append(groupPLD->triangleGeomGroup);
      for (auto volumeGroup : groupPLD->volumeGroups)
        append(volumeGroup);
    }
  }
  
  void ModelSlot::updateInstanceTransforms(const affine3f *xfms,
                                           int numInstances)
  {
    if (numInstances != (int)instances.groups.size()) {
      std::cout << "#barney: ignoring transform-only update with mismatched instance count"
                << std::endl;
      return;
    }
    std::copy(xfms, xfms + numInstances, instances.xfms.data());

    updateWorldLightsFromInstances();

    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      if (!pld->instanceGroup)
        continue;

      std::vector<affine3f> rtcTransforms;
      flattenInstancesForDevice(device, nullptr, rtcTransforms, nullptr);

      pld->instanceGroup->setTransforms(rtcTransforms);
      pld->instanceGroup->refitAccel();
    }
  }

  void ModelSlot::build()
  {
    // Keep light extraction identical to transform-only updates.
    updateWorldLightsFromInstances();
  
    // ==================================================================
    // generate all (per device) instance lists. note each BGGroup can
    // contain more than one rtcGroup, so theres's not a one-to-one
    // between barney instance list transform array and rtc instance
    // list transform array
    // ==================================================================
    
    std::vector<int>          inputInstIDs;
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      std::vector<affine3f>     rtcTransforms;
      std::vector<rtc::Group *> rtcGroups;
      bool firstDevice = (device == (*devices)[0]);
      flattenInstancesForDevice(device,
                                &rtcGroups,
                                rtcTransforms,
                                firstDevice ? &inputInstIDs : nullptr);
  
      if (pld->instanceGroup) {
        device->rtc->freeGroup(pld->instanceGroup);
        pld->instanceGroup = 0;
      }
      pld->instanceGroup
        = device->rtc->createInstanceGroup(rtcGroups,
                                           inputInstIDs,
                                           rtcTransforms);
      if (pld->instanceGroup)
        pld->instanceGroup->buildAccel();
    }
  }

}

  
