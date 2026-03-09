// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "World.h"
// std
#include <algorithm>
#include <map>
#include <set>

namespace barney_device {

  World::World(BarneyGlobalState *s)
    : Object(ANARI_WORLD, s),
      m_zeroSurfaceData(this),
      m_zeroVolumeData(this),
      m_zeroLightData(this),
      m_instanceData(this)
  {
    m_zeroGroup = new Group(s);
    m_zeroInstance = new Instance(s);
    m_zeroInstance->setParamDirect("group", m_zeroGroup.ptr);
    m_zeroInstance->commitParameters();
    m_zeroInstance->finalize();

    // never any public ref to these objects
    m_zeroGroup->refDec(helium::RefType::PUBLIC);
    m_zeroInstance->refDec(helium::RefType::PUBLIC);

    int uniqueModelID = deviceState()->nextUniqueModelID++;
    tetheredModel = deviceState()->tether->getOrCreateTetheredModel(uniqueModelID);
  }

  World::~World()
  {
    BANARI_TRACK_LEAKS(std::cout << "#banari: ~World deconstructing"
                       << std::endl);
    tetheredModel = {};
  }

  bool World::getProperty(const std::string_view &name,
                          ANARIDataType type,
                          void *ptr,
                          uint64_t size,
                          uint32_t flags)
  {
    if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
      if (flags & ANARI_WAIT) {
        deviceState()->commitBuffer.flush();
        makeCurrent();
      }
      box3 bounds;
      bounds.invalidate();
      std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
        bounds.insert(inst->bounds());
      });
      std::memcpy(ptr, &bounds, sizeof(bounds));
      return true;
    }

    return Object::getProperty(name, type, ptr, size, flags);
  }

  void World::commitParameters()
  {
    m_zeroSurfaceData = getParamObject<ObjectArray>("surface");
    m_zeroVolumeData = getParamObject<ObjectArray>("volume");
    m_zeroLightData = getParamObject<ObjectArray>("light");
    m_instanceData = getParamObject<ObjectArray>("instance");
  }

  void World::finalize()
  {
    const bool addZeroInstance =
      m_zeroSurfaceData || m_zeroVolumeData || m_zeroLightData;
    if (addZeroInstance)
      reportMessage(ANARI_SEVERITY_DEBUG, "barney::World will add zero instance");

    if (m_zeroSurfaceData) {
      reportMessage(ANARI_SEVERITY_DEBUG,
                    "barney::World found %zu surfaces in zero instance",
                    m_zeroSurfaceData->size());
      m_zeroGroup->setParamDirect("surface", getParamDirect("surface"));
    } else {
      m_zeroGroup->removeParam("surface");
    }

    if (m_zeroVolumeData) {
      reportMessage(ANARI_SEVERITY_DEBUG,
                    "barney::World found %zu volumes in zero instance",
                    m_zeroVolumeData->size());
      m_zeroGroup->setParamDirect("volume", getParamDirect("volume"));
    } else
      m_zeroGroup->removeParam("volume");

    if (m_zeroLightData) {
      reportMessage(ANARI_SEVERITY_DEBUG,
                    "barney::World found %zu lights in zero instance",
                    m_zeroLightData->size());
      m_zeroGroup->setParamDirect("light", getParamDirect("light"));
    } else
      m_zeroGroup->removeParam("light");

    m_zeroInstance->setParam("id", getParam<uint32_t>("id", ~0u));

    m_zeroGroup->commitParameters();
    m_zeroGroup->finalize();

    m_instances.clear();

    if (m_instanceData) {
      std::for_each(m_instanceData->handlesBegin(),
                    m_instanceData->handlesEnd(),
                    [&](auto *o) {
                      if (o && o->isValid())
                        m_instances.push_back((Instance *)o);
                    });
    }

    if (addZeroInstance)
      m_instances.push_back(m_zeroInstance.ptr);
  }

  void World::markFinalized()
  {
    deviceState()->markSceneChanged();
    Object::markFinalized();
  }


  BNModel World::makeCurrent()
  {
    buildBarneyModel();
    return tetheredModel->model;
  }

  void World::uploadInstanceAttributes(const InstanceAttributes &attributes)
  {
    auto barneyModel = tetheredModel->model;
    int  slot    = deviceState()->slot;
    auto context = deviceState()->tether->context;

    for (int i = 0; i < Instance::Attributes::count; i++) {
      if (m_attributesData[i]) {
        bnRelease(m_attributesData[i]);
        m_attributesData[i] = 0;
      }
      m_attributesData[i] =
        bnDataCreate(context, slot, BN_FLOAT4,
                     attributes[i].size(), attributes[i].data());
    }

    for (int i = 0; i < Instance::Attributes::count; i++) {
      std::string attribName = std::string("attribute") + std::to_string(i);
      bnSetInstanceAttributes(barneyModel, slot,
                              attribName.c_str(), m_attributesData[i]);
    }
  }

  void World::buildBarneyModel()
  {
    auto *state = deviceState();
    if (state->objectUpdates.lastSceneChange <= m_lastBarneyModelBuild)
      return;

    bool structural =
      m_lastBarneyModelBuild == 0
      || state->objectUpdates.lastStructuralChange > m_lastBarneyModelBuild;

    if (structural) {
      reportMessage(ANARI_SEVERITY_DEBUG, "barney::World full model rebuild");
      fullRebuild();
    } else {
      reportMessage(ANARI_SEVERITY_DEBUG, "barney::World transform-only update");
      transformOnlyUpdate();
    }

    m_lastBarneyModelBuild = helium::newTimeStamp();
  }

  void World::fullRebuild()
  {
    auto barneyModel = tetheredModel->model;
    auto context = deviceState()->tether->context;
    int defaultSlot = deviceState()->slot;

    // Collect all known slot indices so even empty ones get bnSetInstances
    std::set<int> allSlots;
    allSlots.insert(defaultSlot);
    for (auto &kv : deviceState()->dataRankToSlot)
      allSlots.insert(kv.second);

    struct SlotData {
      std::vector<BNGroup>       barneyGroups;
      std::vector<BNTransform>   barneyTransforms;
      std::vector<int>           instIDs;
      std::vector<math::float4>  attributes[Instance::Attributes::count];
    };
    std::map<int, SlotData> perSlot;
    for (int s : allSlots)
      perSlot[s];

    // Cache: (anariGroup, slot) -> BNGroup
    std::map<std::pair<const Group*, int>, BNGroup> barneyGroupCache;

    auto addInstToSlot = [&](Instance *inst, const Group *ag, int slot) {
      auto key = std::make_pair(ag, slot);
      BNGroup bg = 0;
      auto it = barneyGroupCache.find(key);
      if (it == barneyGroupCache.end()) {
        bg = ag->makeBarneyGroup(slot);
        barneyGroupCache[key] = bg;
      } else {
        bg = it->second;
      }
      if (!bg) return;

      auto &sd = perSlot[slot];
      BNTransform bt;
      sd.instIDs.push_back(inst->m_id);
      inst->writeTransform(&bt);
      sd.barneyTransforms.push_back(bt);
      sd.barneyGroups.push_back(bg);
      if (inst->attributes)
        for (int i = 0; i < Instance::Attributes::count; i++) {
          if (isnan(inst->attributes->values[i].x)) continue;
          while (sd.attributes[i].size() < sd.barneyTransforms.size())
            sd.attributes[i].push_back(math::float4(NAN));
          sd.attributes[i].back() = inst->attributes->values[i];
        }
    };

    for (auto inst : m_instances) {
      if (!inst) continue;
      const Group *ag = inst->group();
      if (!ag) continue;

      int slot = ag->resolvedSlot();

      if (ag->hasLights()) {
        for (int s : allSlots)
          addInstToSlot(inst, ag, s);
      } else {
        addInstToSlot(inst, ag, slot);
      }
    }

    assert(barneyModel);

    // Release old per-slot attribute data
    for (auto &[s, attribs] : m_perSlotAttribs)
      for (auto &d : attribs)
        if (d) { bnRelease(d); d = 0; }
    m_perSlotAttribs.clear();

    for (auto &[slot, sd] : perSlot) {
      bnSetInstances(barneyModel, slot,
                     sd.barneyGroups.data(),
                     sd.barneyTransforms.data(),
                     (int)sd.barneyGroups.size());

      auto &attribs = m_perSlotAttribs[slot];
      attribs.fill(0);
      for (int i = 0; i < Instance::Attributes::count; i++) {
        attribs[i] = bnDataCreate(context, slot, BN_FLOAT4,
                                  sd.attributes[i].size(),
                                  sd.attributes[i].data());
        std::string attribName = std::string("attribute") + std::to_string(i);
        bnSetInstanceAttributes(barneyModel, slot,
                                attribName.c_str(), attribs[i]);
      }

      bnBuild(barneyModel, slot);

      reportMessage(ANARI_SEVERITY_DEBUG,
                    "barney::World built slot %d with %d instances",
                    slot, (int)sd.barneyGroups.size());
    }

    for (auto &[key, bg] : barneyGroupCache)
      bnRelease(bg);
  }

  void World::transformOnlyUpdate()
  {
    auto barneyModel = tetheredModel->model;
    int  slot    = deviceState()->slot;

    std::vector<BNTransform> barneyTransforms;
    InstanceAttributes attributes;
    barneyTransforms.reserve(m_instances.size());
    for (auto &a : attributes)
      a.clear();

    for (auto inst : m_instances) {
      if (!inst) continue;
      if (!inst->group()) continue;

      BNTransform bt;
      inst->writeTransform(&bt);
      barneyTransforms.push_back(bt);

      if (!inst->attributes)
        continue;
      for (int i = 0; i < Instance::Attributes::count; i++) {
        if (isnan(inst->attributes->values[i].x))
          continue;
        while (attributes[i].size() < barneyTransforms.size())
          attributes[i].push_back(math::float4(NAN));
        attributes[i].back() = inst->attributes->values[i];
      }
    }

    assert(barneyModel);
    uploadInstanceAttributes(attributes);
    bnUpdateInstanceTransforms(barneyModel, slot,
                               barneyTransforms.data(),
                               (int)barneyTransforms.size());
  }

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::World *);
