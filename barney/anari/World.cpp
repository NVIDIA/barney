// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "World.h"
// std
#include <algorithm>
#include <map>

namespace BARNEY_NS {
  namespace anari {
    
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

      auto context = deviceState()->tether->context;
      for (int i = 0; i < Instance::Attributes::count; i++) {
        if (m_attributesData[i]) {
          bnRelease(m_attributesData[i]);
          m_attributesData[i] = 0;
        }
      }

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
          if (inst && inst->isValid())
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
                        // Keep invalid instances so a later transform edit
                        // can make them renderable without replacing the array.
                        if (o)
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
#if 1
        // iw, 4/29/26 - the logic below is broken: if only the
        // world::surface[] array changes, but neither of the groups or
        // instaneces therein was changed, then the scnee is nor marked
        // as structurally changed (because it's the group or instance
        // commits that trigger that), but of course a different surface
        // list _is_ a structural change. for now, let's not try to be
        // clever and simply rebuild all.
        true
#else
        m_lastBarneyModelBuild == 0
        || state->objectUpdates.lastStructuralChange > m_lastBarneyModelBuild
#endif
        ;

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
      int  slot    = deviceState()->slot;

      std::vector<BNGroup>     barneyGroups;
      std::vector<BNTransform> barneyTransforms;
      InstanceAttributes attributes;
      barneyGroups.reserve(m_instances.size());
      barneyTransforms.reserve(m_instances.size());

      std::map<const Group *, BNGroup> groupDedup;

      for (auto inst : m_instances) {
        if (!inst || !inst->isValid()) continue;
        const Group *ag = inst->group();
        if (!ag) continue;

        auto [it, inserted] = groupDedup.try_emplace(ag, nullptr);
        if (inserted)
          it->second = ag->makeBarneyGroup();
        BNGroup bg = it->second;
        if (!bg)
          continue;

        BNTransform bt;
        inst->writeTransform(&bt);
        barneyTransforms.push_back(bt);
        barneyGroups.push_back(bg);
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
      bnSetInstances(barneyModel, slot,
                     barneyGroups.data(), barneyTransforms.data(),
                     (int)barneyGroups.size());

      for (auto &[_, bg] : groupDedup)
        bnRelease(bg);

      uploadInstanceAttributes(attributes);
      uploadMotionDeltas();
      bnBuild(barneyModel, slot);
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
        if (!inst || !inst->isValid()) continue;

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
      uploadMotionDeltas();
    }

    void World::uploadMotionDeltas()
    {
      bool anyMotion = false;
      for (auto inst : m_instances) {
        if (inst && inst->isValid() && inst->hasMotionTransform()) {
          anyMotion = true;
          break;
        }
      }
      auto barneyModel = tetheredModel->model;
      int  slot        = deviceState()->slot;
      if (!anyMotion) {
        bnSetInstanceMotionDeltas(barneyModel, slot, nullptr, 0);
        return;
      }
      std::vector<BNTransform> deltas;
      deltas.reserve(m_instances.size());
      for (auto inst : m_instances) {
        if (!inst || !inst->isValid()) continue;
        BNTransform bt;
        inst->writeMotionDelta(&bt);
        deltas.push_back(bt);
      }
      bnSetInstanceMotionDeltas(barneyModel, slot,
                                deltas.data(),
                                (int)deltas.size());
    }
    
  } // ::BARNEY_NS::anari
} // ::BARNEY_NS

BARNEY_ANARI_TYPEFOR_DEFINITION(BARNEY_NS::anari::World *);
