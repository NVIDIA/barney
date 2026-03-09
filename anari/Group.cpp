// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "Group.h"
#include <iostream>
#include "common.h"

namespace barney_device {

  Group::Group(BarneyGlobalState *s)
    : Object(ANARI_GROUP, s),
      m_surfaceData(this),
      m_volumeData(this),
      m_lightData(this)
  {}

  Group::~Group() = default;

  void Group::commitParameters()
  {
    m_surfaceData = getParamObject<ObjectArray>("surface");
    m_volumeData = getParamObject<ObjectArray>("volume");
    m_lightData = getParamObject<ObjectArray>("light");
  }

  void Group::markFinalized()
  {
    deviceState()->markStructuralSceneChanged();
    Object::markFinalized();
  }

  int Group::resolvedSlot() const
  {
    if (m_volumeData) {
      for (auto it = m_volumeData->handlesBegin(); it != m_volumeData->handlesEnd(); ++it) {
        auto *v = (Volume *)*it;
        if (!v || !v->isValid()) continue;
        auto *field = v->field();
        if (!field) continue;
        int dr = field->getParam<int>("dataRank", -1);
        int s = deviceState()->resolveSlot(dr);
        if (s >= 0) return s;
      }
    }
    return deviceState()->slot;
  }

  bool Group::hasLights() const
  {
    return m_lightData && m_lightData->size() > 0;
  }

  BNGroup Group::makeBarneyGroup(int slot) const
  {
    if (slot < 0) slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

    std::vector<BNGeom> barneyGeometries;
    std::vector<Surface *> surfaces;
    std::vector<BNVolume> barneyVolumes;
    std::vector<Volume *> volumes;
    std::vector<BNLight> barneyLights;
    std::vector<Light *> lights;

    // Surfaces //

    if (m_surfaceData) {
      std::for_each(m_surfaceData->handlesBegin(),
                    m_surfaceData->handlesEnd(),
                    [&](auto *o) {
                      auto *s = (Surface *)o;
                      if (s && s->isValid() && s->isVisible())
                        surfaces.push_back(s);
                      else {
                        reportMessage(ANARI_SEVERITY_WARNING,
                                      "encountered invalid surface (%p)",
                                      this,
                                      s);
                      }
                    });
    }

    for (auto s : surfaces) {
      auto bg = s->getBarneyGeom();
      barneyGeometries.push_back(bg);
    }

    // Volumes //

    if (m_volumeData) {
      std::for_each(m_volumeData->handlesBegin(),
                    m_volumeData->handlesEnd(),
                    [&](auto *o) {
                      auto *v = (Volume *)o;
                      if (v && v->isValid() && v->isVisible())
                        volumes.push_back(v);
                      else {
                        reportMessage(ANARI_SEVERITY_WARNING,
                                      "encountered invalid volume (%p)",
                                      this,
                                      v);
                      }
                    });
    }

    for (auto v : volumes)
      barneyVolumes.push_back(v->getBarneyVolume());

    // Lights //

    if (m_lightData) {
      std::for_each(m_lightData->handlesBegin(),
                    m_lightData->handlesEnd(),
                    [&](auto *o) {
                      auto *l = (Light *)o;
                      if (l && l->isValid())
                        lights.push_back(l);
                      else {
                        reportMessage(ANARI_SEVERITY_WARNING,
                                      "encountered invalid volume (%p)",
                                      this,
                                      l);
                      }
                    });
    }

    for (auto l : lights) {
      if (l && l->isValid())
        barneyLights.push_back(l->getBarneyLight());
    }

    BNData lightsData = nullptr;
    if (!barneyLights.empty()) {
      lightsData = bnDataCreate(context, slot, BN_OBJECT,
                                (int)barneyLights.size(), barneyLights.data());
    }

    BNGroup bg = bnGroupCreate(context,slot,
                               barneyGeometries.data(),
                               (int)barneyGeometries.size(),
                               barneyVolumes.data(),
                               (int)barneyVolumes.size());
    if (lightsData) {
      bnSetData(bg, "lights", lightsData);
      bnRelease(lightsData);
      bnCommit(bg);
    }
    bnGroupBuild(bg);

    reportMessage(ANARI_SEVERITY_DEBUG,
                  "barney::Group constructed with %zu surfaces, %zu volumes, and %zu lights",
                  barneyGeometries.size(),
                  barneyVolumes.size(),
                  barneyLights.size());

    return bg;
  }

  box3 Group::bounds() const
  {
    box3 result;
    result.invalidate();
    if (m_surfaceData) {
      std::for_each(m_surfaceData->handlesBegin(),
                    m_surfaceData->handlesEnd(),
                    [&](auto *o) {
                      auto *s = (Surface *)o;
                      if (s && s->isValid())
                        result.insert(s->geometry()->bounds());
                    });
    }

    if (m_volumeData) {
      std::for_each(
                    m_volumeData->handlesBegin(), m_volumeData->handlesEnd(), [&](auto *o) {
                      auto *v = (Volume *)o;
                      if (v && v->isValid())
                        result.insert(v->bounds());
                    });
    }
    return result;
  }

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Group *);
