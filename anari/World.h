// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "Array.h"
#include "Instance.h"
#include <array>
#include <map>

namespace barney_device {

  struct World : public Object
  {
    World(BarneyGlobalState *s);
    ~World() override;

    bool getProperty(const std::string_view &name,
                     ANARIDataType type,
                     void *ptr,
                     uint64_t size,
                     uint32_t flags) override;

    void commitParameters() override;
    void finalize() override;

    BNModel makeCurrent();
    void markFinalized() override;

  private:
    using InstanceAttributes
      = std::array<std::vector<math::float4>, Instance::Attributes::count>;

    void buildBarneyModel();
    void uploadInstanceAttributes(const InstanceAttributes &attributes);
    void fullRebuild();
    void transformOnlyUpdate();

    helium::ChangeObserverPtr<ObjectArray> m_zeroSurfaceData;
    helium::ChangeObserverPtr<ObjectArray> m_zeroVolumeData;
    helium::ChangeObserverPtr<ObjectArray> m_zeroLightData;
    helium::ChangeObserverPtr<ObjectArray> m_instanceData;

    helium::IntrusivePtr<Group> m_zeroGroup;
    helium::IntrusivePtr<Instance> m_zeroInstance;

    std::vector<Instance *> m_instances;

    TetheredModel::SP tetheredModel;

    using AttribArray = std::array<BNData, Instance::Attributes::count>;
    std::map<int, AttribArray> m_perSlotAttribs;
    helium::TimeStamp m_lastBarneyModelBuild{0};
  };

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::World *, ANARI_WORLD);
