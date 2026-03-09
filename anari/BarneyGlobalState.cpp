// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#define ANARI_BARNEY_MATH_DEFINITIONS 1

#include "BarneyGlobalState.h"
#include "Frame.h"

namespace barney_device {

  Tether::~Tether()
  {
    BANARI_TRACK_LEAKS(std::cout << "#banari: tether destructing - "
                       "releasing barney context" << std::endl);
    if (context) { bnContextDestroy(context); context = 0; }
  }

  BarneyGlobalState::BarneyGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d)
  {}

  BarneyGlobalState::~BarneyGlobalState()
  {
    BANARI_TRACK_LEAKS(std::cout << "#banari: barneyglobalstate destructing"
                       " - releasing tether" << std::endl);
  }
  
  void BarneyGlobalState::markSceneChanged()
  {
    objectUpdates.lastSceneChange = helium::newTimeStamp();
  }

  void BarneyGlobalState::markStructuralSceneChanged()
  {
    auto ts = helium::newTimeStamp();
    objectUpdates.lastSceneChange = ts;
    objectUpdates.lastStructuralChange = ts;
  }

  bool Tether::allDevicesPresent()
  {
    for (auto dev : devices)
      if (dev == 0) return false;
    return true;
  }

  TetheredModel::SP Tether::getOrCreateTetheredModel(int uniqueID)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (activeModels.find(uniqueID) != activeModels.end()) {
      BANARI_TRACK_LEAKS(std::cout << "#banari returning already created model "
                         << uniqueID << std::endl);
      return activeModels[uniqueID]->shared_from_this();
    }

    BANARI_TRACK_LEAKS(std::cout << "#banari creating new tethered model "
                       << uniqueID << std::endl);
    TetheredModel::SP newModel = std::make_shared<TetheredModel>(this,uniqueID);
    return newModel;
  }

  TetheredModel::TetheredModel(Tether *tether, int uniqueID)
    : tether(tether),
      uniqueID(uniqueID)
  {
    BANARI_TRACK_LEAKS(std::cout << "#banari: creating new tetherd model ID "
                       << uniqueID << std::endl);
    model = bnModelCreate(tether->context);
    BANARI_TRACK_LEAKS(std::cout << "#banari: created new barney model "
                       << (int*)model << std::endl);
    
    // iw do NOT try to lock tether, it's already locked when it creates us!
    tether->activeModels[uniqueID] = this;
  }
  
  TetheredModel::~TetheredModel()
  {
    BANARI_TRACK_LEAKS(std::cout << "#banari: tethered model is dying" << std::endl);
    std::lock_guard<std::mutex> lock(tether->mutex);
    tether->activeModels.erase(tether->activeModels.find(uniqueID));
    
    if (model) {
      BANARI_TRACK_LEAKS(std::cout << "#banari: releasing barney model handle ID "
                         << uniqueID << std::endl);
      bnRelease(model);
    }
  }
  
} // namespace barney_device
