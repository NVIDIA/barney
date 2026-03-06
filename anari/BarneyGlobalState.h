// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney_math.h"
// helium
#include "helium/BaseGlobalDeviceState.h"
#include <memory>
#include <map>
#include <unordered_map>

namespace barney_device {

  inline void bnSetAndRelease(BNObject o, const char *n, BNObject v)
  {
    bnSetObject(o,n,v);
    bnRelease(v);
  }
  inline void bnSetAndRelease(BNObject o, const char *n, BNData v)
  {
    bnSetData(o,n,v);
    bnRelease(v);
  }

  struct Frame;
  struct World;

  struct BarneyDevice;
  struct Tether;
  
  struct TetheredModel : public std::enable_shared_from_this<TetheredModel> {
    typedef std::shared_ptr<TetheredModel> SP;
    TetheredModel(Tether *tether, int uniqueID);
    ~TetheredModel();
    BNModel model = 0;
    Tether *const tether;
    int     const uniqueID;
  };

  /*! keeps info on multiple (banari-)devices that are tethered
      together onto a singel barney ncontext */
  struct Tether {
    ~Tether();
    
    BNContext context{nullptr};

    bool allDevicesPresent();

    TetheredModel::SP getOrCreateTetheredModel(int uniqueID);
    std::map<int,TetheredModel*> activeModels;
    std::mutex mutex;

    int numRenderCallsOutstanding = 0;
    struct {
      BNCamera camera;
      BNRenderer renderer;
      BNFrameBuffer fb;
      BNModel model;
    } deferredRenderCall;

    std::vector<BarneyDevice *> devices;
  };

  struct BarneyGlobalState : public helium::BaseGlobalDeviceState
  {
    struct ObjectUpdates
    {
      helium::TimeStamp lastSceneChange{0};
    } objectUpdates;

    int slot = -1;

    // Maps data rank IDs to context slot indices when multi-slot mode is active.
    // Populated during context creation when dataGroupIDs are provided.
    std::unordered_map<int, int> dataRankToSlot;

    int resolveSlot(int dataRank) const
    {
      if (dataRank >= 0 && !dataRankToSlot.empty()) {
        auto it = dataRankToSlot.find(dataRank);
        if (it != dataRankToSlot.end())
          return it->second;
      }
      return slot;
    }

    std::shared_ptr<Tether> tether;

    /*! created models get consecutive IDs, which allows us for
        identifying which models created by which (tethered) device(s)
        belong togther. Ie, if two devices A and B are tethered, then
        the i'th model of A is always tethered with the i'th model of
        B (and vice versa) */
    int nextUniqueModelID = 0;

    bool hasBeenCommitted = false;

    // Helper methods //

    BarneyGlobalState(ANARIDevice d);
    ~BarneyGlobalState();
    
    void markSceneChanged();
  };

  // Helper functions/macros ////////////////////////////////////////////////////

  inline BarneyGlobalState *asBarneyState(helium::BaseGlobalDeviceState *s)
  {
    return (BarneyGlobalState *)s;
  }

#define BARNEY_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)   \
  namespace anari {                                             \
    ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);             \
  }

#define BARNEY_ANARI_TYPEFOR_DEFINITION(type)   \
  namespace anari {                             \
    ANARI_TYPEFOR_DEFINITION(type);             \
  }

} // namespace barney_device
