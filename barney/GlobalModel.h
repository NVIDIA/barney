// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/Context.h"
#include "barney/ModelSlot.h"

namespace BARNEY_NS {

  struct ModelSlot;
  
  struct GlobalModel : public barney_api::Model {
    typedef std::shared_ptr<GlobalModel> SP;

    static SP create(Context *ctx) { return std::make_shared<GlobalModel>(ctx); }
    
    GlobalModel(Context *context);
    virtual ~GlobalModel();
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Model{}"; }

    void render(barney_api::Renderer    *renderer,
                barney_api::Camera      *camera,
                barney_api::FrameBuffer *fb) override;

    ModelSlot *getSlot(int whichSlot)
    {
      assert(whichSlot >= 0);
      assert(whichSlot < modelSlots.size());
      return modelSlots[whichSlot].get();
    }
    std::vector<ModelSlot::SP> modelSlots;

    void setInstances(int slot,
                      barney_api::Group **groups,
                      const affine3f *xfms,
                      int numInstances) override
    { getSlot(slot)->setInstances(groups,xfms,numInstances); }

    void updateInstanceTransforms(int slot,
                                  const affine3f *xfms,
                                  int numInstances) override
    { getSlot(slot)->updateInstanceTransforms(xfms,numInstances); }

    void setInstanceAttributes(int slot,
                               const std::string &which,
                               Data::SP data) override
    { getSlot(slot)->setInstanceAttributes(which,data?data->as<PODData>():PODData::SP{}); }

    void build(int slot) override
    { getSlot(slot)->build(); }
      
  };

} // ::BARNEY_NS
