// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/volume/MCGrid.h"
#include "native/volume/Volume.h"

namespace BARNEY_NS {
  namespace native {
    
    MajorantsGrid::MajorantsGrid(MCGrid::SP mcGrid)
      : mcGrid(mcGrid),
        devices(mcGrid->devices)
    {
      perLogical.resize(devices->numLogical);
    }

    MajorantsGrid::~MajorantsGrid()
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        PLD *pld = getPLD(device);
        if (pld->majorantsBuffer) {
          device->rtc->freeBuffer(pld->majorantsBuffer);
          pld->majorantsBuffer = 0;
        }
      }
    }
  
    MCGrid::MCGrid(const DevGroup::SP &devices)
      : devices(devices)
    {
      perLogical.resize(devices->numLogical);
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;
        SetActiveGPU forDuration(device);

        pld->scalarRangesBuffer = rtc->createBuffer(sizeof(range1f));
      }
    }

    MCGrid::~MCGrid()
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        PLD *pld = getPLD(device);
        if (pld->scalarRangesBuffer) {
          device->rtc->freeBuffer(pld->scalarRangesBuffer);
          pld->scalarRangesBuffer = 0;
        }
      }
    }

    /* kernel CODE */
    __rtc_global
    void clearMCs(const rtc::ComputeInterface &ci, MCGrid::DD grid)
    {
#if RTC_DEVICE_CODE
      int ix = ci.getThreadIdx().x
        +ci.getBlockIdx().x*ci.getBlockDim().x;
      if (ix >= grid.dims.x*grid.dims.y*grid.dims.z) return;
    
      grid.scalarRanges[ix] = { +BARNEY_INF, -BARNEY_INF };
#endif
    }
  
    /*! re-set all cells' ranges to "infinite empty" */
    void MCGrid::clearCells()
    {
      size_t numCells = owl::common::reduce_mul(dims);
      const int bs = 1024;
      // cuda num blocks
      // const int nb = (int)divRoundUp((size_t)numCells,(size_t)bs);
      const int nb = (int)dru(numCells,bs);
    
      for (auto device : *devices) {
        auto d_grid = getDD(device);
        __rtc_launch(device->rtc,
                     clearMCs,
                     nb,bs,
                     d_grid);
      }
      for (auto device : *devices) 
        device->sync();
    }
  
    
    __rtc_global
    void mapMCs(const rtc::ComputeInterface &ci,
                MajorantsGrid::DD grid,
                TransferFunction::DD xf)
    {
#if RTC_DEVICE_CODE
      int ix = ci.getThreadIdx().x
        +ci.getBlockIdx().x*ci.getBlockDim().x;
      if (ix >= grid.dims.x*grid.dims.y*grid.dims.z) return;
      range1f scalarRange = grid.scalarRanges[ix];
      grid.majorants[ix] = xf.majorant(scalarRange);
#endif
    }
  
    void MajorantsGrid::computeMajorants(Volume *volume)
    {
      if (dims != mcGrid->dims)
        resize(mcGrid->dims); 
      assert(volume);
      auto dims = mcGrid->dims;
      assert(dims.x > 0);
      assert(dims.y > 0);
      assert(dims.z > 0);
      size_t numCells = owl::common::reduce_mul(dims);
      const int bs = 1024;
      const int nb = (int)dru(numCells,bs);
      for (auto device : *devices) 
        device->sync();
    
      for (auto device : *devices) {
        auto d_xf = volume->xf.getDD(device);
        auto dd = getDD(device);
        __rtc_launch(device->rtc,
                     mapMCs,
                     nb,bs,
                     dd,d_xf);
      }
      for (auto device : *devices) 
        device->sync();
    }
  
    /*! allocate memory for the given grid */
    void MajorantsGrid::resize(vec3i dims)
    {
      assert(dims.x > 0);
      assert(dims.y > 0);
      assert(dims.z > 0);
      if (dims == this->dims)
        return;
      mcGrid->resize(dims);
      size_t numCells = owl::common::reduce_mul(dims);
    
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;
        auto *replacement = rtc->createBuffer(sizeof(float)*numCells);
        if (pld->majorantsBuffer)
          rtc->freeBuffer(pld->majorantsBuffer);
        pld->majorantsBuffer = replacement;
      }
      for (auto device : *devices)
        device->sync();
      // A failed partial multi-device resize must remain retryable.
      this->dims = dims;
    }

    /*! allocate memory for the given grid */
    void MCGrid::resize(vec3i dims)
    {
      assert(dims.x > 0);
      assert(dims.y > 0);
      assert(dims.z > 0);
      if (dims == this->dims)
        return;
      this->dims = dims;
      size_t numCells = owl::common::reduce_mul(dims);
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;
        auto *replacement = rtc->createBuffer(sizeof(range1f)*numCells);
        rtc->freeBuffer(pld->scalarRangesBuffer);
        pld->scalarRangesBuffer = replacement;
      }
      for (auto device : *devices) 
        device->sync();
    }
  
    /*! get cuda-usable device-data for given device ID (relative to
      devices in the devgroup that this gris is in */
    MCGrid::DD MCGrid::getDD(Device *device) 
    {
      PLD *pld = getPLD(device);
    
      MCGrid::DD dd;
    
      dd.scalarRanges
        = (range1f*)pld->scalarRangesBuffer->getDD();

      dd.dims = dims;
      dd.gridOrigin = gridOrigin;
      dd.gridSpacing = gridSpacing;
      return dd;
    }

    /*! get cuda-usable device-data for given device ID (relative to
      devices in the devgroup that this gris is in */
    MajorantsGrid::DD MajorantsGrid::getDD(Device *device) 
    {
      MajorantsGrid::DD dd;

      (MCGrid::DD&)dd = mcGrid->getDD(device);
    
      PLD *pld = getPLD(device);
      dd.majorants
        = (float *)pld->majorantsBuffer->getDD();
      return dd;
    }

  }
} // ::barney



