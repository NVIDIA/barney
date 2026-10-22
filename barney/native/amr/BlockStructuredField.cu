// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/amr/BlockStructuredField.h"
#include "native/Context.h"
#include "native/volume/MCGrid.cuh"
#include "native/amr/BlockStructuredCuBQLSampler.h"

namespace BARNEY_NS {
  namespace native {

    RTC_IMPORT_USER_GEOM(/*file*/BlockStructuredMC,
                         /*name*/BlockStructuredMC,
                         /*geomtype device data */
                         MCVolumeAccel<BlockStructuredCuBQLSampler>::DD,false,false);
    RTC_IMPORT_USER_GEOM(/*file*/BlockStructuredMC,/*name*/BlockStructuredMC_Iso,
                         /*geomtype device data */
                         MCIsoSurfaceAccel<BlockStructuredCuBQLSampler>::DD,false,false);

    enum { MC_GRID_SIZE = 512 };

    BlockStructuredField::PLD *BlockStructuredField::getPLD(Device *device) 
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }

    __rtc_global
    void BSField_rasterGrids(const rtc::ComputeInterface &ci,
                             BlockStructuredField::DD field,
                             MCGrid::DD               grid)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= field.numBlocks) return;

      const Block block = Block::getFrom(field,tid);

      vec3i numCells = block.dims;

      const box3f worldBox = getBox(field.worldBounds);
      for (int z=0;z<numCells.z;z++) {
        for (int y=0;y<numCells.y;y++) {
          for (int x=0;x<numCells.x;x++) {
            const box3f cb3 = block.cellBounds({x,y,z});
            const float scalar = block.getScalar({x,y,z});
            const box4f cellBounds(vec4f(cb3.lower,scalar),
                                   vec4f(cb3.upper,scalar));
            rasterBox(grid,worldBox,cellBounds);
          }
        }
      }
#endif
    }
  
    __rtc_global
    void computeWorldBounds(const rtc::ComputeInterface &ci,
                            box3f *pBounds,
                            const BlockStructuredField::DD dd)
    {
#if RTC_DEVICE_CODE
      int tid = ci.launchIndex().x;
      if (tid >= dd.numBlocks)
        return;

      Block block = Block::getFrom(dd,tid);
      box3f bb = block.getDomain();
      rtc::fatomicMin(&pBounds->lower.x,bb.lower.x);
      rtc::fatomicMin(&pBounds->lower.y,bb.lower.y);
      rtc::fatomicMin(&pBounds->lower.z,bb.lower.z);
      rtc::fatomicMax(&pBounds->upper.x,bb.upper.x);
      rtc::fatomicMax(&pBounds->upper.y,bb.upper.y);
      rtc::fatomicMax(&pBounds->upper.z,bb.upper.z);
#endif
    }
  
    MCGrid::SP BlockStructuredField::buildMCs()
    {
      if (mcGrid) return mcGrid;

      mcGrid = std::make_shared<MCGrid>(devices);
      auto &grid = *mcGrid;

      std::cout << OWL_TERMINAL_BLUE
                << "#bn.amr: building initial macro cell grid"
                << OWL_TERMINAL_DEFAULT << std::endl;
      numBlocks = (int)perBlock.origins->count;

      // =============================================================================
      // compute macro cell grid dims
      // =============================================================================
    
      float maxWidth = reduce_max(getBox(worldBounds).size());
      vec3i dims = 1+vec3i(getBox(worldBounds).size() * ((MC_GRID_SIZE-1) / maxWidth));
      // printf("#bn.amr: chosen macro-cell dims of (%i %i %i)\n",
      //        dims.x,
      //        dims.y,
      //        dims.z);
      grid.resize(dims);
      grid.gridOrigin = worldBounds.lower;
      grid.gridSpacing = worldBounds.size() * rcp(vec3f(dims));
    
      grid.clearCells();

      for (auto device : *devices) {
        __rtc_launch(device->rtc,
                     BSField_rasterGrids,
                     dru(numBlocks,1024),1024,
                     this->getDD(device),grid.getDD(device));
      }
    
      for (auto device : *devices) 
        device->sync();
    
      return mcGrid;
    }
  

    BlockStructuredField::~BlockStructuredField()
    {}
    

  
    BlockStructuredField::BlockStructuredField(Context *context,
                                               const DevGroup::SP &devices)
      : ScalarField(context,devices)
    {
      perLogical.resize(devices->numLogical);
    }

    BlockStructuredField::DD BlockStructuredField::getDD(Device *device)
    {
      BlockStructuredField::DD dd;

      // inherited:
      (ScalarField::DD &)dd = ScalarField::getDD(device);
    
      dd.perBlock.origins = (const vec3i *)perBlock.origins->getDD(device);
      dd.perBlock.dims    = (const vec3i *)perBlock.dims->getDD(device);
      dd.perBlock.levels  = (const int   *)perBlock.levels->getDD(device);
      dd.perBlock.offsets = (const uint64_t *)perBlock.offsets->getDD(device);

      dd.scalars      = (const float *)scalars->getDD(device);
      dd.gridOrigin   = gridOrigin;
      dd.gridSpacing  = gridSpacing;
    
      dd.numBlocks    = (int)perBlock.origins->count;

      return dd;
    }

    VolumeAccel::SP BlockStructuredField::createAccel(Volume *volume)
    {
      if (!sampler)
        sampler
          = std::make_shared<BlockStructuredCuBQLSampler>(this);
      return std::make_shared<MCVolumeAccel<BlockStructuredCuBQLSampler>>
        (volume,
         createGeomType_BlockStructuredMC,
         sampler);
    }

    __rtc_global
    void BSField_computeElementBBs(const rtc::ComputeInterface &ci,
                                   box3f         *d_primBounds,
                                   range1f       *d_primRanges,
                                   BlockStructuredField::DD field)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= field.numBlocks) return;

      Block block = Block::getFrom(field,tid);
      d_primBounds[tid] = block.getDomain();

      if (isnan(d_primBounds[tid].lower.x) ||
          isnan(d_primBounds[tid].lower.y) || 
          isnan(d_primBounds[tid].lower.z) ||
          isnan(d_primBounds[tid].upper.x) ||
          isnan(d_primBounds[tid].upper.y) || 
          isnan(d_primBounds[tid].upper.z) ) {
        printf("NAN box\n");
      }

      if (isinf(d_primBounds[tid].lower.x) ||
          isinf(d_primBounds[tid].lower.y) || 
          isinf(d_primBounds[tid].lower.z) ||
          isinf(d_primBounds[tid].upper.x) ||
          isinf(d_primBounds[tid].upper.y) || 
          isinf(d_primBounds[tid].upper.z) ) {
        printf("INF box %i - org %i %i %i dim %i %i %i cellsize %f\n",tid,
               block.origin.x,
               block.origin.y,
               block.origin.z,
               block.dims.x,
               block.dims.y,
               block.dims.z,
               block.cellSize
               );
      }
    
      if (d_primRanges) d_primRanges[tid] = block.getValueRange();
#endif
    }
    
    /*! computes, on specified device, the bounding boxes and - if
      d_primRanges is non-null - the primitmives ranges. d_primBounds
      and d_primRanges (if non-null) must be pre-allocated and
      writeaable on specified device */
    void BlockStructuredField::computeElementBBs(Device  *device,
                                                 box3f   *d_primBounds,
                                                 range1f *d_primRanges)
    {
      int bs = 128;
      int nb = divRoundUp(numBlocks,bs);
      __rtc_launch(device->rtc, BSField_computeElementBBs,
                   nb,bs,
                   d_primBounds,
                   d_primRanges,
                   getDD(device));
      device->sync();
    }

    bool BlockStructuredField::setData(const std::string &member,
                                       const std::shared_ptr<Data> &value)
    {
      if (ScalarField::setData(member,value)) return true;

      if (member == "grid.origins") {
        perBlock.origins = value->as<PODData>();
        return true;
      }
      if (member == "grid.dims") {
        perBlock.dims = value->as<PODData>();
        return true;
      }
      if (member == "grid.levels") {
        perBlock.levels = value->as<PODData>();
        return true;
      }
      if (member == "grid.offsets") {
        perBlock.offsets = value->as<PODData>();
        return true;
      }
      if (member == "scalars") {
        scalars = value->as<PODData>();
        return true;
      }

      return false;
    }

    bool BlockStructuredField::set3f(const std::string &member,
                                     const vec3f &value)
    {
      if (ScalarField::set3f(member,value)) return true;
      if (member == "gridOrigin") {
        gridOrigin = value;
        return true;
      }
      if (member == "gridSpacing") {
        gridSpacing = value;
        return true;
      }
      return false;
    }

    /*! creates an acceleration structure for a 'isoSurface' geometry
      using this scalar field type */
    IsoSurfaceAccel::SP BlockStructuredField::createIsoAccel(IsoSurface *isoSurface)
    {
      if (!sampler)
        sampler = std::make_shared<BlockStructuredCuBQLSampler>(this);
      return std::make_shared<MCIsoSurfaceAccel<BlockStructuredCuBQLSampler>>
        (isoSurface,
         createGeomType_BlockStructuredMC_Iso,
         sampler);
    }
    
    
    /*! pretty-printer for printf-debugging */
    std::string BlockStructuredField::toString() const
    { return "barney::native::BlockStructuredField"; }
    
    
    void BlockStructuredField::commit()
    {
      assert(perBlock.origins);
      assert(perBlock.dims);
      assert(perBlock.levels);
      assert(perBlock.offsets);

      numBlocks = (int)perBlock.origins->count;
      assert(numBlocks > 0);
      assert(perBlock.dims->count == numBlocks);
      assert(perBlock.offsets->count == numBlocks);
      assert(perBlock.levels->count == numBlocks);

      // ==================================================================
      // compute world bounds
      // ==================================================================
      {
        auto dev = (*devices)[0];
        SetActiveGPU forDuration(dev);
        auto rtc = dev->rtc;
        box3f *d_worldBounds = (box3f *)rtc->allocMem(sizeof(box3f));
        rtc->copy(d_worldBounds,&worldBounds,sizeof(worldBounds));
        __rtc_launch(// dev and kernel
                     rtc,computeWorldBounds,
                     // launch config
                     divRoundUp(numBlocks,128),128,
                     // kernel args
                     d_worldBounds,getDD(dev));
        rtc->sync();
        rtc->copy(&worldBounds,d_worldBounds,sizeof(worldBounds));
      }
    
    }
  }
}
