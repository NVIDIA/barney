// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Object.h"
#include "native/ModelSlot.h"
#include "native/volume/MCAccelerator.h"

namespace BARNEY_NS {
  namespace native {
    
    struct Block;
    struct BlockStructuredCuBQLSampler;
    
    struct BlockStructuredField : public ScalarField
    {
      typedef std::shared_ptr<BlockStructuredField> SP;

      struct PLD {
        // rtc::ComputeKernel1D *mcRasterBlocks = 0;
        // rtc::ComputeKernel1D *computeElementBBs = 0;
        Block *blocks;
        float *scalars;
      };
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;

      /*! pretty-printer for printf-debugging */
      std::string toString() const override;

      /*! the sampler we create(d) for this object; lets us share this
          across multiple volumes and iso-surfaces */
      std::shared_ptr<BlockStructuredCuBQLSampler> sampler;
      
      /*! creates an acceleration structure for a 'isoSurface' geometry
        using this scalar field type */
      IsoSurfaceAccel::SP createIsoAccel(IsoSurface *isoSurface) override;
    
      
      struct DD : public ScalarField::DD {

#if RTC_DEVICE_CODE
        /* compute basis function contribution of given block at point P, and add
           that to 'sumWeightedValues' and 'sumWeights'. returns true if P is
           inside the block *filter domain*, false if outside (in which case the
           out params are not defined) */
        inline __rtc_device bool addBasisFunctions(float &sumWeightedValues,
                                                   float &sumWeights,
                                                   uint32_t bid,
                                                   vec3f P) const;
#endif
        const float   *scalars;
        struct {
          const vec3i *origins;
          const vec3i *dims;
          const int   *levels;
          const uint64_t *offsets;
        } perBlock;
        vec3f gridOrigin;
        vec3f gridSpacing;
        int numBlocks;
      };

      BlockStructuredField(Context *context,
                           const DevGroup::SP &devices);
      virtual ~BlockStructuredField() override;
    
      DD getDD(Device *device);
    
      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      bool setData(const std::string &member,
                   const std::shared_ptr<Data> &value) override;
      bool set3f(const std::string &member, const vec3f &value) override;
      /*! @} */
      // ------------------------------------------------------------------
    
      MCGrid::SP buildMCs() override;
    
      /*! computes, on specified device, the array of bounding box and
        value ranges for cubql bvh consturction; one box and one value
        range per each block */
      void computeElementBBs(Device *device,
                             box3f *d_primBounds,
                             range1f *d_primRanges);
    
      VolumeAccel::SP createAccel(Volume *volume) override;

      struct {
        PODData::SP/*3i*/ origins    = 0;
        PODData::SP/*3i*/ dims       = 0;
        PODData::SP/*1i*/ levels     = 0;
        PODData::SP/*1l*/ offsets    = 0;
      } perBlock;
      PODData::SP/*1f*/   scalars     = 0;
      vec3f               gridOrigin  {0.f};
      vec3f               gridSpacing {1.f};
      int                 numBlocks   = 0;
    };


    struct Block
    {
#if RTC_DEVICE_CODE
      static
      inline __rtc_device Block getFrom(const BlockStructuredField::DD &dd, int blockID, bool dbg=false);
    
      inline __rtc_device float getScalar(const vec3i cellID) const;
      inline __rtc_device box3f cellBounds(const vec3i cellID) const;
      inline __rtc_device box3f getDomain() const;
      inline __rtc_device range1f getValueRange() const;
#endif
      vec3i origin;
      vec3i dims;
      int   level;
      float cellSize;
      vec3f gridOrigin;
      vec3f gridSpacing;
      const float *scalars;
    };
  
  
#if RTC_DEVICE_CODE
    /* compute basis function contribution of given block at point P, and add
       that to 'sumWeightedValues' and 'sumWeights'. returns true if P is inside
       the block *filter domain*, false if outside (in which case the out params
       are not defined) */
    inline __rtc_device
    bool BlockStructuredField::DD::addBasisFunctions(float &sumWeightedValues,
                                                     float &sumWeights,
                                                     uint32_t bid,
                                                     vec3f P) const
    {
      const auto block = Block::getFrom(*this,bid);
      const box3f domain = block.getDomain();

      if (!domain.contains(P)) return false;

      const vec3f logicalP = (P-block.gridOrigin)/block.gridSpacing;
      const vec3f cellCenter000
        = (vec3f(block.origin)+.5f)*block.cellSize;
      const vec3f localPos
        = (logicalP-cellCenter000) / block.cellSize;
    
      vec3f floor_localPos(floorf(localPos.x),
                           floorf(localPos.y),
                           floorf(localPos.z));
      vec3i idx_lo   = vec3i(floor_localPos);
      idx_lo = max(vec3i(-1), idx_lo);
      const vec3i idx_hi   = idx_lo + vec3i(1);
      const vec3f frac     = localPos - floor_localPos;
      const vec3f neg_frac = vec3f(1.f) - frac;

      if (idx_lo.z >= 0 && idx_lo.z < block.dims.z) {
        if (idx_lo.y >= 0 && idx_lo.y < block.dims.y) {
          if (idx_lo.x >= 0 && idx_lo.x < block.dims.x) {
            const float scalar = block.getScalar({idx_lo.x,idx_lo.y,idx_lo.z});
            const float weight = (neg_frac.z)*(neg_frac.y)*(neg_frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
          if (idx_hi.x >= 0 && idx_hi.x < block.dims.x) {
            const float scalar = block.getScalar({idx_hi.x,idx_lo.y,idx_lo.z});
            const float weight = (neg_frac.z)*(neg_frac.y)*(frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
        }
        if (idx_hi.y >= 0 && idx_hi.y < block.dims.y) {
          if (idx_lo.x >= 0 && idx_lo.x < block.dims.x) {
            const float scalar = block.getScalar({idx_lo.x,idx_hi.y,idx_lo.z});
            const float weight = (neg_frac.z)*(frac.y)*(neg_frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
          if (idx_hi.x >= 0 && idx_hi.x < block.dims.x) {
            const float scalar = block.getScalar({idx_hi.x,idx_hi.y,idx_lo.z});
            const float weight = (neg_frac.z)*(frac.y)*(frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
        }
      }
        
      if (idx_hi.z >= 0 && idx_hi.z < block.dims.z) {
        if (idx_lo.y >= 0 && idx_lo.y < block.dims.y) {
          if (idx_lo.x >= 0 && idx_lo.x < block.dims.x) {
            const float scalar = block.getScalar({idx_lo.x,idx_lo.y,idx_hi.z});
            const float weight = (frac.z)*(neg_frac.y)*(neg_frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
          if (idx_hi.x < block.dims.x) {
            const float scalar = block.getScalar({idx_hi.x,idx_lo.y,idx_hi.z});
            const float weight = (frac.z)*(neg_frac.y)*(frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
        }
        if (idx_hi.y >= 0 && idx_hi.y < block.dims.y) {
          if (idx_lo.x >= 0 && idx_lo.x < block.dims.x) {
            const float scalar = block.getScalar({idx_lo.x,idx_hi.y,idx_hi.z});
            const float weight = (frac.z)*(frac.y)*(neg_frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
          if (idx_hi.x >= 0 && idx_hi.x < block.dims.x) {
            const float scalar = block.getScalar({idx_hi.x,idx_hi.y,idx_hi.z});
            const float weight = (frac.z)*(frac.y)*(frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
        }
      }
      return true;
    }

    inline __rtc_device
    float Block::getScalar(const vec3i cellID) const
    {
      const int idx
        = 
        + cellID.x
        + cellID.y * dims.x
        + cellID.z * dims.x*dims.y;
      return scalars[idx];
    }

    inline __rtc_device
    box3f Block::cellBounds(const vec3i cellID) const
    {
      box3f cb;
      const vec3f logicalLower = (vec3f(origin+cellID)-.5f)*cellSize;
      cb.lower = gridOrigin + gridSpacing*logicalLower;
      cb.upper = gridOrigin
        + gridSpacing*(logicalLower + 2.f*vec3f(cellSize));
      return cb;
    }

    inline __rtc_device
    range1f Block::getValueRange() const
    {
      range1f range;
      for (int i=0;i<dims.x*dims.y*dims.z;i++)
        range.extend(scalars[i]);
      return range;
    }
  
    inline __rtc_device
    box3f Block::getDomain() const
    {
      box3f cb;
      cb.lower = gridOrigin
        + gridSpacing*((vec3f(origin)-.5f)*cellSize);
      cb.upper = gridOrigin
        + gridSpacing*((vec3f(origin+dims)+.5f)*cellSize);
      return cb;
    }

    inline __rtc_device
    Block Block::getFrom(const BlockStructuredField::DD &dd, int blockID, bool dbg)
    {
      Block block;
      block.origin   = dd.perBlock.origins[blockID];
      block.dims     = dd.perBlock.dims[blockID];
      block.level    = dd.perBlock.levels[blockID];
      block.cellSize =
        1.f/(1<<block.level);
      block.gridOrigin = dd.gridOrigin;
      block.gridSpacing = dd.gridSpacing;
      block.scalars  = dd.scalars+dd.perBlock.offsets[blockID];
      return block;
    }
#endif
  }
}
