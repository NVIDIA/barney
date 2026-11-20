// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/DeviceGroup.h"
#include "native/volume/TransferFunction.h"
#include "native/common/math.h"
#include "native/volume/PrincipledVolume.h"

namespace BARNEY_NS {
  namespace native {
    
    struct Volume;

    /*! a grid of macro-cells, with each macro-cell storing both value
      range of the underlying field, and majorant after mapping
      through transfer functoin (though for efficiency reasons the
      value ranges and majorants are stored in separate arrasy). This
      class is "passive" in the sense that it only serves as a tool
      for others to create, resize, traverse, etc; it does not do any
      of that by itself (it cannot know if an accelerator wants to
      traverse it via DDA or build an optx BVH over active cells, for
      example), but provides a common interface for different fields
      to be able to compute and store macro-cells for accelerators to
      do something with. The only thing a MC grid can "acively" do is
      map its scalar ranges through a transfer functoin to compute the
      majorants. */
    struct MCGrid {
      typedef std::shared_ptr<MCGrid> SP;
    
      /*! device data for this class - grid of per-cell ranges, grid of
        majorants, and dimensionality of grid */
      struct DD {
        // not actually exported to optix programs, only used by cuda
        // kernels
        range1f *scalarRanges;
        vec3i    dims;
        vec3f    gridOrigin;
        vec3f    gridSpacing;

#if RTC_DEVICE_CODE
        inline __rtc_device
        range1f scalarRange(vec3i cellID) const
        {
          return scalarRanges[cellID.x+dims.x*(cellID.y+dims.y*cellID.z)];
        }
      
        inline __rtc_device int numCells() const
        { return dims.x*dims.y*dims.z; }
      
        inline __rtc_device vec3i cellID(int linearID) const
        {
          vec3i mcID;
          mcID.x = linearID % dims.x;
          mcID.y = (linearID / dims.x) % dims.y;
          mcID.z = linearID / (dims.x*dims.y);
          return mcID;
        }
      
        /*! returns the bounding box of the given cell */
        inline __rtc_device box3f cellBounds(vec3i cellID,
                                             const box3f &worldBounds) const
        {
          box3f bounds;
          bounds.lower = gridOrigin + vec3f(cellID)*gridSpacing;
          bounds.upper = min(bounds.lower+gridSpacing,worldBounds.upper);
          return bounds;
        }
#endif
      
        // static void addVars(std::vector<OWLVarDecl> &vars, int base);
      };

      MCGrid(const DevGroup::SP &devices);
      ~MCGrid();

      /*! get cuda-usable device-data for given device ID (relative to
        devices in the devgroup that this gris is in */
      DD getDD(Device *device);

      // void setVariables(OWLGeom geom);
    
      /*! allocate memory for the given grid */
      void resize(vec3i dims);

      /*! re-set all cells' ranges to "infinite empty" */
      void clearCells();
    
      /*! checks if this macro-cell grid has already been
        allocated/built - mostly for sanity checking nd debugging */
      inline bool built() const { return (dims != vec3i(0)); }

      struct PLD {
        /* buffer of range1f's, the min/max scalar values per cell */
        rtc::Buffer *scalarRangesBuffer = 0;
      };
      PLD *getPLD(Device *device) 
      { return &perLogical[device->localRank()]; } 
      std::vector<PLD> perLogical;

    
      vec3i     dims { 0,0,0 };
      vec3f     gridOrigin;
      vec3f     gridSpacing;
      const DevGroup::SP devices;
    };


  
    struct MajorantsGrid {
      typedef std::shared_ptr<MajorantsGrid> SP;
      struct DD : public MCGrid::DD {
        float *majorants;
      
        inline __rtc_device
        float majorant(vec3i cellID) const
        {
          return majorants[cellID.x+dims.x*(cellID.y+dims.y*cellID.z)];
        }
      };

      struct PLD {
        /* buffer of floats, the actual per-cell majorants */
        rtc::Buffer *majorantsBuffer = 0;
      };
      PLD *getPLD(Device *device) 
      { return &perLogical[device->localRank()]; } 
      std::vector<PLD> perLogical;
    
      MajorantsGrid(MCGrid::SP mcGrid);
      ~MajorantsGrid();

      /*! given the current per-cell scalar ranges, map each such cell's
        range through the transfer functoin to compute a majorant */
      void computeMajorants(Volume *volume);

      /*! allocate memory for the given grid */
      void resize(vec3i dims);
    
      /*! get cuda-usable device-data for given device ID (relative to
        devices in the devgroup that this gris is in */
      DD getDD(Device *device);

      vec3i dims { 0, 0, 0 };
      MCGrid::SP const mcGrid;
      const DevGroup::SP devices;
    };
    
  }  
}


