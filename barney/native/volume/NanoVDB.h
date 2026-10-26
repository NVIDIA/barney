// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney_rtc.h"
#if BARNEY_HAVE_NANOVDB
#include "native/ModelSlot.h"
#include "native/common/Texture.h"
#include "native/volume/MCAccelerator.h"
// nanovdb
#include <nanovdb/NanoVDB.h>
#include "nanovdb/math/Math.h"
#if !defined(BARNEY_DEVICE_PROGRAM)
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#endif

#include "native/volume/NanoVDBTypes.h"
#include "native/volume/NanoVDBMath.h"

namespace BARNEY_NS {
  namespace native {

    struct NanoVDBData : public ScalarField
    {
      /*! device data for this class. nothis special for a structured
        data object; all sampling related stuff will be in the
        StructuredDataSampler */
      struct DD : public ScalarField::DD {
        vec3f voxelSize;
        vec3i gridSize;
        box3i indexBounds;
        nanovdb::GridType gridType;
        const void *gridData;
      };

      /*! construct a new NanoDVB scalar field; will not do anything -
        or have any data - untile 'set' and 'commit'ed
      */
      NanoVDBData(Context *context,
                  const DevGroup::SP &devices);
      virtual ~NanoVDBData() = default;

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      bool setData(const std::string &member, const Data::SP &value) override;
      bool set1i(const std::string &member, const int &value) override;
      void commit() override;
      /*! @} */
      // ------------------------------------------------------------------
    
      DD getDD(Device *device);
      VolumeAccel::SP createAccel(Volume *volume) override;
      IsoSurfaceAccel::SP createIsoAccel(IsoSurface *isoSurface) override;
      MCGrid::SP buildMCs() override;

      PODData::SP data;
      BNTextureFilterMode filterMode { BN_TEXTURE_LINEAR };
      box3i indexBounds;
      vec3f voxelSize;
      vec3i gridSize;
      nanovdb::GridType gridType;
    };

    /*! sampler object for a StructuredData object, using a 3d texture */
    template<typename T>
    struct NanoVDBDataSampler : public ScalarFieldSampler {
      using NVDBGridT = typename nanovdb::Grid<nanovdb::NanoTree<T>>;
      using NVDBAccessorT = typename NVDBGridT::AccessorType;
    
      NanoVDBDataSampler(NanoVDBData *const sf)
        : sf(sf)
      {}
    
      struct DD {
#if RTC_DEVICE_CODE
        inline __rtc_device float sample(const vec3f P, bool dbg=false) const;
#endif
        NVDBGridT *nvdbGrid;
        box3i indexBounds;
        BNTextureFilterMode filterMode { BN_TEXTURE_LINEAR };
      };

      void build() override;
    
      DD getDD(Device *device);
      NanoVDBData *const sf;
    };
  
#if RTC_DEVICE_CODE
    template<typename T>
    inline __rtc_device
    float NanoVDBDataSampler<T>::DD::sample(const vec3f P, bool dbg) const
    {
      const auto nvdbLoc = nanovdb::Vec3f(P.x, P.y, P.z);

      // NanoVDB's worldBBox max is computed as map(indexBBox.max + 1), i.e. the
      // upper corner of the last voxel, not its center. So worldToIndexF maps
      // a world position to the voxel's lower-left corner index. We subtract 0.5
      // to shift into voxel-center index coordinates for correct trilinear
      // sampling. This matches the workaround in VisRTX for cell-centered NanoVDB.
      const auto mapped = nvdbGrid->worldToIndexF(nvdbLoc);
      nanovdb::Vec3d indexPos(mapped[0], mapped[1], mapped[2]);
      for (int axis = 0; axis < 3; ++axis) {
        if (!barney_nanovdb::isFinite(indexPos[axis])) {
          // Finite float maps/positions can overflow in the subtraction or
          // dot product (including 0*inf). Widen the same stored coefficients,
          // not a different inverse, before any arithmetic or integer cast.
          const auto &map = nvdbGrid->map();
          const double x = double(P.x) - map.mVecF[0];
          const double y = double(P.y) - map.mVecF[1];
          const double z = double(P.z) - map.mVecF[2];
          const auto *m = map.mInvMatF + 3*axis;
          indexPos[axis] = double(m[0])*x + (double(m[1])*y + double(m[2])*z) - 0.5;
        } else {
          indexPos[axis] = mapped[axis] - 0.5f;
        }
      }

      // Clamp to the valid voxel range so boundary rays extrapolate the outermost
      // voxel value rather than blending with the background (inactive) value.
      vec3i base;
      vec3f fraction;
      for (int axis = 0; axis < 3; ++axis) {
        // Compare before conversion: float(INT_MAX) is outside the int range,
        // and float-rounded bounds can otherwise reach voxels outside the box.
        if (double(indexPos[axis]) <= indexBounds.lower[axis]) {
          base[axis] = indexBounds.lower[axis];
          fraction[axis] = 0.f;
        } else if (double(indexPos[axis]) >= indexBounds.upper[axis]) {
          base[axis] = indexBounds.upper[axis];
          fraction[axis] = 0.f;
        } else {
          base[axis] = nanovdb::math::Floor(indexPos[axis]);
          fraction[axis] = float(double(indexPos[axis]) - base[axis]);
        }
      }
      const int x0 = base.x, y0 = base.y, z0 = base.z;
      const float fx = fraction.x, fy = fraction.y, fz = fraction.z;
      const int x1 = x0 < indexBounds.upper.x ? x0 + 1 : x0;
      const int y1 = y0 < indexBounds.upper.y ? y0 + 1 : y0;
      const int z1 = z0 < indexBounds.upper.z ? z0 + 1 : z0;

      auto acc = nvdbGrid->getAccessor();
      if (filterMode == BN_TEXTURE_NEAREST)
        return acc.getValue(nanovdb::Coord(fx < 0.5f ? x0 : x1,
                                          fy < 0.5f ? y0 : y1,
                                          fz < 0.5f ? z0 : z1));
      const float v000 = acc.getValue(nanovdb::Coord(x0, y0, z0));
      const float v100 = acc.getValue(nanovdb::Coord(x1, y0, z0));
      const float v010 = acc.getValue(nanovdb::Coord(x0, y1, z0));
      const float v110 = acc.getValue(nanovdb::Coord(x1, y1, z0));
      const float v001 = acc.getValue(nanovdb::Coord(x0, y0, z1));
      const float v101 = acc.getValue(nanovdb::Coord(x1, y0, z1));
      const float v011 = acc.getValue(nanovdb::Coord(x0, y1, z1));
      const float v111 = acc.getValue(nanovdb::Coord(x1, y1, z1));

      const float v00 = v000 + fx * (v100 - v000);
      const float v10 = v010 + fx * (v110 - v010);
      const float v01 = v001 + fx * (v101 - v001);
      const float v11 = v011 + fx * (v111 - v011);
      const float v0 = v00 + fy * (v10 - v00);
      const float v1 = v01 + fy * (v11 - v01);
      return v0 + fz * (v1 - v0);
    }
#endif
  }
}

#endif

