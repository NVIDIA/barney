// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if BARNEY_HAVE_NANOVDB
#include "barney/ModelSlot.h"
#include "barney/common/Texture.h"
#include "barney/volume/MCAccelerator.h"
// nanovdb
#include <nanovdb/NanoVDB.h>
#include "nanovdb/math/Math.h"
#include "nanovdb/math/SampleFromVoxels.h"
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Math.h>

namespace BARNEY_NS {

  struct ModelSlot;

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
    void commit() override;
    /*! @} */
    // ------------------------------------------------------------------
    
    DD getDD(Device *device);
    VolumeAccel::SP createAccel(Volume *volume) override;
    IsoSurfaceAccel::SP createIsoAccel(IsoSurface *isoSurface) override;
    MCGrid::SP buildMCs() override;

    PODData::SP data;
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
    using NVDBSamplerT = nanovdb::math::SampleFromVoxels<NVDBAccessorT, 1>;
    
    NanoVDBDataSampler(NanoVDBData *const sf)
      : sf(sf)
    {}
    
    struct DD {
#if RTC_DEVICE_CODE
      inline __rtc_device float sample(const vec3f P, bool dbg=false) const;
#endif
      NVDBGridT *nvdbGrid;
      box3i indexBounds;
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
    const auto nvdbLoc = nanovdb::Vec3d(P.x, P.y, P.z);

    // NanoVDB's worldBBox max is computed as map(indexBBox.max + 1), i.e. the
    // upper corner of the last voxel, not its center. So worldToIndexF maps
    // a world position to the voxel's lower-left corner index. We subtract 0.5
    // to shift into voxel-center index coordinates for correct trilinear
    // sampling. This matches the workaround in VisRTX for cell-centered NanoVDB.
    auto indexPos = nvdbGrid->worldToIndexF(nvdbLoc);
    indexPos[0] -= 0.5;
    indexPos[1] -= 0.5;
    indexPos[2] -= 0.5;

    // Clamp to the valid voxel range so boundary rays extrapolate the outermost
    // voxel value rather than blending with the background (inactive) value.
    indexPos[0] = nanovdb::math::Clamp(indexPos[0],
        (double)indexBounds.lower.x, (double)indexBounds.upper.x);
    indexPos[1] = nanovdb::math::Clamp(indexPos[1],
        (double)indexBounds.lower.y, (double)indexBounds.upper.y);
    indexPos[2] = nanovdb::math::Clamp(indexPos[2],
        (double)indexBounds.lower.z, (double)indexBounds.upper.z);

    auto acc = nvdbGrid->getAccessor();
    auto sampler = nanovdb::math::createSampler<1>(acc);
    float res = sampler(nanovdb::math::Vec3d(indexPos));
    return res;
  }
#endif
}

#endif

