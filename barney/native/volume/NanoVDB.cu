// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/volume/NanoVDB.h" // Keep host and device NanoVDB code in sync.

#if BARNEY_HAVE_NANOVDB

#include "native/Context.h"
#include "native/common/Texture.h"
#include "native/volume/MCGrid.cuh"
#include "native/volume/NanoVDBHelpers.h"
#include <nanovdb/HostBuffer.h>
#include <nanovdb/GridHandle.h>

namespace BARNEY_NS {
  namespace native {

#define NANOVDB_IMPORT_GEOM(BuildType, Suffix, EnumName)                \
    RTC_IMPORT_USER_GEOM(NanoVDB, NanoVDBMC_##Suffix,                   \
                         MCVolumeAccel<NanoVDBDataSampler<BuildType>>::DD,false,false);

    BARNEY_NANOVDB_FLOAT_TYPES(NANOVDB_IMPORT_GEOM)
#undef NANOVDB_IMPORT_GEOM

    template<typename BuildType>
    __rtc_global
    void NanoVDBData_computeMCs(const rtc::ComputeInterface &ci,
                                MCGrid::DD mcGrid,
                                NanoVDBData::DD dd)
    {
#if RTC_DEVICE_CODE
      using GridT = typename nanovdb::Grid<nanovdb::NanoTree<BuildType>>;

      vec3i mcDims = mcGrid.dims;
      int tid = ci.launchIndex().x;
      if (tid >= mcDims.x*mcDims.y*mcDims.z) return;
      vec3i mcID(tid  %  mcDims.x,
                 (tid /  mcDims.x) % mcDims.y,
                 tid  / (mcDims.x  * mcDims.y));

      if (mcID.x >= mcGrid.dims.x) return;
      if (mcID.y >= mcGrid.dims.y) return;
      if (mcID.z >= mcGrid.dims.z) return;

      nanovdb::Vec3d lo, hi;
      for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = double(mcGrid.gridOrigin[axis])
          + double(mcID[axis]) * mcGrid.gridSpacing[axis];
        hi[axis] = double(mcGrid.gridOrigin[axis])
          + double(mcID[axis] + 1) * mcGrid.gridSpacing[axis];
      }
      const auto range = barney_nanovdb::scalarRange(
        *(const GridT *)dd.gridData, nanovdb::Vec3dBBox(lo, hi));
      mcGrid.scalarRanges[tid] = {range.lower, range.upper};
#endif
    }
  
    NanoVDBData::NanoVDBData(Context *context,
                             const DevGroup::SP &devices)
      : ScalarField(context,devices)
    {}

    template<typename T>
    void NanoVDBDataSampler<T>::build()
    {
      // nothing to do here, we have no other data structure in addition
      // to nanovdb
    }

    MCGrid::SP NanoVDBData::buildMCs() 
    {
#if BARNEY_HAVE_NANOVDB
      MCGrid::SP mcGrid = std::make_shared<MCGrid>(devices);
      const auto dims = barney_nanovdb::macrocellDims(nanovdb::CoordBBox(
        nanovdb::Coord(indexBounds.lower.x, indexBounds.lower.y, indexBounds.lower.z),
        nanovdb::Coord(indexBounds.upper.x, indexBounds.upper.y, indexBounds.upper.z)));
      const vec3i mcDims(dims[0], dims[1], dims[2]);
      const size_t count = size_t(mcDims.x) * mcDims.y * mcDims.z;
      mcGrid->gridOrigin = worldBounds.lower;
      mcGrid->gridSpacing = worldBounds.size() * rcp(vec3f(mcDims));
      for (int axis = 0; axis < 3; ++axis) {
        const float spacing = mcGrid->gridSpacing[axis];
        if (!std::isfinite(worldBounds.lower[axis])
            || !std::isfinite(worldBounds.upper[axis])
            || !(spacing > 0.f) || !std::isfinite(spacing)
            || !std::isfinite(1.f / spacing))
          throw std::runtime_error("Unrepresentable NanoVDB macrocell domain");
      }
      mcGrid->resize(mcDims);

      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        __rtc_launch(device->rtc, (NanoVDBData_computeMCs<float>),
                     int((count + 127) / 128), 128,
                     mcGrid->getDD(device), getDD(device));
      }
      for (auto device : *devices)
        device->sync();
      return mcGrid;
#else
      throw std::runtime_error("NanoVDB support not compiled in");
#endif
    }

    template<typename T>
    typename NanoVDBDataSampler<T>::DD NanoVDBDataSampler<T>::getDD(Device *device)
    {
      DD dd;

      assert(sf);
      assert(sf->data);
      dd.nvdbGrid    = (NVDBGridT*)sf->data->getDD(device);
      dd.indexBounds = sf->indexBounds;
      dd.filterMode = sf->filterMode;

      assert(dd.nvdbGrid);
      return dd;
    }

    NanoVDBData::DD NanoVDBData::getDD(Device *device)
    {
      assert(this->data);
    
      DD dd;
      dd.worldBounds = worldBounds;
      dd.voxelSize   = voxelSize;
      dd.gridType    = gridType;
      dd.gridSize    = gridSize;
      dd.gridData    = data->getDD(device);
      dd.indexBounds = indexBounds;
    
      assert(dd.gridData);
      return dd;
    }
  
    VolumeAccel::SP NanoVDBData::createAccel(Volume *volume)
    {
      switch (gridType) {
#define CREATE_ACCEL(BuildType, Suffix, EnumName)                       \
        case nanovdb::GridType::EnumName: {                             \
          auto sampler = std::make_shared<NanoVDBDataSampler<BuildType>>(this); \
          return std::make_shared<MCVolumeAccel<NanoVDBDataSampler<BuildType>>> \
            (volume, createGeomType_NanoVDBMC_##Suffix, sampler);       \
        }
        BARNEY_NANOVDB_FLOAT_TYPES(CREATE_ACCEL)
#undef CREATE_ACCEL
      default:
          throw std::runtime_error
          ("barney::NanoVDBData::createAccel: unsupported grid type");
      }
    }
  
    IsoSurfaceAccel::SP NanoVDBData::createIsoAccel(IsoSurface *isoSurface)
    {
      throw std::runtime_error("NanoVDB iso-surfaces are not supported");
    }
  
    // ==================================================================
    bool NanoVDBData::setData(const std::string &member,
                              const Data::SP &value) 
    {
      if (member == "data") {
        data = value->as<PODData>();
        assert(data);
        return true;
      }
      return false;
    }

    bool NanoVDBData::set1i(const std::string &member, const int &value)
    {
      if (member == "filterMode") {
        filterMode = value == BN_TEXTURE_NEAREST
          ? BN_TEXTURE_NEAREST : BN_TEXTURE_LINEAR;
        return true;
      }
      return false;
    }

    // ==================================================================
    void NanoVDBData::commit() 
    {
      // Data might not be aligned, make sure we get something that works for
      // nanovdb.
      size_t numBytes = data->size();
    
      auto dev0 = (*devices)[0];
      auto nvHostBuffer = nanovdb::HostBuffer::create(numBytes);
      data->download(dev0,nvHostBuffer.data());
    
      auto gridHandle = nanovdb::GridHandle<>(std::move(nvHostBuffer));
      const auto *floatGrid = gridHandle.grid<float>();
      if (!floatGrid)
        throw std::runtime_error("barney::NanoVDBData: expected Float grid");
      barney_nanovdb::validateMap(floatGrid->map());
      // GridMetaData pointer casts require the root immediately after the tree.
      const nanovdb::GridMetaData metadata(*floatGrid);
      const auto *gridMetadata = &metadata;
      auto boundsMin = gridMetadata->worldBBox().min();
      auto boundsMax = gridMetadata->worldBBox().max();
    
      worldBounds
        = box3f(vec3f(boundsMin[0], boundsMin[1], boundsMin[2]),
                vec3f(boundsMax[0], boundsMax[1], boundsMax[2]));
      (nanovdb::CoordBBox&)indexBounds = gridMetadata->indexBBox();
      nanovdb::Vec3d nvVoxelSize = gridMetadata->voxelSize();
      voxelSize = vec3f((const vec3d&)nvVoxelSize);
    
      auto nvGridSize = gridMetadata->indexBBox().dim();
      gridSize.x = nvGridSize[0];
      gridSize.y = nvGridSize[1];
      gridSize.z = nvGridSize[2];
    
      gridType = gridMetadata->gridType();
      bool supported = false;
#define CHECK_TYPE(BuildType, Suffix, EnumName)                         \
      supported = supported || (gridType == nanovdb::GridType::EnumName);
      BARNEY_NANOVDB_FLOAT_TYPES(CHECK_TYPE)
#undef CHECK_TYPE
        if (!supported)
          throw std::runtime_error
            ("barney::NanoVDBData: unsupported grid type");
      // Construction failure must reach the adapter's commit exception handler,
      // not surface later while building the world with a supposedly valid field.
      mcGrid = buildMCs();
    }

  }
}

#endif
