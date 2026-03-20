// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
// & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// include barney.h first, so we know whether BARNEY_HAVE_NANOVDB is set 
#include "barney/barney.h"

#if BARNEY_HAVE_NANOVDB

#include "barney/Context.h"
#include "barney/volume/NanoVDB.h"
#include "barney/common/Texture.h"
#include "rtcore/ComputeInterface.h"
#include "barney/volume/MCGrid.cuh"

namespace BARNEY_NS {

#define NANOVDB_IMPORT_GEOM(BuildType, Suffix, EnumName) \
  RTC_IMPORT_USER_GEOM(NanoVDB, NanoVDBMC_##Suffix, \
    MCVolumeAccel<NanoVDBDataSampler<BuildType>>::DD,false,false); \
  RTC_IMPORT_USER_GEOM(NanoVDB, NanoVDBMC_Iso_##Suffix, \
    MCIsoSurfaceAccel<NanoVDBDataSampler<BuildType>>::DD,false,false);

  BARNEY_NANOVDB_FLOAT_TYPES(NANOVDB_IMPORT_GEOM)
#undef NANOVDB_IMPORT_GEOM

  template<typename BuildType>
  __rtc_global
  void NanoVDBData_computeMCs(const rtc::ComputeInterface &ci,
                              MCGrid::DD mcGrid,
                              NanoVDBData::DD dd)
  {
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

    vec3i lo = dd.indexBounds.lower+(vec3i(mcID)*dd.gridSize) / mcGrid.dims;
    vec3i hi = dd.indexBounds.lower+(vec3i(mcID+1)*dd.gridSize) / mcGrid.dims;
    nanovdb::CoordBBox bbox({lo.x,lo.y,lo.z},
                            {hi.x+1,hi.y+1,hi.z+1});

    GridT *gridPtr = (GridT *)dd.gridData;
    auto acc = gridPtr->getAccessor();

    range1f scalarRange;
    for (nanovdb::CoordBBox::Iterator iter = bbox.begin(); iter; ++iter) {
      float value = acc.getValue(*iter);
      scalarRange.extend(value);
    }
    int cellIdx = mcID.x + mcGrid.dims.x * (mcID.y + mcGrid.dims.y * (mcID.z));
    mcGrid.scalarRanges[cellIdx] = scalarRange;
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
#if 1
    vec3i mcDims = divRoundUp(gridSize,vec3i(16));
#else
    vec3i mcDims = vec3i(64);
#endif
    mcGrid->resize(mcDims);
    mcGrid->gridOrigin = worldBounds.lower;
    mcGrid->gridSpacing = worldBounds.size() * rcp(vec3f(mcDims));
    
    mcGrid->clearCells();
    for (auto device : *devices) {
      size_t lc64 = (size_t)mcDims.x*(size_t)mcDims.y*(size_t)mcDims.z;
      int lc = int(lc64);
      assert(lc == lc64);
      int bs = 128;
      int nb = divRoundUp(lc,bs);

      auto mcDD = mcGrid->getDD(device);
      auto dd = getDD(device);

      switch (gridType) {
#define LAUNCH_MC(BuildType, Suffix, EnumName) \
      case nanovdb::GridType::EnumName: \
        __rtc_launch(device->rtc, \
                     (NanoVDBData_computeMCs<BuildType>), \
                     nb,bs, mcDD, dd); \
        break;
      BARNEY_NANOVDB_FLOAT_TYPES(LAUNCH_MC)
#undef LAUNCH_MC
      default:
        throw std::runtime_error
          ("barney::NanoVDBData: unsupported grid type for macro-cell build");
      }
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
#define CREATE_ACCEL(BuildType, Suffix, EnumName) \
    case nanovdb::GridType::EnumName: { \
      auto sampler = std::make_shared<NanoVDBDataSampler<BuildType>>(this); \
      return std::make_shared<MCVolumeAccel<NanoVDBDataSampler<BuildType>>> \
        (volume, createGeomType_NanoVDBMC_##Suffix, sampler); \
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
    switch (gridType) {
#define CREATE_ISO_ACCEL(BuildType, Suffix, EnumName) \
    case nanovdb::GridType::EnumName: { \
      auto sampler = std::make_shared<NanoVDBDataSampler<BuildType>>(this); \
      return std::make_shared<MCIsoSurfaceAccel<NanoVDBDataSampler<BuildType>>> \
        (isoSurface, createGeomType_NanoVDBMC_Iso_##Suffix, sampler); \
    }
    BARNEY_NANOVDB_FLOAT_TYPES(CREATE_ISO_ACCEL)
#undef CREATE_ISO_ACCEL
    default:
      throw std::runtime_error
        ("barney::NanoVDBData::createIsoAccel: unsupported grid type");
    }
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
    const nanovdb::GridMetaData *gridMetadata = gridHandle.gridMetaData();
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
#define CHECK_TYPE(BuildType, Suffix, EnumName) \
    supported = supported || (gridType == nanovdb::GridType::EnumName);
    BARNEY_NANOVDB_FLOAT_TYPES(CHECK_TYPE)
#undef CHECK_TYPE
    if (!supported)
      throw std::runtime_error
        ("barney::NanoVDBData: unsupported grid type");
  }
  
}

#endif
