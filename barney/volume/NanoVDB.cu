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

  RTC_IMPORT_USER_GEOM(/*file*/NanoVDB,/*name*/NanoVDBMC_float,
                       /*geomtype device data */
                       MCVolumeAccel<NanoVDBDataSampler<float>>::DD,false,false);
  RTC_IMPORT_USER_GEOM(/*file*/NanoVDB,/*name*/NanoVDBMC_Iso_float,
                       /*geomtype device data */
                       MCIsoSurfaceAccel<NanoVDBDataSampler<float>>::DD,false,false);

  /*! compute kernel that computes macro-cell information for a 3D
    structured data grid */
  __rtc_global
  void NanoVDBData_float_computeMCs(const rtc::ComputeInterface &ci,
                                    /* kernel ARGS */
                                    MCGrid::DD mcGrid,
                                    NanoVDBData::DD dd)
  {
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

    using GridType = typename nanovdb::Grid<nanovdb::NanoTree<float>>;
    GridType *gridPtr = (GridType *)dd.gridData;
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
      __rtc_launch(device->rtc,
                   NanoVDBData_float_computeMCs,
                   nb,bs,
                   mcGrid->getDD(device),
                   getDD(device));
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
    auto sampler = std::make_shared<NanoVDBDataSampler<float>>(this);
    return std::make_shared<MCVolumeAccel<NanoVDBDataSampler<float>>>
      (volume,
       createGeomType_NanoVDBMC_float,
       sampler);
  }
  
  IsoSurfaceAccel::SP NanoVDBData::createIsoAccel(IsoSurface *isoSurface) 
  {
    auto sampler = std::make_shared<NanoVDBDataSampler<float>>(this);
    return std::make_shared<MCIsoSurfaceAccel<NanoVDBDataSampler<float>>>
      (isoSurface,
       createGeomType_NanoVDBMC_Iso_float,
       sampler);
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
    if (gridType != nanovdb::GridType::Float)
      throw std::runtime_error
        ("barney::NanoVDBData currently implemented only for float grids");
  }
  
}

#endif
