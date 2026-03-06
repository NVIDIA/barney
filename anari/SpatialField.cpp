// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


// std
#include <cfloat>
#include <numeric>
#include <optional>
// ours
#include "Array.h"
#include "SpatialField.h"

// nanovdb
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Math.h>

// glm
#include <glm/ext/vector_float3.hpp>
#include <glm/gtx/component_wise.hpp>


namespace barney_device {

  SpatialField::SpatialField(BarneyGlobalState *s)
    : Object(ANARI_SPATIAL_FIELD, s)
  {}

  SpatialField::~SpatialField() = default;

  SpatialField *SpatialField::createInstance(std::string_view subtype,
                                             BarneyGlobalState *s)
  {
    if (subtype == "unstructured")
      return new UnstructuredField(s);
    else if (subtype == "amr")
      return new BlockStructuredField(s);
    else if (subtype == "nanovdb")
      return new NanoVDBSpatialField(s);
    else if (subtype == "amr")
      return new BlockStructuredField(s);
    else if (subtype == "structuredRegular")
      return new StructuredRegularField(s);
    else {
      // Try to create a custom Barney scalar field by type name
      // This supports custom field types registered via ScalarFieldRegistry
      auto *customField = new CustomSpatialField(s, std::string(subtype));
      if (customField->isValid()) {
        return customField;
      }
      delete customField;
      return (SpatialField *)new UnknownObject(ANARI_SPATIAL_FIELD, subtype, s);
    }
  }

  void SpatialField::markFinalized()
  {
    deviceState()->markSceneChanged();
    Object::markFinalized();
  }

  // Subtypes ///////////////////////////////////////////////////////////////////

  // StructuredRegularField //

  StructuredRegularField::StructuredRegularField(BarneyGlobalState *s)
    : SpatialField(s), m_data(this)
  {}

  void StructuredRegularField::commitParameters()
  {
    Object::commitParameters();
    m_data = getParamObject<helium::Array3D>("data");
    m_origin = getParam<helium::float3>("origin", helium::float3(0.f));
    m_spacing = getParam<helium::float3>("spacing", helium::float3(1.f));
    m_dims = m_data->size();
  }

  void StructuredRegularField::finalize()
  {
    if (!m_data) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'data' on 'structuredRegular' field");
      return;
    }

    const auto dims = m_data->size();
    m_coordUpperBound = helium::float3(std::nextafterf((float)dims.x - 1, 0),
                                       std::nextafterf((float)dims.y - 1, 0),
                                       std::nextafterf((float)dims.z - 1, 0));

    //=======================================================
    // get (or create) and populate bn field
    //=======================================================

    int slot = deviceState()->resolveSlot(getParam<int>("dataRank", -1));
    auto context = deviceState()->tether->context;

    BNScalarField sf = getBarneyScalarField();

    BNDataType barneyType;
    switch (m_data->elementType()) {
    case ANARI_FLOAT32:
      barneyType = BN_FLOAT;
      break;
    case ANARI_UFIXED8:
    case ANARI_UINT8:
      barneyType = BN_UFIXED8;
      break;
    default:
      throw std::runtime_error("scalar type not implemented ...");
    }

#if 1
    BNTextureData td = bnTextureData3DCreate(
                                             context, slot, barneyType, dims.x, dims.y, dims.z, m_data->data());
    bnSetObject(sf, "textureData", td);
    bnRelease(td);
#else
    BNTexture3D texture = bnTexture3DCreate(context,
                                            slot,
                                            barneyType,
                                            dims.x,
                                            dims.y,
                                            dims.z,
                                            m_data->data(),
                                            BN_TEXTURE_LINEAR,
                                            BN_TEXTURE_CLAMP);
    bnSetObject(sf, "texture", texture);
    bnRelease(texture);
#endif
    bnSet3i(sf, "dims", dims.x, dims.y, dims.z);
    bnSetVec(sf, "gridOrigin", m_origin);
    bnSetVec(sf, "gridSpacing", m_spacing);
    bnCommit(sf);
  }

  bool StructuredRegularField::isValid() const
  {
    return m_data;
  }

  BNScalarField StructuredRegularField::createBarneyScalarField() const
  {
    if (!isValid())
      return {};

    int slot = deviceState()->resolveSlot(getParam<int>("dataRank", -1));
    auto context = deviceState()->tether->context;

    BNScalarField sf = bnScalarFieldCreate(context, slot, "structured");
    return sf;
  }

  box3 StructuredRegularField::bounds() const
  {
    return isValid()
      ? box3(m_origin, m_origin + ((helium::float3(m_dims) - 1.f) * m_spacing))
      : box3{};
  }
 
  // NanoVDBSpatialField //
  NanoVDBSpatialField::NanoVDBSpatialField(BarneyGlobalState *s)
    : SpatialField(s), m_data(this)
  {
  }
  
  void NanoVDBSpatialField::commitParameters()
  {
    m_filter = getParamString("filter", "linear");
    m_data = getParamObject<Array1D>("data");
  }
  
  void NanoVDBSpatialField::finalize()
  {
    ANARIDataType format = m_data->elementType();
    if (format != ANARI_UINT8) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "invalid data array type encountered "
                    "in NanoVDB spatial field(%s)",
                    anari::toString(format));
      return;
    }

    // Data might not be aligned, make sure we get something that works for
    // nanovdb.
    auto hostbuffer = nanovdb::HostBuffer::create(m_data->size());
    // std::memcpy(hostbuffer.data(), m_data->data(AddressSpace::HOST), m_data->size());
    std::memcpy(hostbuffer.data(), m_data->data(), m_data->size());

    auto gridHandle = nanovdb::GridHandle<>(std::move(hostbuffer));
    std::optional<nanovdb::GridMetaData> m_gridMetadata;
    m_gridMetadata = *gridHandle.gridMetaData();

    // m_deviceBuffer.upload(
    //                       static_cast<const std::byte *>(gridHandle.data()), gridHandle.size());

    if (gridHandle.gridCount() != 1) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "VisRTX NanoVDB support's a single grid per file");
      return;
    }

    auto boundsMin = m_gridMetadata->worldBBox().min();
    auto boundsMax = m_gridMetadata->worldBBox().max();
    (box3&)m_bounds
      = box3(math::float3(boundsMin[0], boundsMin[1], boundsMin[2]),
             math::float3(boundsMax[0], boundsMax[1], boundsMax[2]));
    auto voxelSize = m_gridMetadata->voxelSize();
    (glm::vec3&)m_voxelSize
      = glm::vec3(voxelSize[0], voxelSize[1], voxelSize[2]);



    //=======================================================
    // get (or create) and populate bn field
    //=======================================================
    int slot = deviceState()->resolveSlot(getParam<int>("dataRank", -1));
    auto context = deviceState()->tether->context;
    
    BNScalarField sf = getBarneyScalarField();
    BNData bd = bnDataCreate(context,slot,
                             BN_UINT8,
                             gridHandle.size(),
                             gridHandle.data());
    assert(bd);
    assert(sf);
    bnSetData(sf,"data",bd);
    bnCommit(sf);
  }

  BNScalarField NanoVDBSpatialField::createBarneyScalarField() const
  {
    int slot = deviceState()->resolveSlot(getParam<int>("dataRank", -1));
    auto context = deviceState()->tether->context;

    BNScalarField sf = bnScalarFieldCreate(context, slot, "NanoVDB");
    assert(sf);
    return sf;
  }

  box3 NanoVDBSpatialField::bounds() const
  {
    return m_bounds;
  }
  
  bool NanoVDBSpatialField::isValid() const
  {
    return m_data && m_data->elementType() == ANARI_UINT8;
  }
  
  // UnstructuredField //

  UnstructuredField::UnstructuredField(BarneyGlobalState *s)
    : SpatialField(s), m_params(this) {}

  void UnstructuredField::commitParameters()
  {
    Object::commitParameters();

    m_params.vertexPosition = getParamObject<helium::Array1D>("vertex.position");
    m_params.vertexData = getParamObject<helium::Array1D>("vertex.data");
    m_params.cellData = getParamObject<helium::Array1D>("cell.data");
    m_params.index = getParamObject<helium::Array1D>("index");
    m_params.cellType = getParamObject<helium::Array1D>("cell.type");
    m_params.cellBegin = getParamObject<helium::Array1D>("cell.begin");
    if (!m_params.cellBegin) // some older apps use "cell.index"
      m_params.cellBegin = getParamObject<helium::Array1D>("cell.index");
  }

  void UnstructuredField::finalize()
  {
    if (!m_params.vertexPosition) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'vertex.position' "
                    "on unstructured spatial field");
      return;
    }
    if (m_params.vertexPosition->elementType() != ANARI_FLOAT32_VEC3) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "'unstructured::vertex.position' must be ANARI_FLOAT32_VEC3 (is %i) ",
                    m_params.vertexPosition->elementType());
      return;
    }

    if (!m_params.vertexData && !m_params.cellData) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'vertex.data' OR"
                    " 'cell.data' on unstructured spatial field");
      return;
    }

    if (m_params.vertexData && m_params.cellData) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "this ANARI program is setting *both* 'cell.data' "
                    "*and* 'vertex.data' "
                    "on an unstructured spatial field; this is legal "
                    "in terms of the spec, but doesn't "
                    "make much sense in terms of the data; so might "
                    "indicate something is fishy in the application"
                    );
    }

    if (!m_params.index) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'index' on unstructured spatial field");
      return;
    }

    if (!m_params.cellType) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'cell.type' on unstructured spatial field");
      return;
    }

    if (!m_params.cellBegin) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'cell.begin' on unstructured spatial field");
      return;
    }

    m_bounds.invalidate();

    auto *vertexPositions = m_params.vertexPosition->beginAs<math::float3>();
    int numVertices = (int)m_params.vertexPosition->size();
    auto *vertexData =
      m_params.vertexData ? m_params.vertexData->beginAs<float>() : nullptr;
    auto *cellData =
      m_params.cellData ? m_params.cellData->beginAs<float>() : nullptr;

    /* ANARI Spec: Sampled cell values can be specified either
       per-vertex (via vertex.data) or per-cell (via cell.data). If
       both arrays are explicitly set, vertex.data takes precedence */
    if (vertexData) cellData = nullptr;
    
    int numScalars =
      int(cellData ? m_params.cellData->size() : m_params.vertexData->size());

    for (int i = 0; i < numVertices; i++)
      m_bounds.insert(vertexPositions[i]);

    uint32_t *index32{nullptr};
    // uint64_t *index64{nullptr};
    if (m_params.index->elementType() == ANARI_UINT32)
      index32 = (uint32_t *)m_params.index->beginAs<uint32_t>();
    else if (m_params.index->elementType() == ANARI_UINT64)
      //      index64 = (uint64_t *)m_params.index->beginAs<uint64_t>();
      reportMessage(ANARI_SEVERITY_ERROR,
                    "'unstructured:index' - we only support 32-bit indices");
    else {
      reportMessage(ANARI_SEVERITY_ERROR,
                    "parameter 'index' on unstructured spatial field has wrong element type");
      return;
    }
    if (m_params.cellBegin->elementType() == ANARI_UINT32)
      index32 = (uint32_t *)m_params.index->beginAs<uint32_t>();
    else if (m_params.cellBegin->elementType() == ANARI_UINT64)
      //      index64 = (uint64_t *)m_params.index->beginAs<uint64_t>();
      reportMessage(ANARI_SEVERITY_ERROR,
                    "'unstructured:index' - we only support 32-bit indices");
    else {
      reportMessage(ANARI_SEVERITY_ERROR,
                    "parameter 'index' on unstructured spatial field has wrong element type");
      return;
    }

    //=======================================================
    // get (or create) and populate bn field
    //=======================================================

    int slot = deviceState()->resolveSlot(getParam<int>("dataRank", -1));
    auto context = deviceState()->tether->context;

    BNScalarField sf = getBarneyScalarField();

    if (!m_bnData.vertices) {
      m_bnData.vertices =
        bnDataCreate(context, slot, BN_FLOAT3, numVertices, vertexPositions);
    } else {
      bnDataSet(m_bnData.vertices, numVertices, vertexPositions);
    }

    if (!m_bnData.scalars) {
      m_bnData.scalars = bnDataCreate(
                                      context, slot, BN_FLOAT, numScalars, vertexData ? vertexData : cellData);
    } else {
      bnDataSet(m_bnData.scalars, numScalars, vertexData ? vertexData : cellData);
    }

    if (!m_bnData.indices) {
      m_bnData.indices = bnDataCreate(context,
                                      slot,
                                      BN_INT,
                                      m_params.index->size(),
                                      (const int *)m_params.index->data());
    } else {
      bnDataSet(m_bnData.indices, m_params.index->size(), (const int *)m_params.index->data());
    }

    if (!m_bnData.cellType) {
      m_bnData.cellType = bnDataCreate(context,
                                       slot,
                                       BN_UINT8,
                                       m_params.cellType->size(),
                                       (const int *)m_params.cellType->data());
    } else {
      bnDataSet(m_bnData.cellType, m_params.cellType->size(), (const int *)m_params.cellType->data());
    }

    if (!m_bnData.elementOffsets) {
      m_bnData.elementOffsets = bnDataCreate(context,
                                             slot,
                                             BN_INT,
                                             m_params.cellBegin->size(),
                                             (const int *)m_params.cellBegin->data());
    } else {
      bnDataSet(m_bnData.elementOffsets, m_params.cellBegin->size(), (const int *)m_params.cellBegin->data());
    }

    bnSetData(sf, "vertex.position", m_bnData.vertices);
    if (vertexData) {
      // this will atomatically set cell.data to 0 on barney side
      bnSetData(sf, "vertex.data", m_bnData.scalars);
    } else {
      // this will atomatically set vertex.data to 0 on barney side
      bnSetData(sf, "cell.data", m_bnData.scalars);
    }
    bnSetData(sf, "index", m_bnData.indices);
    bnSetData(sf, "cell.index", m_bnData.elementOffsets);
    bnSetData(sf, "cell.type", m_bnData.cellType);
    bnCommit(sf);
  }

  BNScalarField UnstructuredField::createBarneyScalarField() const
  {
    int slot = deviceState()->resolveSlot(getParam<int>("dataRank", -1));
    auto context = deviceState()->tether->context;

    BNScalarField sf = bnScalarFieldCreate(context, slot, "unstructured");
    return sf;
  }

  box3 UnstructuredField::bounds() const
  {
    return m_bounds;
  }

  bool UnstructuredField::isValid() const
  {
    return m_params.vertexPosition && m_params.index && m_params.cellBegin
      && m_params.cellType && (m_params.vertexData || m_params.cellData);
  }

  // BlockStructuredField //

  BlockStructuredField::BlockStructuredField(BarneyGlobalState *s)
    : SpatialField(s), m_params(this)
  {}

  void BlockStructuredField::commitParameters()
  {
    Object::commitParameters();
    m_params.refinementRatio = getParamObject<helium::Array1D>("refinementRatio");
    m_params.blockBounds = getParamObject<helium::Array1D>("block.bounds");
    m_params.blockLevel = getParamObject<helium::Array1D>("block.level");
    m_params.data = getParamObject<helium::Array1D>("data");
  }

  void BlockStructuredField::finalize()
  {
    if (!m_params.blockBounds) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'block.bounds' on amr spatial field");
      return;
    }

    if (!m_params.blockLevel) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'block.level' on amr spatial field");
      return;
    }

    if (!m_params.data) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'data' on amr spatial field");
      return;
    }

    size_t numBlocks = m_params.blockLevel->totalSize();
    auto *blockBounds = m_params.blockBounds->beginAs<box3i>();
    auto *blockLevels = m_params.blockLevel->beginAs<int>();

    m_generatedBlockOrigins.clear();
    m_generatedBlockDims.clear();
    m_generatedBlockLevels.clear();
    m_generatedBlockOffsets.clear();

    m_bounds.invalidate();

    int maxLevel = 0;
    for (size_t i = 0; i < numBlocks; ++i) {
      const box3i bounds = *(blockBounds + i);
      const int level = *(blockLevels + i);

      math::int3 dims = bounds.upper - bounds.lower + math::int3(1);

      m_generatedBlockOrigins.push_back(bounds.lower);
      m_generatedBlockDims.push_back(dims);
      m_generatedBlockLevels.push_back(level);
      m_generatedBlockOffsets.push_back(dims.x * size_t(dims.y) * dims.z);
      maxLevel = std::max(maxLevel, level);

      box3 worldBounds;
      worldBounds.lower = math::float3(float(bounds.lower.x * (1 << level)),
                                       float(bounds.lower.y * (1 << level)),
                                       float(bounds.lower.z * (1 << level)));
      worldBounds.upper = math::float3(float((bounds.upper.x + 1) * (1 << level)),
                                       float((bounds.upper.y + 1) * (1 << level)),
                                       float((bounds.upper.z + 1) * (1 << level)));
      m_bounds.insert(worldBounds);
    }

    m_generatedRefinements.resize(maxLevel+1, 2);

    std::exclusive_scan(m_generatedBlockOffsets.begin(),
                        m_generatedBlockOffsets.end(),
                        m_generatedBlockOffsets.begin(),
                        (uint64_t)0);

    //=======================================================
    // get (or create) and populate bn field
    //=======================================================

    int slot = deviceState()->resolveSlot(getParam<int>("dataRank", -1));
    auto context = deviceState()->tether->context;

    BNScalarField sf = getBarneyScalarField();

    size_t numScalars = m_params.data->size();
    size_t numLevels = m_generatedRefinements.size();

    if (!m_bnData.scalars) {
      m_bnData.scalars =
        bnDataCreate(context, slot, BN_FLOAT32, numScalars, m_params.data->beginAs<float>());
    } else {
      bnDataSet(m_bnData.scalars, numScalars, m_params.data->beginAs<float>());
    }


    if (!m_bnData.blockOrigins) {
      m_bnData.blockOrigins =
        bnDataCreate(context, slot, BN_INT32_VEC3, numBlocks, m_generatedBlockOrigins.data());
    } else {
      bnDataSet(m_bnData.blockOrigins, numBlocks, m_generatedBlockOrigins.data());
    }


    if (!m_bnData.blockDims) {
      m_bnData.blockDims =
        bnDataCreate(context, slot, BN_INT32_VEC3, numBlocks, m_generatedBlockDims.data());
    } else {
      bnDataSet(m_bnData.blockDims, numBlocks, m_generatedBlockDims.data());
    }


    if (!m_bnData.blockLevels) {
      m_bnData.blockLevels =
        bnDataCreate(context, slot, BN_INT32, numBlocks, m_generatedBlockLevels.data());
    } else {
      bnDataSet(m_bnData.blockLevels, numBlocks, m_generatedBlockLevels.data());
    }


    if (!m_bnData.blockOffsets) {
      m_bnData.blockOffsets =
        bnDataCreate(context, slot, BN_UINT64, numBlocks, m_generatedBlockOffsets.data());
    } else {
      bnDataSet(m_bnData.blockOffsets, numBlocks, m_generatedBlockOffsets.data());
    }

    if (!m_bnData.levelRefinements) {
      m_bnData.levelRefinements =
        bnDataCreate(context, slot, BN_INT32, numLevels, m_generatedRefinements.data());
    } else {
      bnDataSet(m_bnData.levelRefinements, numLevels, m_generatedRefinements.data());
    }

    bnSetData(sf, "scalars", m_bnData.scalars);
    bnSetData(sf, "grid.origins", m_bnData.blockOrigins);
    bnSetData(sf, "grid.dims", m_bnData.blockDims);
    bnSetData(sf, "grid.levels", m_bnData.blockLevels);
    bnSetData(sf, "grid.offsets", m_bnData.blockOffsets);
    bnSetData(sf, "level.refinements", m_bnData.levelRefinements);
    bnCommit(sf);
  }

  BNScalarField BlockStructuredField::createBarneyScalarField() const
  {
    std::cout
      << "=================================================================="
      << std::endl;
    std::cout << "BANARI: CREATING AMR DATA" << std::endl;
    std::cout
      << "=================================================================="
      << std::endl;

    int slot = deviceState()->resolveSlot(getParam<int>("dataRank", -1));
    auto context = deviceState()->tether->context;

    BNScalarField sf = bnScalarFieldCreate(context, slot, "BlockStructuredAMR");
    return sf;
  }

  box3 BlockStructuredField::bounds() const
  {
    return m_bounds;
  }

  // CustomSpatialField //

  CustomSpatialField::CustomSpatialField(BarneyGlobalState *s, const std::string &type)
    : SpatialField(s), m_fieldType(type)
  {
  }

  void CustomSpatialField::commitParameters()
  {
    Object::commitParameters();
  }

  void CustomSpatialField::finalize()
  {
    // Set bounds to a reasonable default sphere
    const float radius = 1.0f;
    m_bounds = box3(math::float3(-radius), math::float3(radius));
    
    // Ensure the Barney field is created before applying parameters
    if (!m_bnField) {
      m_bnField = createBarneyScalarField();
      if (!m_bnField) {
        return;
      }
    }
    
    // Apply parameters to the Barney field
    applyParametersToField();
  }

  void CustomSpatialField::markFinalized()
  {
    // Call base class to mark scene as changed
    SpatialField::markFinalized();
  }

  void CustomSpatialField::applyParametersToField()
  {
    if (!m_bnField)
    {
        return;
    }
    
    int slot = deviceState()->resolveSlot(getParam<int>("dataRank", -1));
    auto context = deviceState()->tether->context;
    
    // Iterate over all parameters and forward them to Barney
    for (auto it = params_begin(); it != params_end(); ++it) {
      const std::string& paramName = it->first;
      ANARIDataType paramType = it->second.type();
      
      // Handle array parameters
      if (paramType == ANARI_ARRAY3D) {
        auto array = getParamObject<helium::Array3D>(paramName);
        if (!array || !array->data()) {
          continue;
        }
        
        auto dims = array->size();
        auto elemType = array->elementType();
        
        BNDataType barneyType;
        switch (elemType) {
          case ANARI_FLOAT32: barneyType = BN_FLOAT; break;
          case ANARI_INT32: barneyType = BN_INT; break;
          case ANARI_UINT8:
          case ANARI_UFIXED8: barneyType = BN_UFIXED8; break;
          default:
            continue;
        }
        
        BNTextureData td = bnTextureData3DCreate(context, slot, barneyType,
                                                  dims.x, dims.y, dims.z,
                                                  array->data());
        bnSetObject(m_bnField, paramName.c_str(), td);
        bnRelease(td);
      }
      else if (paramType == ANARI_ARRAY2D) {
        auto array = getParamObject<helium::Array2D>(paramName);
        if (!array || !array->data()) {
          continue;
        }
        
        auto dims = array->size();
        auto elemType = array->elementType();
        
        BNDataType barneyType;
        switch (elemType) {
          case ANARI_FLOAT32: barneyType = BN_FLOAT; break;
          case ANARI_INT32: barneyType = BN_INT; break;
          case ANARI_UINT8:
          case ANARI_UFIXED8: barneyType = BN_UFIXED8; break;
          default:
            continue;
        }
        
        BNTextureData td = bnTextureData2DCreate(context, slot, barneyType,
                                                  dims.x, dims.y,
                                                  array->data());
        bnSetObject(m_bnField, paramName.c_str(), td);
        bnRelease(td);
      }
      else if (paramType == ANARI_ARRAY1D) {
        auto array = getParamObject<helium::Array1D>(paramName);
        if (!array || !array->data()) {
          continue;
        }
        
        auto elemType = array->elementType();
        size_t numElements = array->totalSize();
        
        BNDataType barneyType;
        switch (elemType) {
          case ANARI_FLOAT32: barneyType = BN_FLOAT; break;
          case ANARI_INT32: barneyType = BN_INT; break;
          case ANARI_UINT8:
          case ANARI_UFIXED8: barneyType = BN_UFIXED8; break;
          default:
            continue;
        }
        
        BNData data = bnDataCreate(context, slot, barneyType, numElements, array->data());
        bnSetObject(m_bnField, paramName.c_str(), data);
        bnRelease(data);
      }
      // Handle scalar parameters
      else {
        switch (paramType) {
          case ANARI_FLOAT32:
            bnSet1f(m_bnField, paramName.c_str(), getParam<float>(paramName, 0.0f));
            break;
          case ANARI_FLOAT32_VEC2: {
            auto val = getParam<math::float2>(paramName, math::float2(0.0f));
            bnSet2f(m_bnField, paramName.c_str(), val.x, val.y);
            break;
          }
          case ANARI_FLOAT32_VEC3: {
            auto val = getParam<math::float3>(paramName, math::float3(0.0f));
            bnSet3f(m_bnField, paramName.c_str(), val.x, val.y, val.z);
            break;
          }
          case ANARI_FLOAT32_VEC4: {
            auto val = getParam<math::float4>(paramName, math::float4(0.0f));
            bnSet4f(m_bnField, paramName.c_str(), val.x, val.y, val.z, val.w);
            break;
          }
          case ANARI_INT32:
            bnSet1i(m_bnField, paramName.c_str(), getParam<int>(paramName, 0));
            break;
          case ANARI_UINT32:
            bnSet1i(m_bnField, paramName.c_str(), (int)getParam<uint32_t>(paramName, 0));
            break;
          case ANARI_BOOL:
            bnSet1i(m_bnField, paramName.c_str(), getParam<bool>(paramName, false) ? 1 : 0);
            break;
          default:
            break;
        }
      }
    }
    
    // Commit the field with all parameters
    bnCommit(m_bnField);
  }

  BNScalarField CustomSpatialField::createBarneyScalarField() const
  {
    int slot = deviceState()->resolveSlot(getParam<int>("dataRank", -1));
    auto context = deviceState()->tether->context;
    
    // Create a Barney scalar field using the registered type name
    BNScalarField sf = bnScalarFieldCreate(context, slot, m_fieldType.c_str());
    
    if (!sf) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "Failed to create Barney scalar field of type '%s' - field type not registered?",
                    m_fieldType.c_str());
    } else {
      reportMessage(ANARI_SEVERITY_INFO,
                    "Successfully created Barney scalar field of type '%s'",
                    m_fieldType.c_str());
    }
    
    return sf;
  }

  box3 CustomSpatialField::bounds() const
  {
    return m_bounds;
  }

  bool CustomSpatialField::isValid() const
  {
    return !m_fieldType.empty();
  }

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::SpatialField *);
