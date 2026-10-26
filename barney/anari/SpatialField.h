// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include "helium/array/Array1D.h"
#include "helium/array/Array3D.h"
#include "helium/array/ObjectArray.h"
// ours
#include "Object.h"

namespace BARNEY_NS {
  namespace anari {
    
    struct SpatialField : public Object
    {
      SpatialField(BarneyGlobalState *s);
      ~SpatialField() override;

      static SpatialField *createInstance(std::string_view subtype,
                                          BarneyGlobalState *s);

      void markFinalized() override;

      virtual BNScalarField createBarneyScalarField() const = 0;

      void cleanup()
      {
        if (m_bnField) {
          bnRelease(m_bnField);
          m_bnField = nullptr;
        }
      }

      BNScalarField getBarneyScalarField()
      {
        if (!isValid())
          return {};
        if (!m_bnField)
          m_bnField = createBarneyScalarField();
        return m_bnField;
      }

      virtual box3 bounds() const = 0;

      BNScalarField m_bnField = 0;
    };

    // Subtypes ///////////////////////////////////////////////////////////////////

    struct UnstructuredField : public SpatialField
    {
      UnstructuredField(BarneyGlobalState *s);
      ~UnstructuredField() override;

      void commitParameters() override;
      void finalize() override;

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;
      bool isValid() const override;

    private:
      struct Parameters
      {
        Parameters(helium::BaseObject *observer)
          : vertexPosition(observer),
            vertexData(observer),
            cellData(observer),
            index(observer),
            cellType(observer),
            cellBegin(observer)
        {}
        helium::ChangeObserverPtr<helium::Array1D> vertexPosition;
        helium::ChangeObserverPtr<helium::Array1D> vertexData;
        helium::ChangeObserverPtr<helium::Array1D> cellData;
        helium::ChangeObserverPtr<helium::Array1D> index;
        helium::ChangeObserverPtr<helium::Array1D> cellType;
        helium::ChangeObserverPtr<helium::Array1D> cellBegin;
      } m_params;

      struct BarneyData
      {
        BNData vertices{nullptr};
        BNData scalars{nullptr};
        BNData indices{nullptr};
        BNData cellType{nullptr};
        BNData elementOffsets{nullptr};
      } m_bnData;

      box3 m_bounds;
    };

    struct BlockStructuredField : public SpatialField
    {
      BlockStructuredField(BarneyGlobalState *s);
      ~BlockStructuredField() override;
      void commitParameters() override;
      void finalize() override;

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;

      struct Parameters
      {
        Parameters(helium::BaseObject *observer)
          : blockDims(observer),
            blockOrigins(observer),
            blockLevel(observer),
            data(observer)
        {}
        helium::ChangeObserverPtr<helium::Array1D> blockDims;
        helium::ChangeObserverPtr<helium::Array1D> blockOrigins;
        helium::ChangeObserverPtr<helium::Array1D> blockLevel;
        helium::ChangeObserverPtr<helium::Array1D> data;
        helium::float3 origin {0.f};
        helium::float3 spacing {1.f};
      } m_params;

      struct BarneyData
      {
        BNData scalars{nullptr};
        BNData blockOrigins{nullptr};
        BNData blockDims{nullptr};
        BNData blockLevels{nullptr};
        BNData blockOffsets{nullptr};
      } m_bnData;

      std::vector<uint64_t> m_generatedBlockOffsets;

      box3 m_bounds;
    };

    struct StructuredRegularField : public SpatialField
    {
      StructuredRegularField(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;
      bool isValid() const override;

      math::uint3 m_dims{0u};
      math::float3 m_origin;
      math::float3 m_spacing;
      math::float3 m_coordUpperBound;

      helium::ChangeObserverPtr<helium::Array3D> m_data;
    };

    struct NanoVDBSpatialField : public SpatialField
    {
      NanoVDBSpatialField(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;
      bool isValid() const override;

      std::string m_filter;
      helium::ChangeObserverPtr<helium::Array1D> m_data;

      box3 m_bounds;
      math::float3 m_voxelSize;
    };

    // Generic wrapper for custom Barney scalar field types
    // This allows ANARI to use fields registered via ScalarFieldRegistry
    struct CustomSpatialField : public SpatialField
    {
      CustomSpatialField(BarneyGlobalState *s, const std::string &type);
      void commitParameters() override;
      void finalize() override;
      void markFinalized() override; // Apply parameters after field is created

      BNScalarField createBarneyScalarField() const override;

      box3 bounds() const override;
      bool isValid() const override;

      void applyParametersToField(); // Apply collected parameters to the Barney field

      std::string m_fieldType;
      box3 m_bounds;
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::SpatialField *,
                                    ANARI_SPATIAL_FIELD);
