// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Object.h"
#include "SpatialField.h"

#include <vector>

namespace BARNEY_NS {
  namespace anari {
    
    struct Volume : public Object
    {
      Volume(BarneyGlobalState *s);
      ~Volume() override;

      static Volume *createInstance(std::string_view subtype, BarneyGlobalState *s);

      void markFinalized() override;

      BNVolume getBarneyVolume();

      virtual box3 bounds() const = 0;
      void commitParameters() override;

      bool isVisible() const;

    protected:
      virtual BNVolume createBarneyVolume() = 0;
      virtual void setBarneyParameters() = 0;
      void cleanup();

      uint32_t m_id{~0u};
      bool m_visible{true};
      BNVolume m_bnVolume{nullptr};
    };

    // Subtypes ///////////////////////////////////////////////////////////////////

    struct FieldMappedVolume : public Volume
    {
      FieldMappedVolume(BarneyGlobalState *s);

      void commitParameters() override;
      bool isValid() const override;

      BNVolume createBarneyVolume() override;
      box3 bounds() const override;

    protected:
      void setBarneyParameters() override;
      void invalidateBarneyVolumeIfFieldChanged();
      void finalizeFieldMappedVolume();

      helium::ChangeObserverPtr<SpatialField> m_field;
      const SpatialField *m_boundField{nullptr};
      BNScalarField m_boundNativeField{nullptr};

      box3 m_bounds;
      box1 m_valueRange{0.f, 1.f};
      float m_unitDistance{1.f};
      float m_densityScale{1.f};
      float m_anisotropy{0.6f};
      float m_scatteringAlbedo{0.9f};
      std::vector<math::float4> m_rgbaMap;
    };

    struct TransferFunction1D : public FieldMappedVolume
    {
      TransferFunction1D(BarneyGlobalState *s);

      void commitParameters() override;
      void finalize() override;

    private:
      math::float4 m_uniformColor{1.f, 1.f, 1.f, 1.f};
      float m_uniformOpacity{1.f};

      helium::ChangeObserverPtr<helium::Array1D> m_colorData;
      helium::ChangeObserverPtr<helium::Array1D> m_opacityData;
    };

    struct PrincipledVolume : public FieldMappedVolume
    {
      PrincipledVolume(BarneyGlobalState *s);

      void commitParameters() override;
      void finalize() override;

    protected:
      void setBarneyParameters() override;

    private:
      float m_emissionStrength{0.f};

      math::float4 m_uniformColor{1.f, 1.f, 1.f, 1.f};
      float m_uniformOpacity{1.f};
      helium::ChangeObserverPtr<helium::Array1D> m_colorData;
      helium::ChangeObserverPtr<helium::Array1D> m_opacityData;
      helium::ChangeObserverPtr<helium::Array1D> m_emissionMapData;
      std::vector<math::float4> m_emissionRGBAMap;

      void rebuildRGBAMapFromTransferFunction();
      void rebuildEmissionMap();
    };

// #else

//     struct TransferFunction1D : public Volume
//     {
//       TransferFunction1D(BarneyGlobalState *s);

//       void commitParameters() override;
//       void finalize() override;
//       bool isValid() const override;

//       BNVolume createBarneyVolume() override;

//       box3 bounds() const override;

//     private:
//       void setBarneyParameters() override;
//       void invalidateBarneyVolumeIfFieldChanged();

//       helium::ChangeObserverPtr<SpatialField> m_field;
//       const SpatialField *m_boundField{nullptr};

//       box3 m_bounds;

//       box1 m_valueRange{0.f, 1.f};
//       float m_unitDistance{1.f};
//       float m_densityScale{1.f};
//       math::float4 m_uniformColor{1.f, 1.f, 1.f, 1.f};
//       float m_uniformOpacity{1.f};

//       helium::ChangeObserverPtr<helium::Array1D> m_colorData;
//       helium::ChangeObserverPtr<helium::Array1D> m_opacityData;
//       bool needsOpacityData;

//       std::vector<math::float4> m_rgbaMap;
//     };


  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Volume *, ANARI_VOLUME);
