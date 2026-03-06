// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "Array.h"
#include "Object.h"
#include "SpatialField.h"

namespace barney_device {

  struct Volume : public Object
  {
    Volume(BarneyGlobalState *s);
    ~Volume() override;

    static Volume *createInstance(std::string_view subtype, BarneyGlobalState *s);

    void markFinalized() override;

    BNVolume getBarneyVolume();
    virtual SpatialField *field() const { return nullptr; }

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

  struct TransferFunction1D : public Volume
  {
    TransferFunction1D(BarneyGlobalState *s);

    void commitParameters() override;
    void finalize() override;
    bool isValid() const override;

    BNVolume createBarneyVolume() override;
    SpatialField *field() const override { return m_field.get(); }

    box3 bounds() const override;

  private:
    void setBarneyParameters() override;

    helium::ChangeObserverPtr<SpatialField> m_field;

    box3 m_bounds;

    box1 m_valueRange{0.f, 1.f};
    float m_unitDistance{1.f};
    float m_densityScale{1.f};
    math::float4 m_uniformColor{1.f, 1.f, 1.f, 1.f};
    float m_uniformOpacity{1.f};

    helium::ChangeObserverPtr<helium::Array1D> m_colorData;
    helium::ChangeObserverPtr<helium::Array1D> m_opacityData;
    bool needsOpacityData;

    std::vector<math::float4> m_rgbaMap;
  };

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Volume *, ANARI_VOLUME);
