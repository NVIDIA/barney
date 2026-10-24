// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Volume.h"
// std
#include <algorithm>
#include <numeric>

namespace BARNEY_NS {
  namespace anari {
    
    Volume::Volume(BarneyGlobalState *s) : Object(ANARI_VOLUME, s) {}

    Volume::~Volume()
    {
      cleanup();
    }

    Volume *Volume::createInstance(std::string_view subtype, BarneyGlobalState *s)
    {
      if (subtype == "transferFunction1D")
        return new TransferFunction1D(s);
      /* iw - renamed this from pythonic to anaric spelling case */
      if (subtype == "principledVolume")
        return new PrincipledVolume(s);
      return (Volume *)new UnknownObject(ANARI_VOLUME, subtype, s);
    }

    void Volume::markFinalized()
    {
      deviceState()->markStructuralSceneChanged();
      Object::markFinalized();
    }

    BNVolume Volume::getBarneyVolume()
    {
      if (!isValid())
        return {};
      if (!m_bnVolume) {
        m_bnVolume = createBarneyVolume();
        setBarneyParameters();
      }
      return m_bnVolume;
    }

    void Volume::cleanup()
    {
      if (m_bnVolume) {
        bnRelease(m_bnVolume);
        m_bnVolume = nullptr;
      }
    }

    void Volume::commitParameters()
    {
      m_id = getParam<uint32_t>("id", ~0u);
      m_visible = getParam<bool>("visible", true);
    }

    bool Volume::isVisible() const
    {
      return m_visible;
    }

    // FieldMappedVolume //////////////////////////////////////////////////////////

    FieldMappedVolume::FieldMappedVolume(BarneyGlobalState *s)
      : Volume(s), m_field(this)
    {}

    bool FieldMappedVolume::isValid() const
    {
      return m_field && m_field->isValid() && !m_field->isEmpty();
    }

    void FieldMappedVolume::commitParameters()
    {
      Volume::commitParameters();
      m_field = getParamObject<SpatialField>("value");
      m_valueRange = getParam<box1>("valueRange", box1{0.f, 1.f});
      m_unitDistance = getParam<float>("unitDistance", 1.f);
      m_anisotropy = getParam<float>("anisotropy", 0.6f);
      m_scatteringAlbedo = getParam<float>("scatteringAlbedo", 0.9f);
      invalidateBarneyVolumeIfFieldChanged();
    }

    void FieldMappedVolume::finalizeFieldMappedVolume()
    {
      invalidateBarneyVolumeIfFieldChanged();
      if (!m_field) {
        reportMessage(ANARI_SEVERITY_WARNING,
                      "no spatial field provided to volume");
        return;
      }

      if (m_unitDistance <= 0.f) {
        reportMessage(ANARI_SEVERITY_WARNING,
                      "invalid density scale calculated from 'unitDistance',"
                      " setting to 1");
        m_densityScale = 1.f;
      } else
        m_densityScale = 1.f / m_unitDistance;

      m_bounds = m_field->bounds();
      if (auto *field = dynamic_cast<NanoVDBSpatialField *>(m_field.get())) {
        try {
          // Prepare all dependent storage during finalization, before any
          // group can publish geometry from this shared field.
          if (!m_bnVolume)
            getBarneyVolume();
          else
            setBarneyParameters();
        } catch (const std::exception &e) {
          cleanup();
          field->invalidate(e.what());
          m_bounds.invalidate();
        }
      } else {
        setBarneyParameters();
      }
    }

    BNVolume FieldMappedVolume::createBarneyVolume()
    {
      int slot = deviceState()->slot;
      auto context = deviceState()->tether->context;

      BNVolume volume = m_field
        ? bnVolumeCreate(context, slot, m_field->getBarneyScalarField())
        : BNVolume{};

      if (volume) {
        m_boundField = m_field.get();
        m_boundNativeField = m_field->getBarneyScalarField();
      }

      return volume;
    }

    box3 FieldMappedVolume::bounds() const
    {
      return m_bounds;
    }

    void FieldMappedVolume::setBarneyParameters()
    {
      if (!isValid() || !m_bnVolume)
        return;
      BNVolume vol = getBarneyVolume();
      bnSet1i(vol, "userID", m_id);
      bnSet1f(vol, "anisotropy", m_anisotropy);
      bnSet1f(vol, "scatteringAlbedo", m_scatteringAlbedo);
      bnSet1i(vol, "principledVolume", 0);
      bnVolumeSetXF(vol,
                    (bn_float2 &)m_valueRange,
                    (const bn_float4 *)m_rgbaMap.data(),
                    (int)m_rgbaMap.size(),
                    m_densityScale);
      bnCommit(vol);
    }

    void FieldMappedVolume::invalidateBarneyVolumeIfFieldChanged()
    {
      const SpatialField *newField = m_field.get();
      const BNScalarField newNativeField = isValid()
        ? m_field->m_bnField : nullptr;
      if (m_boundField == newField && m_boundNativeField == newNativeField)
        return;

      cleanup();
      m_boundField = nullptr;
      m_boundNativeField = nullptr;
    }

    // TransferFunction1D ///////////////////////////////////////////////////////////

    TransferFunction1D::TransferFunction1D(BarneyGlobalState *s)
      : FieldMappedVolume(s), m_colorData(this), m_opacityData(this)
    {}

    void TransferFunction1D::commitParameters()
    {
      FieldMappedVolume::commitParameters();
      m_colorData = getParamObject<helium::Array1D>("color");
      math::float4 uniformColor(1.f);
      getParam("color", ANARI_FLOAT32_VEC3, &uniformColor);
      getParam("color", ANARI_FLOAT32_VEC4, &uniformColor);
      m_uniformColor = uniformColor;
      m_opacityData = getParamObject<helium::Array1D>("opacity");
      m_uniformOpacity = getParam<float>("opacity", 1.f) * m_uniformColor.w;
    }

    void TransferFunction1D::finalize()
    {
      if (!m_field) {
        reportMessage(ANARI_SEVERITY_WARNING,
                      "no spatial field provided to transferFunction1D volume");
        return;
      }

      size_t numColorChannels{4};
      if (m_colorData) {
        if (m_colorData->elementType() == ANARI_FLOAT32_VEC3)
          numColorChannels = 3;
      }

      float *colorData = m_colorData ? (float *)m_colorData->data() : nullptr;
      float *opacityData = m_opacityData ? (float *)m_opacityData->data() : nullptr;

      size_t numColors = m_colorData ? m_colorData->size() : 1;
      size_t numOpacities = m_opacityData ? m_opacityData->size() : 1;
      size_t tfSize = std::max(numColors, numOpacities);

      m_rgbaMap.resize(tfSize);
      for (size_t i = 0; i < tfSize; ++i) {
        float colorPos = tfSize > 1 ? (float(i) / (tfSize - 1)) * (numColors - 1) : 0.f;
        float colorFrac = colorPos - floorf(colorPos);

        math::float4 color0(m_uniformColor.xyz(), m_uniformOpacity);
        math::float4 color1(m_uniformColor.xyz(), m_uniformOpacity);
        if (colorData) {
          if (numColorChannels == 3) {
            math::float3 *colors = (math::float3 *)colorData;
            color0 = math::float4(colors[int(floorf(colorPos))], m_uniformOpacity);
            color1 = math::float4(colors[int(ceilf(colorPos))], m_uniformOpacity);
          } else if (numColorChannels == 4) {
            math::float4 *colors = (math::float4 *)colorData;
            color0 = colors[int(floorf(colorPos))];
            color1 = colors[int(ceilf(colorPos))];
          }
        }

        math::float4 color = (1.f - colorFrac) * color0 + colorFrac * color1;

        if (opacityData) {
          float alphaPos = tfSize > 1 ? (float(i) / (tfSize - 1)) * (numOpacities - 1) : 0.f;
          float alphaFrac = alphaPos - floorf(alphaPos);

          float alpha0 = opacityData[int(floorf(alphaPos))];
          float alpha1 = opacityData[int(ceilf(alphaPos))];

          color.w *= (1.f - alphaFrac) * alpha0 + alphaFrac * alpha1;
        }

        m_rgbaMap[i] = color;
      }

      FieldMappedVolume::finalizeFieldMappedVolume();
    }

    // PrincipledVolume ///////////////////////////////////////////////////////////

    PrincipledVolume::PrincipledVolume(BarneyGlobalState *s)
      : FieldMappedVolume(s),
        m_colorData(this),
        m_opacityData(this),
        m_emissionMapData(this)
    {}

    void PrincipledVolume::rebuildRGBAMapFromTransferFunction()
    {
      size_t numColorChannels{4};
      if (m_colorData) {
        if (m_colorData->elementType() == ANARI_FLOAT32_VEC3)
          numColorChannels = 3;
      }

      float *colorData = m_colorData ? (float *)m_colorData->data() : nullptr;
      float *opacityData = m_opacityData ? (float *)m_opacityData->data() : nullptr;

      size_t numColors = m_colorData ? m_colorData->size() : 1;
      size_t numOpacities = m_opacityData ? m_opacityData->size() : 1;
      size_t tfSize = std::max(numColors, numOpacities);

      m_rgbaMap.resize(tfSize);
      for (size_t i = 0; i < tfSize; ++i) {
        float colorPos = tfSize > 1 ? (float(i) / (tfSize - 1)) * (numColors - 1) : 0.f;
        float colorFrac = colorPos - floorf(colorPos);

        math::float4 color0(m_uniformColor.xyz(), m_uniformOpacity);
        math::float4 color1(m_uniformColor.xyz(), m_uniformOpacity);
        if (colorData) {
          if (numColorChannels == 3) {
            math::float3 *colors = (math::float3 *)colorData;
            color0 = math::float4(colors[int(floorf(colorPos))], m_uniformOpacity);
            color1 = math::float4(colors[int(ceilf(colorPos))], m_uniformOpacity);
          } else if (numColorChannels == 4) {
            math::float4 *colors = (math::float4 *)colorData;
            color0 = colors[int(floorf(colorPos))];
            color1 = colors[int(ceilf(colorPos))];
          }
        }

        math::float4 color = (1.f - colorFrac) * color0 + colorFrac * color1;

        if (opacityData) {
          float alphaPos = tfSize > 1 ? (float(i) / (tfSize - 1)) * (numOpacities - 1) : 0.f;
          float alphaFrac = alphaPos - floorf(alphaPos);

          float alpha0 = opacityData[int(floorf(alphaPos))];
          float alpha1 = opacityData[int(ceilf(alphaPos))];

          color.w *= (1.f - alphaFrac) * alpha0 + alphaFrac * alpha1;
        }

        m_rgbaMap[i] = color;
      }
    }

    void PrincipledVolume::rebuildEmissionMap()
    {
      m_emissionRGBAMap.clear();
      if (!m_emissionMapData || m_emissionMapData->size() == 0)
        return;

      const size_t n = m_emissionMapData->size();
      switch (m_emissionMapData->elementType()) {
      case ANARI_FLOAT32_VEC4: {
        auto *src = (const math::float4 *)m_emissionMapData->data();
        m_emissionRGBAMap.assign(src, src + n);
        break;
      }
      case ANARI_FLOAT32_VEC3: {
        auto *src = (const math::float3 *)m_emissionMapData->data();
        m_emissionRGBAMap.resize(n);
        for (size_t i = 0; i < n; ++i)
          m_emissionRGBAMap[i] = math::float4(src[i], 1.f);
        break;
      }
      case ANARI_FLOAT32: {
        auto *src = (const float *)m_emissionMapData->data();
        m_emissionRGBAMap.resize(n);
        for (size_t i = 0; i < n; ++i)
          m_emissionRGBAMap[i] = math::float4(1.f, 1.f, 1.f, src[i]);
        break;
      }
      default:
        reportMessage(ANARI_SEVERITY_WARNING,
                      "principledVolume 'emissionMap' must be FLOAT32,"
                      " FLOAT32_VEC3 or FLOAT32_VEC4 -- ignoring");
        break;
      }
    }

    void PrincipledVolume::commitParameters()
    {
      FieldMappedVolume::commitParameters();
      m_emissionStrength = getParam<float>("emissionStrength", 0.f);

      m_colorData = getParamObject<helium::Array1D>("color");
      math::float4 uniformColor(1.f);
      getParam("color", ANARI_FLOAT32_VEC3, &uniformColor);
      getParam("color", ANARI_FLOAT32_VEC4, &uniformColor);
      m_uniformColor = uniformColor;
      m_opacityData = getParamObject<helium::Array1D>("opacity");
      m_uniformOpacity = getParam<float>("opacity", 1.f) * m_uniformColor.w;
      m_emissionMapData = getParamObject<helium::Array1D>("emissionMap");
      rebuildRGBAMapFromTransferFunction();
      rebuildEmissionMap();

      if (m_unitDistance <= 0.f)
        m_densityScale = 1.f;
      else
        m_densityScale = 1.f / m_unitDistance;

      if (m_bnVolume)
        setBarneyParameters();
    }

    void PrincipledVolume::setBarneyParameters()
    {
      if (!isValid())
        return;

      if (m_unitDistance <= 0.f)
        m_densityScale = 1.f;
      else
        m_densityScale = 1.f / m_unitDistance;

      if (!m_bnVolume)
        return;

      BNVolume vol = m_bnVolume;
      bnSet1i(vol, "userID", m_id);
      bnSet1f(vol, "anisotropy", m_anisotropy);
      bnSet1i(vol, "principledVolume", 1);
      bnSet1f(vol, "principledEmissionStrength", m_emissionStrength);

      if (!m_rgbaMap.empty()) {
        bnVolumeSetXF(vol,
                      (bn_float2 &)m_valueRange,
                      (const bn_float4 *)m_rgbaMap.data(),
                      (int)m_rgbaMap.size(),
                      m_densityScale);
      }
      if (!m_emissionRGBAMap.empty()) {
        bnVolumeSetEmissionXF(vol,
                              (bn_float2 &)m_valueRange,
                              (const bn_float4 *)m_emissionRGBAMap.data(),
                              (int)m_emissionRGBAMap.size());
      } else {
        bnVolumeClearEmissionXF(vol);
      }
      bnCommit(vol);
    }

    void PrincipledVolume::finalize()
    {
      if (!m_field) {
        reportMessage(ANARI_SEVERITY_WARNING,
                      "no spatial field provided to principled_volume");
        return;
      }

      rebuildRGBAMapFromTransferFunction();
      rebuildEmissionMap();
      finalizeFieldMappedVolume();
    }

  }
}

BARNEY_ANARI_TYPEFOR_DEFINITION(BARNEY_NS::anari::Volume *);
