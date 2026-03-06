// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "Volume.h"
// std
#include <numeric>

namespace barney_device {

  Volume::Volume(BarneyGlobalState *s) : Object(ANARI_VOLUME, s) {}

  Volume::~Volume() = default;

  Volume *Volume::createInstance(std::string_view subtype, BarneyGlobalState *s)
  {
    if (subtype == "transferFunction1D")
      return new TransferFunction1D(s);
    else
      return (Volume *)new UnknownObject(ANARI_VOLUME, subtype, s);
  }

  void Volume::markFinalized()
  {
    deviceState()->markSceneChanged();
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

  // Subtypes ///////////////////////////////////////////////////////////////////

  TransferFunction1D::TransferFunction1D(BarneyGlobalState *s)
    : Volume(s), m_field(this), m_colorData(this), m_opacityData(this)
  {}

  bool TransferFunction1D::isValid() const
  {
    return m_field && m_field->isValid();
  }

  void TransferFunction1D::commitParameters()
  {
    Volume::commitParameters();
    m_field = getParamObject<SpatialField>("value");
    m_valueRange = getParam<box1>("valueRange", box1{0.f, 1.f});
    m_colorData = getParamObject<helium::Array1D>("color");
    m_uniformColor = math::float4(1.f);
    getParam("color", ANARI_FLOAT32_VEC3, &m_uniformColor);
    getParam("color", ANARI_FLOAT32_VEC4, &m_uniformColor);
    m_opacityData = getParamObject<helium::Array1D>("opacity");
    m_uniformOpacity = getParam<float>("opacity", 1.f) * m_uniformColor.w;
    m_unitDistance = getParam<float>("unitDistance", 1.f);
  }

  void TransferFunction1D::finalize()
  {
    if (!m_field) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "no spatial field provided to transferFunction1D volume");
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

    size_t numColorChannels{4};
    if (m_colorData) { // TODO: more types
      if (m_colorData->elementType() == ANARI_FLOAT32_VEC3)
        numColorChannels = 3;
    }

    float *colorData = m_colorData ? (float *)m_colorData->data() : nullptr;
    float *opacityData = m_opacityData ? (float *)m_opacityData->data() : nullptr;

    size_t numColors = m_colorData ? m_colorData->size() : 1;
    size_t numOpacities = m_opacityData ? m_opacityData->size() : 1;
    size_t tfSize = std::max(numColors, numOpacities);

    m_rgbaMap.resize(tfSize);
    for (size_t i=0; i<tfSize; ++i) {
      float colorPos = tfSize > 1 ? (float(i)/(tfSize-1))*(numColors-1) : 0.f;
      float colorFrac = colorPos-floorf(colorPos);

      math::float4 color0(m_uniformColor.xyz(), m_uniformOpacity);
      math::float4 color1(m_uniformColor.xyz(), m_uniformOpacity);
      if (colorData) {
        if (numColorChannels == 3) {
          math::float3 *colors = (math::float3 *)colorData;
          color0 = math::float4(colors[int(floorf(colorPos))], m_uniformOpacity);
          color1 = math::float4(colors[int(ceilf(colorPos))], m_uniformOpacity);
        }
        else if (numColorChannels == 4) {
          math::float4 *colors = (math::float4 *)colorData;
          color0 = colors[int(floorf(colorPos))];
          color1 = colors[int(ceilf(colorPos))];
        }
      }

      math::float4 color = (1.f-colorFrac)*color0 + colorFrac*color1;

      if (opacityData) {
        float alphaPos = tfSize > 1 ? (float(i)/(tfSize-1))*(numOpacities-1) : 0.f;
        float alphaFrac = alphaPos-floorf(alphaPos);

        float alpha0 = opacityData[int(floorf(alphaPos))];
        float alpha1 = opacityData[int(ceilf(alphaPos))];

        color.w *= (1.f-alphaFrac)*alpha0 + alphaFrac*alpha1;
      }

      m_rgbaMap[i] = color;
    }

    setBarneyParameters();
  }

  BNVolume TransferFunction1D::createBarneyVolume()
  {
    // Inherit slot from the field's dataRank so the volume and its
    // scalar field live on the same Barney data partition.
    int dataRank = m_field ? m_field->getParam<int>("dataRank", -1) : -1;
    int slot = deviceState()->resolveSlot(dataRank);
    auto context = deviceState()->tether->context;

    return m_field
      ? bnVolumeCreate(context, slot, m_field->getBarneyScalarField())
      : BNVolume{};
  }

  box3 TransferFunction1D::bounds() const
  {
    return m_bounds;
  }

  void TransferFunction1D::setBarneyParameters()
  {
    if (!isValid() || !m_bnVolume)
      return;
    BNVolume vol = getBarneyVolume();
    bnSet1i(vol,"userID",m_id);
    bnVolumeSetXF(vol,
                  (bn_float2 &)m_valueRange,
                  (const bn_float4 *)m_rgbaMap.data(),
                  (int)m_rgbaMap.size(),
                  m_densityScale);
    bnCommit(vol);
  }

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Volume *);
