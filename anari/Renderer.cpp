// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "Renderer.h"

namespace barney_device {

Renderer::Renderer(BarneyGlobalState *s)
    : Object(ANARI_RENDERER, s), m_backgroundImage(this)
{
  barneyRenderer = bnRendererCreate(deviceState()->tether->context, "default");
}

Renderer::~Renderer()
{
  bnRelease(barneyRenderer);
}

void Renderer::commitParameters()
{
  m_pixelSamples = getParam<int>("pixelSamples", 1);
  m_ambientRadiance = getParam<float>("ambientRadiance", 1.f);
  m_crosshairs = getParam<bool>("crosshairs", false);
  m_denoise = getParam<bool>("denoise", true);
  m_background = getParam<math::float4>("background", math::float4(0, 0, 0, 1));
  m_backgroundImage = getParamObject<Array2D>("background");
  m_cutPlane = getParam<math::float4>("cutPlane", math::float4(0, 0, 0, 0));
}

void Renderer::finalize()
{
  bnSetVec(barneyRenderer, "bgColor", m_background);
  bnSet1i(barneyRenderer, "crosshairs", (int)m_crosshairs);
  bnSet1i(barneyRenderer, "pathsPerPixel", (int)m_pixelSamples);
  bnSet1f(barneyRenderer, "ambientRadiance", m_ambientRadiance);
  bnSet4f(barneyRenderer, "cutPlane",
          m_cutPlane.x, m_cutPlane.y, m_cutPlane.z, m_cutPlane.w);

  if (m_backgroundImage) {
    int sx = m_backgroundImage->size().x;
    int sy = m_backgroundImage->size().y;
    const bn_float4 *texels
      = (const bn_float4 *)m_backgroundImage->data();
    barneyBackgroundImage
      = bnTexture2DCreate(deviceState()->tether->context,-1,
                          BN_FLOAT4,sx,sy,
                          texels,
                          BN_TEXTURE_LINEAR,
                          BN_TEXTURE_CLAMP,BN_TEXTURE_CLAMP);
    bnSetObject(barneyRenderer,"bgTexture",barneyBackgroundImage);
  } else {
    if (barneyBackgroundImage) {
      bnRelease(barneyBackgroundImage);
      barneyBackgroundImage = 0;
      bnSetObject(barneyRenderer,"bgTexture",0);
    }
  }
  bnCommit(barneyRenderer);
}

bool Renderer::crosshairs() const
{
  return m_crosshairs;
}

bool Renderer::denoise() const
{
  return m_denoise;
}

bool Renderer::isValid() const
{
  return barneyRenderer != 0;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Renderer *);
