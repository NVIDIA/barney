// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Renderer.h"

namespace BARNEY_NS {
  namespace anari {
    
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
      m_denoise = getParam<bool>("denoise", false);
      m_fadeOutDenoiser = getParam<bool>("fadeOutDenoiser", true);
      m_upscale = getParam<bool>("upscale", false);
      m_background = getParam<math::float4>("background", math::float4(0, 0, 0, 1));
      m_backgroundImage = getParamObject<Array2D>("background");
      m_cutPlane = getParam<math::float4>("cutPlane", math::float4(0, 0, 0, 0));
      /* iw - @ap: using this to _both_ set max bounces _and_
         enabel/disable multiscatter (if set to 0). imho there shold
         be only two possible configs for multiscatter: which voluem
         type (ie which phase function), and then how many bounces */
      m_maxVolumeBounces = getParam<int>("maxVolumeBounces", 0);
      m_sampleLimit = std::clamp(getParam<int>("sampleLimit", 1024), 1, INT_MAX);
      // m_maxVolumeBounces = getParam<int>("maxVolumeBounces", 8);
    }

    void Renderer::finalize()
    {
      bnSetVec(barneyRenderer, "bgColor", m_background);
      bnSet1i(barneyRenderer, "crosshairs", (int)m_crosshairs);
      bnSet1i(barneyRenderer, "pathsPerPixel", (int)m_pixelSamples);
      bnSet1f(barneyRenderer, "ambientRadiance", m_ambientRadiance);
      bnSet1i(barneyRenderer, "maxVolumeBounces", m_maxVolumeBounces);
      bnSet1i(barneyRenderer, "sampleLimit", m_sampleLimit);
      // bnSet1i(barneyRenderer, "volumeMultiScatter", (int)m_volumeMultiScatter);
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

    bool Renderer::fadeOutDenoiser() const
    {
      return m_fadeOutDenoiser;
    }

    bool Renderer::upscale() const
    {
      return m_upscale;
    }

    bool Renderer::isValid() const
    {
      return barneyRenderer != 0;
    }

  }
}

BARNEY_ANARI_TYPEFOR_DEFINITION(BARNEY_NS::anari::Renderer *);
