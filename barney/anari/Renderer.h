// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Object.h"

namespace BARNEY_NS {
  namespace anari {
    
    struct Renderer : public Object
    {
      Renderer(BarneyGlobalState *s);
      ~Renderer() override;

      void commitParameters() override;
      void finalize() override;

      bool crosshairs() const;
      bool denoise() const;
      bool fadeOutDenoiser() const;
      bool upscale() const;
      bool isValid() const override;

      BNRenderer barneyRenderer{nullptr};

    private:
      BNTexture2D barneyBackgroundImage{nullptr};

      int m_pixelSamples{1};
      float m_ambientRadiance{0.8f};
      bool m_crosshairs{false};
      bool m_denoise{true};
      bool m_fadeOutDenoiser{false};
      bool m_upscale{false};
      anari::math::float4 m_background{0.f, 0.f, 0.f, 1.f};
      anari::math::float4 m_cutPlane{0.f, 0.f, 0.f, -1e30f};
#if BARNEY_USE_MULTI_SCATTERING
      int m_maxVolumeBounces{8};
      bool m_volumeMultiScatter{false};
#endif
      int m_sampleLimit{1024};
      helium::ChangeObserverPtr<Array2D> m_backgroundImage;
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Renderer *, ANARI_RENDERER);
