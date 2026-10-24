// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/volume/TransferFunction.h"

// Shared CPU/CUDA numerical seam: production TF mapping and majorants applied
// to production NanoVDB local ranges and reconstructed samples.
inline __rtc_device bool conservativeMajorants(BARNEY_NS::native::range1f range, float value)
{
  using namespace BARNEY_NS;
  using namespace BARNEY_NS::native;
  using namespace owl::common;
  rtc::float4 a[5] = {{0,0,0,0}, {0,0,0,.05f}, {0,0,0,0},
                      {0,0,0,.1f}, {0,0,0,0}};
  rtc::float4 b[5] = {{0,0,0,.1f}, {0,0,0,0}, {0,0,0,.4f},
                      {0,0,0,0}, {0,0,0,.1f}};
  TransferFunction::DD first, second;
  first.values = a;
  second.values = b;
  first.numValues = second.numValues = 5;
  first.domain = second.domain = range1f(0.f, 1.f);
  first.baseDensity = 2.f;
  second.baseDensity = 3.f;
  for (int update = 0; update < 2; ++update) {
    if (update) {
      a[1].w = .9f;
      a[3].w = .7f;
    }
    const float majorants[] = {first.majorant(range), second.majorant(range)};
    const float densities[] = {first.map(value).w, second.map(value).w};
    for (int i = 0; i < 2; ++i)
      if (!(densities[i] <= majorants[i] + 1e-6f)) return false;
    // Interior peaks, not just range endpoints, determine the full majorant.
    const float expected = update ? 1.8f : .2f;
    if (fabsf(first.majorant(range1f(-7.f, 9.f))-expected) > 1e-6f
        || fabsf(second.majorant(range1f(-7.f, 9.f))-1.2f) > 1e-6f)
      return false;
  }
  // Worked subinterval maxima and clamped endpoint densities, independent of
  // the production interpolation implementation.
  return fabsf(first.majorant(range1f(.1f,.2f))-1.44f) < 1e-6f
      && fabsf(second.majorant(range1f(.1f,.2f))-.18f) < 1e-6f
      && first.map(-7.f).w == 0.f && first.map(9.f).w == 0.f
      && fabsf(second.map(-7.f).w-.3f) < 1e-6f
      && fabsf(second.map(9.f).w-.3f) < 1e-6f;
}
