// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/math.h"

namespace BARNEY_NS {
  namespace native {
    
    /*! HitIDs are used for a special "ID pass" that computes the
      closest intersection along a (primary) ray, WITHOUT any opacity
      or transparency taken into effect. As such it requires its own
      depth value: a ray hitting a semi-transparent object will
      randomly decide whether or not to accept the hit, so if we were
      to rely on ray.t to determine depth channel that value would
      randomly pick the front surface or another surface behind it.
      Note that even though 'traceRays' will fill all those values we
      do _not_ necessarily have to send them over the network for
      frame assembly - though these get written by trace as a array of
      structs, in the TiledFB we actually store them as separate tiles
      (one per value), and only those that are enabled in anari will
      actually get sent/copied to dev0 for assembly */
    struct HitIDs {
      float depth = BARNEY_INF;
      int primID = -1;
      int instID = -1;
      int objID  = -1;
    };

  }
} // ::BARNEY_NS
