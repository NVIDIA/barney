// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/barney-common.h"
#include <cmath>

namespace BARNEY_NS {
  namespace native {

    // Instance traversal needs a representable inverse, not a pseudoinverse
    // or an arbitrary normal for collapsed geometry. No determinant epsilon:
    // small but representable scales are still valid.
    inline bool usableInstanceTransform(const affine3f &xfm)
    {
      auto finite = [](const vec3f &v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
      };
      if (!finite(xfm.l.vx) || !finite(xfm.l.vy) || !finite(xfm.l.vz)
          || !finite(xfm.p))
        return false;
      const float det = xfm.l.det();
      if (!std::isfinite(det) || det == 0.f)
        return false;
      const affine3f inverse = rcp(xfm);
      return finite(inverse.l.vx) && finite(inverse.l.vy)
        && finite(inverse.l.vz) && finite(inverse.p);
    }

  }
}
