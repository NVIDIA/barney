// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/volume/NanoVDBMath.h"
#include <cmath>

__hostdev__ inline bool checkFiniteClassification()
{
  const double finite[] = {
      0., -0., 1., -1., DBL_MAX, -DBL_MAX, DBL_MIN, -DBL_MIN,
      DBL_MIN / 2., -DBL_MIN / 2.};
  for (double value : finite)
    if (!barney_nanovdb::isFinite(value))
      return false;
  return !barney_nanovdb::isFinite(double(INFINITY))
      && !barney_nanovdb::isFinite(-double(INFINITY))
      && !barney_nanovdb::isFinite(double(NAN));
}
