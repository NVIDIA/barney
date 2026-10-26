// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nanovdb/NanoVDB.h>
#include <cfloat>

namespace barney_nanovdb {

__hostdev__ inline bool isFinite(double value)
{
  // MSVC's std::isfinite overload is host-only with CUDA 12.6.
  // Ordered comparisons reject NaN as well as either infinity.
  return value >= -DBL_MAX && value <= DBL_MAX;
}

} // namespace barney_nanovdb
