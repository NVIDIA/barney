// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "NanoVDBTypes.h"
#include "NanoVDBMath.h"
#include <cmath>
#include <climits>
#include <stdexcept>

namespace barney_nanovdb {

inline nanovdb::Coord macrocellDims(const nanovdb::CoordBBox &bounds)
{
  nanovdb::Coord dims;
  size_t count = 1;
  for (int axis = 0; axis < 3; ++axis) {
    const int64_t size = int64_t(bounds.max()[axis]) - bounds.min()[axis] + 1;
    if (size <= 0 || size > INT_MAX)
      throw std::runtime_error("Unrepresentable NanoVDB dimensions");
    dims[axis] = int(1 + (size - 1) / 16);
    // All existing MC/DDA and majorant indices are signed ints.
    if (size_t(dims[axis]) > size_t(INT_MAX) / count)
      throw std::runtime_error("NanoVDB macrocell count exceeds RTC indexing");
    count *= size_t(dims[axis]);
  }
  if (count > (SIZE_MAX - NANOVDB_DATA_ALIGNMENT) / (2 * sizeof(float)))
    throw std::runtime_error("NanoVDB macrocell allocation overflow");
  return dims;
}

template<typename T>
inline bool invertible(const T *m)
{
  for (int i = 0; i < 9; ++i)
    if (!std::isfinite(m[i])) return false;
  const long double det =
      (long double)m[0] * ((long double)m[4]*m[8] - (long double)m[5]*m[7])
    - (long double)m[1] * ((long double)m[3]*m[8] - (long double)m[5]*m[6])
    + (long double)m[2] * ((long double)m[3]*m[7] - (long double)m[4]*m[6]);
  return det != 0;
}

inline void validateMap(const nanovdb::Map &map)
{
  if (!invertible(map.mMatD) || !invertible(map.mMatF)
      || !invertible(map.mInvMatD) || !invertible(map.mInvMatF))
    throw std::runtime_error("Singular or nonfinite NanoVDB affine map");
  for (int i = 0; i < 3; ++i)
    if (!std::isfinite(map.mVecD[i]) || !std::isfinite(map.mVecF[i]))
      throw std::runtime_error("Nonfinite NanoVDB translation");
}

struct Interval { float lower, upper; };

__hostdev__ inline float down(double x)
{
  return nextafterf(float(x), -INFINITY);
}
__hostdev__ inline float up(double x)
{
  return nextafterf(float(x), INFINITY);
}

// Enclose the actual float inverse map, including subtraction, products,
// additions (fused or unfused), centering and clamping. Mapping only index
// blocks to world cells is not conservative under rotations or shear.
template<typename BuildT>
__hostdev__ inline Interval scalarRange(const nanovdb::NanoGrid<BuildT> &grid,
                                       const nanovdb::Vec3dBBox &world)
{
  const auto &map = grid.map();
  const auto &bounds = grid.indexBBox();
  Interval translated[3];
  for (int j = 0; j < 3; ++j)
    translated[j] = {down(double(down(world.min()[j])) - map.mVecF[j]),
                     up(double(up(world.max()[j])) - map.mVecF[j])};
  nanovdb::Coord lo, hi;
  for (int i = 0; i < 3; ++i) {
    Interval sum{0.f, 0.f};
    for (int j = 2; j >= 0; --j) {
      const double a = double(map.mInvMatF[3*i+j]) * translated[j].lower;
      const double b = double(map.mInvMatF[3*i+j]) * translated[j].upper;
      const Interval product{down(fmin(a,b)), up(fmax(a,b))};
      sum = {down(double(sum.lower) + product.lower),
             up(double(sum.upper) + product.upper)};
    }
    // A whole-axis fallback handles overflow without a near-singular cutoff.
    double lower = down(double(sum.lower) - 0.5);
    double upper = up(double(sum.upper) - 0.5);
    if (!isFinite(lower) || !isFinite(upper)) {
      lo[i] = bounds.min()[i];
      hi[i] = bounds.max()[i];
    } else {
      lower = fmax(double(bounds.min()[i]), fmin(double(bounds.max()[i]), lower));
      upper = fmax(double(bounds.min()[i]), fmin(double(bounds.max()[i]), upper));
      lo[i] = int32_t(floor(lower));
      hi[i] = int32_t(fmin(double(bounds.max()[i]), floor(upper) + 1.0));
    }
  }
  auto acc = grid.getAccessor();
  Interval range{INFINITY, -INFINITY};
  // int64 counters also terminate safely at INT_MAX. Accessors include stored
  // inactive values, tiles and background; active-only statistics cannot.
  for (int64_t z = lo[2]; z <= hi[2]; ++z)
    for (int64_t y = lo[1]; y <= hi[1]; ++y)
      for (int64_t x = lo[0]; x <= hi[0]; ++x) {
        const float value = acc.getValue(nanovdb::Coord(int(x), int(y), int(z)));
        range.lower = fminf(range.lower, value);
        range.upper = fmaxf(range.upper, value);
      }
  // Three levels of float lerps can round slightly outside the convex hull.
  const double error = 32.0 * 0x1p-23 * fmax(fabs(double(range.lower)), fabs(double(range.upper)));
  return {down(double(range.lower) - error), up(double(range.upper) + error)};
}

} // namespace barney_nanovdb
