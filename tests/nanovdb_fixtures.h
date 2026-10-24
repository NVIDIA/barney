// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nanovdb/tools/CreateNanoGrid.h>
#include <cmath>

namespace nanovdb_test {

template<typename BuildT>
inline void setAffine(nanovdb::NanoGrid<BuildT> &grid, const double *matrix,
                      const double *inverse, const nanovdb::Vec3d &translation)
{
  for (int i = 0; i < 9; ++i) {
    grid.mMap.mMatF[i] = float(grid.mMap.mMatD[i] = matrix[i]);
    grid.mMap.mInvMatF[i] = float(grid.mMap.mInvMatD[i] = inverse[i]);
  }
  for (int i = 0; i < 3; ++i) {
    grid.mMap.mVecF[i] = float(grid.mMap.mVecD[i] = translation[i]);
    grid.mVoxelSize[i] = std::sqrt(matrix[i]*matrix[i]
        + matrix[i+3]*matrix[i+3] + matrix[i+6]*matrix[i+6]);
  }
  nanovdb::Vec3dBBox world;
  for (int corner = 0; corner < 8; ++corner) {
    nanovdb::Vec3d p;
    for (int i = 0; i < 3; ++i)
      p[i] = corner & (1 << i) ? double(grid.indexBBox().max()[i])+1
                              : grid.indexBBox().min()[i];
    world.expand(grid.indexToWorld(p));
  }
  grid.mWorldBBox = world;
}

template<typename BuildT>
inline void setComposedAffine(nanovdb::NanoGrid<BuildT> &grid,
                             const nanovdb::Vec3d &translation = nanovdb::Vec3d(7.25,0,0))
{
  const double matrix[9] = {0,-2,0, 1,0,.5, 0,0,.75};
  const double inverse[9] = {0,1,-2.0/3, -.5,0,0, 0,0,4.0/3};
  setAffine(grid, matrix, inverse, translation);
}

inline auto extremeAffineGrid(int minX = 5)
{
  nanovdb::tools::build::Grid<float> source(0.f);
  for (int x = minX; x <= 5; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        source.getAccessor().setValue(nanovdb::Coord(x,y,z), 10.f+20*(x-5)+20*y+40*z);
  auto handle = nanovdb::tools::createNanoGrid(source);
  const double matrix[9] = {1e38,0,0, 0,1,0, 0,0,1};
  const double inverse[9] = {1e-38,0,0, 0,1,0, 0,0,1};
  setAffine(*handle.grid<float>(), matrix, inverse, nanovdb::Vec3d(-3e38,0,0));
  return handle;
}

} // namespace nanovdb_test
