// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "native/volume/NanoVDB.h"
#include "native/volume/NanoVDBHelpers.h"
#include <nanovdb/tools/CreateNanoGrid.h>
#include <cstdio>
#include <stdexcept>

using namespace BARNEY_NS::native;
using namespace owl::common;

int main()
{
  nanovdb::tools::build::Grid<float> source(9.f);
  auto acc = source.getAccessor();
  acc.setValue(nanovdb::Coord(-19, 2, -3), 1.f);
  acc.setValue(nanovdb::Coord(15, 6, 1), 2.f);
  auto handle = nanovdb::tools::createNanoGrid(source, nanovdb::tools::StatsMode::BBox);
  auto *grid = handle.grid<float>();
  auto dims = barney_nanovdb::macrocellDims(grid->indexBBox());
  if (dims != nanovdb::Coord(3, 1, 1))
    throw std::runtime_error("round up partial and small dimensions");
  NanoVDBDataSampler<float>::DD sampler;
  sampler.nvdbGrid = grid;
  sampler.indexBounds = box3i(vec3i(-19, 2, -3), vec3i(15, 6, 1));
  auto range = barney_nanovdb::scalarRange(*grid, grid->worldBBox());
  if (range.lower > 1.f || range.upper < 9.f)
    throw std::runtime_error("sparse background outside active extrema");
  if (sampler.sample(vec3f(-18.5f, 2.5f, -2.5f)) != 1.f
      || sampler.sample(vec3f(-18.f, 2.5f, -2.5f)) != 5.f)
    throw std::runtime_error("production linear sample at center and boundary");
  std::puts("PASS: partial dimensions, sparse background, production sampling");
  // A stored inactive value is not part of NanoVDB's active-value statistics.
  auto *leaf = const_cast<nanovdb::NanoLeaf<float> *>(grid->getAccessor().probeLeaf(nanovdb::Coord(-18, 2, -3)));
  leaf->setValueOnly(nanovdb::Coord(-18, 2, -3), -7.f);
  if (grid->tree().isActive(nanovdb::Coord(-18, 2, -3)))
    throw std::runtime_error("fixture must contain an inactive value");
  range = barney_nanovdb::scalarRange(*grid, grid->worldBBox());
  if (range.lower > -7.f) throw std::runtime_error("inactive value excluded");

  const double matrices[][9] = {
    {1,0,0, 0,1,0, 0,0,1},
    {2,0,0, 0,0.5,0, 0,0,3},
    {0,-1,0, 1,0,0, 0,0,1},
    {1,2,0, 0,1,0.5, 0,0,1},
    {0,-2,0, 1,0,0.5, 0,0,0.75}
  };
  const double inverses[][9] = {
    {1,0,0, 0,1,0, 0,0,1},
    {0.5,0,0, 0,2,0, 0,0,1.0/3},
    {0,1,0, -1,0,0, 0,0,1},
    {1,-2,1, 0,1,-0.5, 0,0,1},
    {0,1,-2.0/3, -0.5,0,0, 0,0,4.0/3}
  };
  for (int transform = 0; transform < 5; ++transform) {
    auto &map = grid->mMap;
    for (int i = 0; i < 9; ++i) {
      map.mMatF[i] = float(map.mMatD[i] = matrices[transform][i]);
      map.mInvMatF[i] = float(map.mInvMatD[i] = inverses[transform][i]);
    }
    for (int i = 0; i < 3; ++i)
      map.mVecF[i] = float(map.mVecD[i] = 7.25 - 3.5*i);
    nanovdb::Vec3dBBox world;
    for (int corner = 0; corner < 8; ++corner) {
      nanovdb::Vec3d p;
      for (int i = 0; i < 3; ++i)
        p[i] = corner & (1 << i) ? double(grid->indexBBox().max()[i])+1 : grid->indexBBox().min()[i];
      world.expand(grid->indexToWorld(p));
    }
    barney_nanovdb::validateMap(map);
    for (int cell = 0; cell < dims[0]; ++cell) {
      auto lo = world.min(), hi = world.max();
      lo[0] = world.min()[0] + cell * (world.max()[0]-world.min()[0])/dims[0];
      hi[0] = world.min()[0] + (cell+1) * (world.max()[0]-world.min()[0])/dims[0];
      const auto local = barney_nanovdb::scalarRange(*grid, nanovdb::Vec3dBBox(lo, hi));
      // Include faces, edges, corners and interior points, not just voxel centers.
      for (int z = 0; z <= 8; ++z)
        for (int y = 0; y <= 8; ++y)
          for (int x = 0; x <= 8; ++x) {
            const vec3f p(float(lo[0]+(hi[0]-lo[0])*x/8),
                          float(lo[1]+(hi[1]-lo[1])*y/8),
                          float(lo[2]+(hi[2]-lo[2])*z/8));
            const float value = sampler.sample(p);
            if (value < local.lower || value > local.upper)
              throw std::runtime_error("affine local range underbounds production sample");
          }
    }
  }
  // A finite invertible map must not be rejected by a determinant epsilon.
  nanovdb::Map tiny(1e-12);
  barney_nanovdb::validateMap(tiny);
  std::puts("PASS: inactive values, translated/scaled/rotated/sheared coverage");
  nanovdb::tools::build::Grid<float> rampSource(99.f);
  auto rampAccessor = rampSource.getAccessor();
  for (int x = -19; x <= 15; ++x)
    for (int y = 2; y <= 6; ++y)
      for (int z = -3; z <= 1; ++z)
        rampAccessor.setValue(nanovdb::Coord(x,y,z), float(x+20));
  auto ramp = nanovdb::tools::createNanoGrid(rampSource);
  const auto firstCell = barney_nanovdb::scalarRange(*ramp.grid<float>(),
      nanovdb::Vec3dBBox(nanovdb::Vec3d(-19,2,-3), nanovdb::Vec3d(-19+35.0/3,7,2)));
  if (firstCell.lower > 1.f || firstCell.upper < 13.f || firstCell.upper >= 14.f)
    throw std::runtime_error("local range must include interpolation halo without global extrema");
  std::puts("PASS: dense ramp local tightness and partial-block halo");
  nanovdb::tools::build::Grid<float> farSource(9.f);
  farSource.getAccessor().setValue(nanovdb::Coord(16777217, 0, 0), 1.f);
  auto far = nanovdb::tools::createNanoGrid(farSource);
  sampler.nvdbGrid = far.grid<float>();
  sampler.indexBounds = box3i(vec3i(16777217,0,0), vec3i(16777217,0,0));
  if (sampler.sample(vec3f(16777218.f,0.5f,0.5f)) != 1.f)
    throw std::runtime_error("float-rounded index bound must not sample outside the grid");
  std::puts("PASS: small grid at an origin beyond exact float integers");
}
