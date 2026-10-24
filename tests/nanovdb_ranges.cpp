// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "native/volume/NanoVDB.h"
#include "native/volume/NanoVDBHelpers.h"
#include "nanovdb_majorants.h"
#include "nanovdb_fixtures.h"
#include <nanovdb/tools/CreateNanoGrid.h>
#include <cstdio>
#include <stdexcept>

using namespace BARNEY_NS;
using namespace BARNEY_NS::native;
using namespace owl::common;

#include "nanovdb_quantized.h"

int main()
{
  checkQuantized<nanovdb::Fp4>();
  checkQuantized<nanovdb::Fp8>();
  checkQuantized<nanovdb::Fp16>();
  checkQuantized<nanovdb::FpN>();
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
  // Independent worked values: one corner is 1, its x neighbor is -7,
  // and the other six are 9. Quarter weights give 3.375 and 8.125.
  const vec3f points[] = {vec3f(-18.5f,2.5f,-2.5f),
    vec3f(-18.25f,2.5f,-2.5f), vec3f(-17.75f,2.5f,-2.5f),
    vec3f(-18.25f,2.75f,-2.25f), vec3f(-17.75f,3.25f,-1.75f),
    vec3f(-20.f,1.f,-4.f), vec3f(-18.25f,3.25f,-2.25f),
    vec3f(-18.25f,2.75f,-1.75f), vec3f(-17.75f,2.75f,-2.25f)};
  const float linear[] = {1.f, -1.f, -5.f, 3.375f, 8.125f, 1.f, 7.125f, 7.125f, 1.125f};
  const float nearest[] = {1.f, 1.f, -7.f, 1.f, 9.f, 1.f, 9.f, 9.f, -7.f};
  for (auto filter : {BN_TEXTURE_LINEAR, BN_TEXTURE_NEAREST}) {
    sampler.filterMode = filter;
    for (int i = 0; i < 9; ++i)
      if (sampler.sample(points[i]) != (filter == BN_TEXTURE_LINEAR ? linear[i] : nearest[i]))
        throw std::runtime_error("known Float nearest/linear reconstruction");
  }
  std::puts("PASS: numerical nearest/linear oracle, inactive neighbors and clamp");

  const double matrices[][9] = {
    {1,0,0, 0,1,0, 0,0,1},
    {2,0,0, 0,0.5,0, 0,0,3},
    {0,-1,0, 1,0,0, 0,0,1},
    {1,2,0, 0,1,0.5, 0,0,1}
  };
  const double inverses[][9] = {
    {1,0,0, 0,1,0, 0,0,1},
    {0.5,0,0, 0,2,0, 0,0,1.0/3},
    {0,1,0, -1,0,0, 0,0,1},
    {1,-2,1, 0,1,-0.5, 0,0,1}
  };
  for (int transform = 0; transform < 5; ++transform) {
    const nanovdb::Vec3d translation(7.25,3.75,.25);
    if (transform == 4)
      nanovdb_test::setComposedAffine(*grid, translation);
    else
      nanovdb_test::setAffine(*grid, matrices[transform], inverses[transform], translation);
    const auto world = grid->worldBBox();
    barney_nanovdb::validateMap(grid->map());
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
            for (auto filter : {BN_TEXTURE_LINEAR, BN_TEXTURE_NEAREST}) {
              sampler.filterMode = filter;
              const float value = sampler.sample(p);
              if (value < local.lower || value > local.upper)
                throw std::runtime_error("affine local range underbounds production sample");
              if (!conservativeMajorants(range1f(local.lower, local.upper), value))
                throw std::runtime_error("non-monotonic TF majorant underbounds density");
            }
          }
    }
  }
  // A finite invertible map must not be rejected by a determinant epsilon.
  nanovdb::Map tiny(1e-12);
  barney_nanovdb::validateMap(tiny);
  std::puts("PASS: inactive values, affine ranges and independent non-monotonic majorants");
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
  // Match the API filter-update fixture: a first-cell nearest-only range
  // would be [0,0], but linear can reach the neighboring impulse at z=12.
  nanovdb::tools::build::Grid<float> impulseSource(0.f);
  auto impulseAccessor = impulseSource.getAccessor();
  for (int z = 0; z < 35; ++z)
    impulseAccessor.setValue(nanovdb::Coord(0,0,z), z == 12 ? 1.f : 0.f);
  auto impulse = nanovdb::tools::createNanoGrid(impulseSource);
  sampler.nvdbGrid = impulse.grid<float>();
  sampler.indexBounds = box3i(vec3i(0), vec3i(0,0,34));
  const auto halo = barney_nanovdb::scalarRange(*sampler.nvdbGrid,
      nanovdb::Vec3dBBox(nanovdb::Vec3d(0), nanovdb::Vec3d(1,1,35.0/3)));
  sampler.filterMode = BN_TEXTURE_NEAREST;
  if (sampler.sample(vec3f(0.5f,0.5f,11.625f)) != 0.f)
    throw std::runtime_error("nearest must not reach the adjacent-cell impulse");
  sampler.filterMode = BN_TEXTURE_LINEAR;
  if (sampler.sample(vec3f(0.5f,0.5f,11.625f)) != 0.125f
      || halo.lower > 0.f || halo.upper < 1.f)
    throw std::runtime_error("shared range must cover the wider linear footprint");
  std::puts("PASS: nearest-to-linear impulse footprint remains conservative");
  nanovdb::tools::build::Grid<float> farSource(9.f);
  farSource.getAccessor().setValue(nanovdb::Coord(16777217, 0, 0), 1.f);
  auto far = nanovdb::tools::createNanoGrid(farSource);
  sampler.nvdbGrid = far.grid<float>();
  sampler.indexBounds = box3i(vec3i(16777217,0,0), vec3i(16777217,0,0));
  for (auto filter : {BN_TEXTURE_LINEAR, BN_TEXTURE_NEAREST}) {
    sampler.filterMode = filter;
    if (sampler.sample(vec3f(16777218.f,0.5f,0.5f)) != 1.f)
      throw std::runtime_error("float-rounded index bound must not sample outside the grid");
  }
  std::puts("PASS: small grid at an origin beyond exact float integers");

  for (int minX : {5, 4}) {
    auto extreme = nanovdb_test::extremeAffineGrid(minX);
    auto *extremeGrid = extreme.grid<float>();
    barney_nanovdb::validateMap(extremeGrid->map());
    sampler.nvdbGrid = extremeGrid;
    sampler.indexBounds = box3i(vec3i(minX,0,0), vec3i(5,1,1));
    const auto extremeRange = barney_nanovdb::scalarRange(*extremeGrid, extremeGrid->worldBBox());
    for (auto filter : {BN_TEXTURE_LINEAR, BN_TEXTURE_NEAREST}) {
      sampler.filterMode = filter;
      // x=2.25e38 maps to centered index 4.75, not the upper boundary.
      const float value = sampler.sample(vec3f(2.25e38f,0.75f,0.75f));
      const float expected = filter == BN_TEXTURE_NEAREST ? 10.f : minX == 5 ? 25.f : 20.f;
      if (!std::isfinite(value) || std::abs(value - expected) > 1e-4f
          || value < extremeRange.lower || value > extremeRange.upper)
        throw std::runtime_error("finite extreme affine sampling and conservative range");
    }
  }
  std::puts("PASS: extreme finite affine inverse overflow retains correct samples");
}
