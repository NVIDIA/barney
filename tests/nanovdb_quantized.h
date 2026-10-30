// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "nanovdb_fixtures.h"

// Included by the CPU numerical client; uses its production sampling seam.
template<typename BuildT>
static void checkQuantized()
{
  nanovdb::tools::build::Grid<float> source(9.f);
  source.getAccessor().setValue(nanovdb::Coord(-19,2,-3), 1.f);
  source.getAccessor().setValue(nanovdb::Coord(15,6,1), 2.f);
  auto original = nanovdb::tools::createNanoGrid(source);
  auto *input = original.grid<float>();
  auto *leaf = const_cast<nanovdb::NanoLeaf<float> *>(
      input->getAccessor().probeLeaf(nanovdb::Coord(-18,2,-3)));
  leaf->setValueOnly(nanovdb::Coord(-18,2,-3), -7.f);
  for (auto stats : {nanovdb::tools::StatsMode::BBox, nanovdb::tools::StatsMode::All}) {
    auto handle = nanovdb::tools::createNanoGrid<nanovdb::NanoGrid<float>, BuildT>(*input, stats);
    auto *grid = handle.template grid<BuildT>();
    typename NanoVDBDataSampler<BuildT>::DD sampler;
    sampler.nvdbGrid = grid;
    sampler.indexBounds = box3i(vec3i(-19,2,-3), vec3i(15,6,1));
    auto acc = grid->getAccessor();
    const float a = acc.getValue(nanovdb::Coord(-19,2,-3));
    const float b = acc.getValue(nanovdb::Coord(-18,2,-3));
    if (acc.isActive(nanovdb::Coord(-18,2,-3)))
      throw std::runtime_error("quantized fixture lost inactive value");
    sampler.filterMode = BN_TEXTURE_NEAREST;
    if (sampler.sample(vec3f(-18.25f,2.5f,-2.5f)) != a)
      throw std::runtime_error("quantized nearest differs from NanoVDB decode");
    sampler.filterMode = BN_TEXTURE_LINEAR;
    if (std::abs(sampler.sample(vec3f(-18.25f,2.5f,-2.5f)) - (.75f*a+.25f*b)) > 1e-5f)
      throw std::runtime_error("quantized linear differs from decoded corner weights");
    for (int affine = 0; affine < 2; ++affine) {
      if (affine)
        nanovdb_test::setComposedAffine(*grid);
      const auto world = grid->worldBBox();
      for (int cell = 0; cell < 3; ++cell) {
        auto lo = world.min(), hi = world.max();
        lo[0] += cell*(world.max()[0]-world.min()[0])/3;
        hi[0] = world.min()[0]+(cell+1)*(world.max()[0]-world.min()[0])/3;
        const auto range = barney_nanovdb::scalarRange(*grid, nanovdb::Vec3dBBox(lo,hi));
        if (!affine && cell == 0 && (range.lower > b || range.upper < 9.f))
          throw std::runtime_error("quantized range omits inactive/background values");
        for (int z = 0; z <= 4; ++z)
          for (int y = 0; y <= 4; ++y)
            for (int x = 0; x <= 4; ++x)
              for (auto filter : {BN_TEXTURE_LINEAR, BN_TEXTURE_NEAREST}) {
                sampler.filterMode = filter;
                const float value = sampler.sample(vec3f(
                    float(lo[0]+(hi[0]-lo[0])*x/4),
                    float(lo[1]+(hi[1]-lo[1])*y/4),
                    float(lo[2]+(hi[2]-lo[2])*z/4)));
                if (!std::isfinite(value) || value < range.lower || value > range.upper
                    || !conservativeMajorants(range1f(range.lower,range.upper), value))
                  throw std::runtime_error("quantized affine range/majorant underbound");
              }
      }
    }
    std::printf("PASS: quantized CPU decode/filter/affine/ranges/majorants type=%d stats=%d\n",
                int(grid->gridType()), int(stats));
  }
}
