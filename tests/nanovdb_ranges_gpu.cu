// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "native/volume/NanoVDB.h"
#include "native/volume/NanoVDBHelpers.h"
#include "nanovdb_majorants.h"
#include <nanovdb/tools/CreateNanoGrid.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

using namespace BARNEY_NS;
using namespace BARNEY_NS::native;
using namespace owl::common;

static void check(cudaError_t error)
{
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<typename BuildT>
__global__ void checkRanges(nanovdb::NanoGrid<BuildT> *grid,
                           nanovdb::Coord dims, bool identity,
                           float a, float b, float c, int *failures)
{
#if RTC_DEVICE_CODE
  const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= dims[0]*dims[1]*dims[2]) return;
  const int cell[3] = {tid % dims[0], (tid/dims[0]) % dims[1], tid/(dims[0]*dims[1])};
  nanovdb::Vec3d lo, hi;
  const auto world = grid->worldBBox();
  for (int i = 0; i < 3; ++i) {
    const double spacing = (world.max()[i]-world.min()[i])/dims[i];
    lo[i] = world.min()[i] + cell[i]*spacing;
    hi[i] = world.min()[i] + (cell[i]+1)*spacing;
  }
  const auto range = barney_nanovdb::scalarRange(*grid, nanovdb::Vec3dBBox(lo,hi));
  typename NanoVDBDataSampler<BuildT>::DD sampler;
  sampler.nvdbGrid = grid;
  const auto bounds = grid->indexBBox();
  sampler.indexBounds = box3i(vec3i(bounds.min()[0],bounds.min()[1],bounds.min()[2]),
                             vec3i(bounds.max()[0],bounds.max()[1],bounds.max()[2]));
  if (identity && tid == 0) {
    if (fabsf(sampler.sample(vec3f(-18.5f,2.5f,-2.5f)) - a) > 1e-5f
        || fabsf(sampler.sample(vec3f(-18.f,2.5f,-2.5f)) - (a+b)*.5f) > 1e-5f
        || fabsf(sampler.sample(vec3f(-17.5f,2.5f,-2.5f)) - b) > 1e-5f
        || range.lower > b || range.upper < c)
      atomicAdd(failures, 1);
    // Hand-calculated quarter weights for corners 1, -7 and six 9s.
    const vec3f points[] = {vec3f(-18.5f,2.5f,-2.5f),
      vec3f(-18.25f,2.5f,-2.5f), vec3f(-17.75f,2.5f,-2.5f),
      vec3f(-18.25f,2.75f,-2.25f), vec3f(-17.75f,3.25f,-1.75f),
      vec3f(-20.f,1.f,-4.f), vec3f(-18.25f,3.25f,-2.25f),
      vec3f(-18.25f,2.75f,-1.75f), vec3f(-17.75f,2.75f,-2.25f)};
    // Worked trilinear weights applied to NanoVDB-decoded corners.
    const float linear[] = {a, .75f*a+.25f*b, .25f*a+.75f*b,
      .421875f*a+.140625f*b+.4375f*c,
      .015625f*a+.046875f*b+.9375f*c, a,
      .140625f*a+.046875f*b+.8125f*c,
      .140625f*a+.046875f*b+.8125f*c,
      .140625f*a+.421875f*b+.4375f*c};
    const float nearest[] = {a, a, b, a, c, a, c, c, b};
    for (int mode = 0; mode < 2; ++mode) {
      sampler.filterMode = mode ? BN_TEXTURE_NEAREST : BN_TEXTURE_LINEAR;
      for (int i = 0; i < 9; ++i)
        if (fabsf(sampler.sample(points[i]) - (mode ? nearest[i] : linear[i])) > 1e-5f)
          atomicAdd(failures, 1);
    }
  }
  for (int z = 0; z <= 4; ++z)
    for (int y = 0; y <= 4; ++y)
      for (int x = 0; x <= 4; ++x) {
        for (int mode = 0; mode < 2; ++mode) {
          sampler.filterMode = mode ? BN_TEXTURE_NEAREST : BN_TEXTURE_LINEAR;
          const float value = sampler.sample(vec3f(float(lo[0]+(hi[0]-lo[0])*x/4),
              float(lo[1]+(hi[1]-lo[1])*y/4), float(lo[2]+(hi[2]-lo[2])*z/4)));
          if (!isfinite(value) || value < range.lower || value > range.upper
              || !conservativeMajorants(range1f(range.lower, range.upper), value))
            atomicAdd(failures, 1);
        }
      }
#endif
}

template<typename BuildT>
static void checkType()
{
  int devices = 0;
  auto error = cudaGetDeviceCount(&devices);
  if (error != cudaSuccess || devices == 0) {
    std::printf("SKIP: CUDA device unavailable: %s\n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA device required");
  }
  cudaDeviceProp properties;
  check(cudaGetDeviceProperties(&properties, 0));
  std::printf("CUDA device: %s, compute %d.%d; devices=%d\n",
              properties.name, properties.major, properties.minor, devices);
  nanovdb::tools::build::Grid<float> source(9.f);
  auto accessor = source.getAccessor();
  accessor.setValue(nanovdb::Coord(-19,2,-3), 1.f);
  accessor.setValue(nanovdb::Coord(109,66,61), 2.f);
  for (auto stats : {nanovdb::tools::StatsMode::BBox, nanovdb::tools::StatsMode::All}) {
    auto floatHandle = nanovdb::tools::createNanoGrid(source, stats);
    auto *floatGrid = floatHandle.grid<float>();
    auto *leaf = const_cast<nanovdb::NanoLeaf<float> *>(floatGrid->getAccessor().probeLeaf(nanovdb::Coord(-18,2,-3)));
    leaf->setValueOnly(nanovdb::Coord(-18,2,-3), -7.f);
    auto handle = nanovdb::tools::createNanoGrid<nanovdb::NanoGrid<float>, BuildT>(*floatGrid, stats);
    auto *grid = handle.template grid<BuildT>();
    const auto acc = grid->getAccessor();
    const float a = acc.getValue(nanovdb::Coord(-19,2,-3));
    const float b = acc.getValue(nanovdb::Coord(-18,2,-3));
    const float c = acc.getValue(nanovdb::Coord(-19,3,-3));
    if (grid->tree().isActive(nanovdb::Coord(-18,2,-3)))
      throw std::runtime_error("inactive fixture lost its mask");
    for (int affine = 0; affine < 2; ++affine) {
      if (affine) {
        const double matrix[9] = {0,-2,0, 1,0,0.5, 0,0,0.75};
        const double inverse[9] = {0,1,-2.0/3, -0.5,0,0, 0,0,4.0/3};
        for (int i = 0; i < 9; ++i) {
          grid->mMap.mMatF[i] = float(grid->mMap.mMatD[i] = matrix[i]);
          grid->mMap.mInvMatF[i] = float(grid->mMap.mInvMatD[i] = inverse[i]);
        }
        grid->mMap.mVecF[0] = float(grid->mMap.mVecD[0] = 7.25);
        nanovdb::Vec3dBBox world;
        for (int corner = 0; corner < 8; ++corner)
          world.expand(grid->indexToWorld(nanovdb::Vec3d(corner & 1 ? 110 : -19,
              corner & 2 ? 67 : 2, corner & 4 ? 62 : -3)));
        grid->mWorldBBox = world;
      }
      nanovdb::NanoGrid<BuildT> *deviceGrid;
      int *failures;
      check(cudaMalloc(&deviceGrid, handle.size()));
      check(cudaMallocManaged(&failures, sizeof(int)));
      *failures = 0;
      check(cudaMemcpy(deviceGrid, handle.data(), handle.size(), cudaMemcpyHostToDevice));
      const auto dims = barney_nanovdb::macrocellDims(grid->indexBBox());
      const int count = dims[0]*dims[1]*dims[2];
      checkRanges<<<(count+127)/128,128>>>(deviceGrid, dims, !affine, a, b, c, failures);
      check(cudaGetLastError());
      check(cudaDeviceSynchronize());
      if (*failures) throw std::runtime_error("GPU local ranges underbound production samples");
      check(cudaFree(deviceGrid));
      check(cudaFree(failures));
      std::printf("PASS: GPU ranges/samples/non-monotonic majorants: type=%d %d cells, stats=%d affine=%d\n", int(grid->gridType()), count, int(stats), affine);
    }
  }
}

int main()
{
  checkType<float>();
  checkType<nanovdb::Fp4>();
  checkType<nanovdb::Fp8>();
  checkType<nanovdb::Fp16>();
  checkType<nanovdb::FpN>();
}
