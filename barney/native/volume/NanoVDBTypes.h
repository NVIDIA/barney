// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nanovdb/NanoVDB.h>

// Build type, device-symbol suffix, GridType enumerator.
#define BARNEY_NANOVDB_FLOAT_TYPES(MACRO) \
  MACRO(float, float, Float) \
  MACRO(nanovdb::Fp4, fp4, Fp4) \
  MACRO(nanovdb::Fp8, fp8, Fp8) \
  MACRO(nanovdb::Fp16, fp16, Fp16) \
  MACRO(nanovdb::FpN, fpn, FpN)

namespace barney_nanovdb {

inline size_t fixedRootSize(nanovdb::GridType type)
{
  switch (type) {
#define ROOT_SIZE(BuildType, Suffix, EnumName) \
    case nanovdb::GridType::EnumName: \
      return sizeof(nanovdb::NanoGrid<BuildType>::TreeType::RootType);
    BARNEY_NANOVDB_FLOAT_TYPES(ROOT_SIZE)
#undef ROOT_SIZE
    default: return 0;
  }
}

} // namespace barney_nanovdb
