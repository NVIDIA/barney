// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tangent_hits.h"

namespace tangent_test {
// Compile with the backend language so host-side half decoding uses the same
// representation as the device rather than requiring CPU Float16 symbols.
vec3f mappedNormal(const Ray &hit, bool clearcoat)
{
  require(hit.bsdfType == PackedBSDF::TYPE_PhysicallyBased,"expected PBR hit");
  return clearcoat ? vec3f(hit.hitBSDF.physicallyBased.clearcoatNormal)
                   : vec3f(hit.hitBSDF.physicallyBased.normal);
}
}
