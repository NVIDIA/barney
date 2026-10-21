// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/render/GeometryAttributes.h"

namespace BARNEY_NS {
namespace native {

// Attribute0 is the UV source. Output is per triangle corner so mirrored UV
// orientation and face-varying seams never have to share a vertex tangent.
struct TangentMesh
{
  const vec3f *positions;
  const vec3i *indices;
  int numVertices;
  int numTriangles;
  GeometryAttribute::DD normals;
  GeometryAttribute::DD uvs;
};

// Inputs and output stay on the supplied device. CPU uses the same kernels.
// Angle-weighted approximation, not exact MikkTSpace; bucket sum order may vary.
// The caller owns the returned buffer and releases it with device->freeBuffer().
rtc::Buffer *generateTangentFrames(rtc::Device *device, const TangentMesh &mesh);

} // namespace native
} // namespace BARNEY_NS
