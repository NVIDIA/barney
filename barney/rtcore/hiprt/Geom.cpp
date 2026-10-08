// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#include "rtcore/hiprt/Geom.h"
#include "rtcore/hiprt/GeomType.h"

namespace BARNEY_NS {
  namespace rtc {

    Geom::Geom(GeomType *gt)
      : gt(gt), data(gt->sizeOfDD)
    {}

    Geom::~Geom() {}

    void Geom::setDD(const void *dd)
    { memcpy(data.data(),dd,data.size()); }

    TrianglesGeom::TrianglesGeom(GeomType *gt) : Geom(gt) {}

    void TrianglesGeom::setVertices(Buffer *vertices, int numVertices)
    { this->vertices = vertices; this->numVertices = numVertices; }

    void TrianglesGeom::setIndices(Buffer *indices, int numIndices)
    { this->indices = indices; this->numIndices = numIndices; }

    UserGeom::UserGeom(GeomType *gt) : Geom(gt) {}

    void UserGeom::setPrimCount(int primCount)
    { this->primCount = primCount; }

  }
}
