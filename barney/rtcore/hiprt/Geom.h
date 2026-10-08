// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#pragma once

#include "rtcore/hiprt/Buffer.h"
#include "rtcore/hiprt/GeomType.h"

namespace BARNEY_NS {
  namespace rtc {

    struct GeomType;

    struct Geom {
      Geom(GeomType *gt);
      virtual ~Geom();
      void setDD(const void *dd);

      // the per-geom shading record: the same function-pointer SBT the software
      // backend uses. HIPRT yields the hit; barney's megakernel calls these.
      struct SBTHeader {
        AHProg ah;
        CHProg ch;
        union {
          struct {
            const vec3f *vertices;
            const vec3i *indices;
          } triangles;
          struct {
            IntersectProg intersect;
          } user;
        };
      };

      virtual void setPrimCount(int primCount) = 0;
      virtual void setVertices(Buffer *vertices, int numVertices) = 0;
      virtual void setIndices(Buffer *indices, int numIndices) = 0;

      GeomType *const gt;
      std::vector<uint8_t> data;
    };

    struct TrianglesGeom : public Geom {
      TrianglesGeom(GeomType *gt);
      void setPrimCount(int primCount) override { assert(0); }
      void setVertices(Buffer *vertices, int numVertices) override;
      void setIndices(Buffer *indices, int numIndices) override;

      Buffer *vertices = 0;
      int numVertices = 0;
      Buffer *indices = 0;
      int numIndices = 0;
    };

    struct UserGeom : public Geom {
      UserGeom(GeomType *gt);
      void setPrimCount(int primCount) override;
      void setVertices(Buffer *vertices, int numVertices) override { assert(0); }
      void setIndices(Buffer *indices, int numIndices) override { assert(0); }

      int primCount = 0;
    };

  }
}
