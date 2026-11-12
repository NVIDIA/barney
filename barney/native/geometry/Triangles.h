// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/geometry/Geometry.h"
#include <array>

namespace BARNEY_NS {
  namespace native {

    // ==================================================================
    /*! Scalar field made of 3D structured data, constting of Nx*Ny*Nz
      scalars.

      Supported settable fields:

      - "vertices"  (BNData<float3>)
      - "indices"   (BNData<int3>)

      Per-vertex normals and texture coordinates arrive through the
      ANARI attribute system (vertex.normal -> normalAttribute,
      texcoords -> attribute0), not bespoke arrays.
    */
    struct Triangles : public Geometry {
      typedef std::shared_ptr<Triangles> SP;

      struct DD : public Geometry::DD {
        const vec3i *indices;
        const vec3f *vertices;
        // const vec4f *vertexAttribute[5];
      };
    
      Triangles(Context *context, DevGroup::SP devices);
      virtual ~Triangles();
    
      /*! pretty-printer for printf-debugging */
      std::string toString() const override
      { return "Triangles{}"; }
    
      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      void writeDD(Triangles::DD &dd, Device *device);
      bool setData(const std::string &member,
                   const Data::SP &value) override;
      /*! @} */
      // ------------------------------------------------------------------

      PODData::SP vertices;
      PODData::SP indices;

    private:
      std::array<PODData::SP,8> tangentInputs;
      std::array<uint64_t,8> tangentRevisions{};
      std::vector<rtc::Buffer *> tangentBuffers;
    };

  }
}

