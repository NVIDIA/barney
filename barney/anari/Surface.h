// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Geometry.h"
#include "Material.h"

namespace BARNEY_NS {
  namespace anari {

    struct Surface : public Object
    {
      Surface(BarneyGlobalState *s);
      ~Surface() override;

      void commitParameters() override;
      void finalize() override;
      void markFinalized() override;

      uint32_t id() const;
      const Geometry *geometry() const;
      const Material *material() const;

      BNGeom getBarneyGeom();

      bool isValid() const override;
      bool isVisible() const;

    private:
      void setBarneyParameters();
      void cleanup();

      uint32_t m_id{~0u};
      bool m_visible{true};
      helium::ChangeObserverPtr<Geometry> m_geometry;
      helium::ChangeObserverPtr<Material> m_material;

      const Geometry *m_boundMesh{nullptr};
      helium::TimeStamp m_meshFinalized{0};
      BNGeom m_bnGeom{nullptr};
      BNMaterial m_bnMat{nullptr};
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Surface *, ANARI_SURFACE);
