// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Surface.h"

namespace BARNEY_NS {
  namespace anari {

    Surface::Surface(BarneyGlobalState *s)
      : Object(ANARI_SURFACE, s), m_geometry(this), m_material(this)
    {}

    Surface::~Surface()
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: ~Surface deconstructing"
                         << std::endl);
      cleanup();
    }

    void Surface::commitParameters()
    {
      m_id = getParam<uint32_t>("id", ~0u);
      m_geometry = getParamObject<Geometry>("geometry");
      m_material = getParamObject<Material>("material");
      m_visible = getParam<bool>("visible", true);
    }

    void Surface::finalize()
    {
      if (!m_material) {
        reportMessage(ANARI_SEVERITY_WARNING, "missing 'material' on ANARISurface");
        return;
      }

      if (!m_geometry) {
        reportMessage(ANARI_SEVERITY_WARNING, "missing 'geometry' on ANARISurface");
        return;
      }

      setBarneyParameters();
    }

    void Surface::markFinalized()
    {
      deviceState()->markStructuralSceneChanged();
      Object::markFinalized();
    }

    const Geometry *Surface::geometry() const
    {
      return m_geometry.get();
    }

    const Material *Surface::material() const
    {
      return m_material.get();
    }

    BNGeom Surface::getBarneyGeom()
    {
      int slot = deviceState()->slot;
      auto context = deviceState()->tether->context;

      if (!m_bnGeom) {
        cleanup();
        m_bnGeom = bnGeometryCreate(context, slot, m_geometry->bnSubtype());
      }
      setBarneyParameters();

      return m_bnGeom;
    }

    bool Surface::isValid() const
    {
      return
        m_geometry &&
        m_material &&
        m_geometry->isValid() &&
        m_material->isValid();
    }

    bool Surface::isVisible() const
    {
      return m_visible;
    }

    void Surface::setBarneyParameters()
    {
      if (!isValid() || !m_bnGeom)
        return;
      bnSetObject(m_bnGeom, "material", m_material->getBarneyMaterial());
      bnSet1i(m_bnGeom, "userID", m_id);
      // Rebinding mesh arrays would replace triangulated/converted data handles
      // and invalidate device tangents on instance-only world rebuilds.
      const bool mesh = std::string_view(m_geometry->bnSubtype()) == "triangles";
      if (!mesh || m_boundMesh != m_geometry.get()
          || m_meshFinalized != m_geometry->lastFinalized()) {
        m_geometry->setBarneyParameters(m_bnGeom);
        m_boundMesh = m_geometry.get();
        m_meshFinalized = m_geometry->lastFinalized();
      } else {
        m_geometry->setAttributes(m_bnGeom, Geometry::AttributeUpdate::MaterialAttributes);
      }
      bnCommit(m_bnGeom);
    }

    void Surface::cleanup()
    {
      if (m_bnGeom)
        bnRelease(m_bnGeom);
      m_bnGeom = nullptr;
      m_boundMesh = nullptr;
      m_meshFinalized = 0;
    }

  }
}

BARNEY_ANARI_TYPEFOR_DEFINITION(BARNEY_NS::anari::Surface *);
