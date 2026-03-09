// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "Surface.h"

namespace barney_device {

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
    m_geometry->setBarneyParameters(m_bnGeom);
    bnCommit(m_bnGeom);
  }

  void Surface::cleanup()
  {
    if (m_bnGeom)
      bnRelease(m_bnGeom);
    m_bnGeom = nullptr;
  }

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Surface *);
