// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"
#include "common.h"
#include <iostream>
#include <limits>

namespace BARNEY_NS {
  namespace anari {

    // Helper functions ///////////////////////////////////////////////////////////

    template <typename T>
    inline MaterialParameter<T>
    getMaterialHelper(Object *o, const char *p, T defaultValue)
    {
      MaterialParameter<T> retval;
      retval.value = o->getParam<T>(p, defaultValue);
      retval.attribute = o->getParamString(p, "");
      retval.sampler = o->getParamObject<Sampler>(p);
      return retval;
    }

    template <>
    inline MaterialParameter<math::float4>
    getMaterialHelper(Object *o, const char *p, math::float4 defaultValue)
    {
      MaterialParameter<math::float4> retval;
      retval.value = defaultValue;
      o->getParam(p, ANARI_FLOAT32_VEC3, &retval.value);
      o->getParam(p, ANARI_FLOAT32_VEC4, &retval.value);
      retval.attribute = o->getParamString(p, "");
      retval.sampler = o->getParamObject<Sampler>(p);
      return retval;
    }

    template <typename T>
    inline void setBNMaterialUniform(BNMaterial m, const char *p, T v)
    {
      throw std::runtime_error("unhandled setBNMaterialUniform type");
    }

    template <>
    inline void setBNMaterialUniform(BNMaterial m, const char *p, float v)
    {
      bnSet1f(m, p, v);
    }

    template <>
    inline void setBNMaterialUniform(BNMaterial m, const char *p, math::float3 v)
    {
      bnSet3f(m, p, v.x, v.y, v.z);
    }

    template <>
    inline void setBNMaterialUniform(BNMaterial m, const char *p, math::float4 v)
    {
      bnSet4f(m, p, v.x, v.y, v.z, v.w);
    }

    template <typename T>
    inline void setBNMaterialHelper
    (BNMaterial m, const char *p, MaterialParameter<T> &mp)
    {
      if (mp.sampler) {
        BNSampler s = mp.sampler->getBarneySampler();
        bnSetObject(m, p, s);
      } else if (!mp.attribute.empty())
        bnSetString(m, p, mp.attribute.c_str());
      else {
        setBNMaterialUniform(m, p, mp.value);
      }
    }

    inline void setBNCoverage(BNMaterial m, Object *o)
    {
      // 'bn:'-prefixed bookkeeping flags consumed by HostMaterial::set1i;
      // kept namespaced away from application parameter names.
      if (o->hasParam("alphaMode"))
        bnSetString(m, "alphaMode", o->getParamString("alphaMode", "").c_str());
      bnSet1i(m, "bn:alphaModeSet", o->hasParam("alphaMode") ? 1 : 0);
      if (o->hasParam("alphaCutoff"))
        bnSet1f(m, "alphaCutoff", o->getParam<float>("alphaCutoff", 0.f));
      bnSet1i(m, "bn:alphaCutoffSet", o->hasParam("alphaCutoff") ? 1 : 0);
    }

    // Material definitions ///////////////////////////////////////////////////////

    Material::Material(BarneyGlobalState *s) : Object(ANARI_MATERIAL, s) {}

    Material::~Material()
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: ~Material deconstructing" << std::endl);
      if (m_bnMat)
        bnRelease(m_bnMat);
      m_bnMat = nullptr;
    }

    Material *Material::createInstance(std::string_view subtype,
                                       BarneyGlobalState *s)
    {
      if (subtype == "matte")
        return new Matte(s);
      else if (subtype == "physicallyBased")
        return new PhysicallyBased(s);
      else
        return (Material *)new UnknownObject(ANARI_MATERIAL, subtype, s);
    }

    void Material::finalize()
    {
      setBarneyParameters();
    }

    BNMaterial Material::getBarneyMaterial()
    {
      int slot = deviceState()->slot;
      auto context = deviceState()->tether->context;

      if (!m_bnMat)
        m_bnMat = bnMaterialCreate(context, slot, bnSubtype());
      setBarneyParameters();
      return m_bnMat;
    }

    // Subtypes ///////////////////////////////////////////////////////////////////

    // Matte //

    Matte::Matte(BarneyGlobalState *s) : Material(s)
    {
      this->commitParameters(); // init with defaults for scalar values
    }

    void Matte::commitParameters()
    {
      Object::commitParameters();
      m_color   = getMaterialHelper(this, "color",
                                    math::float4(0.8f, 0.8f, 0.8f, 1.f));
      m_opacity = getMaterialHelper(this, "opacity", 1.f);
    }

    bool Matte::isValid() const
    {
      return !m_color.sampler || m_color.sampler->isValid();
    }

    const char *Matte::bnSubtype() const
    {
      return "AnariMatte";
    }

    void Matte::setBarneyParameters()
    {
      if (!m_bnMat)
        return;

      setBNMaterialHelper(m_bnMat, "color",   m_color);
      setBNMaterialHelper(m_bnMat, "opacity", m_opacity);
      setBNCoverage(m_bnMat, this);
      bnCommit(m_bnMat);
    }

    // PhysicallyBased //

    PhysicallyBased::PhysicallyBased(BarneyGlobalState *s) : Material(s)
    {
      this->commitParameters(); // init with defaults for scalar values
    }

    void PhysicallyBased::commitParameters()
    {
      Object::commitParameters();
      m_baseColor     = getMaterialHelper(this, "baseColor",
                                          math::float4(1, 1, 1, 1));
      m_emissive      = getMaterialHelper(this, "emissive",
                                          math::float3(0, 0, 0));
      m_specularColor = getMaterialHelper(this, "specularColor",
                                          math::float3(1, 1, 1));
      m_opacity       = getMaterialHelper(this, "opacity",      1.f);
      m_metallic      = getMaterialHelper(this, "metallic",     1.f);
      m_roughness     = getMaterialHelper(this, "roughness",    1.f);
      m_specular      = getMaterialHelper(this, "specular",     0.f);
      m_transmission  = getMaterialHelper(this, "transmission", 0.f);
      m_occlusion     = getMaterialHelper(this, "occlusion",    1.f);
      m_clearcoat     = getMaterialHelper(this, "clearcoat",    0.f);
      m_clearcoatRoughness = getMaterialHelper(this, "clearcoatRoughness",
                                               0.f);
      m_attenuationColor = getMaterialHelper(this, "attenuationColor",
                                             math::float3(1, 1, 1));
      m_thickness     = getMaterialHelper(this, "thickness",    0.f);
      m_attenuationDistance = getMaterialHelper(this, "attenuationDistance",
                                                std::numeric_limits<float>::infinity());
      m_sheenColor    = getMaterialHelper(this, "sheenColor",
                                          math::float3(0, 0, 0));
      m_sheenRoughness = getMaterialHelper(this, "sheenRoughness", 0.f);
      m_iridescence   = getMaterialHelper(this, "iridescence",  0.f);
      m_iridescenceThickness = getMaterialHelper(this, "iridescenceThickness",
                                                 0.f);
      // Sampler-only: normal / clearcoatNormal (no scalar default).
      m_normal        = getMaterialHelper(this, "normal",
                                          math::float4(0, 0, 1, 1));
      m_clearcoatNormal = getMaterialHelper(this, "clearcoatNormal",
                                            math::float4(0, 0, 1, 1));
      m_ior           = getParam<float>("ior", 1.5f);
      m_iridescenceIor = getParam<float>("iridescenceIor", 1.3f);
    }

    const char *PhysicallyBased::bnSubtype() const
    {
      return "physicallyBased";
    }

    void PhysicallyBased::setBarneyParameters()
    {
      if (!m_bnMat)
        return;

      setBNMaterialHelper(m_bnMat, "baseColor",    m_baseColor);
      setBNMaterialHelper(m_bnMat, "emissive",     m_emissive);
      setBNMaterialHelper(m_bnMat, "specularColor",m_specularColor);
      setBNMaterialHelper(m_bnMat, "metallic",     m_metallic);
      setBNMaterialHelper(m_bnMat, "roughness",    m_roughness);
      setBNMaterialHelper(m_bnMat, "specular",     m_specular);
      setBNMaterialHelper(m_bnMat, "transmission", m_transmission);
      setBNMaterialHelper(m_bnMat, "opacity",      m_opacity);
      setBNMaterialHelper(m_bnMat, "occlusion",    m_occlusion);
      setBNMaterialHelper(m_bnMat, "clearcoat",    m_clearcoat);
      setBNMaterialHelper(m_bnMat, "clearcoatRoughness",m_clearcoatRoughness);
      setBNMaterialHelper(m_bnMat, "attenuationColor",m_attenuationColor);
      setBNMaterialHelper(m_bnMat, "thickness",    m_thickness);
      setBNMaterialHelper(m_bnMat, "attenuationDistance",m_attenuationDistance);
      setBNMaterialHelper(m_bnMat, "sheenColor",   m_sheenColor);
      setBNMaterialHelper(m_bnMat, "sheenRoughness",m_sheenRoughness);
      setBNMaterialHelper(m_bnMat, "iridescence",  m_iridescence);
      setBNMaterialHelper(m_bnMat, "iridescenceThickness",m_iridescenceThickness);
      // normal / clearcoatNormal have no scalar form in barney (sampler or
      // attribute only); forwarding the uniform default would just produce
      // an ignored-member warning, so only set them when actually mapped.
      if (m_normal.sampler || !m_normal.attribute.empty())
        setBNMaterialHelper(m_bnMat, "normal",       m_normal);
      if (m_clearcoatNormal.sampler || !m_clearcoatNormal.attribute.empty())
        setBNMaterialHelper(m_bnMat, "clearcoatNormal",m_clearcoatNormal);

      bnSet1f(m_bnMat, "ior", m_ior);
      bnSet1f(m_bnMat, "iridescenceIor", m_iridescenceIor);
      setBNCoverage(m_bnMat, this);
      bnCommit(m_bnMat);
    }

  }
}

BARNEY_ANARI_TYPEFOR_DEFINITION(BARNEY_NS::anari::Material *);
