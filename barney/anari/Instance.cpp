// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Instance.h"
#include "native/common/InstanceTransform.h"
// anari
#include <anari/anari_cpp/ext/linalg.h>

namespace BARNEY_NS {
  namespace anari {
    
    Instance::Instance(BarneyGlobalState *s) : Object(ANARI_INSTANCE, s) {}

    Instance::~Instance()
    {
      if (attributes) delete attributes;
    }

    void Instance::commitParameters()
    {
      math::mat4 xfm = anari::math::identity;
      getParam("transform", ANARI_FLOAT32_MAT4, &xfm);

      math::mat4 prevXfm = xfm;
      m_hasPrevTransform = getParam("motion.transform",
                                    ANARI_FLOAT32_MAT4, &prevXfm);
      
      Attributes attributes;
      for (int i=0;i<attributes.count;i++)
        attributes.values[i] = math::float4(NAN);
      assert(attributes.count == 5);
      getParam("attribute0",ANARI_FLOAT32_VEC4, (void *)&attributes.values[0]);
      getParam("attribute1",ANARI_FLOAT32_VEC4, (void *)&attributes.values[1]);
      getParam("attribute2",ANARI_FLOAT32_VEC4, (void *)&attributes.values[2]);
      getParam("attribute3",ANARI_FLOAT32_VEC4, (void *)&attributes.values[3]);
      getParam("color",     ANARI_FLOAT32_VEC4, (void *)&attributes.values[4]);
      m_id = getParam("id",~0);
    
      if (isnan(attributes.values[0].x) &&
          isnan(attributes.values[1].x) &&
          isnan(attributes.values[2].x) &&
          isnan(attributes.values[3].x) &&
          isnan(attributes.values[4].x)) {
        // no attributes worth storing ....
        if (this->attributes) {
          delete this->attributes;
          this->attributes = 0;
        }
      } else {
        // we *do* have to store attributes      
        if (!this->attributes) this->attributes = new Attributes;
        *this->attributes = attributes;
      }
    
      m_transform     = xfm;
      m_prevTransform = prevXfm;
      m_previousTransformValid = m_transformValid;
      const owl::common::affine3f nativeXfm(
        owl::common::LinearSpace3f(
          owl::common::vec3f(xfm[0].x, xfm[0].y, xfm[0].z),
          owl::common::vec3f(xfm[1].x, xfm[1].y, xfm[1].z),
          owl::common::vec3f(xfm[2].x, xfm[2].y, xfm[2].z)),
        owl::common::vec3f(xfm[3].x, xfm[3].y, xfm[3].z));
      m_transformValid = native::usableInstanceTransform(nativeXfm);
      m_previousGroup = m_group.ptr;
      m_group = getParamObject<Group>("group");
    }

    void Instance::finalize()
    {
      if (!m_group)
        reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");
      if (!m_transformValid)
        reportMessage(ANARI_SEVERITY_WARNING,
                      "omitting ANARIInstance: transform requires a finite inverse");
    }

    void Instance::markFinalized()
    {
      if (m_group.ptr != m_previousGroup
          || m_transformValid != m_previousTransformValid)
        deviceState()->markStructuralSceneChanged();
      else
        deviceState()->markSceneChanged();
      Object::markFinalized();
    }

    bool Instance::isValid() const
    {
      return m_group && m_transformValid;
    }

    const Group *Instance::group() const
    {
      return m_group.ptr;
    }

    void Instance::writeTransform(BNTransform *out) const
    {
      out->l.vx = (const bn_float3&)m_transform[0];
      out->l.vy = (const bn_float3&)m_transform[1];
      out->l.vz = (const bn_float3&)m_transform[2];
      out->p    = (const bn_float3&)m_transform[3];
    };
  
    void Instance::writeMotionDelta(BNTransform *out) const
    {
      const math::mat4 delta = m_hasPrevTransform
        ? math::mul(m_prevTransform, math::inverse(m_transform))
        : math::mat4(anari::math::identity);
      out->l.vx = (const bn_float3&)delta[0];
      out->l.vy = (const bn_float3&)delta[1];
      out->l.vz = (const bn_float3&)delta[2];
      out->p    = (const bn_float3&)delta[3];
    }
  
    box3 Instance::bounds() const
    {
      math::mat4 xfm = m_transform;

      box3 result = group()->bounds();

      math::float4 lower(result.lower, 1.f);
      math::float4 upper(result.upper, 1.f);

      lower = mul(xfm, lower);
      upper = mul(xfm, upper);

      result.lower = math::float3(lower.x, lower.y, lower.z);
      result.upper = math::float3(upper.x, upper.y, upper.z);

      return result;
    }

  }
}

BARNEY_ANARI_TYPEFOR_DEFINITION(BARNEY_NS::anari::Instance *);
