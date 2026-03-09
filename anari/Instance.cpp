// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "Instance.h"
// anari
#include <anari/anari_cpp/ext/linalg.h>

namespace barney_device {

  Instance::Instance(BarneyGlobalState *s) : Object(ANARI_INSTANCE, s) {}

  Instance::~Instance()
  {
    if (attributes) delete attributes;
  }

  void Instance::commitParameters()
  {
    math::mat4 xfm = anari::math::identity;
    getParam("transform", ANARI_FLOAT32_MAT4, &xfm);

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
    
    m_transform = xfm;
    m_previousGroup = m_group.ptr;
    m_group = getParamObject<Group>("group");
  }

  void Instance::finalize()
  {
    if (!m_group)
      reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");
  }

  void Instance::markFinalized()
  {
    if (m_group.ptr != m_previousGroup)
      deviceState()->markStructuralSceneChanged();
    else
      deviceState()->markSceneChanged();
    Object::markFinalized();
  }

  bool Instance::isValid() const
  {
    return m_group;
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

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Instance *);
