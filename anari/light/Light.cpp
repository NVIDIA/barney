// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "light/Light.h"
#include "light/Directional.h"
#include "light/HDRI.h"
#include "light/Point.h"

namespace barney_device {

Light::Light(BarneyGlobalState *s) : Object(ANARI_LIGHT, s) {}

Light::~Light()
{
  BANARI_TRACK_LEAKS(std::cout << "#banari: ~Light deconstructing"
                     << std::endl);
  if (m_bnLight)
    bnRelease(m_bnLight);
  m_bnLight = nullptr;
}

Light *Light::createInstance(std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "directional")
    return new Directional(s);
  if (subtype == "hdri")
    return new HDRILight(s);
  if (subtype == "point")
    return new PointLight(s);
  else
    return (Light *)new UnknownObject(ANARI_LIGHT, subtype, s);
}

void Light::markFinalized()
{
  deviceState()->markStructuralSceneChanged();
  Object::markFinalized();
}

void Light::commitParameters()
{
  m_color = getParam<math::float3>("color", math::float3(1.f, 1.f, 1.f));
}

void Light::finalize()
{
  setBarneyParameters();
}

BNLight Light::getBarneyLight()
{
  int slot = deviceState()->slot;
  auto context = deviceState()->tether->context;

  if (!m_bnLight)
    m_bnLight = bnLightCreate(context, slot, bnSubtype());
  setBarneyParameters();
  return m_bnLight;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Light *);
