// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "Geometry.h"
#include "common.h"
// std
#include <cassert>
#include <iostream>
#include <numeric>

namespace barney_device {

  // Helper functions ///////////////////////////////////////////////////////////

  static void addAttribute(BNGeom geom,
                           BNContext context,
                           int slot,
                           const helium::IntrusivePtr<Array1D> &attribute,
                           const std::string &name,
                           Object *object)
  {
    if (!attribute)
      return;

    if (BNData attr = makeBarneyData(context, slot, attribute, object); attr)
      bnSetAndRelease(geom, name.c_str(), attr);
  }

  // Base Geometry definitions //////////////////////////////////////////////////

  Geometry::Geometry(BarneyGlobalState *s)
    : Object(ANARI_GEOMETRY, s)
  {
  }

  Geometry::~Geometry()
  {
    BANARI_TRACK_LEAKS(std::cout << "#banari::Geometry is dying" << std::endl);
  }

  Geometry *Geometry::createInstance(std::string_view subtype,
                                     BarneyGlobalState *s)
  {
    if (subtype == "sphere")
      return new Sphere(s);
    if (subtype == "isosurface")
      return new IsoSurface(s);
    if (subtype == "cylinder")
      return new Cylinder(s);
    if (subtype == "cone")
      return new Cone(s);
    if (subtype == "curve")
      return new Curve(s);
    if (subtype == "quad")
      return new Quad(s);
    if (subtype == "triangle")
      return new Triangle(s);
    return (Geometry *)new UnknownObject(ANARI_GEOMETRY, subtype, s);
  }

  void Geometry::setAttributes(BNGeom geom)
  {
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

    bnSet4f(geom,
            "attribute0",
            m_constantAttributes[0].x,
            m_constantAttributes[0].y,
            m_constantAttributes[0].z,
            m_constantAttributes[0].w);
    bnSet4f(geom,
            "attribute1",
            m_constantAttributes[1].x,
            m_constantAttributes[1].y,
            m_constantAttributes[1].z,
            m_constantAttributes[1].w);
    bnSet4f(geom,
            "attribute2",
            m_constantAttributes[2].x,
            m_constantAttributes[2].y,
            m_constantAttributes[2].z,
            m_constantAttributes[2].w);
    bnSet4f(geom,
            "attribute3",
            m_constantAttributes[3].x,
            m_constantAttributes[3].y,
            m_constantAttributes[3].z,
            m_constantAttributes[3].w);
    bnSet4f(geom,
            "color",
            m_constantAttributes[4].x,
            m_constantAttributes[4].y,
            m_constantAttributes[4].z,
            m_constantAttributes[4].w);

    addAttribute(geom, context, slot,
                 m_vertexAttributes[0], "vertex.attribute0", this);
    addAttribute(geom, context, slot,
                 m_vertexAttributes[1], "vertex.attribute1", this);
    addAttribute(geom, context, slot,
                 m_vertexAttributes[2], "vertex.attribute2", this);
    addAttribute(geom, context, slot,
                 m_vertexAttributes[3], "vertex.attribute3", this);
    addAttribute(geom, context, slot,
                 m_vertexAttributes[4], "vertex.color", this);

    addAttribute(geom, context, slot,
                 m_primitiveAttributes[0], "primitive.attribute0", this);
    addAttribute(geom, context, slot,
                 m_primitiveAttributes[1], "primitive.attribute1", this);
    addAttribute(geom, context, slot,
                 m_primitiveAttributes[2], "primitive.attribute2", this);
    addAttribute(geom, context, slot,
                 m_primitiveAttributes[3], "primitive.attribute3", this);
    addAttribute(geom, context, slot,
                 m_primitiveAttributes[4], "primitive.color", this);
  }

  void Triangle::setAttributes(BNGeom geom)
  {
    Geometry::setAttributes(geom);
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

    addAttribute(geom, context, slot,
                 m_faceVaryingAttributes[0],
                 "faceVarying.attribute0", this);
    addAttribute(geom, context, slot,
                 m_faceVaryingAttributes[1],
                 "faceVarying.attribute1", this);
    addAttribute(geom, context, slot,
                 m_faceVaryingAttributes[2],
                 "faceVarying.attribute2", this);
    addAttribute(geom, context, slot,
                 m_faceVaryingAttributes[3],
                 "faceVarying.attribute3", this);
    addAttribute(geom, context, slot,
                 m_faceVaryingAttributes[4],
                 "faceVarying.color", this);
    addAttribute(geom, context, slot,
                 m_faceVaryingAttributes[5],
                 "faceVarying.normal", this);
  }

  void Geometry::commitParameters()
  {
    math::float4 invalidAttr(NAN, NAN, NAN, NAN);
    m_constantAttributes[0] = getParam<math::float4>("attribute0", invalidAttr);
    m_constantAttributes[1] = getParam<math::float4>("attribute1", invalidAttr);
    m_constantAttributes[2] = getParam<math::float4>("attribute2", invalidAttr);
    m_constantAttributes[3] = getParam<math::float4>("attribute3", invalidAttr);
    m_constantAttributes[4] = getParam<math::float4>("color", invalidAttr);

    m_primitiveAttributes[0] = getParamObject<Array1D>("primitive.attribute0");
    m_primitiveAttributes[1] = getParamObject<Array1D>("primitive.attribute1");
    m_primitiveAttributes[2] = getParamObject<Array1D>("primitive.attribute2");
    m_primitiveAttributes[3] = getParamObject<Array1D>("primitive.attribute3");
    m_primitiveAttributes[4] = getParamObject<Array1D>("primitive.color");

    m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
    m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
    m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
    m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
    m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");
  }

  void Geometry::markFinalized()
  {
    deviceState()->markStructuralSceneChanged();
    Object::markFinalized();
  }

  // Subtypes ///////////////////////////////////////////////////////////////////

  // Isosurface //

  IsoSurface::IsoSurface(BarneyGlobalState *s)
    : Geometry(s),
      m_field(this),
      m_isoValues(this)
  {}

  void IsoSurface::commitParameters()
  {
    Geometry::commitParameters();
    m_isoValue = getParam<float>("isovalue", (float)NAN);
    m_isoValues = getParamObject<Array1D>("isovalues");
    m_field = getParamObject<SpatialField>("field");
  }

  void IsoSurface::finalize()
  {
    if (!m_field) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'field' on isosurface geometry");
      return;
    }
    if (isnan(m_isoValue) && !m_isoValues) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'isovalue' or 'isovalues' on isosurface geometry");
      return;
    }
  }

  void IsoSurface::setBarneyParameters(BNGeom geom)
  {
    bnSet1f(geom, "isoValue", m_isoValue);
    if (m_isoValues)
      bnSetData(geom, "isoValues", m_isoValues->barneyData());
    else
      bnSetData(geom, "isoValues", (BNData)nullptr);
    bnSetObject(geom, "scalarField", m_field->getBarneyScalarField());

    setAttributes(geom);
  }

  bool IsoSurface::isValid() const
  {
    return m_field && (m_isoValues || !isnan(m_isoValue));
  }

  const char *IsoSurface::bnSubtype() const
  {
    return "iso_surface";
  }

  box3 IsoSurface::bounds() const
  {
    if (!isValid())
      return {};

    return m_field->bounds();
  }

  
  // Sphere //

  Sphere::Sphere(BarneyGlobalState *s)
    : Geometry(s), m_index(this), m_vertexPosition(this), m_vertexRadius(this)
  {}

  void Sphere::commitParameters()
  {
    Geometry::commitParameters();
    m_index = getParamObject<Array1D>("primitive.index");
    m_vertexPosition = getParamObject<Array1D>("vertex.position");
    m_vertexRadius = getParamObject<Array1D>("vertex.radius");
    m_globalRadius = getParam<float>("radius", 0.01f);
    m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
    m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
    m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
    m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
    m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");
  }

  void Sphere::finalize()
  {
    if (!m_vertexPosition) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'vertex.position' on sphere geometry");
      return;
    }
  }

  void Sphere::setBarneyParameters(BNGeom geom)
  {
    bnSetData(geom, "origins", m_vertexPosition->barneyData());
    if (m_vertexRadius)
      bnSetData(geom, "radii", m_vertexRadius->barneyData());
    else
      bnSet1f(geom, "radius", m_globalRadius);

    setAttributes(geom);
  }

  bool Sphere::isValid() const
  {
    return m_vertexPosition;
  }

  const char *Sphere::bnSubtype() const
  {
    return "spheres";
  }

  box3 Sphere::bounds() const
  {
    if (!isValid())
      return {};

    box3 result;
    if (m_index) {
      std::for_each(m_index->beginAs<uint32_t>(),
                    m_index->beginAs<uint32_t>() + m_index->totalSize(),
                    [&](uint32_t index) {
                      math::float3 v = *(m_vertexPosition->beginAs<math::float3>() + index);
                      float r = m_vertexRadius ? *(m_vertexRadius->beginAs<float>() + index)
                        : m_globalRadius;
                      result.insert(math::float3{v.x - r, v.y - r, v.z - r});
                      result.insert(math::float3{v.x + r, v.y + r, v.z + r});
                    });
    } else {
      for (size_t i = 0; i < m_vertexPosition->totalSize(); ++i) {
        math::float3 v = *(m_vertexPosition->beginAs<math::float3>() + i);
        float r = m_vertexRadius ? *(m_vertexRadius->beginAs<float>() + i)
          : m_globalRadius;
        result.insert(math::float3{v.x - r, v.y - r, v.z - r});
        result.insert(math::float3{v.x + r, v.y + r, v.z + r});
      }
    }
    return result;
  }

  // Cylinder //

  Cylinder::Cylinder(BarneyGlobalState *s)
    : Geometry(s), m_index(this), m_radius(this), m_vertexPosition(this)
  {}

  void Cylinder::commitParameters()
  {
    Geometry::commitParameters();
    m_index = getParamObject<Array1D>("primitive.index");
    m_radius = getParamObject<Array1D>("primitive.radius");
    m_vertexPosition = getParamObject<Array1D>("vertex.position");
    m_globalRadius = getParam<float>("radius", 1.f);
  }

  void Cylinder::finalize()
  {
    if (!m_vertexPosition) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'vertex.position' on cylinder geometry");
      return;
    }

    m_generatedIndices.clear();
    if (!m_index) {
      m_generatedIndices.resize(m_vertexPosition->totalSize() / 2);
      for (size_t i = 0; i < m_generatedIndices.size(); ++i) {
        m_generatedIndices[i] = math::uint2((uint32_t)i * 2, (uint32_t)i * 2 + 1);
      }
    }

    m_generatedRadii.clear();
    if (!m_radius) {
      int numIndices =
        m_index ? (int)m_index->size() : (int)m_generatedIndices.size();
      
      m_generatedRadii.resize(numIndices);
      for (size_t i = 0; i < numIndices; ++i) {
        m_generatedRadii[i] = m_globalRadius;
      }
    }
  }

  void Cylinder::setBarneyParameters(BNGeom geom)
  {
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

    int numIndices =
      m_index ? (int)m_index->size() : (int)m_generatedIndices.size();
    const bn_int2 *indices = m_index ? (const bn_int2 *)m_index->data()
      : (const bn_int2 *)m_generatedIndices.data();
    const float *radii = m_radius ? (const float *)m_radius->data()
      : (const float *)m_generatedRadii.data();

    BNData _indices = bnDataCreate(context, slot, BN_INT2, numIndices, indices);
    bnSetAndRelease(geom, "indices", _indices);

    BNData _radii = bnDataCreate(context, slot, BN_FLOAT, numIndices, radii);
    bnSetAndRelease(geom, "radii", _radii);

    bnSetData(geom, "vertices", m_vertexPosition->barneyData());

    setAttributes(geom);
  }

  bool Cylinder::isValid() const
  {
    return m_vertexPosition;
  }

  const char *Cylinder::bnSubtype() const
  {
    return "cylinders";
  }

  box3 Cylinder::bounds() const
  {
    if (!isValid())
      return {};

    box3 result;
    if (m_index) {
      for (size_t i = 0; i < m_index->totalSize(); ++i) {
        math::uint2 index = *(m_index->beginAs<math::uint2>() + i);
        math::float3 v1 = *(m_vertexPosition->beginAs<math::float3>() + index.x);
        math::float3 v2 = *(m_vertexPosition->beginAs<math::float3>() + index.y);
        float r = m_radius ? *(m_radius->beginAs<float>() + i) : m_globalRadius;
        result.insert(math::float3{v1.x - r, v1.y - r, v1.z - r});
        result.insert(math::float3{v1.x + r, v1.y + r, v1.z + r});
        result.insert(math::float3{v2.x - r, v2.y - r, v2.z - r});
        result.insert(math::float3{v2.x + r, v2.y + r, v2.z + r});
      }
    } else {
      for (size_t i = 0; i < m_vertexPosition->totalSize(); i += 2) {
        math::float3 v1 = *(m_vertexPosition->beginAs<math::float3>() + i);
        math::float3 v2 = *(m_vertexPosition->beginAs<math::float3>() + i + 1);
        float r =
          m_radius ? *(m_radius->beginAs<float>() + i / 2) : m_globalRadius;
        result.insert(math::float3{v1.x - r, v1.y - r, v1.z - r});
        result.insert(math::float3{v1.x + r, v1.y + r, v1.z + r});
        result.insert(math::float3{v2.x - r, v2.y - r, v2.z - r});
        result.insert(math::float3{v2.x + r, v2.y + r, v2.z + r});
      }
    }
    return result;
  }

  // Cone //

  Cone::Cone(BarneyGlobalState *s)
    : Geometry(s), m_index(this), m_vertexPosition(this), m_vertexRadius(this)
  {}

  void Cone::commitParameters()
  {
    Geometry::commitParameters();
    m_index = getParamObject<Array1D>("primitive.index");
    m_vertexPosition = getParamObject<Array1D>("vertex.position");
    m_vertexRadius = getParamObject<Array1D>("vertex.radius");
  }

  void Cone::finalize()
  {
    if (!m_vertexPosition) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'vertex.position' on cone geometry");
      return;
    }

    if (!m_vertexRadius) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'vertex.radius' on cone geometry");
      return;
    }

    m_generatedIndices.clear();
    if (!m_index) {
      m_generatedIndices.resize(m_vertexPosition->totalSize() / 2);
      for (size_t i = 0; i < m_generatedIndices.size(); ++i) {
        m_generatedIndices[i] = math::uint2((uint32_t)i * 2, (uint32_t)i * 2 + 1);
      }
    }
  }

  void Cone::setBarneyParameters(BNGeom geom)
  {
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

    int numIndices =
      m_index ? (int)m_index->size() : (int)m_generatedIndices.size();
    const bn_int2 *indices = m_index ? (const bn_int2 *)m_index->data()
      : (const bn_int2 *)m_generatedIndices.data();
    BNData _indices = bnDataCreate(context, slot, BN_INT2, numIndices, indices);
    bnSetAndRelease(geom, "indices", _indices);

    bnSetData(geom, "vertices", m_vertexPosition->barneyData());
    bnSetData(geom, "radii", m_vertexRadius->barneyData());

    setAttributes(geom);
  }

  bool Cone::isValid() const
  {
    return m_vertexPosition && m_vertexRadius;
  }

  const char *Cone::bnSubtype() const
  {
    return "cones";
  }

  box3 Cone::bounds() const
  {
    if (!isValid())
      return {};

    box3 result;
    if (m_index) {
      for (size_t i = 0; i < m_index->totalSize(); ++i) {
        math::uint2 index = *(m_index->beginAs<math::uint2>() + i);
        math::float3 v1 = *(m_vertexPosition->beginAs<math::float3>() + index.x);
        math::float3 v2 = *(m_vertexPosition->beginAs<math::float3>() + index.y);
        float r1 = *(m_vertexRadius->beginAs<float>() + index.x);
        float r2 = *(m_vertexRadius->beginAs<float>() + index.y);
        result.insert(math::float3{v1.x - r1, v1.y - r1, v1.z - r1});
        result.insert(math::float3{v1.x + r1, v1.y + r1, v1.z + r1});
        result.insert(math::float3{v2.x - r2, v2.y - r2, v2.z - r2});
        result.insert(math::float3{v2.x + r2, v2.y + r2, v2.z + r2});
      }
    } else {
      for (size_t i = 0; i < m_vertexPosition->totalSize(); i += 2) {
        math::float3 v1 = *(m_vertexPosition->beginAs<math::float3>() + i);
        math::float3 v2 = *(m_vertexPosition->beginAs<math::float3>() + i + 1);
        float r1 = *(m_vertexRadius->beginAs<float>() + i);
        float r2 = *(m_vertexRadius->beginAs<float>() + i + 1);
        result.insert(math::float3{v1.x - r1, v1.y - r1, v1.z - r1});
        result.insert(math::float3{v1.x + r1, v1.y + r1, v1.z + r1});
        result.insert(math::float3{v2.x - r2, v2.y - r2, v2.z - r2});
        result.insert(math::float3{v2.x + r2, v2.y + r2, v2.z + r2});
      }
    }
    return result;
  }

  // Curve //

  Curve::Curve(BarneyGlobalState *s)
    : Geometry(s), m_index(this), m_vertexPosition(this), m_vertexRadius(this)
  {}

  void Curve::commitParameters()
  {
    Geometry::commitParameters();

    m_index = getParamObject<Array1D>("primitive.index");
    m_vertexPosition = getParamObject<Array1D>("vertex.position");
    m_vertexRadius = getParamObject<Array1D>("vertex.radius");
    m_globalRadius = getParam<float>("radius", 0.01f);

    if (!m_vertexPosition) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'vertex.position' on curve geometry");
      return;
    }
  }

  void Curve::finalize()
  {
    if (!m_vertexPosition) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'vertex.position' on curve geometry");
      return;
    }
  }

  void Curve::setBarneyParameters(BNGeom geom)
  {
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

    int numVertices = (int)m_vertexPosition->totalSize();
    const bn_float3 *in_vertex = (const bn_float3 *)m_vertexPosition->data();
  
    std::vector<math::float4> vertex(numVertices);
    if (m_vertexRadius && m_vertexRadius->totalSize() > 0) {
      assert(m_vertexRadius->totalSize() == m_vertexPosition->totalSize());
      const float *in_radius = (const float *)m_vertexRadius->data();
      for (int i = 0; i < numVertices; i++)
        vertex[i] = math::float4(
                                 in_vertex[i].x, in_vertex[i].y, in_vertex[i].z, in_radius[i]);
    } else {
      // Use global radius for all vertices
      for (int i = 0; i < numVertices; i++)
        vertex[i] = math::float4(
                                 in_vertex[i].x, in_vertex[i].y, in_vertex[i].z, m_globalRadius);
    }

    BNData vertices =
      bnDataCreate(context, slot, BN_FLOAT4, numVertices, vertex.data());
    bnSetData(geom, "vertices", vertices);

    int numIndices = (int)m_index->totalSize();
    std::vector<math::int2> index(numIndices);
    const int *in_index = (const int *)m_index->data();
    for (int i = 0; i < numIndices; i++) {
      index[i] = math::int2(in_index[i], in_index[i] + 1);
    }

#if 0
    // dump geometry to create test cases:
    std::cout << "std::vector<vec4f> vertices = {" << std::endl;
    for (auto v : vertex)
      std::cout << "{"<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<"}," << std::endl;
    std::cout << "};" << std::endl;
    std::cout << "std::vector<vec2i> index = {" << std::endl;
    for (auto v : index)
      std::cout << "{"<<v.x<<","<<v.y<<"}," << std::endl;
    std::cout << "};" << std::endl;
#endif

    BNData indices = bnDataCreate(
                                  context, slot, BN_INT2, index.size(), (const int *)index.data());
    bnSetData(geom, "indices", indices);

    setAttributes(geom);
  }

  bool Curve::isValid() const
  {
    return m_vertexPosition;
  }

  const char *Curve::bnSubtype() const
  {
    return "capsules";
  }

  box3 Curve::bounds() const
  {
    if (!isValid())
      return {};

    box3 result;
    for (size_t i = 0; i < m_vertexPosition->totalSize(); ++i) {
      math::float3 v = *(m_vertexPosition->beginAs<math::float3>() + i);
      float r = *(m_vertexRadius->beginAs<float>() + i);
      result.insert(math::float3{v.x - r, v.y - r, v.z - r});
      result.insert(math::float3{v.x + r, v.y + r, v.z + r});
    }
    return result;
  }

  // Quad //

  Quad::Quad(BarneyGlobalState *s)
    : Geometry(s), m_index(this), m_vertexPosition(this), m_vertexNormal(this)
  {}

  void Quad::commitParameters()
  {
    Geometry::commitParameters();
    m_index = getParamObject<Array1D>("primitive.index");
    m_vertexPosition = getParamObject<Array1D>("vertex.position");
    m_vertexNormal = getParamObject<Array1D>("vertex.normal");
    m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
    m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
    m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
    m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
    m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");
  }

  void Quad::finalize()
  {
    if (!m_vertexPosition) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'vertex.position' on triangle geometry");
      return;
    }

    m_generatedIndices.clear();
    if (!m_index) {
      size_t numQuads = m_vertexPosition->totalSize() / 4;
      for (size_t i = 0; i < numQuads; ++i) {
        // tri1
        m_generatedIndices.push_back(int(i * 4));
        m_generatedIndices.push_back(int(i * 4 + 1));
        m_generatedIndices.push_back(int(i * 4 + 2));
        // tri2
        m_generatedIndices.push_back(int(i * 4));
        m_generatedIndices.push_back(int(i * 4 + 2));
        m_generatedIndices.push_back(int(i * 4 + 3));
      }
    } else {
      for (size_t i = 0; i < m_index->totalSize(); ++i) {
        math::uint4 index = *(m_index->beginAs<math::uint4>() + i);
        // tri1
        m_generatedIndices.push_back(int(index.x));
        m_generatedIndices.push_back(int(index.y));
        m_generatedIndices.push_back(int(index.z));
        // tri2
        m_generatedIndices.push_back(int(index.x));
        m_generatedIndices.push_back(int(index.z));
        m_generatedIndices.push_back(int(index.w));
      }
    }
  }

  bool Quad::isValid() const
  {
    return m_vertexPosition;
  }

  void Quad::setBarneyParameters(BNGeom geom)
  {
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

    int numIndices = (int)(m_generatedIndices.size() / 3);
    const bn_int3 *indices = (const bn_int3 *)m_generatedIndices.data();
    BNData _indices = bnDataCreate(context, slot, BN_INT3, numIndices, indices);
    bnSetAndRelease(geom, "indices", _indices);

    bnSetData(geom, "vertices", m_vertexPosition->barneyData());
    if (m_vertexNormal)
      bnSetData(geom, "normals", m_vertexNormal->barneyData());

    setAttributes(geom);
  }

  const char *Quad::bnSubtype() const
  {
    return "triangles";
  }

  box3 Quad::bounds() const
  {
    if (!isValid())
      return {};

    box3 result;
    if (m_index) {
      std::for_each(m_index->beginAs<math::uint4>(),
                    m_index->beginAs<math::uint4>() + m_index->totalSize(),
                    [&](math::uint4 index) {
                      math::float3 v1 =
                        *(m_vertexPosition->beginAs<math::float3>() + index.x);
                      math::float3 v2 =
                        *(m_vertexPosition->beginAs<math::float3>() + index.y);
                      math::float3 v3 =
                        *(m_vertexPosition->beginAs<math::float3>() + index.z);
                      math::float3 v4 =
                        *(m_vertexPosition->beginAs<math::float3>() + index.w);
                      result.insert(v1);
                      result.insert(v2);
                      result.insert(v3);
                      result.insert(v4);
                    });
    } else {
      std::for_each(m_vertexPosition->beginAs<math::float3>(),
                    m_vertexPosition->beginAs<math::float3>()
                    + m_vertexPosition->totalSize(),
                    [&](math::float3 v) { result.insert(v); });
    }
    return result;
  }

  // Triangle //

  Triangle::Triangle(BarneyGlobalState *s)
    : Geometry(s),
      m_index(this),
      m_vertexPosition(this),
      m_vertexNormal(this)
  {
  }

  void Triangle::commitParameters()
  {
    Geometry::commitParameters();
    m_index = getParamObject<Array1D>("primitive.index");
    m_vertexPosition = getParamObject<Array1D>("vertex.position");
    m_vertexNormal = getParamObject<Array1D>("vertex.normal");
    m_vertexAttributes[0] = getParamObject<Array1D>("vertex.attribute0");
    m_vertexAttributes[1] = getParamObject<Array1D>("vertex.attribute1");
    m_vertexAttributes[2] = getParamObject<Array1D>("vertex.attribute2");
    m_vertexAttributes[3] = getParamObject<Array1D>("vertex.attribute3");
    m_vertexAttributes[4] = getParamObject<Array1D>("vertex.color");
    m_faceVaryingAttributes[0] = getParamObject<Array1D>("faceVarying.attribute0");
    m_faceVaryingAttributes[1] = getParamObject<Array1D>("faceVarying.attribute1");
    m_faceVaryingAttributes[2] = getParamObject<Array1D>("faceVarying.attribute2");
    m_faceVaryingAttributes[3] = getParamObject<Array1D>("faceVarying.attribute3");
    m_faceVaryingAttributes[4] = getParamObject<Array1D>("faceVarying.color");
    m_faceVaryingAttributes[5] = getParamObject<Array1D>("faceVarying.normal");
  }

  void Triangle::finalize()
  {
    if (!m_vertexPosition) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'vertex.position' on triangle geometry");
      return;
    }

    m_generatedIndices.clear();
    if (!m_index) {
      m_generatedIndices.resize(m_vertexPosition->totalSize());
      std::iota(m_generatedIndices.begin(), m_generatedIndices.end(), 0);
    }
  }

  bool Triangle::isValid() const
  {
    return m_vertexPosition;
  }

  void Triangle::setBarneyParameters(BNGeom geom)
  {
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

    int numIndices =
      m_index ? (int)m_index->size() : (int)(m_generatedIndices.size() / 3);
    const bn_int3 *indices = m_index ? (const bn_int3 *)m_index->data()
      : (const bn_int3 *)m_generatedIndices.data();
    BNData _indices = bnDataCreate(context, slot, BN_INT3, numIndices, indices);
    bnSetAndRelease(geom, "indices", _indices);

    bnSetData(geom, "vertices", m_vertexPosition->barneyData());
    if (m_vertexNormal)
      bnSetData(geom, "normals", m_vertexNormal->barneyData());

    setAttributes(geom);
  }

  const char *Triangle::bnSubtype() const
  {
    return "triangles";
  }

  box3 Triangle::bounds() const
  {
    if (!isValid())
      return {};

    box3 result;
    if (m_index) {
      std::for_each(m_index->beginAs<math::uint3>(),
                    m_index->beginAs<math::uint3>() + m_index->totalSize(),
                    [&](math::uint3 index) {
                      math::float3 v1 =
                        *(m_vertexPosition->beginAs<math::float3>() + index.x);
                      math::float3 v2 =
                        *(m_vertexPosition->beginAs<math::float3>() + index.y);
                      math::float3 v3 =
                        *(m_vertexPosition->beginAs<math::float3>() + index.z);
                      result.insert(v1);
                      result.insert(v2);
                      result.insert(v3);
                    });
    } else {
      std::for_each(m_vertexPosition->beginAs<math::float3>(),
                    m_vertexPosition->beginAs<math::float3>()
                    + m_vertexPosition->totalSize(),
                    [&](math::float3 v) { result.insert(v); });
    }
    return result;
  }

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Geometry *);
