// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Object.h"
#include "SpatialField.h"

namespace BARNEY_NS {
  namespace anari {
    
    struct Geometry : public Object
    {
      Geometry(BarneyGlobalState *s);
      ~Geometry() override;

      static Geometry *createInstance(std::string_view subtype,
                                      BarneyGlobalState *s);

      void commitParameters() override;
      void markFinalized() override;

      virtual const char *bnSubtype() const = 0;
      virtual void setBarneyParameters(BNGeom geom) = 0;
      virtual box3 bounds() const = 0;
      enum class AttributeUpdate { All, MaterialAttributes };
      virtual void setAttributes(BNGeom geom,
                                 AttributeUpdate update = AttributeUpdate::All);

    protected:
      std::array<math::float4, 5>                  m_constantAttributes;
      std::array<helium::IntrusivePtr<Array1D>, 5> m_primitiveAttributes;
      std::array<helium::IntrusivePtr<Array1D>, 5> m_vertexAttributes;
    };

    // Subtypes /////////////////////////////////////////////////////////////////

    struct Sphere : public Geometry
    {
      Sphere(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      const char *bnSubtype() const override;
      box3 bounds() const override;

    private:
      helium::ChangeObserverPtr<Array1D> m_index;
      helium::ChangeObserverPtr<Array1D> m_vertexPosition;
      helium::ChangeObserverPtr<Array1D> m_vertexRadius;
      float m_globalRadius{0.f};
    };

    struct IsoSurface : public Geometry
    {
      IsoSurface(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      const char *bnSubtype() const override;
      box3 bounds() const override;

    private:
      helium::ChangeObserverPtr<Array1D> m_isoValues;
      helium::ChangeObserverPtr<SpatialField> m_field;
      float m_isoValue{NAN};
    };

    struct Cylinder : public Geometry
    {
      Cylinder(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      const char *bnSubtype() const override;
      box3 bounds() const override;

    private:
      helium::ChangeObserverPtr<Array1D> m_index;
      helium::ChangeObserverPtr<Array1D> m_radius;
      helium::ChangeObserverPtr<Array1D> m_vertexPosition;
      float m_globalRadius{0.f};
      std::vector<math::uint2> m_generatedIndices;
      std::vector<float> m_generatedRadii;
    };

    struct Cone : public Geometry
    {
      Cone(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      const char *bnSubtype() const override;
      box3 bounds() const override;

    private:
      helium::ChangeObserverPtr<Array1D> m_index;
      helium::ChangeObserverPtr<Array1D> m_vertexPosition;
      helium::ChangeObserverPtr<Array1D> m_vertexRadius;
      std::vector<math::uint2> m_generatedIndices;
    };

    struct Curve : public Geometry
    {
      Curve(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      const char *bnSubtype() const override;
      box3 bounds() const override;

    private:
      helium::ChangeObserverPtr<Array1D> m_index;
      helium::ChangeObserverPtr<Array1D> m_vertexPosition;
      helium::ChangeObserverPtr<Array1D> m_vertexRadius;
      float m_globalRadius{0.f};
    };

    struct Quad : public Geometry
    {
      Quad(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      const char *bnSubtype() const override;
      box3 bounds() const override;

      void setAttributes(BNGeom geom, AttributeUpdate update) override;

    private:
      helium::ChangeObserverPtr<Array1D> m_index;
      helium::ChangeObserverPtr<Array1D> m_vertexPosition;
      helium::ChangeObserverPtr<Array1D> m_vertexNormal;
      helium::ChangeObserverPtr<Array1D> m_vertexTangent;
      helium::ChangeObserverPtr<Array1D> m_faceVaryingTangent;
      helium::ChangeObserverPtr<Array1D> m_vertexUV;
      helium::ChangeObserverPtr<Array1D> m_faceUV;
      helium::ChangeObserverPtr<Array1D> m_faceNormal;
      std::vector<int> m_generatedIndices;
      bool m_validVertexTangent = false;
      bool m_validFaceTangent = false;
    };

    struct Triangle : public Geometry
    {
      Triangle(BarneyGlobalState *s);
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

      void setBarneyParameters(BNGeom geom) override;
      const char *bnSubtype() const override;
      box3 bounds() const override;

      void setAttributes(BNGeom geom, AttributeUpdate update) override;
    
    private:
      helium::ChangeObserverPtr<Array1D> m_index;
      helium::ChangeObserverPtr<Array1D> m_vertexPosition;
      helium::ChangeObserverPtr<Array1D> m_vertexNormal;
      helium::ChangeObserverPtr<Array1D> m_vertexTangent;
      helium::ChangeObserverPtr<Array1D> m_faceVaryingTangent;
      helium::ChangeObserverPtr<Array1D> m_vertexUV;
      helium::ChangeObserverPtr<Array1D> m_faceUV;
      helium::ChangeObserverPtr<Array1D> m_faceNormal;
      std::array<helium::IntrusivePtr<Array1D>, 6> m_faceVaryingAttributes;
      std::vector<int> m_generatedIndices;
      bool m_validVertexTangent = false;
      bool m_validFaceTangent = false;
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Geometry *, ANARI_GEOMETRY);
