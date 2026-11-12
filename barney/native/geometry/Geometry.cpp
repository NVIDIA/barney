// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/geometry/Geometry.h"
#include "native/ModelSlot.h"
#include "native/Context.h"

#include "native/geometry/Triangles.h"
#include "native/geometry/Spheres.h"
#include "native/geometry/Cones.h"
#include "native/geometry/Cylinders.h"
#include "native/geometry/Capsules.h"
#include "native/geometry/IsoSurface.h"

namespace BARNEY_NS {
  namespace native {

    Geometry::PLD *Geometry::getPLD(Device *device)
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }

    Geometry::SP Geometry::create(Context *context,
                                  DevGroup::SP devices,
                                  const std::string &type)
    {
      if (type == "spheres")
        return std::make_shared<Spheres>(context,devices);
      if (type == "iso_surface")
        return std::make_shared<IsoSurface>(context,devices);
      if (type == "cones")
        return std::make_shared<Cones>(context,devices);
      if (type == "cylinders")
        return std::make_shared<Cylinders>(context,devices);
      if (type == "capsules")
        return std::make_shared<Capsules>(context,devices);
      if (type == "triangles")
        return std::make_shared<Triangles>(context,devices);
    
      context->warn_unsupported_object("Geometry",type);
      return {};
    }

    Geometry::Geometry(Context *context,
                       DevGroup::SP devices)
      : Object(context),
        devices(devices)
    {
      perLogical.resize(devices->numLogical);
    }

    bool Geometry::set1i(const std::string &member,
                         const int   &value) 
    {
      if (member == "userID") {
        userID = value;
        return true; 
      } 
    
      return false;
    }
  
    Geometry::~Geometry()
    {
      BN_TRACK_LEAKS(std::cout << "#barney: ~Geometry deconstructing"
                     << std::endl);
      material = {};
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        for (auto &geom : pld->triangleGeoms)
          if (geom) {
            device->rtc->freeGeom(geom);
            geom = 0;
          }
        for (auto &geom : pld->userGeoms)
          if (geom) {
            device->rtc->freeGeom(geom);
            geom = 0;
          }
      }
    }

    HostMaterial::SP Geometry::getMaterial() const
    {
      assert(this->material);
      return this->material;
    }
  
    void Geometry::setMaterial(HostMaterial::SP mat)
    {
      if (mat) {
        assert(mat->hasBeenCommittedAtLeastOnce);
      }
      this->material = mat;
    }
    
    void Geometry::writeDD(Geometry::DD &dd,
                           Device *device)
    {
      setAttributesOn(dd,device);
      dd.userID     = userID;
      dd.attributes = attributes.getDD(device);
      dd.materialID = getMaterial()->materialID;
    }  
  
    void Geometry::setAttributesOn(Geometry::DD &dd,
                                   Device *device)
    {
      dd.attributes = attributes.getDD(device);
      dd.materialID = material->materialID;  
    }
  
    bool Geometry::set1f(const std::string &member, const float &value)
    {
      return false;
    }
  
    bool Geometry::set3f(const std::string &member, const vec3f &value)
    {
      if (member == "attribute0") {
        attributes.attribute[0].constant = vec4f(value.x,value.y,value.z,1.f);
        return true;
      }
      if (member == "attribute1") {
        attributes.attribute[1].constant = vec4f(value.x,value.y,value.z,1.f);
        return true;
      }
      if (member == "attribute2") {
        attributes.attribute[2].constant = vec4f(value.x,value.y,value.z,1.f);
        return true;
      }
      if (member == "attribute3") {
        attributes.attribute[3].constant = vec4f(value.x,value.y,value.z,1.f);
        return true;
      }
      if (member == "color") {
        attributes.colorAttribute.constant = vec4f(value.x,value.y,value.z,1.f);
        return true;
      }
    
      return false;
    }
  
    bool Geometry::set4f(const std::string &member, const vec4f &value)
    {
      if (member == "attribute0") {
        attributes.attribute[0].constant = value;
        return true;
      }
      if (member == "attribute1") {
        attributes.attribute[1].constant = value;
        return true;
      }
      if (member == "attribute2") {
        attributes.attribute[2].constant = value;
        return true;
      }
      if (member == "attribute3") {
        attributes.attribute[3].constant = value;
        return true;
      }
      if (member == "color") {
        attributes.colorAttribute.constant = value;
        return true;
      }
    
      return false;
    }
  
    bool Geometry::setData(const std::string &member, const Data::SP &value)
    {
      // ----------- per prim -----------
      if (member == "primitive.attribute0") {
        attributes.attribute[0].perPrim = value->as<PODData>();
        return true;
      }
      if (member == "primitive.attribute1") {
        attributes.attribute[1].perPrim = value->as<PODData>();
        return true;
      }
      if (member == "primitive.attribute2") {
        attributes.attribute[2].perPrim = value->as<PODData>();
        return true;
      }
      if (member == "primitive.attribute3") {
        attributes.attribute[3].perPrim = value->as<PODData>();
        return true;
      }
      if (member == "primitive.color") {
        attributes.colorAttribute.perPrim = value->as<PODData>();
        return true;
      }
      if (member == "primitive.normal") {
        attributes.normalAttribute.perPrim = value->as<PODData>();
        return true;
      }

      // ----------- per vertex -----------
      if (member == "vertex.attribute0") {
        attributes.attribute[0].perVertex = value ? value->as<PODData>() : nullptr;
        return true;
      }
      if (member == "vertex.attribute1") {
        attributes.attribute[1].perVertex = value->as<PODData>();
        return true;
      }
      if (member == "vertex.attribute2") {
        attributes.attribute[2].perVertex = value->as<PODData>();
        return true;
      }
      if (member == "vertex.attribute3") {
        attributes.attribute[3].perVertex = value->as<PODData>();
        return true;
      }
      if (member == "vertex.color") {
        attributes.colorAttribute.perVertex = value->as<PODData>();
        return true;
      }
      if (member == "vertex.normal") {
        attributes.normalAttribute.perVertex = value ? value->as<PODData>() : nullptr;
        return true;
      }
      if (member == "vertex.tangent") {
        attributes.tangentAttribute.perVertex = value ? value->as<PODData>() : nullptr;
        return true;
      }

      // ----------- per face-varying -----------
      if (member == "faceVarying.attribute0") {
        attributes.attribute[0].faceVarying = value ? value->as<PODData>() : nullptr;
        return true;
      }
      if (member == "faceVarying.attribute1") {
        attributes.attribute[1].faceVarying = value->as<PODData>();
        return true;
      }
      if (member == "faceVarying.attribute2") {
        attributes.attribute[2].faceVarying = value->as<PODData>();
        return true;
      }
      if (member == "faceVarying.attribute3") {
        attributes.attribute[3].faceVarying = value->as<PODData>();
        return true;
      }
      if (member == "faceVarying.color") {
        attributes.colorAttribute.faceVarying = value->as<PODData>();
        return true;
      }
      if (member == "faceVarying.normal") {
        attributes.normalAttribute.faceVarying = value ? value->as<PODData>() : nullptr;
        return true;
      }
      if (member == "faceVarying.tangent") {
        attributes.tangentAttribute.faceVarying = value ? value->as<PODData>() : nullptr;
        return true;
      }
    
      return false;
    }
  
    bool Geometry::setObject(const std::string &member, const Object::SP &value)
    {
      if (member == "material") {
        HostMaterial::SP newMaterial = value->as<HostMaterial>();
        if (value && !newMaterial) {
          throw std::runtime_error("invalid material in geometry::set(\"material\"");
        }
        setMaterial(newMaterial);
        return true;
      }
      return false;
    }

  }
}

